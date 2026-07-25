import asyncio
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form

from src.chat.index import index_chunks
from src.core.config import settings
from src.core.models import IngestRequest, IngestResponse, SourceFormat
from src.delta_engine.engine import run_delta
from src.ingestion.canonicalizer import canonicalize
from src.observability.logging import get_logger, set_correlation_id
from src.reporting.generator import generate_report

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/delta", tags=["Delta"])

import json

def _get_jobs_db_path() -> Path:
    db_path = Path(settings.DATA_DIR).parent / "jobs_db.json"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path

def _load_jobs_from_disk() -> Dict[str, dict]:
    path = _get_jobs_db_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_jobs_to_disk(jobs_dict: Dict[str, dict]):
    path = _get_jobs_db_path()
    path.write_text(json.dumps(jobs_dict, indent=2), encoding="utf-8")

# In-memory job store backed by disk persistence.
# Stores job status and results keyed by job_id.
_jobs: Dict[str, dict] = _load_jobs_from_disk()

# ProcessPoolExecutor for CPU-heavy parsing (OpenCV, Tesseract, RANSAC)
# Max workers = 2 to avoid overwhelming the server; each parse job is single-file.
_executor = ProcessPoolExecutor(max_workers=2)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest, bg_tasks: BackgroundTasks):
    """
    Upload two document revisions and trigger the full delta pipeline.

    The pipeline runs in a background task (non-blocking):
    1. Canonicalize Rev A → List[CanonicalEntity]
    2. Canonicalize Rev B → List[CanonicalEntity]
    3. Run Delta Engine → List[DeltaEntry]
    4. Generate JSON + HTML + RAG chunk reports
    5. Index all chunks into Pinecone under the job_id namespace

    Returns immediately with job_id and status="processing".
    Poll GET /api/v1/delta/{job_id}/status to check completion.
    """
    job_id = str(uuid.uuid4())
    cid = set_correlation_id(job_id)

    # Validate both files exist before accepting the job
    for path_str in [request.pid_a_path, request.pid_b_path]:
        if not Path(path_str).exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {path_str}",
            )

    _jobs[job_id] = {"status": "processing", "label": request.job_label}
    _save_jobs_to_disk(_jobs)
    logger.info(
        "ingest_job_accepted",
        job_id=job_id,
        pid_a=request.pid_a_path,
        pid_b=request.pid_b_path,
    )

    # Run the heavy pipeline in a background task so we return immediately
    async def run_and_update():
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                _executor,
                _run_pipeline_sync,
                job_id,
                request.pid_a_path,
                request.pid_b_path,
            )
            _jobs[job_id].update(result)
        except Exception as e:
            _jobs[job_id].update({"status": "failed", "error": str(e)})
        finally:
            _save_jobs_to_disk(_jobs)

    bg_tasks.add_task(run_and_update)

    return IngestResponse(
        job_id=job_id,
        status="processing",
        message="Job accepted. Use GET /api/v1/delta/{job_id}/status to poll.",
    )


def _run_pipeline_sync(job_id: str, pid_a_path: str, pid_b_path: str):
    """
    The synchronous pipeline that runs in the ProcessPoolExecutor.
    This function is the only place where all stages are wired together.
    """
    set_correlation_id(job_id)
    try:
        # Stage 1: Ingest
        entities_a, format_a = canonicalize(pid_a_path)
        entities_b, format_b = canonicalize(pid_b_path)

        # Stage 2: Delta
        delta_entries = run_delta(entities_a, entities_b)

        # Stage 3: Report
        report_bundle = generate_report(delta_entries, job_id)
        output_dir = str(Path(settings.DATA_DIR) / job_id)
        report_bundle.save_to_disk(output_dir)

        # Stage 4: Index into Pinecone
        # Convert entities to text chunks for Rev A and Rev B
        rev_a_chunks = [
            f"[Rev A | Page {e.page_number} | {e.entity_type.value}] {e.text_content}"
            for e in entities_a if e.text_content
        ]
        rev_b_chunks = [
            f"[Rev B | Page {e.page_number} | {e.entity_type.value}] {e.text_content}"
            for e in entities_b if e.text_content
        ]

        index_chunks(rev_a_chunks, job_id, "rev_a")
        index_chunks(rev_b_chunks, job_id, "rev_b")
        index_chunks(report_bundle.rag_chunks, job_id, "delta_report")

        logger.info("pipeline_complete", job_id=job_id, delta_count=len(delta_entries))
        return {
            "status": "complete",
            "format_a": format_a.value,
            "format_b": format_b.value,
            "entity_count_a": len(entities_a),
            "entity_count_b": len(entities_b),
            "delta_count": len(delta_entries),
            "output_dir": output_dir,
        }

    except Exception as e:
        logger.error("pipeline_failed", job_id=job_id, error=str(e), exc_info=True)
        return {"status": "failed", "error": str(e)}


@router.get("/{job_id}/status", response_model=IngestResponse)
def get_job_status(job_id: str):
    """Poll the status of a previously submitted ingest job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return IngestResponse(
        job_id=job_id,
        status=job["status"],
        pid_a_format=SourceFormat(job["format_a"]) if job.get("format_a") else None,
        pid_b_format=SourceFormat(job["format_b"]) if job.get("format_b") else None,
        entity_count_a=job.get("entity_count_a"),
        entity_count_b=job.get("entity_count_b"),
        delta_count=job.get("delta_count"),
        message=job.get("error", ""),
    )


@router.get("/{job_id}/report/json")
def get_json_report(job_id: str):
    """
    Return the machine-parseable JSON delta report for a completed job.
    The JSON is read from disk (written by generate_report during pipeline).
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail=f"Job status: {job['status']}")

    import json
    report_path = Path(job["output_dir"]) / f"{job_id}_delta.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found on disk.")

    return json.loads(report_path.read_text(encoding="utf-8"))


@router.get("/{job_id}/report/html")
def get_html_report(job_id: str):
    """Return the human-readable HTML delta report for a completed job."""
    from fastapi.responses import HTMLResponse

    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail=f"Job status: {job['status']}")

    report_path = Path(job["output_dir"]) / f"{job_id}_delta.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found on disk.")

    return HTMLResponse(content=report_path.read_text(encoding="utf-8"))
