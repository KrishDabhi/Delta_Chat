import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.core.models import DeltaEntry
from src.observability.logging import get_logger
from src.observability.tracing import trace_stage
from src.reporting.formatters.html_formatter import to_html
from src.reporting.formatters.json_formatter import to_json
from src.reporting.formatters.rag_chunker import to_rag_chunks

logger = get_logger(__name__)


@dataclass
class ReportBundle:
    """
    The complete output of a single delta comparison job.
    All three formats are generated together and returned as one object.
    """
    job_id: str
    json_report: str        # Machine-parseable — for API consumers
    html_report: str        # Human-readable — for the browser report view
    rag_chunks: List[str]   # Natural language sentences — for vector embedding

    def save_to_disk(self, output_dir: str) -> dict:
        """
        Write the JSON and HTML reports to disk.
        RAG chunks are not written to disk — they are passed directly
        to the vector store (chat/index.py) after generation.

        Returns a dict of {format: file_path} for logging purposes.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / f"{self.job_id}_delta.json"
        html_path = out / f"{self.job_id}_delta.html"

        json_path.write_text(self.json_report, encoding="utf-8")
        html_path.write_text(self.html_report, encoding="utf-8")

        logger.info(
            "reports_saved",
            job_id=self.job_id,
            json_path=str(json_path),
            html_path=str(html_path),
        )

        return {
            "json": str(json_path),
            "html": str(html_path),
        }


@trace_stage("report_generation")
def generate_report(
    delta_entries: List[DeltaEntry],
    job_id: str,
) -> ReportBundle:
    """
    Generate all three report formats from the delta entries.

    Parameters
    ----------
    delta_entries : List[DeltaEntry]
        Output of delta_engine.engine.run_delta().
    job_id : str
        Unique identifier for this comparison job.

    Returns
    -------
    ReportBundle
        Contains json_report, html_report, and rag_chunks.
    """
    logger.info("report_generation_start", job_id=job_id, change_count=len(delta_entries))

    json_report = to_json(delta_entries, job_id)
    html_report = to_html(delta_entries, job_id)
    rag_chunks = to_rag_chunks(delta_entries, job_id)

    logger.info(
        "report_generation_complete",
        job_id=job_id,
        rag_chunk_count=len(rag_chunks),
    )

    return ReportBundle(
        job_id=job_id,
        json_report=json_report,
        html_report=html_report,
        rag_chunks=rag_chunks,
    )
