from fastapi import APIRouter, HTTPException

from eval.run_eval import generate_eval_report
from src.core.models import EvalReportResponse
from src.observability.logging import get_logger, set_correlation_id

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/eval", tags=["Evaluation"])

@router.post("", response_model=EvalReportResponse)
def run_evaluation_endpoint():
    """
    Trigger the evaluation harness against the internal dataset.
    This runs the pipeline on all configured evaluation pairs and returns a structured
    report containing delta precision/recall/F1 metrics and chat keyword coverage.
    """
    import uuid
    set_correlation_id(f"eval-api-{uuid.uuid4().hex[:8]}")
    logger.info("eval_api_started")
    
    try:
        report = generate_eval_report()
        logger.info(
            "eval_api_completed",
            total_pairs=len(report.pair_results),
            failures=len(report.failures)
        )
        return report
    except Exception as e:
        logger.error("eval_api_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
