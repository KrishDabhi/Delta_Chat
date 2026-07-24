from fastapi import APIRouter, HTTPException

from src.chat.answer import get_grounded_answer
from src.core.models import ChatRequest, ChatResponse
from src.observability.logging import get_logger, set_correlation_id

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Ask a grounded question about the documents and delta report
    for a specific ingestion job.

    The job must be in "complete" status (i.e., ingestion pipeline has
    finished and all chunks have been indexed into Pinecone).

    Request body
    ------------
    {
        "job_id": "uuid-of-completed-job",
        "query": "What changed on page 3?"
    }

    Response
    --------
    {
        "answer": "On page 3, the text 'Pump A' was changed... [Source: Delta Report | Page 3 | top-left]",
        "citations": ["[Source: Delta Report | Page 3 | top-left]"],
        "retrieval_chunk_count": 8
    }
    """
    import uuid
    cid = set_correlation_id()
    logger.info("chat_request", job_id=request.job_id, query=request.query[:80])

    try:
        response = get_grounded_answer(query=request.query, job_id=request.job_id)
        return response
    except Exception as e:
        logger.error(
            "chat_endpoint_failed",
            job_id=request.job_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))
