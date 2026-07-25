from fastapi import FastAPI

from src.api.routes import delta as delta_router
from src.api.routes import chat as chat_router
from src.api.routes import eval as eval_router
from src.core.config import settings
from src.observability.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Document Delta & Grounded Chat API",
    version="1.0.0",
    description=(
        "Ingests two document revisions (Native PDF, Scanned PDF, or DWG), "
        "computes a structured delta, generates a delta report, and exposes "
        "a grounded chat interface over both documents and the delta report."
    ),
)

app.include_router(delta_router.router)
app.include_router(chat_router.router)
app.include_router(eval_router.router)


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "llm_model": settings.LLM_MODEL,
        "environment": settings.ENVIRONMENT,
        "iou_threshold": settings.IOU_THRESHOLD,
        "raster_dpi": settings.RASTER_DPI,
    }
