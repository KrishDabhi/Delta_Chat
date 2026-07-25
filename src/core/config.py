from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── LLM (Groq via LangChain-Groq) ──────────────────────────────────────
    GROQ_API_KEY: str = Field(..., description="Groq API key for LLM calls")
    LLM_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model name passed to Groq. Swappable without code changes.",
    )
    LLM_TEMPERATURE: float = Field(
        default=0.0,
        description="0.0 = deterministic output for grounded chat answers.",
    )

    # ── Vector Store (Pinecone) ─────────────────────────────────────────────
    PINECONE_API_KEY: str = Field(..., description="Pinecone API key")
    PINECONE_INDEX_NAME: str = Field(
        default="delta-chat-index",
        description="Pinecone index that stores embeddings for Rev A, Rev B, and delta report chunks.",
    )

    # ── Delta Engine Thresholds ─────────────────────────────────────────────
    IOU_THRESHOLD: float = Field(
        default=0.5,
        description=(
            "Minimum Intersection over Union score for two bounding boxes to be "
            "considered a spatial match. Values below this threshold are treated as "
            "distinct, unmatched entities (ADDED / REMOVED)."
        ),
    )
    TEXT_SIMILARITY_THRESHOLD: float = Field(
        default=0.85,
        description=(
            "Minimum normalized text similarity (1 - Levenshtein distance ratio) for "
            "matched entities to be classified as NO_CHANGE. Below this = MODIFIED."
        ),
    )

    # ── Raster PDF (pdf_raster_adapter) ────────────────────────────────────
    RASTER_DPI: int = Field(
        default=300,
        description=(
            "DPI at which scanned PDF pages are rasterized before OpenCV processing. "
            "300 is the minimum for reliable Tesseract OCR on engineering drawings."
        ),
    )
    RANSAC_RESIDUAL_THRESHOLD: float = Field(
        default=2.0,
        description=(
            "Maximum pixel distance from a RANSAC-fitted line for a point to be "
            "considered an inlier. Lower = tighter fit, fewer points classified as "
            "belonging to the line."
        ),
    )
    RANSAC_MIN_SAMPLES: int = Field(
        default=2,
        description="Minimum number of points required to fit a line in RANSAC.",
    )
    RANSAC_MIN_LINE_LENGTH: int = Field(
        default=50,
        description="Minimum pixel length of a reconstructed line segment to be kept.",
    )
    TESSERACT_CMD: str = Field(
        default="tesseract",
        description=(
            "Path to the Tesseract executable. On Windows, set this to the full path, "
            "e.g. C:/Program Files/Tesseract-OCR/tesseract.exe"
        ),
    )

    # ── Coordinate Normalization ────────────────────────────────────────────
    COORDINATE_PRECISION: int = Field(
        default=6,
        description="Decimal places kept when normalizing coordinates to [0.0, 1.0] space.",
    )

    # ── Observability ───────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging verbosity. Set to DEBUG to see per-entity extraction details.",
    )
    LOG_FORMAT: str = Field(
        default="json",
        description="'json' for structured production logs, 'console' for human-readable dev output.",
    )

    # ── Application ─────────────────────────────────────────────────────────
    ENVIRONMENT: str = Field(default="development", description="development | production")
    DATA_DIR: str = Field(
        default="data/samples",
        description="Directory where uploaded document pairs are stored.",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Extra fields from existing .env (e.g. DEFAULT_TONE) are ignored cleanly
        extra = "ignore"


settings = Settings()

# Push API keys to os.environ for libraries (like Langchain) that expect them there
import os
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
