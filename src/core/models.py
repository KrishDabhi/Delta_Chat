import hashlib
import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations — strict, closed sets used across the pipeline
# ─────────────────────────────────────────────────────────────────────────────

class EntityType(str, Enum):
    """What kind of content this canonical entity represents."""
    TEXT = "TEXT"           # A block of plain text (annotation, note, label)
    DIMENSION = "DIMENSION" # A dimension string (e.g., "1200mm", "4'-6"")
    TABLE_CELL = "TABLE_CELL"
    GEOMETRY = "GEOMETRY"   # A vector or reconstructed line / shape
    TITLE_BLOCK = "TITLE_BLOCK"  # Drawing title, revision block, sheet info


class SourceFormat(str, Enum):
    """Which adapter produced this entity — for traceability."""
    PDF_NATIVE = "PDF_NATIVE"
    PDF_RASTER = "PDF_RASTER"
    DWG = "DWG"


class DeltaType(str, Enum):
    """The five and only five outcomes the Delta Engine can produce."""
    ADDED = "ADDED"       # Exists in Rev B, not in Rev A
    REMOVED = "REMOVED"   # Exists in Rev A, not in Rev B
    MODIFIED = "MODIFIED" # Matched spatially, content changed
    MOVED = "MOVED"       # Same content hash, different bounding box location
    NO_CHANGE = "NO_CHANGE"


# ─────────────────────────────────────────────────────────────────────────────
# BoundingBox
# All coordinates are normalized to [0.0, 1.0] relative to page dimensions.
# This makes bounding boxes directly comparable regardless of source DPI or units.
# ─────────────────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x0: float = Field(..., ge=0.0, le=1.0, description="Left edge, normalized")
    y0: float = Field(..., ge=0.0, le=1.0, description="Top edge, normalized")
    x1: float = Field(..., ge=0.0, le=1.0, description="Right edge, normalized")
    y1: float = Field(..., ge=0.0, le=1.0, description="Bottom edge, normalized")

    @computed_field
    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    @computed_field
    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @computed_field
    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# CanonicalEntity
# The single, format-agnostic unit that all adapters produce.
# The Delta Engine works entirely in CanonicalEntity space.
# ─────────────────────────────────────────────────────────────────────────────

class CanonicalEntity(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this entity within a single document revision.",
    )
    page_number: int = Field(..., ge=1, description="1-indexed page / sheet number.")
    entity_type: EntityType
    source_format: SourceFormat
    bbox: BoundingBox
    text_content: str = Field(
        default="",
        description="The raw text string extracted from this entity. Empty for pure geometry.",
    )
    content_hash: str = Field(
        default="",
        description=(
            "SHA-256 of text_content (for TEXT/DIMENSION/TABLE_CELL) or a "
            "perceptual hash of the image crop (for GEOMETRY). "
            "Used by the Delta Engine to detect MODIFIED vs MOVED."
        ),
    )
    geometry_descriptor: Optional[str] = Field(
        default=None,
        description=(
            "For GEOMETRY entities: a JSON string encoding reconstructed line "
            "endpoints [x0,y0,x1,y1] in normalized space after RANSAC fitting."
        ),
    )
    raw_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the extraction: 1.0 for native vector data, "
            "Tesseract word-level confidence / 100 for raster OCR."
        ),
    )

    def compute_content_hash(self) -> str:
        """Deterministically hash the text_content. Call this after setting text_content."""
        return hashlib.sha256(self.text_content.strip().encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# DeltaEntry
# One classified change output by the Delta Engine.
# This is the data structure that feeds the JSON report, HTML report,
# RAG chunker, and the chat citations system.
# ─────────────────────────────────────────────────────────────────────────────

class DeltaEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    delta_type: DeltaType

    # Entities involved — source is from Rev A, target is from Rev B.
    # For ADDED: source_entity is None. For REMOVED: target_entity is None.
    source_entity: Optional[CanonicalEntity] = None
    target_entity: Optional[CanonicalEntity] = None

    # Location context — human-readable for the report and for RAG citations
    page_number: int = Field(..., description="The page where this change occurred.")
    region_description: str = Field(
        default="",
        description=(
            "Auto-generated human-readable description of where on the page this "
            "change occurred, e.g. 'top-left quadrant' or 'title block area'."
        ),
    )

    # Confidence is deterministic — it is the IoU score of the spatial match.
    # ADDED and REMOVED always have confidence 1.0 (no ambiguity).
    confidence: float = Field(..., ge=0.0, le=1.0)

    # Human-readable description for the report
    description: str = Field(
        default="",
        description=(
            "Auto-generated sentence describing this change. "
            "Example: 'Text changed from \"Pump A\" to \"Pump B\".' "
            "Used directly as the RAG chunk text."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# API Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Sent to POST /api/v1/delta/ingest. Both files must already be on the server."""
    pid_a_path: str = Field(..., description="Absolute or relative path to Rev A document.")
    pid_b_path: str = Field(..., description="Absolute or relative path to Rev B document.")
    job_label: Optional[str] = Field(
        default=None,
        description="Optional human label for this comparison job, e.g. 'Pump Room Rev1 vs Rev2'.",
    )


class IngestResponse(BaseModel):
    job_id: str
    status: str  # "processing" | "complete" | "failed"
    pid_a_format: Optional[SourceFormat] = None
    pid_b_format: Optional[SourceFormat] = None
    entity_count_a: Optional[int] = None
    entity_count_b: Optional[int] = None
    delta_count: Optional[int] = None
    message: str = ""


class ChatRequest(BaseModel):
    job_id: str = Field(..., description="The job_id returned by the ingest endpoint.")
    query: str = Field(..., description="The user's natural language question.")


class ChatResponse(BaseModel):
    answer: str
    citations: list[str] = Field(
        default_factory=list,
        description=(
            "List of source references grounding the answer. "
            "Format: 'Rev A | Page 3 | TEXT entity' or 'Delta Report | MODIFIED | Page 2'."
        ),
    )
    retrieval_chunk_count: int = Field(
        default=0, description="Number of RAG chunks retrieved and fed to the LLM."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation API Models
# ─────────────────────────────────────────────────────────────────────────────

class DeltaMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


class ChatMetrics(BaseModel):
    avg_keyword_coverage: float


class FailureCase(BaseModel):
    pair_id: str
    stage: str
    error: str


class PairEvalResult(BaseModel):
    pair_id: str
    delta_metrics: Optional[DeltaMetrics] = None
    chat_metrics: Optional[ChatMetrics] = None
    status: str


class EvalReportResponse(BaseModel):
    overall_delta_precision: float
    overall_delta_recall: float
    overall_delta_f1: float
    overall_chat_keyword_coverage: float
    pair_results: list[PairEvalResult]
    failures: list[FailureCase]
