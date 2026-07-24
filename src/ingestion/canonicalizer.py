from pathlib import Path
from typing import List

from src.core.models import CanonicalEntity, SourceFormat
from src.ingestion.adapters.base_adapter import BaseAdapter
from src.ingestion.adapters.dwg_adapter import DwgAdapter
from src.ingestion.adapters.pdf_raster_adapter import PdfRasterAdapter
from src.ingestion.adapters.pdf_vector_adapter import PdfVectorAdapter
from src.observability.logging import get_logger
from src.observability.tracing import trace_stage

logger = get_logger(__name__)

# Magic bytes for format detection — reads the first few bytes of the file.
# This is more reliable than file extensions (which can be wrong or missing).
_PDF_MAGIC = b"%PDF"
_DWG_MAGIC = b"AC10"   # All DWG files start with "AC10xx" (e.g. AC1015 = AutoCAD 2000)


def detect_format(file_path: Path) -> SourceFormat:
    """
    Detect the document format by reading the file's magic bytes.
    For PDFs, further distinguishes between native (vector) and raster
    by checking whether the first page contains extractable text.

    Returns
    -------
    SourceFormat.PDF_NATIVE   — Born-digital PDF with text layer
    SourceFormat.PDF_RASTER   — Image-only PDF (scanned)
    SourceFormat.DWG          — AutoCAD DWG binary file
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
    except OSError as e:
        raise FileNotFoundError(f"Cannot read file: {file_path}") from e

    if header[:4] == _DWG_MAGIC or header[:4] == b"AC10":
        logger.info("format_detected", path=str(file_path), format="DWG")
        return SourceFormat.DWG

    if header[:4] == _PDF_MAGIC:
        # Distinguish native vs raster by attempting text extraction on page 1
        source_format = _is_native_pdf(file_path)
        logger.info("format_detected", path=str(file_path), format=source_format.value)
        return source_format

    raise ValueError(
        f"Unsupported file format for {file_path.name}. "
        "Only Native PDF, Scanned PDF, and DWG are supported."
    )


def _is_native_pdf(file_path: Path) -> SourceFormat:
    """
    Open the PDF with PyMuPDF and attempt to extract text from page 1.
    If fewer than 20 characters are returned, the page is almost certainly
    a raster image with no embedded text layer.

    Threshold: 20 characters is below any meaningful text block but above
    the occasional empty-string artifacts that appear in some native PDFs.
    """
    import fitz

    try:
        doc = fitz.open(str(file_path))
        page = doc[0]
        text = page.get_text("text").strip()
        doc.close()
        if len(text) >= 20:
            return SourceFormat.PDF_NATIVE
        return SourceFormat.PDF_RASTER
    except Exception as e:
        logger.warning(
            "native_pdf_check_failed",
            path=str(file_path),
            error=str(e),
            fallback="PDF_RASTER",
        )
        return SourceFormat.PDF_RASTER


def _get_adapter(source_format: SourceFormat) -> BaseAdapter:
    """Return the correct adapter instance for the detected format."""
    if source_format == SourceFormat.PDF_NATIVE:
        return PdfVectorAdapter()
    if source_format == SourceFormat.PDF_RASTER:
        return PdfRasterAdapter()
    if source_format == SourceFormat.DWG:
        return DwgAdapter()
    raise ValueError(f"No adapter registered for format: {source_format}")


@trace_stage("canonicalize", capture_result_len=True)
def canonicalize(file_path: str) -> tuple[List[CanonicalEntity], SourceFormat]:
    """
    The single public entry point for the ingestion layer.

    Given a path to any supported document, this function:
    1. Validates the file exists.
    2. Detects the format by reading magic bytes.
    3. Instantiates the correct adapter.
    4. Calls extract() on the adapter.
    5. Returns (entities, detected_format) to the caller.

    The delta engine and API layer only call this function — they never
    touch individual adapters directly.

    Parameters
    ----------
    file_path : str
        Path to the document file.

    Returns
    -------
    tuple[List[CanonicalEntity], SourceFormat]
        The extracted canonical entities and the detected format label.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    source_format = detect_format(path)
    adapter = _get_adapter(source_format)

    logger.info(
        "canonicalize_start",
        path=str(path),
        format=source_format.value,
        adapter=type(adapter).__name__,
    )

    entities = adapter.extract(path)

    logger.info(
        "canonicalize_complete",
        path=str(path),
        format=source_format.value,
        entity_count=len(entities),
    )

    return entities, source_format
