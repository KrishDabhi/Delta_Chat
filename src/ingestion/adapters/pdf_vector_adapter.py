from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from src.core.models import (
    BoundingBox,
    CanonicalEntity,
    EntityType,
    SourceFormat,
)
from src.ingestion.adapters.base_adapter import BaseAdapter
from src.ingestion.utils import normalize_bbox
from src.observability.logging import get_logger
from src.observability.tracing import trace_stage

logger = get_logger(__name__)


class PdfVectorAdapter(BaseAdapter):
    """
    Extracts canonical entities from born-digital (native) PDFs.

    Tools used
    ----------
    PyMuPDF (fitz): The only tool needed. Native PDFs contain an embedded text
    layer and vector path descriptions. PyMuPDF reads them directly from the
    PDF dictionary — no image processing, no OCR, no guessing.

    What we extract per page
    ------------------------
    1. Words (TEXT / DIMENSION):
       page.get_text("words") returns every word with its exact bounding box
       in PDF user-space units (points). We group adjacent words on the same
       line into logical text blocks using a Y-axis proximity threshold.

    2. Vector paths (GEOMETRY):
       page.get_drawings() returns every path element (lines, rectangles,
       curves) with exact coordinates. We keep only paths whose bounding box
       is large enough to represent a meaningful engineering element (not a
       hairline rule or tick mark).

    Coordinate normalization
    ------------------------
    PyMuPDF returns coordinates in PDF user-space (points, origin top-left).
    We pass them to normalize_bbox() in utils.py which maps them to [0.0, 1.0]
    relative to the page rectangle.
    """

    # Minimum bounding box area (in normalized units) for a path to be kept.
    # Filters out micro-artifacts like hairlines and tick marks.
    _MIN_GEOMETRY_AREA: float = 0.0001

    # Words whose Y-midpoint is within this fraction of page height of each
    # other are merged into one text block.
    _LINE_MERGE_TOLERANCE: float = 0.005

    @trace_stage("pdf_vector_extraction", capture_result_len=True)
    def extract(self, file_path: Path) -> List[CanonicalEntity]:
        entities: List[CanonicalEntity] = []

        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            logger.error("pdf_open_failed", path=str(file_path), error=str(e))
            raise

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1
            page_rect = page.rect  # width and height in PDF points

            try:
                text_entities = self._extract_text(page, page_num, page_rect)
                geo_entities = self._extract_geometry(page, page_num, page_rect)
                entities.extend(text_entities)
                entities.extend(geo_entities)
                logger.debug(
                    "page_extracted",
                    page=page_num,
                    text_count=len(text_entities),
                    geo_count=len(geo_entities),
                )
            except Exception as e:
                # Log and skip this page. A single bad page must not kill the job.
                logger.error(
                    "page_extraction_failed",
                    path=str(file_path),
                    page=page_num,
                    error=str(e),
                )

        doc.close()
        return self._sort_entities(entities)

    def _extract_text(
        self, page: fitz.Page, page_num: int, page_rect: fitz.Rect
    ) -> List[CanonicalEntity]:
        """
        Extract word-level text and merge into line-level blocks.
        PyMuPDF 'words' format: (x0, y0, x1, y1, word, block_no, line_no, word_no)
        """
        raw_words = page.get_text("words")
        if not raw_words:
            return []

        # Group words by (block_no, line_no) to reconstruct full lines
        lines: dict = {}
        for x0, y0, x1, y1, word, block_no, line_no, word_no in raw_words:
            key = (block_no, line_no)
            if key not in lines:
                lines[key] = {"words": [], "x0": x0, "y0": y0, "x1": x1, "y1": y1}
            else:
                lines[key]["x0"] = min(lines[key]["x0"], x0)
                lines[key]["y0"] = min(lines[key]["y0"], y0)
                lines[key]["x1"] = max(lines[key]["x1"], x1)
                lines[key]["y1"] = max(lines[key]["y1"], y1)
            lines[key]["words"].append(word)

        entities = []
        for line_data in lines.values():
            text = " ".join(line_data["words"]).strip()
            if not text:
                continue

            bbox = normalize_bbox(
                x0=line_data["x0"],
                y0=line_data["y0"],
                x1=line_data["x1"],
                y1=line_data["y1"],
                page_width=page_rect.width,
                page_height=page_rect.height,
            )

            entity_type = self._classify_text(text)
            entity = CanonicalEntity(
                page_number=page_num,
                entity_type=entity_type,
                source_format=SourceFormat.PDF_NATIVE,
                bbox=bbox,
                text_content=text,
                raw_confidence=1.0,  # Native text — no OCR uncertainty
            )
            entity.content_hash = entity.compute_content_hash()
            entities.append(entity)

        return entities

    def _extract_geometry(
        self, page: fitz.Page, page_num: int, page_rect: fitz.Rect
    ) -> List[CanonicalEntity]:
        """
        Extract vector paths (lines, rectangles, curves) from the PDF.
        page.get_drawings() returns a list of dicts, each describing one path.
        We use the path's 'rect' (bounding box) for spatial indexing.
        """
        drawings = page.get_drawings()
        entities = []

        for drawing in drawings:
            rect = drawing.get("rect")
            if rect is None:
                continue

            bbox = normalize_bbox(
                x0=rect.x0,
                y0=rect.y0,
                x1=rect.x1,
                y1=rect.y1,
                page_width=page_rect.width,
                page_height=page_rect.height,
            )

            # Skip micro-artifacts (hairlines, tick marks, separators)
            if bbox.area < self._MIN_GEOMETRY_AREA:
                continue

            # Build a geometry descriptor from the path's items list
            # Each item is ("l", p1, p2) for a line or ("c", ...) for a curve
            descriptor = self._build_geometry_descriptor(drawing.get("items", []), page_rect)

            entity = CanonicalEntity(
                page_number=page_num,
                entity_type=EntityType.GEOMETRY,
                source_format=SourceFormat.PDF_NATIVE,
                bbox=bbox,
                text_content="",
                geometry_descriptor=descriptor,
                raw_confidence=1.0,
            )
            # Hash the geometry descriptor so the delta engine can detect changes
            import hashlib
            entity.content_hash = hashlib.sha256(
                (descriptor or "").encode("utf-8")
            ).hexdigest()
            entities.append(entity)

        return entities

    def _classify_text(self, text: str) -> EntityType:
        """
        Deterministically classify a text string as TEXT or DIMENSION.
        A DIMENSION contains a numeric value followed by a unit of measure.
        This is a simple rule-based heuristic — no LLM involved.
        """
        import re
        # Matches patterns like: 1200mm, 4'-6", 3.5m, 200 mm, DN50
        dimension_pattern = re.compile(
            r"\b\d+[\d\.\-\s\'\"]*\s*"
            r"(mm|cm|m|ft|in|inch|inches|'|\"|\"|DN|NPS|kg|kPa|psi|bar|°C|°F)\b",
            re.IGNORECASE,
        )
        if dimension_pattern.search(text):
            return EntityType.DIMENSION
        return EntityType.TEXT

    def _build_geometry_descriptor(self, items: list, page_rect: fitz.Rect) -> str:
        """
        Build a compact JSON string from a path's line segments.
        Only processes straight line segments ("l" type items).
        Coordinates are normalized to [0.0, 1.0].
        """
        import json

        segments = []
        for item in items:
            if item[0] == "l" and len(item) == 3:
                p1, p2 = item[1], item[2]
                segments.append(
                    [
                        round(p1.x / page_rect.width, 6),
                        round(p1.y / page_rect.height, 6),
                        round(p2.x / page_rect.width, 6),
                        round(p2.y / page_rect.height, 6),
                    ]
                )
        return json.dumps(segments)
