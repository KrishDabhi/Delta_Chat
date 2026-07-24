import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import fitz  # PyMuPDF — used only to rasterize the PDF page to an image
import numpy as np
import pytesseract
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

from src.core.config import settings
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

# Configure Tesseract executable path from config (critical for Windows)
pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


class PdfRasterAdapter(BaseAdapter):
    """
    Extracts canonical entities from scanned (raster) PDFs.

    Three-stage deterministic pipeline — NO generative LLM at any step:

    Stage 1 — Rasterize
        PyMuPDF renders each PDF page to a numpy image at settings.RASTER_DPI.
        This is the raw pixel array we process.

    Stage 2a — Text Extraction (OpenCV → Tesseract)
        OpenCV finds bounding boxes of text regions via adaptive thresholding
        and contour detection. Tesseract runs only on those cropped regions,
        returning text strings with per-word confidence scores.

    Stage 2b — Geometry Extraction (Canny → RANSAC)
        OpenCV Canny edge detection produces a set of edge pixel coordinates.
        scikit-learn RANSAC fits clean line segments through the noisy pixels,
        discarding outliers (dirt, scan artifacts). The result is a set of
        clean normalized line endpoints — equivalent to native CAD vectors.

    Stage 3 — Canonicalization
        Both text and geometry results are assembled into CanonicalEntity objects
        with bounding boxes normalized to [0.0, 1.0] page-relative coordinates.
    """

    def __init__(self):
        self._dpi = settings.RASTER_DPI
        self._ransac_residual = settings.RANSAC_RESIDUAL_THRESHOLD
        self._ransac_min_samples = settings.RANSAC_MIN_SAMPLES
        self._min_line_length_px = settings.RANSAC_MIN_LINE_LENGTH

    @trace_stage("pdf_raster_extraction", capture_result_len=True)
    def extract(self, file_path: Path) -> List[CanonicalEntity]:
        entities: List[CanonicalEntity] = []

        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            logger.error("raster_pdf_open_failed", path=str(file_path), error=str(e))
            raise

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1

            try:
                image = self._rasterize_page(page)
                page_h, page_w = image.shape[:2]

                text_entities = self._extract_text_regions(image, page_num, page_w, page_h)
                geo_entities = self._extract_geometry_lines(image, page_num, page_w, page_h)

                entities.extend(text_entities)
                entities.extend(geo_entities)

                logger.debug(
                    "raster_page_extracted",
                    page=page_num,
                    text_count=len(text_entities),
                    geo_count=len(geo_entities),
                )
            except Exception as e:
                logger.error(
                    "raster_page_failed",
                    path=str(file_path),
                    page=page_num,
                    error=str(e),
                )

        doc.close()
        return self._sort_entities(entities)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1: Rasterize
    # ──────────────────────────────────────────────────────────────────────────

    def _rasterize_page(self, page: fitz.Page) -> np.ndarray:
        """
        Render one PDF page to a numpy uint8 BGR image at self._dpi.
        Matrix scale factor = DPI / 72 (PDF base resolution is 72 dpi).
        """
        scale = self._dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2a: Text Extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_text_regions(
        self, image: np.ndarray, page_num: int, page_w: int, page_h: int
    ) -> List[CanonicalEntity]:
        """
        1. Convert to grayscale and apply adaptive thresholding to get a
           binary image where text pixels are black on white background.
        2. Use morphological dilation to connect adjacent characters into word-blobs.
        3. Find external contours — each contour is a candidate text region.
        4. Filter out contours that are too small (noise) or too large (page border).
        5. Run Tesseract image_to_data on each cropped region to get text + confidence.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold — handles uneven lighting across scanned pages
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=8,
        )

        # Dilate horizontally to merge characters into word blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        entities = []
        min_area = 100  # px² — discard specks
        max_area = page_w * page_h * 0.5  # discard page-sized blobs

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < min_area or area > max_area:
                continue

            # Add a small padding so Tesseract doesn't clip edge characters
            pad = 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(page_w, x + w + pad)
            y2 = min(page_h, y + h + pad)

            crop = image[y1:y2, x1:x2]
            text, confidence = self._run_tesseract(crop)

            if not text:
                continue

            bbox = normalize_bbox(
                x0=x1, y0=y1, x1=x2, y1=y2,
                page_width=page_w, page_height=page_h,
            )

            entity_type = self._classify_text(text)
            entity = CanonicalEntity(
                page_number=page_num,
                entity_type=entity_type,
                source_format=SourceFormat.PDF_RASTER,
                bbox=bbox,
                text_content=text,
                raw_confidence=confidence,
            )
            entity.content_hash = entity.compute_content_hash()
            entities.append(entity)

        return entities

    def _run_tesseract(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Run Tesseract on a single cropped image region.
        Returns (cleaned_text, mean_confidence).
        Confidence is the mean of per-word confidences, normalized to [0.0, 1.0].
        Returns ("", 0.0) if Tesseract finds nothing above its threshold.
        """
        try:
            data = pytesseract.image_to_data(
                crop,
                output_type=pytesseract.Output.DICT,
                config="--psm 6",  # psm 6 = uniform block of text
            )
        except Exception as e:
            logger.warning("tesseract_failed", error=str(e))
            return "", 0.0

        words = []
        confidences = []
        for i, word in enumerate(data["text"]):
            conf = int(data["conf"][i])
            if conf > 0 and word.strip():  # conf = -1 means Tesseract skipped it
                words.append(word.strip())
                confidences.append(conf)

        if not words:
            return "", 0.0

        text = " ".join(words)
        mean_conf = round(sum(confidences) / len(confidences) / 100.0, 4)
        return text, mean_conf

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2b: Geometry Extraction (Canny + RANSAC)
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_geometry_lines(
        self, image: np.ndarray, page_num: int, page_w: int, page_h: int
    ) -> List[CanonicalEntity]:
        """
        Detect clean line vectors from the scanned image using RANSAC.

        Steps:
        1. Convert to grayscale and run Canny edge detection to get edge pixels.
        2. Extract (x, y) coordinates of all white pixels in the edge map.
        3. Iteratively run RANSAC to fit lines through the point cloud:
           - Each RANSAC run finds the dominant line among remaining points.
           - Inlier points (within RANSAC_RESIDUAL_THRESHOLD pixels of the line)
             are removed before the next iteration.
           - Stop when fewer than 10 edge pixels remain.
        4. Each fitted line is validated: it must span at least
           RANSAC_MIN_LINE_LENGTH pixels to be kept.
        5. Kept lines are built into GEOMETRY CanonicalEntities with normalized
           bounding boxes and geometry descriptors.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Canny parameters: lower threshold, upper threshold
        # These values work well for 300 DPI engineering drawings
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)

        # Get pixel coordinates of all detected edges
        edge_points = np.column_stack(np.where(edges > 0))  # shape: (N, 2) → (row, col)
        if len(edge_points) < self._ransac_min_samples:
            return []

        # RANSAC iterative line fitting
        fitted_lines = self._ransac_iterative(edge_points)

        entities = []
        for (x0_px, y0_px, x1_px, y1_px) in fitted_lines:
            # Compute line bounding box
            bx0 = min(x0_px, x1_px)
            by0 = min(y0_px, y1_px)
            bx1 = max(x0_px, x1_px)
            by1 = max(y0_px, y1_px)

            bbox = normalize_bbox(
                x0=bx0, y0=by0, x1=bx1, y1=by1,
                page_width=page_w, page_height=page_h,
            )

            # Build normalized geometry descriptor
            descriptor = json.dumps([
                [
                    round(x0_px / page_w, 6),
                    round(y0_px / page_h, 6),
                    round(x1_px / page_w, 6),
                    round(y1_px / page_h, 6),
                ]
            ])

            entity = CanonicalEntity(
                page_number=page_num,
                entity_type=EntityType.GEOMETRY,
                source_format=SourceFormat.PDF_RASTER,
                bbox=bbox,
                text_content="",
                geometry_descriptor=descriptor,
                raw_confidence=1.0,  # RANSAC is deterministic — no confidence uncertainty
            )
            entity.content_hash = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()
            entities.append(entity)

        return entities

    def _ransac_iterative(
        self, points: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Iteratively apply RANSAC to find all dominant lines in the edge point cloud.

        Each iteration:
        1. Fit a linear model (y = mx + b) using RANSAC over the remaining points.
        2. Extract inlier points (those within RANSAC_RESIDUAL_THRESHOLD of the line).
        3. Compute the endpoints of the fitted line segment from the inlier bounding box.
        4. Remove inliers from the working set.
        5. Repeat until too few points remain.

        Returns a list of (x0, y0, x1, y1) pixel tuples for each found line.
        """
        # points is (row, col) = (y, x) from np.where
        remaining = points.copy().astype(float)
        fitted_lines = []

        while len(remaining) >= max(self._ransac_min_samples, 10):
            X = remaining[:, 1].reshape(-1, 1)  # col = x
            y = remaining[:, 0]                 # row = y

            try:
                ransac = RANSACRegressor(
                    min_samples=self._ransac_min_samples,
                    residual_threshold=self._ransac_residual,
                    max_trials=100,
                    random_state=42,  # Fixed seed for reproducibility
                )
                ransac.fit(X, y)
            except Exception:
                break

            inlier_mask = ransac.inlier_mask_
            inliers = remaining[inlier_mask]

            if len(inliers) < 2:
                break

            # Endpoints are the leftmost and rightmost inlier points
            sorted_by_x = inliers[inliers[:, 1].argsort()]
            x0_px = int(sorted_by_x[0, 1])
            y0_px = int(sorted_by_x[0, 0])
            x1_px = int(sorted_by_x[-1, 1])
            y1_px = int(sorted_by_x[-1, 0])

            # Check minimum line length in pixels
            length = np.hypot(x1_px - x0_px, y1_px - y0_px)
            if length >= self._min_line_length_px:
                fitted_lines.append((x0_px, y0_px, x1_px, y1_px))

            # Remove inliers from remaining set for next iteration
            remaining = remaining[~inlier_mask]

        return fitted_lines

    # ──────────────────────────────────────────────────────────────────────────
    # Helper: Text Classification
    # ──────────────────────────────────────────────────────────────────────────

    def _classify_text(self, text: str) -> EntityType:
        """Same rule-based dimension detector as the native adapter."""
        import re
        dimension_pattern = re.compile(
            r"\b\d+[\d\.\-\s\'\"]*\s*"
            r"(mm|cm|m|ft|in|inch|inches|'|\"|\"|DN|NPS|kg|kPa|psi|bar|°C|°F)\b",
            re.IGNORECASE,
        )
        if dimension_pattern.search(text):
            return EntityType.DIMENSION
        return EntityType.TEXT
