from src.core.config import settings
from src.core.models import BoundingBox


def normalize_bbox(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page_width: float,
    page_height: float,
) -> BoundingBox:
    """
    Convert raw coordinates into a BoundingBox normalized to [0.0, 1.0].

    Works for ALL coordinate systems:
    - PDF user-space points (from PyMuPDF native extraction)
    - Pixel coordinates (from OpenCV / Tesseract raster extraction)
    - DWG world-space units (once ODA/ezdxf extraction is implemented)

    The caller is responsible for passing the correct page_width and page_height
    in the same unit system as x0, y0, x1, y1.

    Clamps output to [0.0, 1.0] to handle any floating-point overflow from
    bounding boxes that marginally exceed the page boundary.

    Parameters
    ----------
    x0, y0 : float
        Top-left corner of the bounding box in raw units.
    x1, y1 : float
        Bottom-right corner of the bounding box in raw units.
    page_width : float
        Full width of the page in the same raw units.
    page_height : float
        Full height of the page in the same raw units.

    Returns
    -------
    BoundingBox
        Normalized to [0.0, 1.0] with COORDINATE_PRECISION decimal places.
    """
    precision = settings.COORDINATE_PRECISION

    def clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    norm_x0 = clamp(round(x0 / page_width, precision))
    norm_y0 = clamp(round(y0 / page_height, precision))
    norm_x1 = clamp(round(x1 / page_width, precision))
    norm_y1 = clamp(round(y1 / page_height, precision))

    # Ensure x0 < x1 and y0 < y1 regardless of coordinate direction
    return BoundingBox(
        x0=min(norm_x0, norm_x1),
        y0=min(norm_y0, norm_y1),
        x1=max(norm_x0, norm_x1),
        y1=max(norm_y0, norm_y1),
    )


def region_description(bbox: BoundingBox) -> str:
    """
    Convert a normalized BoundingBox into a human-readable page region label.
    Used by the Delta Engine and the report formatter for citation text.

    The page is divided into a 3×3 grid (top/middle/bottom × left/center/right).
    The label is determined by the center point of the bounding box.

    Examples
    --------
    - center at (0.1, 0.1) → "top-left"
    - center at (0.5, 0.5) → "center"
    - center at (0.9, 0.8) → "bottom-right"
    """
    cx = bbox.center_x
    cy = bbox.center_y

    if cy < 0.33:
        vert = "top"
    elif cy < 0.67:
        vert = "middle"
    else:
        vert = "bottom"

    if cx < 0.33:
        horiz = "left"
    elif cx < 0.67:
        horiz = "center"
    else:
        horiz = "right"

    if vert == "middle" and horiz == "center":
        return "center"
    if horiz == "center":
        return vert
    if vert == "middle":
        return horiz
    return f"{vert}-{horiz}"
