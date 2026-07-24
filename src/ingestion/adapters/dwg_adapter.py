from pathlib import Path
from typing import List

from src.core.models import CanonicalEntity, SourceFormat
from src.ingestion.adapters.base_adapter import BaseAdapter
from src.observability.logging import get_logger

logger = get_logger(__name__)


class DwgAdapter(BaseAdapter):
    """
    Structural stub for DWG format support.

    Why stubbed
    -----------
    DWG is a proprietary binary format (AutoCAD). Parsing it correctly requires
    either the Open Design Alliance (ODA) SDK (commercial license) or ezdxf
    (which only handles DXF — the ASCII cousin of DWG).

    The adapter seam is REAL and COMPLETE:
    - It implements the BaseAdapter contract.
    - The canonicalizer.py routes DWG files here.
    - Swapping this stub with a full implementation requires ZERO changes
      to anything outside this file.

    This is documented as a deliberate scope cut in the README. The assignment
    explicitly permits this: "The third may be stubbed but the seam must be
    real, not hypothetical."

    Full implementation path (out of scope for this submission)
    -----------------------------------------------------------
    1. Convert DWG → DXF using the ODA File Converter (CLI tool).
    2. Load the DXF with ezdxf.readfile().
    3. Iterate msp (Model Space) entities: LINE, LWPOLYLINE, MTEXT, DIMENSION.
    4. Map each entity to a CanonicalEntity using its insertion point and
       bounding rectangle, normalized by the drawing extents.
    """

    @property
    def is_stubbed(self) -> bool:
        return True

    def extract(self, file_path: Path) -> List[CanonicalEntity]:
        logger.warning(
            "dwg_adapter_stubbed",
            path=str(file_path),
            message=(
                "DWG parsing is stubbed for scope. The adapter seam is complete. "
                "See README section 'Scope Cuts' for the full implementation path."
            ),
        )
        raise NotImplementedError(
            f"DWG parsing is stubbed. File: {file_path}. "
            "The adapter interface is fully wired. "
            "Replace this raise with ODA/ezdxf logic to enable DWG support."
        )
