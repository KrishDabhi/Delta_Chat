from typing import List, Optional

from shapely.geometry import box
from shapely.strtree import STRtree

from src.core.models import BoundingBox, CanonicalEntity
from src.observability.logging import get_logger

logger = get_logger(__name__)


class SpatialIndex:
    """
    Wraps Shapely's STRtree for efficient bounding-box spatial queries.

    Why STRtree
    -----------
    STRtree (Sort-Tile-Recursive tree) is an R-tree variant that is extremely
    fast for static datasets (no insertions after build). Since Rev B entities
    are loaded once and then queried N times (once per Rev A entity), STRtree
    is the correct data structure.

    Coordinate system
    -----------------
    All BoundingBoxes are already normalized to [0.0, 1.0] by the adapters.
    Shapely uses those values directly — no further conversion needed.
    """

    def __init__(self, entities: List[CanonicalEntity]):
        """
        Build the spatial index from a list of CanonicalEntity objects.
        The index stores entities from a single revision (Rev B in the
        Delta Engine's workflow).

        Parameters
        ----------
        entities : List[CanonicalEntity]
            All entities extracted from one document revision.
        """
        self._entities = entities

        # Build a Shapely box for each entity's bounding box
        self._geometries = [
            box(e.bbox.x0, e.bbox.y0, e.bbox.x1, e.bbox.y1)
            for e in entities
        ]

        # STRtree takes the list of geometries and builds the index in O(N log N)
        self._tree = STRtree(self._geometries)

        logger.debug(
            "spatial_index_built",
            entity_count=len(entities),
        )

    def query_candidates(
        self,
        query_bbox: BoundingBox,
        page_number: int,
    ) -> List[CanonicalEntity]:
        """
        Return all entities from the indexed revision whose bounding boxes
        intersect with query_bbox AND are on the same page.

        This is a fast pre-filter — it returns ALL intersecting entities.
        The matcher.py then computes precise IoU scores to select the best match.

        Parameters
        ----------
        query_bbox : BoundingBox
            The bounding box to search against (from a Rev A entity).
        page_number : int
            Only candidates on this page are returned.

        Returns
        -------
        List[CanonicalEntity]
            Candidate entities from Rev B that spatially overlap with query_bbox
            on the given page. May be empty if no overlap exists.
        """
        query_geom = box(
            query_bbox.x0, query_bbox.y0,
            query_bbox.x1, query_bbox.y1,
        )

        # STRtree.query() returns integer indices of intersecting geometries
        candidate_indices = self._tree.query(query_geom)

        candidates = []
        for idx in candidate_indices:
            entity = self._entities[idx]
            if entity.page_number == page_number:
                candidates.append(entity)

        return candidates
