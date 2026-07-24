from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.core.config import settings
from src.core.models import CanonicalEntity
from src.delta_engine.metrics import calculate_iou, calculate_text_similarity
from src.delta_engine.spatial_index import SpatialIndex
from src.observability.logging import get_logger

logger = get_logger(__name__)

# Type alias for a matched pair and its combined score
MatchedPair = Tuple[CanonicalEntity, CanonicalEntity, float]


def build_cost_matrix(
    rev_a_entities: List[CanonicalEntity],
    rev_b_index: SpatialIndex,
    rev_b_entities: List[CanonicalEntity],
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Build a cost matrix for bipartite matching between Rev A and Rev B entities
    on a single page.

    The cost for a pair (entity_a, entity_b) is:
        cost = 1.0 - combined_score

    where combined_score = 0.6 * IoU + 0.4 * text_similarity

    This weighting means spatial overlap dominates (60%), but content
    similarity contributes strongly enough to avoid false matches where
    two different things happen to overlap on the same page.

    For entity pairs where IoU = 0.0 (no spatial overlap), the cost is set
    to 1.0 (maximum cost) so the optimizer never selects them.

    Parameters
    ----------
    rev_a_entities : List[CanonicalEntity]
        Entities from Rev A on a single page.
    rev_b_index : SpatialIndex
        STRtree index of all Rev B entities for fast candidate lookup.
    rev_b_entities : List[CanonicalEntity]
        Full list of Rev B entities — used as the column axis of the matrix.

    Returns
    -------
    Tuple[np.ndarray, List[int], List[int]]
        - cost_matrix: shape (len_a, len_b), values in [0.0, 1.0]
        - row_ids: indices into rev_a_entities for the matrix rows
        - col_ids: indices into rev_b_entities for the matrix columns
    """
    n_a = len(rev_a_entities)
    n_b = len(rev_b_entities)

    # Build a lookup from entity id → index in rev_b_entities
    b_id_to_idx: Dict[str, int] = {e.id: i for i, e in enumerate(rev_b_entities)}

    cost_matrix = np.ones((n_a, n_b), dtype=np.float64)  # Default: max cost

    for i, entity_a in enumerate(rev_a_entities):
        candidates = rev_b_index.query_candidates(entity_a.bbox, entity_a.page_number)

        for candidate_b in candidates:
            j = b_id_to_idx.get(candidate_b.id)
            if j is None:
                continue

            iou = calculate_iou(entity_a.bbox, candidate_b.bbox)
            if iou < settings.IOU_THRESHOLD:
                continue  # Not a valid spatial match

            # Only compute text similarity if both entities have text content
            if entity_a.text_content and candidate_b.text_content:
                text_sim = calculate_text_similarity(
                    entity_a.text_content, candidate_b.text_content
                )
            else:
                # For geometry-only pairs, similarity is purely spatial
                text_sim = iou

            combined_score = 0.6 * iou + 0.4 * text_sim
            cost_matrix[i, j] = 1.0 - combined_score

    return cost_matrix, list(range(n_a)), list(range(n_b))


def match_entities(
    rev_a_entities: List[CanonicalEntity],
    rev_b_entities: List[CanonicalEntity],
    rev_b_index: SpatialIndex,
) -> Tuple[List[MatchedPair], List[CanonicalEntity], List[CanonicalEntity]]:
    """
    Run bipartite matching using the Hungarian algorithm (scipy linear_sum_assignment)
    to find the globally optimal one-to-one pairing between Rev A and Rev B entities.

    Works page by page — entities on different pages are never matched.

    Returns
    -------
    matched_pairs : List[MatchedPair]
        List of (entity_a, entity_b, combined_score) for each matched pair.
        Only pairs where IoU >= IOU_THRESHOLD are included.
    unmatched_a : List[CanonicalEntity]
        Entities in Rev A with no match in Rev B → candidates for REMOVED.
    unmatched_b : List[CanonicalEntity]
        Entities in Rev B with no match in Rev A → candidates for ADDED.
    """
    # Group entities by page number for per-page matching
    pages = set(e.page_number for e in rev_a_entities) | set(
        e.page_number for e in rev_b_entities
    )

    all_matched: List[MatchedPair] = []
    all_unmatched_a: List[CanonicalEntity] = []
    all_unmatched_b: List[CanonicalEntity] = []

    for page in sorted(pages):
        a_page = [e for e in rev_a_entities if e.page_number == page]
        b_page = [e for e in rev_b_entities if e.page_number == page]

        if not a_page:
            all_unmatched_b.extend(b_page)
            continue
        if not b_page:
            all_unmatched_a.extend(a_page)
            continue

        matched, unmatched_a, unmatched_b = _match_page(a_page, b_page, rev_b_index)
        all_matched.extend(matched)
        all_unmatched_a.extend(unmatched_a)
        all_unmatched_b.extend(unmatched_b)

        logger.debug(
            "page_matching_complete",
            page=page,
            matched=len(matched),
            unmatched_a=len(unmatched_a),
            unmatched_b=len(unmatched_b),
        )

    return all_matched, all_unmatched_a, all_unmatched_b


def _match_page(
    a_page: List[CanonicalEntity],
    b_page: List[CanonicalEntity],
    rev_b_index: SpatialIndex,
) -> Tuple[List[MatchedPair], List[CanonicalEntity], List[CanonicalEntity]]:
    """
    Bipartite matching for a single page.

    Uses scipy.optimize.linear_sum_assignment (Hungarian algorithm) to find
    the optimal assignment that minimizes total cost. This guarantees that
    every entity is matched at most once — no double assignments.
    """
    cost_matrix, _, _ = build_cost_matrix(a_page, rev_b_index, b_page)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched: List[MatchedPair] = []
    matched_a_indices = set()
    matched_b_indices = set()

    for row, col in zip(row_ind, col_ind):
        cost = cost_matrix[row, col]
        if cost >= 1.0:
            # Cost == 1.0 means IoU was below threshold — not a valid match
            continue

        combined_score = 1.0 - cost
        matched.append((a_page[row], b_page[col], combined_score))
        matched_a_indices.add(row)
        matched_b_indices.add(col)

    unmatched_a = [e for i, e in enumerate(a_page) if i not in matched_a_indices]
    unmatched_b = [e for i, e in enumerate(b_page) if i not in matched_b_indices]

    return matched, unmatched_a, unmatched_b
