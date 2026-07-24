from typing import List, Tuple

from src.core.config import settings
from src.core.models import CanonicalEntity, DeltaEntry, DeltaType
from src.delta_engine.matcher import MatchedPair, match_entities
from src.delta_engine.metrics import is_same_content, is_same_location
from src.delta_engine.spatial_index import SpatialIndex
from src.ingestion.utils import region_description
from src.observability.logging import get_logger
from src.observability.tracing import trace_stage

logger = get_logger(__name__)


def _classify_pair(
    entity_a: CanonicalEntity,
    entity_b: CanonicalEntity,
    combined_score: float,
) -> DeltaEntry:
    """
    Apply the five deterministic classification rules to a matched pair.

    Rules (applied in order, first match wins)
    -------------------------------------------
    1. Same content hash AND same location (IoU >= threshold) → NO_CHANGE
    2. Same content hash AND different location (IoU < threshold but matched
       by content proximity) → MOVED
    3. Different content hash AND same location (IoU >= threshold) → MODIFIED
    4. Catch-all for matched pairs with low confidence → MODIFIED

    Note: ADDED and REMOVED are not handled here — they come from unmatched
    entities and are processed separately in run_delta().

    Confidence
    ----------
    - NO_CHANGE: 1.0 (exact hash + location match)
    - MOVED: combined_score (IoU of matched spatial region)
    - MODIFIED: combined_score (IoU of spatial match)
    """
    same_content = is_same_content(entity_a.content_hash, entity_b.content_hash)
    same_loc = is_same_location(entity_a.bbox, entity_b.bbox, settings.IOU_THRESHOLD)

    if same_content and same_loc:
        delta_type = DeltaType.NO_CHANGE
        confidence = 1.0
        description = ""  # No-change entries are not included in the report

    elif same_content and not same_loc:
        delta_type = DeltaType.MOVED
        confidence = round(combined_score, 4)
        description = (
            f"Entity moved from {region_description(entity_a.bbox)} "
            f"to {region_description(entity_b.bbox)} on page {entity_a.page_number}. "
            f'Content: "{entity_a.text_content or "[geometry]"}"'
        )

    else:
        # Different content — regardless of location precision
        delta_type = DeltaType.MODIFIED
        confidence = round(combined_score, 4)
        if entity_a.text_content or entity_b.text_content:
            description = (
                f'Text changed from "{entity_a.text_content}" '
                f'to "{entity_b.text_content}" '
                f"at {region_description(entity_b.bbox)}, page {entity_b.page_number}."
            )
        else:
            description = (
                f"Geometry modified at {region_description(entity_b.bbox)}, "
                f"page {entity_b.page_number}."
            )

    return DeltaEntry(
        delta_type=delta_type,
        source_entity=entity_a,
        target_entity=entity_b,
        page_number=entity_a.page_number,
        region_description=region_description(entity_b.bbox),
        confidence=confidence,
        description=description,
    )


@trace_stage("delta_engine", capture_result_len=True)
def run_delta(
    rev_a_entities: List[CanonicalEntity],
    rev_b_entities: List[CanonicalEntity],
) -> List[DeltaEntry]:
    """
    Main orchestrator for the Delta Engine.

    Flow
    ----
    1. Load all Rev B entities into an STRtree spatial index.
    2. Run bipartite matching (matcher.py) to pair Rev A ↔ Rev B entities.
    3. Classify each matched pair with the five-rule classifier above.
    4. All unmatched Rev A entities → REMOVED (confidence = 1.0).
    5. All unmatched Rev B entities → ADDED (confidence = 1.0).
    6. Filter out NO_CHANGE entries (they carry no information for the report).
    7. Return sorted list of DeltaEntry objects.

    Parameters
    ----------
    rev_a_entities : List[CanonicalEntity]
        All canonical entities from Rev A (the base document).
    rev_b_entities : List[CanonicalEntity]
        All canonical entities from Rev B (the revised document).

    Returns
    -------
    List[DeltaEntry]
        All classified changes (ADDED, REMOVED, MODIFIED, MOVED).
        Sorted by page_number, then by y0 of the relevant bounding box.
    """
    logger.info(
        "delta_run_start",
        rev_a_count=len(rev_a_entities),
        rev_b_count=len(rev_b_entities),
    )

    # Step 1: Build spatial index on Rev B
    rev_b_index = SpatialIndex(rev_b_entities)

    # Step 2: Bipartite matching
    matched_pairs, unmatched_a, unmatched_b = match_entities(
        rev_a_entities, rev_b_entities, rev_b_index
    )

    delta_entries: List[DeltaEntry] = []

    # Step 3: Classify matched pairs
    for entity_a, entity_b, combined_score in matched_pairs:
        entry = _classify_pair(entity_a, entity_b, combined_score)
        if entry.delta_type != DeltaType.NO_CHANGE:
            delta_entries.append(entry)

    # Step 4: REMOVED — in Rev A, not in Rev B
    for entity_a in unmatched_a:
        delta_entries.append(
            DeltaEntry(
                delta_type=DeltaType.REMOVED,
                source_entity=entity_a,
                target_entity=None,
                page_number=entity_a.page_number,
                region_description=region_description(entity_a.bbox),
                confidence=1.0,
                description=(
                    f'Removed: "{entity_a.text_content or "[geometry]"}" '
                    f"from {region_description(entity_a.bbox)}, "
                    f"page {entity_a.page_number}."
                ),
            )
        )

    # Step 5: ADDED — in Rev B, not in Rev A
    for entity_b in unmatched_b:
        delta_entries.append(
            DeltaEntry(
                delta_type=DeltaType.ADDED,
                source_entity=None,
                target_entity=entity_b,
                page_number=entity_b.page_number,
                region_description=region_description(entity_b.bbox),
                confidence=1.0,
                description=(
                    f'Added: "{entity_b.text_content or "[geometry]"}" '
                    f"at {region_description(entity_b.bbox)}, "
                    f"page {entity_b.page_number}."
                ),
            )
        )

    # Step 6: Sort by page, then by top-most entity bbox position
    delta_entries.sort(
        key=lambda d: (
            d.page_number,
            d.target_entity.bbox.y0 if d.target_entity else d.source_entity.bbox.y0,
        )
    )

    logger.info(
        "delta_run_complete",
        total_changes=len(delta_entries),
        added=sum(1 for d in delta_entries if d.delta_type == DeltaType.ADDED),
        removed=sum(1 for d in delta_entries if d.delta_type == DeltaType.REMOVED),
        modified=sum(1 for d in delta_entries if d.delta_type == DeltaType.MODIFIED),
        moved=sum(1 for d in delta_entries if d.delta_type == DeltaType.MOVED),
    )

    return delta_entries
