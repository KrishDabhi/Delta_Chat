from typing import List

from src.core.models import DeltaEntry, DeltaType


def to_rag_chunks(delta_entries: List[DeltaEntry], job_id: str) -> List[str]:
    """
    Convert DeltaEntry objects into plain-text chunks ready for embedding
    into the vector store.

    Why plain text (not JSON)
    -------------------------
    Embedding models are trained on natural language. A sentence like
    "On Page 2, the text 'Pump A' was changed to 'Pump B' in the top-left area."
    produces a semantically meaningful embedding that can be retrieved by
    queries like "what changed near the pump?" or "did any labels change on page 2?".

    A raw JSON blob produces a worse embedding because the model sees
    keys and brackets as noise.

    Chunk format
    ------------
    Each chunk is a single sentence or short paragraph.
    Chunks include: job_id reference, page number, region, delta type,
    and the before/after content. This gives the LLM everything it needs
    to generate a grounded, cited answer.

    Returns
    -------
    List[str]
        One text chunk per DeltaEntry. NO_CHANGE entries are excluded.
        The first element is always the summary chunk for the whole job.
    """
    chunks = []

    # Summary chunk — retrieved when user asks "how many things changed?"
    summary_counts = {t: 0 for t in DeltaType}
    for entry in delta_entries:
        summary_counts[entry.delta_type] += 1

    summary_chunk = (
        f"[Delta Report | Job: {job_id}] "
        f"Total changes detected: {len(delta_entries)}. "
        f"Added: {summary_counts[DeltaType.ADDED]}, "
        f"Removed: {summary_counts[DeltaType.REMOVED]}, "
        f"Modified: {summary_counts[DeltaType.MODIFIED]}, "
        f"Moved: {summary_counts[DeltaType.MOVED]}."
    )
    chunks.append(summary_chunk)

    # One chunk per change
    for entry in delta_entries:
        chunk = _entry_to_text(entry, job_id)
        if chunk:
            chunks.append(chunk)

    return chunks


def _entry_to_text(entry: DeltaEntry, job_id: str) -> str:
    """
    Build a natural language sentence for one DeltaEntry.
    The sentence is designed to be retrievable by user questions
    about what changed, where, and how.
    """
    page = entry.page_number
    region = entry.region_description
    confidence = int(entry.confidence * 100)
    prefix = f"[Delta Report | Job: {job_id} | Page {page} | {region}]"

    if entry.delta_type == DeltaType.ADDED:
        content = entry.target_entity.text_content if entry.target_entity else "[geometry]"
        return (
            f"{prefix} ADDED (confidence: {confidence}%): "
            f'The content "{content}" was added to the document in revision B.'
        )

    if entry.delta_type == DeltaType.REMOVED:
        content = entry.source_entity.text_content if entry.source_entity else "[geometry]"
        return (
            f"{prefix} REMOVED (confidence: {confidence}%): "
            f'The content "{content}" was present in revision A but removed in revision B.'
        )

    if entry.delta_type == DeltaType.MODIFIED:
        src = entry.source_entity.text_content if entry.source_entity else "[geometry]"
        tgt = entry.target_entity.text_content if entry.target_entity else "[geometry]"
        return (
            f"{prefix} MODIFIED (confidence: {confidence}%): "
            f'Content changed from "{src}" in revision A to "{tgt}" in revision B.'
        )

    if entry.delta_type == DeltaType.MOVED:
        content = entry.source_entity.text_content if entry.source_entity else "[geometry]"
        src_region = entry.source_entity.bbox if entry.source_entity else None
        tgt_region = entry.target_entity.bbox if entry.target_entity else None
        return (
            f"{prefix} MOVED (confidence: {confidence}%): "
            f'The content "{content}" moved position between revision A and revision B. '
            f"Same content was detected at a different location on page {page}."
        )

    return ""
