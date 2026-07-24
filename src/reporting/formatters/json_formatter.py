import json
from typing import Any, Dict, List

from src.core.models import DeltaEntry, DeltaType


def to_json(delta_entries: List[DeltaEntry], job_id: str) -> str:
    """
    Serialize the list of DeltaEntry objects to a machine-parseable JSON string.

    Output structure
    ----------------
    {
        "job_id": "...",
        "summary": {
            "total": N,
            "added": N,
            "removed": N,
            "modified": N,
            "moved": N
        },
        "changes": [
            {
                "id": "...",
                "delta_type": "ADDED" | "REMOVED" | "MODIFIED" | "MOVED",
                "page_number": 1,
                "region": "top-left",
                "confidence": 0.95,
                "description": "...",
                "source": { text_content, bbox } | null,
                "target": { text_content, bbox } | null
            },
            ...
        ]
    }

    Only meaningful fields are serialized — internal entity IDs and
    raw_confidence are excluded to keep the output clean for consumers.
    """
    def _entity_summary(entity) -> Dict[str, Any] | None:
        if entity is None:
            return None
        return {
            "text_content": entity.text_content,
            "entity_type": entity.entity_type.value,
            "bbox": {
                "x0": entity.bbox.x0,
                "y0": entity.bbox.y0,
                "x1": entity.bbox.x1,
                "y1": entity.bbox.y1,
            },
        }

    summary = {
        "total": len(delta_entries),
        "added": sum(1 for d in delta_entries if d.delta_type == DeltaType.ADDED),
        "removed": sum(1 for d in delta_entries if d.delta_type == DeltaType.REMOVED),
        "modified": sum(1 for d in delta_entries if d.delta_type == DeltaType.MODIFIED),
        "moved": sum(1 for d in delta_entries if d.delta_type == DeltaType.MOVED),
    }

    changes = []
    for entry in delta_entries:
        changes.append({
            "id": entry.id,
            "delta_type": entry.delta_type.value,
            "page_number": entry.page_number,
            "region": entry.region_description,
            "confidence": entry.confidence,
            "description": entry.description,
            "source": _entity_summary(entry.source_entity),
            "target": _entity_summary(entry.target_entity),
        })

    output = {
        "job_id": job_id,
        "summary": summary,
        "changes": changes,
    }

    return json.dumps(output, indent=2, ensure_ascii=False)
