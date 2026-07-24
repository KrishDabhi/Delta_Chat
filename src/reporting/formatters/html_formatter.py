from collections import defaultdict
from typing import List

from src.core.models import DeltaEntry, DeltaType

# Badge colors for each delta type — standard redline convention
_TYPE_STYLE = {
    DeltaType.ADDED:    ("🟢", "added",    "#2fbf71"),
    DeltaType.REMOVED:  ("🔴", "removed",  "#ef5d5d"),
    DeltaType.MODIFIED: ("🟡", "modified", "#f5a524"),
    DeltaType.MOVED:    ("🔵", "moved",    "#4c8dff"),
}


def to_html(delta_entries: List[DeltaEntry], job_id: str) -> str:
    """
    Generate a human-readable HTML delta report grouped by page number.

    Structure
    ---------
    - Header: job ID and summary counts per delta type.
    - One section per page that has changes.
    - Within each page section: changes grouped by type (ADDED, REMOVED, etc.).
    - Each change row shows: badge, confidence bar, description, and the
      before/after text content when applicable.

    This HTML is self-contained (inline CSS) so it renders correctly in any
    browser without external stylesheets.
    """
    # Group entries by page
    by_page: dict = defaultdict(list)
    for entry in delta_entries:
        by_page[entry.page_number].append(entry)

    summary = {
        "total": len(delta_entries),
        "added": sum(1 for d in delta_entries if d.delta_type == DeltaType.ADDED),
        "removed": sum(1 for d in delta_entries if d.delta_type == DeltaType.REMOVED),
        "modified": sum(1 for d in delta_entries if d.delta_type == DeltaType.MODIFIED),
        "moved": sum(1 for d in delta_entries if d.delta_type == DeltaType.MOVED),
    }

    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="UTF-8">',
        f"<title>Delta Report — {job_id}</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
        "background:#0f1115;color:#e7ecf3;margin:0;padding:24px}",
        "h1{font-size:22px;margin-bottom:4px}",
        ".sub{color:#9aa7b8;font-size:14px;margin-bottom:20px}",
        ".summary{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:28px}",
        ".chip{padding:8px 14px;border-radius:10px;font-size:13px;font-weight:600;"
        "border:1px solid rgba(255,255,255,0.1)}",
        ".page-section{border:1px solid #2a323e;border-radius:12px;"
        "padding:16px 20px;margin-bottom:20px}",
        ".page-title{font-size:16px;font-weight:700;margin-bottom:12px;color:#9aa7b8}",
        ".entry{padding:10px 14px;border-radius:8px;margin-bottom:8px;"
        "background:#161a21;border:1px solid #2a323e}",
        ".badge{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;"
        "border-radius:999px;letter-spacing:.04em;margin-right:8px}",
        ".conf{font-size:11px;color:#9aa7b8;float:right}",
        ".desc{font-size:13px;margin-top:4px;color:#c7d2e0}",
        ".diff{font-size:12px;margin-top:6px;color:#9aa7b8}",
        "del{color:#ef5d5d;text-decoration:line-through}",
        "ins{color:#2fbf71;text-decoration:none}",
        "</style></head><body>",
        f'<h1>Delta Report</h1>',
        f'<p class="sub">Job ID: {job_id} · Total changes: {summary["total"]}</p>',
    ]

    # Summary chips
    lines.append('<div class="summary">')
    chip_colors = {
        "added": "#2fbf71", "removed": "#ef5d5d",
        "modified": "#f5a524", "moved": "#4c8dff"
    }
    for key, color in chip_colors.items():
        lines.append(
            f'<div class="chip" style="color:{color};border-color:{color}40">'
            f'{summary[key]} {key.upper()}</div>'
        )
    lines.append("</div>")

    # Per-page sections
    for page_num in sorted(by_page.keys()):
        page_entries = by_page[page_num]
        lines.append('<div class="page-section">')
        lines.append(f'<div class="page-title">Page {page_num} — {len(page_entries)} change(s)</div>')

        for entry in page_entries:
            emoji, label, color = _TYPE_STYLE.get(
                entry.delta_type, ("⚪", entry.delta_type.value.lower(), "#9aa7b8")
            )
            conf_pct = int(entry.confidence * 100)

            lines.append('<div class="entry">')
            lines.append(
                f'<span class="badge" style="background:{color}22;color:{color};'
                f'border:1px solid {color}55">{emoji} {label.upper()}</span>'
                f'<span class="conf">Confidence: {conf_pct}%</span>'
            )
            lines.append(f'<div class="desc">{entry.description}</div>')

            # Show before / after for MODIFIED entries
            if entry.delta_type == DeltaType.MODIFIED:
                src_text = entry.source_entity.text_content if entry.source_entity else ""
                tgt_text = entry.target_entity.text_content if entry.target_entity else ""
                if src_text or tgt_text:
                    lines.append(
                        f'<div class="diff">'
                        f'<del>{_escape(src_text)}</del> → '
                        f'<ins>{_escape(tgt_text)}</ins>'
                        f"</div>"
                    )

            lines.append("</div>")  # .entry

        lines.append("</div>")  # .page-section

    lines.append("</body></html>")
    return "\n".join(lines)


def _escape(text: str) -> str:
    """Minimal HTML escaping for user-supplied content."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
