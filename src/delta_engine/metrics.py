from src.core.models import BoundingBox


def calculate_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """
    Calculate Intersection over Union (IoU) for two normalized BoundingBoxes.

    IoU = area(intersection) / area(union)

    Range: 0.0 (no overlap) to 1.0 (perfect overlap).
    This is the primary spatial similarity metric used by the matcher.

    Parameters
    ----------
    box_a, box_b : BoundingBox
        Both must be normalized to [0.0, 1.0].

    Returns
    -------
    float
        IoU score in [0.0, 1.0].
    """
    # Intersection rectangle
    inter_x0 = max(box_a.x0, box_b.x0)
    inter_y0 = max(box_a.y0, box_b.y0)
    inter_x1 = min(box_a.x1, box_b.x1)
    inter_y1 = min(box_a.y1, box_b.y1)

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    intersection = inter_w * inter_h

    if intersection == 0.0:
        return 0.0

    union = box_a.area + box_b.area - intersection
    if union <= 0.0:
        return 0.0

    return round(intersection / union, 6)


def calculate_text_similarity(text_a: str, text_b: str) -> float:
    """
    Calculate normalized text similarity between two strings using
    Levenshtein edit distance.

    similarity = 1.0 - (edit_distance / max_len)

    Range: 0.0 (completely different) to 1.0 (identical).

    We use python-Levenshtein (C extension) for speed. Falls back to
    a pure-Python implementation if the library is unavailable, so the
    system never fails due to a missing optional dependency.

    Parameters
    ----------
    text_a, text_b : str
        Raw text strings from two matched CanonicalEntities.

    Returns
    -------
    float
        Similarity score in [0.0, 1.0].
    """
    if text_a == text_b:
        return 1.0

    a = text_a.strip()
    b = text_b.strip()
    max_len = max(len(a), len(b))

    if max_len == 0:
        return 1.0

    try:
        import Levenshtein
        distance = Levenshtein.distance(a, b)
    except ImportError:
        # Pure Python fallback — DP implementation of edit distance
        distance = _levenshtein_pure(a, b)

    return round(1.0 - (distance / max_len), 6)


def _levenshtein_pure(s1: str, s2: str) -> int:
    """
    Pure Python Levenshtein edit distance.
    Only used as fallback if python-Levenshtein C extension is not installed.
    """
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n]


def is_same_location(box_a: BoundingBox, box_b: BoundingBox, iou_threshold: float) -> bool:
    """
    Returns True if the IoU of two boxes is at or above the configured threshold.
    Used by the classifier to distinguish MODIFIED (same location) from MOVED
    (different location but same content).
    """
    return calculate_iou(box_a, box_b) >= iou_threshold


def is_same_content(hash_a: str, hash_b: str) -> bool:
    """
    Returns True if two content hashes are identical.
    For TEXT/DIMENSION: SHA-256 of the stripped text string.
    For GEOMETRY: SHA-256 of the normalized geometry descriptor JSON.
    Exact string match — no fuzzy comparison at the hash level.
    """
    return hash_a == hash_b
