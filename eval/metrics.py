from typing import List, Dict, Tuple


def compute_delta_metrics(
    predicted: List[Dict],
    ground_truth: List[Dict],
) -> Dict[str, float]:
    """
    Compute Precision, Recall, and F1 for the delta engine output.

    A predicted delta is a True Positive if it matches a ground truth entry on:
    - delta_type (exact match)
    - page_number (exact match)
    - text_content substring match (predicted text is contained in ground truth text
      OR ground truth text is contained in predicted text)

    Parameters
    ----------
    predicted : List[Dict]
        List of dicts with keys: delta_type, page_number, text_content.
        These come from the parsed JSON report.
    ground_truth : List[Dict]
        List of dicts from the eval dataset JSON.

    Returns
    -------
    Dict with keys: precision, recall, f1, true_positives, false_positives, false_negatives
    """
    matched_gt = set()
    true_positives = 0

    for pred in predicted:
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            if _is_match(pred, gt):
                true_positives += 1
                matched_gt.add(gt_idx)
                break

    false_positives = len(predicted) - true_positives
    false_negatives = len(ground_truth) - true_positives

    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _is_match(pred: Dict, gt: Dict) -> bool:
    """
    A predicted delta matches a ground truth entry if:
    1. delta_type matches exactly.
    2. page_number matches exactly.
    3. The relevant text content has a substring relationship.
    """
    if pred.get("delta_type") != gt.get("delta_type"):
        return False
    if pred.get("page_number") != gt.get("page_number"):
        return False

    pred_text = (pred.get("target", {}) or {}).get("text_content", "") or ""
    pred_text = pred_text.lower().strip()

    # For ground truth, check both before and after text
    gt_text = (
        gt.get("text_content", "")
        or gt.get("text_content_after", "")
    ).lower().strip()

    if not pred_text or not gt_text:
        # If either is empty (e.g. geometry), match on type + page only
        return True

    return gt_text in pred_text or pred_text in gt_text


def compute_chat_metrics(
    answers: List[str],
    expected_keywords_list: List[List[str]],
) -> Dict[str, float]:
    """
    Evaluate grounded chat answers by checking for expected keyword coverage.

    For each answer, checks what percentage of the expected keywords are
    present (case-insensitive substring match).

    Returns average keyword coverage across all Q&A pairs.
    This is a proxy for correctness — it does not require an LLM-as-judge.
    """
    if not answers:
        return {"avg_keyword_coverage": 0.0}

    coverages = []
    for answer, keywords in zip(answers, expected_keywords_list):
        answer_lower = answer.lower()
        hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
        coverage = hits / len(keywords) if keywords else 0.0
        coverages.append(coverage)

    return {"avg_keyword_coverage": round(sum(coverages) / len(coverages), 4)}
