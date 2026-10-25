"""From the detector's heads to apexes, and apexes against the peaks labeled (SPEC.md 6.2). numpy only.

An apex found matches the unmatched positive nearest it within max(2 points, 0.5 FWHM), the most confident first.
One within that of a peak not scored counts for nothing; one of no peak (a negative included) is a false positive.
"""

import numpy as np

from targets import positions_of

TOLERANCE = (2.0, 0.5)  # points, part of the FWHM
THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


def decode(heat: np.ndarray, offset: np.ndarray, log_widths: np.ndarray, stride: int, *, threshold: float = 0.05,
           length: int | None = None) -> list[dict]:
    """The apexes of one sample's heads (heat as probabilities): the cells at local maxima of the heat above
    threshold, each {"apex", "confidence", "left", "right"}, in points; those past length left out."""
    heat = np.asarray(heat, dtype=np.float64)
    padded = np.concatenate([[-1.0], heat, [-1.0]])
    tops = np.flatnonzero((heat > threshold) & (heat >= padded[:-2]) & (heat > padded[2:]))
    apexes = positions_of(tops, np.asarray(offset)[tops], stride)
    found = []
    for top, apex in zip(tops, apexes):
        if length is not None and not 0 <= apex <= length - 1:
            continue
        found.append({"apex": float(apex), "confidence": float(heat[top]),
                      "left": float(np.exp(log_widths[0, top])), "right": float(np.exp(log_widths[1, top]))})
    return found


def match(found: list[dict], peaks, tolerance=TOLERANCE) -> tuple[list[tuple[float, bool]], list]:
    """(confidence, true) of each apex found that counts, and the positives matched, by the rule above."""
    positives = [peak for peak in peaks if peak.label == "positive"]
    unscored = [peak for peak in peaks if peak.label == "ignore"]

    def near(peak, apex):
        return abs(peak.apex - apex) <= max(tolerance[0], tolerance[1] * peak.fwhm)

    matched, records = set(), []
    for apex in sorted(found, key=lambda a: -a["confidence"]):
        candidates = [i for i, peak in enumerate(positives) if i not in matched and near(peak, apex["apex"])]
        if candidates:
            matched.add(min(candidates, key=lambda i: abs(positives[i].apex - apex["apex"])))
            records.append((apex["confidence"], True))
        elif not any(near(peak, apex["apex"]) for peak in unscored):
            records.append((apex["confidence"], False))
    return records, [positives[i] for i in matched]


def average_precision(records: list[tuple[float, bool]], positives: int) -> float:
    """The area under the precision-recall curve (all points interpolated)."""
    if positives == 0:
        return float("nan")
    if not records:
        return 0.0
    order = sorted(records, key=lambda record: -record[0])
    true = np.cumsum([record[1] for record in order])
    false = np.cumsum([not record[1] for record in order])
    recall = np.concatenate([[0.0], true / positives])
    precision = np.concatenate([[1.0], true / np.maximum(true + false, 1)])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def categories(peak, peaks, meta: dict) -> list[str]:
    """What a positive is, to tell the recall of each."""
    names = [f"shape:{peak.shape}"]
    if peak.components > 1:
        names.append("merged")
    if any(other is not peak and other.label != "negative" and abs(other.apex - peak.apex) <= 1.5 * max(other.fwhm, peak.fwhm)
           for other in peaks):
        names.append("overlapped")
    if peak.saturated:
        names.append("saturated")
    if peak.snr < 20:
        names.append("snr<20")
    if meta.get("coarse", 1) > 1:
        names.append("coarse")
    if peak.fwhm < 6:
        names.append("fwhm<6")
    elif peak.fwhm > 100:
        names.append("fwhm>100")
    return names


def evaluate(results: list[tuple[list[dict], list, dict]], *, recall_threshold: float = 0.5) -> dict:
    """Of (apexes found, peaks, meta) of each sample: the average precision; the precision, recall and F1 at each
    threshold; and the recall of each category at recall_threshold."""
    records, positives = [], 0
    category_counts: dict[str, list[int]] = {}
    for found, peaks, meta in results:
        sample_records, _ = match(found, peaks)
        records += sample_records
        sample_positives = [peak for peak in peaks if peak.label == "positive"]
        positives += len(sample_positives)
        _, matched = match([a for a in found if a["confidence"] >= recall_threshold], peaks)
        matched_ids = {id(peak) for peak in matched}
        for peak in sample_positives:
            for name in categories(peak, peaks, meta):
                counts = category_counts.setdefault(name, [0, 0])
                counts[0] += id(peak) in matched_ids
                counts[1] += 1
    summary = {"ap": average_precision(records, positives), "positives": positives}
    for threshold in THRESHOLDS:
        true = sum(1 for confidence, is_true in records if confidence >= threshold and is_true)
        false = sum(1 for confidence, is_true in records if confidence >= threshold and not is_true)
        precision = true / max(true + false, 1)
        recall = true / max(positives, 1)
        summary[f"precision@{threshold}"] = precision
        summary[f"recall@{threshold}"] = recall
        summary[f"f1@{threshold}"] = 2 * precision * recall / max(precision + recall, 1e-12)
    summary["recall_by"] = {name: (hit / total, total) for name, (hit, total) in sorted(category_counts.items())}
    return summary
