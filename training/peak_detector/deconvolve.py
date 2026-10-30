"""The later stage's parts other than its model (SPEC_deconvolution.md): the peaks whose shapes it is to find, the
candidates it is given, the heights of shapes by least squares, and its scores against the simulator's parts. numpy
and scipy only.

A window here is a dict as data.cut makes it (signal, missing, valid, peaks, meta, and the parts baseline, others and
sigma); whole() makes one of a sample as it is.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import nnls

from decode import assign

LEVEL = 0.05  # of a shape's top, where its range ends: its 5 % widths
RS_EDGES = (0.25, 0.5, 1.0, 1.5)  # of the separation from the nearest peak (SPEC.md 2); isolated from the last on
RATIO_EDGES = (0.1, 1.0, 10.0)  # of the nearest peak's height to the peak's
SNR_EDGES = (20.0, 100.0)
ZERO_RUN = 4  # exact 0s in a row, of a part not measured (SPEC.md 3.6); fewer, of a signal cut at 0
REACH = 4.0  # of a candidate's 5 % widths each side, its span: where the baseline is corrected about it


def whole(sample) -> dict:
    """A sample as a window of its own length."""
    return {"signal": sample.signal, "clean": sample.clean, "missing": sample.missing,
            "valid": np.ones(len(sample.signal), dtype=bool), "peaks": sample.peaks, "meta": sample.meta,
            "baseline": sample.baseline, "others": sample.others, "sigma": sample.sigma}


def shaped(window) -> list:
    """The window's peaks that have a shape to find: the positives and those not scored (SPEC_deconvolution.md 2)."""
    return [peak for peak in window["peaks"] if peak.label != "negative"]


def full(peak, length: int) -> np.ndarray:
    """A peak's profile on all of a window's points."""
    values = np.zeros(length)
    values[peak.start:peak.start + len(peak.profile)] = peak.profile
    return values


def measured(window) -> np.ndarray:
    """The points that tell of the signal: in the window, not missing, and not in a part of 0s."""
    return window["valid"] & ~window["missing"] & (window["sigma"] > 0)


def weighed(signal, missing, valid) -> np.ndarray:
    """The points that tell of the heights, as the signal alone shows them: in the window, measured, and not 0 (of a
    part not measured, or of a signal cut at 0, which tells only that it is no more)."""
    return valid & ~missing & (signal != 0)


def present(signal, missing, valid) -> np.ndarray:
    """The points where a peak may be: in the window, and not in a run of ZERO_RUN or more 0s (a part not measured)."""
    zero = np.concatenate([[False], (signal == 0) & ~missing, [False]])
    edges = np.flatnonzero(np.diff(zero.astype(np.int8)))
    out = valid.copy()
    for first, end in zip(edges[::2], edges[1::2]):
        if end - first >= ZERO_RUN:
            out[first:end] = False
    return out


### candidates ###
def oracle_candidates(window) -> list[dict]:
    """A candidate at each peak with a shape, of its true top and widths: as the front would give them at its best."""
    return [{"apex": peak.apex, "confidence": 1.0, "left": peak.left, "right": peak.right} for peak in shaped(window)]


@dataclass(frozen=True)
class CandidateErrors:
    """How the front's candidates are off (SPEC_deconvolution.md 4), to be set by its own on the simulated data."""
    missed: float = 0.03  # of the peaks
    shift: float = 0.1  # sd, in FWHMs, cut at 3 sd
    width_spread: float = 0.2  # log sd of the widths
    top_duplicates: float = 0.2  # of the peaks, those with a second candidate on the top, 0.15 to 0.5 FWHM off
    tail_duplicates: float = 0.1  # and on a side, 1 to 2.5 times its width there off
    false_rate: float = 1e-3  # a point, of candidates on nothing


def perturbed_candidates(window, rng: np.random.Generator, errors: CandidateErrors = CandidateErrors()) -> list[dict]:
    """Candidates as the front gives them: of the peaks with a shape, some missed, the rest shifted and their widths
    off, some with a second on the top or a side; and some on nothing. Those found less confident."""
    length = int(window["valid"].sum())
    found = []
    for peak in shaped(window):
        if rng.random() < errors.missed:
            continue
        shift = float(np.clip(rng.normal(0, errors.shift), -3 * errors.shift, 3 * errors.shift)) * peak.fwhm
        left = peak.left * float(np.exp(rng.normal(0, errors.width_spread)))
        right = peak.right * float(np.exp(rng.normal(0, errors.width_spread)))
        found.append({"apex": peak.apex + shift, "confidence": rng.uniform(0.3, 1.0), "left": left, "right": right})
        if rng.random() < errors.top_duplicates:
            found.append({"apex": peak.apex + rng.choice([-1, 1]) * rng.uniform(0.15, 0.5) * peak.fwhm,
                          "confidence": rng.uniform(0.05, 0.5), "left": left, "right": right})
        if rng.random() < errors.tail_duplicates:
            side = rng.choice([-1, 1])
            part = rng.uniform(0.3, 1.0)
            found.append({"apex": peak.apex + side * rng.uniform(1.0, 2.5) * (right if side > 0 else left),
                          "confidence": rng.uniform(0.05, 0.4), "left": part * left, "right": part * right})
    for _ in range(rng.poisson(errors.false_rate * length)):
        width = float(np.exp(rng.uniform(np.log(2), np.log(50))))
        found.append({"apex": rng.uniform(0, length - 1), "confidence": rng.uniform(0.05, 0.4), "left": width,
                      "right": width})
    return [candidate for candidate in found if 0 <= candidate["apex"] <= length - 1]


### heights ###
def clusters(candidates, length: int, reach: float = REACH) -> list[tuple[float, float]]:
    """The spans of the clusters of candidates, of those whose spans (reach times their widths each side) overlap:
    from the first's start to the last's end, within the window's length points."""
    spans = sorted((c["apex"] - reach * max(c["left"], 0.5), c["apex"] + reach * max(c["right"], 0.5))
                   for c in candidates)
    out = []
    for low, high in spans:
        if out and low <= out[-1][1]:
            out[-1][1] = max(out[-1][1], high)
        else:
            out.append([low, high])
    return [(max(low, 0.0), min(high, length - 1.0)) for low, high in out]


def corrections(spans, length: int) -> np.ndarray:
    """(2 x spans, length): of each span, 1 in it, and a line through 0 at its middle and 1 at its end; 0 outside.
    The baseline's corrections about the peaks, fitted with their heights: of the errors of a baseline given, an
    offset and a tilt under each cluster, as a baseline is drawn between a cluster's ends."""
    t = np.arange(length)
    out = np.zeros((2 * len(spans), length))
    for i, (low, high) in enumerate(spans):
        inside = (t >= low) & (t <= high)
        out[2 * i] = inside
        out[2 * i + 1] = inside * (t - (low + high) / 2) / max((high - low) / 2, 1.0)
    return out


def heights(residual, shapes, weights, prior=None, strength: float = 0.0,
            free=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The heights a >= 0 of shapes (K x N, each of top 1) that make up residual (N) best, by least squares weighted
    by weights (1/sigma², 0 where nothing is told), their covariance (of those not held at 0, of noise as sigma has
    it), and the amounts of free (M x N, of any sign: the baseline's corrections) fitted with them. With prior (K),
    strength x (a_k - prior_k)² x (what the data tell of a_k alone, the weighted sum of shape k's squares) each
    besides: of a strength well below 1, little against the data of a shape apart from the rest, and what holds the
    shapes too alike to be told apart. Of free, the heights of what free cannot make of the residual, and free's of
    the rest."""
    count = len(shapes)
    root = np.sqrt(weights)
    design = (np.asarray(shapes, dtype=np.float64) * root).T.reshape(len(root), count)
    target = np.asarray(residual, dtype=np.float64) * root
    basis = np.zeros((len(root), 0)) if free is None else (np.asarray(free, dtype=np.float64) * root).T
    told_design, told_target = design, target
    if basis.shape[1]:
        u, singular, _ = np.linalg.svd(basis, full_matrices=False)
        q = u[:, singular > 1e-10 * singular.max()] if singular.max() > 0 else u[:, :0]
        told_design, told_target = design - q @ (q.T @ design), target - q @ (q.T @ target)
    found, covariance = np.zeros(count), np.zeros((count, count))
    if count:
        if prior is not None and strength > 0:
            own = np.sqrt(strength * (design ** 2).sum(axis=0))
            told_design = np.vstack([told_design, np.diag(own)])
            told_target = np.concatenate([told_target, own * np.asarray(prior, dtype=np.float64)])
        found, _ = nnls(told_design, told_target, maxiter=50 * count)
        kept = found > 0
        if kept.any():
            covariance[np.ix_(kept, kept)] = np.linalg.pinv(told_design[:, kept].T @ told_design[:, kept])
    amounts = np.linalg.lstsq(basis, target - design @ found, rcond=None)[0] if basis.shape[1] else np.zeros(0)
    return found, covariance, amounts


### scores ###
def separation(a, b) -> float:
    """Rs of two peaks (SPEC.md 2)."""
    return abs(a.apex - b.apex) / (2 * (a.fwhm + b.fwhm) / 2.3548)


def _bin(value: float, edges, name: str) -> str:
    for low, high in zip((None,) + tuple(edges), tuple(edges) + (None,)):
        if (low is None or value >= low) and (high is None or value < high):
            return f"{name} {'' if low is None else low}-{'' if high is None else high}"
    raise ValueError(value)


def categories(peak, peaks, meta: dict) -> list[str]:
    """What a peak with a shape is, to tell the errors of each."""
    names = ["all", f"shape:{peak.shape}", "scored" if peak.label == "positive" else "not scored",
             _bin(peak.snr, SNR_EDGES, "snr")]
    neighbors = [other for other in peaks if other is not peak]
    nearest = min(neighbors, key=lambda other: separation(peak, other), default=None)
    if nearest is None or separation(peak, nearest) >= RS_EDGES[-1]:
        names.append("isolated")
    else:
        names.append(_bin(separation(peak, nearest), RS_EDGES[:-1], "rs"))
        names.append(_bin(nearest.height / peak.height, RATIO_EDGES, "neighbor x"))
    if peak.saturated:
        names.append("saturated")
    if meta.get("coarse", 1) > 1:
        names.append("coarse")
    if meta.get("zeros"):
        names.append("zeros in the sample")
    if any(kind.startswith("jump") for kind in meta.get("baseline", [])):
        names.append("jump in the sample")
    return names


def _range(shape: np.ndarray) -> tuple[int, int]:
    over = np.flatnonzero(shape >= LEVEL * shape.max())
    return int(over[0]), int(over[-1])


class Scores:
    """Estimates, window by window, against the windows' peaks with a shape; of each window's estimates, each
    {"apex", "confidence", "area", and "sigma" (of the area) and "shape" (on all the window's points) if given},
    matched as the front's apexes are (decode.assign; the peaks not scored taken as peaks here). Of each peak
    matched: the area's relative error, the shape's (the L1 of the shapes of top 1, over the true one's sum), the
    range's ends' (where the shape is down to LEVEL, in FWHMs) and the area's error over the sigma given; the peaks
    matched by none, missed; and the areas of the estimates matched to none, over all the peaks'."""

    def __init__(self):
        self.records, self.false_area, self.true_area = [], 0.0, 0.0

    def add(self, window, estimates) -> None:
        length, valid = len(window["signal"]), window["valid"]
        peaks = shaped(window)
        taken = [replace(peak, label="positive") for peak in peaks]
        matched = {}
        for estimate, peak, _counts in assign(estimates, taken):
            if peak is None:
                self.false_area += max(float(estimate["area"]), 0.0)
            else:
                matched[id(peak)] = estimate
        for peak, as_taken in zip(peaks, taken):
            truth = np.where(valid, full(peak, length), 0.0)
            area = float(truth.sum())
            self.true_area += area
            record = {"categories": categories(peak, peaks, window["meta"])}
            estimate = matched.get(id(as_taken))
            if estimate is None or area <= 0:
                record["missed"] = True
                self.records.append(record)
                continue
            record["area"] = (estimate["area"] - area) / area
            if estimate.get("sigma"):
                record["z"] = (estimate["area"] - area) / estimate["sigma"]
            shape = estimate.get("shape")
            if shape is not None and np.where(valid, shape, 0).max() > 0:
                shape, truth = np.where(valid, shape, 0.0) / np.where(valid, shape, 0.0).max(), truth / truth.max()
                record["shape"] = float(np.abs(shape - truth).sum() / truth.sum())
                record["ends"] = [abs(a - b) / peak.fwhm for a, b in zip(_range(shape), _range(truth))]
            self.records.append(record)

    def summary(self) -> dict:
        """In all, and by category."""
        return {"peaks": len(self.records),
                "false area": self.false_area / self.true_area if self.true_area else 0.0,
                "by category": _summaries(self.records)}


def score(results) -> dict:
    """Scores' summary of (window, estimates) each."""
    scores = Scores()
    for window, estimates in results:
        scores.add(window, estimates)
    return scores.summary()


def _summaries(records) -> dict:
    groups: dict[str, list[dict]] = {}
    for record in records:
        for name in record["categories"]:
            groups.setdefault(name, []).append(record)
    out = {}
    for name in sorted(groups, key=lambda name: (name != "all", name)):
        group = groups[name]
        found = [record for record in group if not record.get("missed")]
        summary = {"peaks": len(group), "missed": 1 - len(found) / len(group)}
        if found:
            errors = np.array([record["area"] for record in found])
            summary |= {"area |error| median": float(np.median(np.abs(errors))),
                        "area |error| p90": float(np.percentile(np.abs(errors), 90)),
                        "area error median": float(np.median(errors))}
        shapes = [record["shape"] for record in found if "shape" in record]
        if shapes:
            summary |= {"shape error median": float(np.median(shapes)),
                        "range end error median (FWHM)": float(np.median([e for r in found if "ends" in r
                                                                         for e in r["ends"]]))}
        z = np.array([record["z"] for record in found if "z" in record])
        if len(z):
            # of the error over the sigma given: about 1, and 95 % within 2, if the sigma is right
            summary |= {"z sd (robust)": float(1.4826 * np.median(np.abs(z - np.median(z)))),
                        "within 2 sigma": float(np.mean(np.abs(z) <= 2))}
        out[name] = summary
    return out
