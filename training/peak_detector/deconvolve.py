"""The later stage's parts other than its model (SPEC_deconvolution.md): the peaks whose shapes it is to find, the
candidates it is given, the heights of shapes by least squares, the calibration of the areas' sigma, and its scores
against the simulator's parts. numpy and scipy only.

A window here is a dict as data.cut makes it (signal, missing, valid, peaks, meta, and the parts baseline, others and
sigma); whole() makes one of a sample as it is.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse
from scipy.optimize import linprog, nnls

from decode import assign

LEVEL = 0.05  # of a shape's top, where its range ends: its 5 % widths
RS_EDGES = (0.25, 0.5, 1.0, 1.5)  # of the separation from the nearest peak (SPEC.md 2); isolated from the last on
RATIO_EDGES = (0.1, 1.0, 10.0)  # of the nearest peak's height to the peak's
SNR_EDGES = (20.0, 100.0)
ZERO_RUN = 4  # exact 0s in a row, of a part not measured (SPEC.md 3.6); fewer, of a signal cut at 0
REACH = 4.0  # of a candidate's 5 % widths each side, its span: where the baseline is corrected about it
# confidences of the candidates roughened() adds: of the front's candidates on nothing, 95 % are below 0.2 (unet v5, on
# the simulated windows, from decode's threshold of 0.05)
ADDED = (0.05, 0.3)
# of a peak's fit, what its area's error is calibrated of (fit_features), and the quantiles of the |relative error|
# fitted: of a normal error, 1 and 2 sigma
FEATURES = ("share", "neighbor", "others", "snr", "residual", "z_rms", "relative_ls", "realness", "prior_ratio",
            "width", "near", "zero")
QUANTILES = (0.6827, 0.9545)


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
    tops = [(peak.apex, peak.fwhm, peak.left, peak.right, None) for peak in shaped(window)]
    return _made_off(tops, int(window["valid"].sum()), rng, errors, lambda _own: rng.uniform(0.3, 1.0),
                     lambda high: rng.uniform(0.05, high))


def roughened(candidates, length: int, rng: np.random.Generator,
              errors: CandidateErrors = CandidateErrors()) -> list[dict]:
    """The front's candidates of a window of length points, worse, as on data unlike those it learned of: some left
    out, the rest shifted and their widths off, some with a second on the top or a side; and some on nothing. Those
    added of confidences as low as the front gives those on nothing (ADDED); the rest of their own."""
    # of a Gaussian, the 5 % widths 2.45 sigmas each side, the FWHM 2.35
    tops = [(c["apex"], 0.48 * (c["left"] + c["right"]), c["left"], c["right"], c["confidence"]) for c in candidates]
    return _made_off(tops, length, rng, errors, lambda own: own,
                     lambda _high: float(np.exp(rng.uniform(np.log(ADDED[0]), np.log(ADDED[1])))))


def _made_off(tops, length: int, rng, errors: CandidateErrors, kept, added) -> list[dict]:
    """Candidates of tops (apex, FWHM, left and right widths, and a confidence of its own or None) as errors has
    them; kept(its own) the confidence of one kept, added(high) of one added (high: as perturbed_candidates has it)."""
    found = []
    for apex, fwhm, left, right, own in tops:
        if rng.random() < errors.missed:
            continue
        shift = float(np.clip(rng.normal(0, errors.shift), -3 * errors.shift, 3 * errors.shift)) * fwhm
        left = left * float(np.exp(rng.normal(0, errors.width_spread)))
        right = right * float(np.exp(rng.normal(0, errors.width_spread)))
        found.append({"apex": apex + shift, "confidence": kept(own), "left": left, "right": right})
        if rng.random() < errors.top_duplicates:
            found.append({"apex": apex + rng.choice([-1, 1]) * rng.uniform(0.15, 0.5) * fwhm,
                          "confidence": added(0.5), "left": left, "right": right})
        if rng.random() < errors.tail_duplicates:
            side = rng.choice([-1, 1])
            part = rng.uniform(0.3, 1.0)
            found.append({"apex": apex + side * rng.uniform(1.0, 2.5) * (right if side > 0 else left),
                          "confidence": added(0.4), "left": part * left, "right": part * right})
    for _ in range(rng.poisson(errors.false_rate * length)):
        width = float(np.exp(rng.uniform(np.log(2), np.log(50))))
        found.append({"apex": rng.uniform(0, length - 1), "confidence": added(0.4), "left": width, "right": width})
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


def heights(residual, shapes, weights, prior=None, strength=0.0,
            free=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The heights a >= 0 of shapes (K x N, each of top 1) that make up residual (N) best, by least squares weighted
    by weights (1/sigma², 0 where nothing is told), their covariance (of those not held at 0, of noise as sigma has
    it), and the amounts of free (M x N, of any sign: the baseline's corrections) fitted with them. With prior (K),
    strength x (a_k - prior_k)² x (what the data tell of a_k alone, the weighted sum of shape k's squares) each
    besides (strength one for all, or one of each shape): of a strength well below 1, little against the data of a
    shape apart from the rest, and what holds the shapes too alike to be told apart. Of free, the heights of what
    free cannot make of the residual, and free's of the rest."""
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
        strength = np.asarray(strength, dtype=np.float64)
        if prior is not None and (strength > 0).any():
            own = np.sqrt(strength * (design ** 2).sum(axis=0))
            told_design = np.vstack([told_design, np.diag(own)])
            told_target = np.concatenate([told_target, own * np.asarray(prior, dtype=np.float64)])
        found, _ = nnls(told_design, told_target, maxiter=50 * count)
        kept = found > 0
        if kept.any():
            covariance[np.ix_(kept, kept)] = np.linalg.pinv(told_design[:, kept].T @ told_design[:, kept])
    amounts = np.linalg.lstsq(basis, target - design @ found, rcond=None)[0] if basis.shape[1] else np.zeros(0)
    return found, covariance, amounts


### the areas' sigma, calibrated ###
def fit_features(shape, height, prior, neighbors, others, residual, noise, valid, realness: float,
                 relative_ls: float) -> np.ndarray:
    """Of a peak taken, the features (FEATURES) of its fit that its area's error is of, as the later stage has them
    when used: of its shape (N, of top 1) at the height found (at prior, the network's, of a height of 0), the shapes
    of the other peaks taken at theirs (M, N), the others, the residual and the noise (N), in the window (valid).

    share: of its area, what the others' shapes take of the points (of each point, in proportion); neighbor: the
    most of the others' shapes in its range (to 5 % of its top), over its top; others: of the others in its range,
    over its area; snr: its top over the noise there; residual: of the residual in its range, the sum over its area;
    z_rms: of the residual over the noise there; relative_ls: the least squares' sigma of its area over it; the
    realness; prior_ratio: its height over the network's; width: of its range, in points; near: of the others, those
    in its range above 5 % of its top; zero: its height 0."""
    zero = height <= 0
    own = (prior if zero else height) * np.where(valid, shape, 0.0)
    neighbors = np.where(valid, np.asarray(neighbors, dtype=np.float64).reshape(-1, len(own)), 0.0)
    fitted = neighbors.sum(0)
    total = max(own.sum(), 1e-30)
    top = max(own.max(), 1e-30)
    inside = own >= LEVEL * top
    noise = np.maximum(noise, 1e-30)
    return np.array([(own * fitted / np.maximum(own + fitted, 1e-30)).sum() / total,
                     (fitted[inside] / top).max(initial=0.0),
                     np.abs(others[inside]).sum() / total,
                     top / noise[int(np.argmax(own))],
                     abs(residual[inside].sum()) / total,
                     np.sqrt(np.mean((residual[inside] / noise[inside]) ** 2)),
                     0.0 if zero else relative_ls,
                     realness,
                     (prior if zero else height) / max(prior, 1e-30),
                     inside.sum(),
                     (neighbors[:, inside] > LEVEL * top).any(axis=1).sum(),
                     float(zero)])


def design(features) -> np.ndarray:
    """Of features (P, FEATURES each), what the calibration is linear in: their logs, mostly."""
    f = np.atleast_2d(np.asarray(features, dtype=np.float64))
    col = {name: f[:, i] for i, name in enumerate(FEATURES)}
    real = np.clip(col["realness"], 1e-4, 1 - 1e-4)
    ratio = np.log(np.maximum(col["prior_ratio"], 1e-6))
    return np.stack([np.log(col["share"] + 1e-3), np.log(col["neighbor"] + 1e-3), np.log(col["others"] + 1e-4),
                     np.log(np.maximum(col["snr"], 1e-3)), np.log(col["residual"] + 1e-4),
                     np.log(np.maximum(col["z_rms"], 1e-3)), np.log(col["relative_ls"] + 1e-4),
                     np.log(real / (1 - real)), ratio, np.abs(ratio), np.log(np.maximum(col["width"], 1.0)),
                     col["near"], col["zero"]], 1)


def fit_calibration(features, errors) -> dict:
    """Of the features of peaks found (P, FEATURES each) and their areas' relative errors (P, over their scale), of
    each of QUANTILES, the quantile of the |error| by linear quantile regression of its log on design(features)
    (standardized): a linear program of the pinball loss (Koenker and Bassett)."""
    x = design(features)
    mean, sd = x.mean(0), x.std(0) + 1e-12
    x = (x - mean) / sd
    y = np.log(np.abs(np.asarray(errors, dtype=np.float64)) + 1e-6)
    n, p = x.shape
    lhs = sparse.hstack([sparse.csr_matrix(np.column_stack([x, np.ones(n)])), sparse.eye(n), -sparse.eye(n)])
    out = {"features": list(FEATURES), "mean": mean.tolist(), "sd": sd.tolist(), "quantiles": {}}
    for q in QUANTILES:
        cost = np.concatenate([np.zeros(p + 1), np.full(n, q), np.full(n, 1 - q)])
        bounds = [(None, None)] * (p + 1) + [(0, None)] * (2 * n)
        solved = linprog(cost, A_eq=lhs.tocsr(), b_eq=y, bounds=bounds, method="highs")
        if not solved.success:
            raise RuntimeError(f"of the quantile {q}: {solved.message}")
        out["quantiles"][str(q)] = {"weights": solved.x[:p].tolist(), "intercept": float(solved.x[p])}
    return out


def calibrated(calibration: dict, features) -> dict[str, np.ndarray]:
    """Of a calibration (fit_calibration's) and features (P, FEATURES each): of each of its quantiles, of the
    |relative error| (P)."""
    x = (design(features) - np.array(calibration["mean"])) / np.array(calibration["sd"])
    return {q: np.exp(x @ np.array(fitted["weights"]) + fitted["intercept"])
            for q, fitted in calibration["quantiles"].items()}


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
    matched by none, missed; and the estimates matched to none, their areas over all the peaks' and their number a
    window."""

    def __init__(self):
        self.records, self.false_area, self.true_area, self.false_peaks, self.windows = [], 0.0, 0.0, 0, 0

    def add(self, window, estimates) -> None:
        length, valid = len(window["signal"]), window["valid"]
        peaks = shaped(window)
        taken = [replace(peak, label="positive") for peak in peaks]
        matched = {}
        self.windows += 1
        for estimate, peak, _counts in assign(estimates, taken):
            if peak is None:
                self.false_area += max(float(estimate["area"]), 0.0)
                self.false_peaks += 1
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
            if estimate.get("interval"):
                record["inside"] = abs(estimate["area"] - area) <= estimate["interval"]
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
                "false peaks a window": self.false_peaks / self.windows if self.windows else 0.0,
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
                        "area error median": float(np.median(errors)),
                        # of those found, given none (a >= 0 held at 0 by the least squares): as good as missed
                        "area 0": float(np.mean(errors <= -1))}
        shapes = [record["shape"] for record in found if "shape" in record]
        if shapes:
            summary |= {"shape error median": float(np.median(shapes)),
                        "range end error median (FWHM)": float(np.median([e for r in found if "ends" in r
                                                                         for e in r["ends"]]))}
        z = np.array([record["z"] for record in found if "z" in record])
        if len(z):
            # of the error over the sigma given: about 1, and 95 % within 2, if the sigma is right
            summary |= {"z sd (robust)": float(1.4826 * np.median(np.abs(z - np.median(z)))),
                        "within 1 sigma": float(np.mean(np.abs(z) <= 1)),
                        "within 2 sigma": float(np.mean(np.abs(z) <= 2))}
        inside = [record["inside"] for record in found if "inside" in record]
        if inside:
            summary["within the interval"] = float(np.mean(inside))  # of a 95 % one, 0.95 if right
        out[name] = summary
    return out
