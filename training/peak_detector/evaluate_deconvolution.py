"""The later stage's references on simulated windows (SPEC_deconvolution.md 5), scored by deconvolve.Scores:

- truth: least squares on the true shapes, the rest of the signal (baseline, others, the peaks that are not) as it is,
  weighted by the true sigma: the error of the areas the noise alone makes
- emg: a sum of EMGs and a straight baseline, fitted unweighted to each cluster of candidates: the classical way
- app: the app's ranges and areas (services/peak_detection._peaks_at, services/quantification.quantify), copied
  here, as this folder does not depend on the app
- model: the later stage trained (train_deconvolution.py), its candidates of realness from --real taken (and those
  from --low fitted to the others); of --calibration (calibrate_deconvolution.py), its areas' sigma calibrated

    python evaluate_deconvolution.py --method truth --count 2000 --out truth.json
    python evaluate_deconvolution.py --method emg --candidates perturbed --count 300 --out emg.json
    python evaluate_deconvolution.py --method emg --candidates front --front runs/unet/best.pt --confidence 0.3
    python evaluate_deconvolution.py --method model --checkpoint runs/deconvolution/best.pt --candidates front \
        --front runs/unet/best.pt

The candidates: the true peaks' (oracle), made off as a front's (perturbed), or a front's (front, of a checkpoint of
train.py, from decode's threshold); of them, those from --confidence only (the app takes the front's from its
threshold; the model all, as it learned of them).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import peak_prominences, peak_widths
from scipy.special import erfc, erfcx

from deconvolve import Scores, full, heights, measured, oracle_candidates, perturbed_candidates, shaped
from simulator import VERSION, Config, simulate

SEED = 1_000_005  # of the windows, apart from the training and the validation ones
REACH = 3.0  # of a candidate's 5 % widths, where its EMG is fitted; those of reaches overlapping, together
APP_END_HEIGHT = 0.8  # services/peak_detection.END_HEIGHT
APP_SNAP = 2  # points about a candidate, of which the highest is its top, as kensert_peaks takes the CNN's


def windows(count: int, length: int, config: Config = Config(), seed: int = SEED):
    """The same windows every time, each with a generator of its own for its candidates."""
    from data import cut  # of torch, which the rest does without

    for i in range(count):
        rng = np.random.default_rng([seed, i])
        yield cut(simulate(rng, config), length, rng), np.random.default_rng([seed, i, 1])


### least squares on the true shapes ###
def truth(window, _candidates=None) -> list[dict]:
    """A peak's estimate each, of the heights of the true shapes by least squares, the rest as it is."""
    length, valid = len(window["signal"]), window["valid"]
    peaks = [peak for peak in shaped(window) if len(peak.profile) and peak.profile.max() > 0]
    rest = window["baseline"] + window["others"]
    for peak in window["peaks"]:
        if peak.label == "negative":
            rest = rest + full(peak, length)
    shapes = np.array([full(peak, length) / peak.profile.max() for peak in peaks]).reshape(len(peaks), length)
    weights = np.where(measured(window), 1 / np.maximum(window["sigma"], 1e-12) ** 2, 0.0)
    found, covariance, _ = heights(window["signal"] - rest, shapes, weights)
    sums = np.where(valid, shapes, 0.0).sum(axis=1)
    return [{"apex": peak.apex, "confidence": 1.0, "area": float(a * total),
             "sigma": float(np.sqrt(max(covariance[k, k], 0.0)) * total) or None, "shape": a * shape}
            for k, (peak, a, total, shape) in enumerate(zip(peaks, found, sums, shapes))]


### EMGs ###
def emg(t, area: float, center: float, sigma: float, tau: float) -> np.ndarray:
    """An exponentially modified Gaussian of that area, of center and sigma its Gaussian's: of a tail to the right of
    tau > 0, to the left of tau < 0 (mirrored)."""
    side, tau = (1.0 if tau >= 0 else -1.0), max(abs(tau), 1e-3 * sigma)
    d = side * (np.asarray(t, dtype=np.float64) - center)
    z = (sigma / tau - d / sigma) / np.sqrt(2)
    out = np.empty_like(d)
    # exp(-d²/2σ²) erfcx(z) where z >= 0, and as defined where not, so that neither overflows
    near = z >= 0
    out[near] = np.exp(-0.5 * (d[near] / sigma) ** 2) * erfcx(z[near])
    out[~near] = np.exp(0.5 * (sigma / tau) ** 2 - d[~near] / tau) * erfc(z[~near])
    return area * out / (2 * tau)


def fit_emg(window, candidates) -> list[dict]:
    """An EMG a candidate, and a straight baseline, fitted by least squares to each cluster of candidates (those
    whose reaches, REACH times their 5 % widths, overlap), from their tops and widths; the noise, of the differences
    of the points beside each other (as nothing tells it), for the areas' sigma."""
    y = window["signal"].astype(np.float64)
    use = measured(window)
    noise = 1.4826 * np.median(np.abs(np.diff(y[use]))) / np.sqrt(2) if use.sum() > 2 else 1.0
    clusters, reach = [], -np.inf
    for candidate in sorted(candidates, key=lambda c: c["apex"]):
        low, high = candidate["apex"] - REACH * candidate["left"] - 2, candidate["apex"] + REACH * candidate["right"] + 2
        if clusters and low <= reach:
            clusters[-1].append(candidate)
            reach = max(reach, high)
        else:
            clusters.append([candidate])
            reach = high
    estimates = []
    for cluster in clusters:
        low = int(max(np.floor(min(c["apex"] - REACH * c["left"] - 2 for c in cluster)), 0))
        high = int(min(np.ceil(max(c["apex"] + REACH * c["right"] + 2 for c in cluster)), len(y) - 1))
        estimates += _fit_cluster(window, y, np.flatnonzero(use[low:high + 1]) + low, cluster, noise or 1.0)
    return estimates


def _fit_cluster(window, y, points, cluster, noise) -> list[dict]:
    if len(points) < 4 * len(cluster) + 2:
        return [{"apex": c["apex"], "confidence": c["confidence"], "area": 0.0} for c in cluster]
    base, middle = float(np.percentile(y[points], 5)), float(points.mean())
    start, lower, upper = [], [], []
    for c in cluster:
        left, right = max(c["left"], 0.5), max(c["right"], 0.5)
        # the 5 % width of a Gaussian 2.45 sigmas; the tail's side the longer by some 3 taus
        sigma = max(min(left, right) / 2.45, 0.3)
        tau = float(np.sign(right - left)) * max(abs(right - left) / 3, 0.05 * sigma)
        height = max(float(y[int(round(np.clip(c["apex"], 0, len(y) - 1)))]) - base, noise)
        # its widths and tail within its 5 % widths' sum: looser, a small one beside a large one, or of a low S/N,
        # grew over the large one's tail and the baseline (areas up to 2000 times its own)
        span = left + right
        start += [height * (2.5066 * sigma + abs(tau)), c["apex"], sigma, tau]
        lower += [0.0, c["apex"] - left - 1, 0.3, -span - 1]
        upper += [np.inf, c["apex"] + right + 1, span + 1, span + 1]
    start, lower, upper = np.array(start + [base, 0.0]), np.array(lower + [-np.inf] * 2), np.array(upper + [np.inf] * 2)
    start = np.clip(start, lower, upper)

    def model(x, at):
        return x[-2] + x[-1] * (at - middle) + sum(emg(at, *x[4 * i:4 * i + 4]) for i in range(len(cluster)))

    def jacobian(x):
        """Each EMG's parameters of its own differences only (of all the EMGs again, a cluster of K took K² times)."""
        out = np.empty((len(points), len(x)))
        for i in range(len(cluster)):
            own = x[4 * i:4 * i + 4]
            at = emg(points, *own)
            for j in range(4):
                step = 1e-6 * max(abs(own[j]), 1e-3)
                moved = own.copy()
                moved[j] += step
                out[:, 4 * i + j] = (emg(points, *moved) - at) / step
        out[:, -2], out[:, -1] = 1.0, points - middle
        return out

    fit = least_squares(lambda x: model(x, points) - y[points], start, jac=jacobian, bounds=(lower, upper),
                        x_scale="jac", max_nfev=100 * len(start))
    covariance = noise ** 2 * np.linalg.pinv(fit.jac.T @ fit.jac)
    t, valid = np.arange(len(y), dtype=np.float64), window["valid"]
    out = []
    for i, c in enumerate(cluster):
        shape = emg(t, *fit.x[4 * i:4 * i + 4])
        area = float(shape[valid].sum())
        part = area / fit.x[4 * i] if fit.x[4 * i] > 0 else 1.0  # of its whole area, in the window
        out.append({"apex": c["apex"], "confidence": c["confidence"], "area": area,
                    "sigma": float(np.sqrt(max(covariance[4 * i, 4 * i], 0.0)) * part) or None, "shape": shape})
    return out


### the app's ###
def app(window, candidates) -> list[dict]:
    """The app's: each candidate's top the highest point within APP_SNAP of it; each top's range from where the trace
    rises to where it falls back (APP_END_HEIGHT of its prominence), not past the lowest point between it and the
    next; its area, of the points in the range, by trapezoids less a flat baseline at their lowest. A candidate whose
    top is a more confident one's, or no higher than the points beside it, has none."""
    length = int(window["valid"].sum())
    y = window["signal"][:length].astype(np.float64)
    tops: dict[int, dict] = {}
    for c in sorted(candidates, key=lambda c: -c["confidence"]):
        near = int(round(c["apex"]))
        low, high = max(near - APP_SNAP, 0), min(near + APP_SNAP + 1, length)
        tops.setdefault(low + int(np.argmax(y[low:high])), c)
    kept = np.array(sorted(top for top in tops if 0 < top < length - 1 and y[top] > y[top - 1] and y[top] > y[top + 1]),
                    dtype=int)
    areas = {}
    if len(kept):
        _widths, _heights, starts, ends = peak_widths(y, kept, rel_height=APP_END_HEIGHT,
                                                      prominence_data=peak_prominences(y, kept))
        for i in range(len(kept) - 1):
            valley = kept[i] + int(np.argmin(y[kept[i]:kept[i + 1] + 1]))
            ends[i], starts[i + 1] = min(ends[i], valley), max(starts[i + 1], valley)
        for top, first, last in zip(kept, starts, ends):
            points = y[int(np.ceil(first)):int(np.floor(last)) + 1]
            areas[top] = float((points[1:] + points[:-1]).sum() / 2 - points.min() * (len(points) - 1)) \
                if len(points) > 1 else 0.0
    area_of = {id(c): areas.get(top, 0.0) for top, c in tops.items()}
    return [{"apex": c["apex"], "confidence": c["confidence"], "area": area_of.get(id(c), 0.0)} for c in candidates]


### the later stage trained ###
def trained(checkpoint: str, device: str = "cpu", real: float | None = None, strength: float | None = None,
            low: float | None = None, calibration: dict | None = None):
    """The later stage of a checkpoint as a method (the candidates it takes), and the real, strength and low it takes
    them of: those it was checked with in training (of a run before low, deconvolve_models.LOW), unless given."""
    import torch

    from deconvolve_models import LOW, deconvolve, load

    model = load(checkpoint, device)
    config = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
    real = config["real"] if real is None else real
    strength = config["strength"] if strength is None else strength
    low = config.get("low", LOW) if low is None else low

    def method(window, candidates) -> list[dict]:
        found = deconvolve(model, window["signal"], window["missing"], window["valid"], candidates, device, real,
                           strength, low, calibration)
        return [peak for peak in found["peaks"] if peak["real"]]

    return method, real, strength, low


METHODS = {"truth": truth, "emg": fit_emg, "app": app}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=(*METHODS, "model"))
    parser.add_argument("--candidates", default="oracle", choices=("oracle", "perturbed", "front"),
                        help="the true peaks' (truth always), as the front's are off (deconvolve.CandidateErrors), "
                             "or the front's (--front)")
    parser.add_argument("--front", help="of --candidates front: a checkpoint of train.py")
    parser.add_argument("--confidence", type=float, default=0.0, help="of the candidates, those from it only")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--checkpoint", help="of the model (train_deconvolution.py)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--real", type=float, help="of the model: the realness from which a candidate is taken")
    parser.add_argument("--strength", type=float, help="of the model: the heights' pull to the network's")
    parser.add_argument("--low", type=float, help="of the model: the realness from which one not taken is fitted to "
                                                  "the others (from --real: none)")
    parser.add_argument("--calibration", help="of the model: of its areas' sigma (calibrate_deconvolution.py)")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.method == "model" and not args.checkpoint:
        parser.error("--method model needs --checkpoint")
    if args.candidates == "front" and not args.front:
        parser.error("--candidates front needs --front")
    method = METHODS.get(args.method)
    if args.method == "model":
        calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else None
        method, args.real, args.strength, args.low = trained(args.checkpoint, args.device, args.real, args.strength,
                                                             args.low, calibration)
    front = None
    if args.candidates == "front" and args.method != "truth":
        from deconvolve_data import Front, inputs  # of torch

        front = Front(args.front, args.device)
    started, scores = time.time(), Scores()
    for window, rng in windows(args.count, args.length):
        if front:
            x = inputs(window["signal"], window["missing"], window["valid"], [])["x"]
            candidates = front(x, int(window["valid"].sum()))
        else:
            candidates = perturbed_candidates(window, rng) if args.candidates == "perturbed" \
                else oracle_candidates(window)
        candidates = [c for c in candidates if c["confidence"] >= args.confidence]
        scores.add(window, method(window, candidates))
    summary = {"method": args.method, "candidates": "oracle" if args.method == "truth" else args.candidates,
               "windows": args.count, "length": args.length, "simulator_version": VERSION,
               "seconds": round(time.time() - started, 1)} | scores.summary()
    if front:
        summary |= {"front": args.front, "confidence": args.confidence}
    if args.method == "model":
        summary |= {"checkpoint": args.checkpoint, "real": args.real, "strength": args.strength, "low": args.low,
                    "calibration": args.calibration}
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
