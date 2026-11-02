"""The later stage's training data (SPEC_deconvolution.md 2, 4), made as it is used: windows of simulated samples
(data.cut), the candidates the front gives of them (Front; in some, worse, deconvolve.roughened), the input channels
(features.py), and the targets of each candidate and of the window. Without a front, candidates made of the peaks as
off as a front's (deconvolve.perturbed_candidates).

Heights, areas, the baseline, the others and the noise are in the linear channel's units: less the window's
baseline (features.Scaling) and over its top.

    x:           (3, N) the input channels
    weighed:     (N) 1 where the points tell of the heights (deconvolve.weighed)
    present:     (N) 1 where a peak may be (deconvolve.present)
    valid:       (N) 1 in the window
    scaling:     (3) the window's baseline, top and noise (features.Scaling), to the signal's units
    candidates:  (K, 4) apex, confidence, left and right widths to 5 %, in points
  the targets, the candidates matched to the peaks with a shape as deconvolve.Scores matches them:
    real:        (K) 1 of a positive's and of a peak not scored's of S/N from REAL_SNR (in the signal for sure: of
                 a close pair the smaller, a wider one's under a narrower one, one at an end), 0 of none's;
                 real_weight 0 of the other peaks not scored (of S/N 2-5), as of one of nothing in the window
    matched:     (K) 1 of a peak with a shape
    shape:       (K, N) its peak's profile over its top, on the window; 0 of none's
    height:      (K) its peak's top
    area:        (K) its peak's profile summed over the window
    spans:       (C, 2) of the clusters of the candidates matched (deconvolve.clusters), where the baseline is
                 corrected
    baseline:    (N)
    others:      (N) with the profiles of the negatives and of the peaks matched by none
    sigma:       (N) of the noise; sigma_mask 1 where it is known (> 0)

A batch (collate) has the candidates of its items padded to the most of them, candidate_mask 1 of those there, and
their shapes one after another, (n, N) of the n there, in candidate_mask's order; the spans padded so, span_mask.
"""

from dataclasses import replace

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from data import cut
from decode import assign, decode
from deconvolve import CandidateErrors, clusters, full, perturbed_candidates, present, roughened, shaped, weighed
from features import features, scaling_of
from models import build as build_front
from simulator import Config, simulate

VALIDATION_SEED = 1_000_007  # apart from the front's (data.py) and the references' (evaluate_deconvolution.py)
REAL_SNR = Config().snr_positive  # of a peak not scored, from which its candidate is to be taken as a positive's
CANDIDATE_KEYS = ("apex", "confidence", "left", "right")
PER_CANDIDATE = ("candidates", "real", "real_weight", "matched", "height", "area")


class Front:
    """The front of a checkpoint of train.py, as it gives the later stage its candidates: its apexes of a window's
    input channels (inputs' x, as the front's training makes them, data.py), from decode's threshold, in the window."""

    def __init__(self, checkpoint: str, device="cpu"):
        self.checkpoint, self.device, self.model = checkpoint, device, None

    def __getstate__(self):
        return self.__dict__ | {"model": None}  # to a worker, the checkpoint only, which it loads of its own

    def __call__(self, x: np.ndarray, length: int) -> list[dict]:
        if self.model is None:
            saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            self.model = build_front(saved["config"]["model"], **saved["config"]["model_options"])
            self.model.load_state_dict(saved["model"])
            self.model = self.model.to(self.device).eval()
        with torch.no_grad():
            out = self.model(torch.from_numpy(x)[None].to(self.device))
        heat = torch.sigmoid(out["heat"][0].float()).cpu().numpy()
        return decode(heat, out["offset"][0].float().cpu().numpy(), out["log_widths"][0].float().cpu().numpy(),
                      self.model.stride, length=length)


def table(candidates) -> np.ndarray:
    """Candidates as the model takes them: (K, 4) of CANDIDATE_KEYS."""
    return np.array([[c[key] for key in CANDIDATE_KEYS] for c in candidates], dtype=np.float32).reshape(-1, 4)


def inputs(signal, missing, valid, candidates) -> dict:
    """The arrays the model takes of a window (valid True in it) and its candidates."""
    signal, missing, valid = np.asarray(signal), np.asarray(missing, dtype=bool), np.asarray(valid, dtype=bool)
    scaling = scaling_of(signal[valid], missing[valid])
    x, _ = features(signal, missing, scaling)
    return {"x": x, "weighed": weighed(signal, missing, valid).astype(np.float32),
            "present": present(signal, missing, valid).astype(np.float32), "valid": valid.astype(np.float32),
            "scaling": np.array([scaling.baseline, scaling.top, scaling.noise]), "candidates": table(candidates)}


def targets(window, candidates, scaling) -> dict:
    """The targets of a window's candidates and of the window, of its scaling (baseline, top, noise)."""
    baseline, top = float(scaling[0]), float(scaling[1])
    length, valid = len(window["signal"]), window["valid"]
    peaks = shaped(window)
    taken = [replace(peak, label="positive") for peak in peaks]
    of = {id(as_taken): peak for as_taken, peak in zip(taken, peaks)}
    index = {id(candidate): k for k, candidate in enumerate(candidates)}
    count = len(candidates)
    out = {name: np.zeros(count, dtype=np.float32) for name in ("real", "matched", "height", "area")}
    out["real_weight"] = np.ones(count, dtype=np.float32)
    out["shape"] = np.zeros((count, length), dtype=np.float32)
    found = set()
    for candidate, peak, _counts in assign(candidates, taken):
        if peak is None:
            continue
        k, peak = index[id(candidate)], of[id(peak)]
        profile = np.where(valid, full(peak, length), 0.0)
        if (peak.label == "positive" or peak.snr >= REAL_SNR) and profile.max() > 0:
            out["real"][k] = 1.0
        else:
            out["real_weight"][k] = 0.0
        if profile.max() > 0:
            found.add(id(peak))
            out["matched"][k], out["height"][k] = 1.0, profile.max() / top
            out["area"][k], out["shape"][k] = profile.sum() / top, profile / profile.max()
    others = window["others"].astype(np.float64)
    for peak in window["peaks"]:
        if id(peak) not in found:
            others = others + full(peak, length)
    sigma = window["sigma"].astype(np.float64)
    spans = clusters([c for c, matched in zip(candidates, out["matched"]) if matched], int(valid.sum()))
    return out | {"spans": np.array(spans, dtype=np.float32).reshape(-1, 2),
                  "baseline": ((window["baseline"] - baseline) / top).astype(np.float32),
                  "others": (others / top).astype(np.float32), "sigma": (sigma / top).astype(np.float32),
                  "sigma_mask": (valid & (sigma > 0)).astype(np.float32)}


def make_item(sample, length: int, rng: np.random.Generator, errors: CandidateErrors = CandidateErrors(),
              front: Front | list[Front] | None = None, rough: float = 0.0) -> tuple[dict, dict, list[dict]]:
    """The arrays of a training item, its window and its candidates, of a sample: the front's candidates (of fronts,
    one's at random), roughened by errors in rough of the windows; without a front, those of
    perturbed_candidates."""
    window = cut(sample, length, rng)
    if isinstance(front, list):
        front = front[rng.integers(len(front))]
    if front is None:
        candidates = perturbed_candidates(window, rng, errors)
        item = inputs(window["signal"], window["missing"], window["valid"], candidates)
    else:
        item = inputs(window["signal"], window["missing"], window["valid"], [])
        count = int(window["valid"].sum())
        candidates = front(item["x"], count)
        if rng.random() < rough:
            candidates = roughened(candidates, count, rng, errors)
        item["candidates"] = table(candidates)
    return item | targets(window, candidates, item["scaling"]), window, candidates


def collate(items: list[dict]) -> dict[str, torch.Tensor]:
    """Items (inputs, with targets or not) as a batch."""
    batch = {}
    for key in items[0]:
        values = [np.asarray(item[key]) for item in items]
        if key == "shape":
            batch[key] = torch.from_numpy(np.concatenate(values))
            continue
        if key in PER_CANDIDATE + ("spans",):
            most = max([1] + [len(v) for v in values])
            values = [np.concatenate([v, np.zeros((most - len(v),) + v.shape[1:], dtype=v.dtype)]) for v in values]
        batch[key] = torch.from_numpy(np.stack(values))
    for key, mask in (("candidates", "candidate_mask"), ("spans", "span_mask")):
        if key in batch:
            counts = np.array([len(item[key]) for item in items])
            batch[mask] = torch.from_numpy(np.arange(batch[key].shape[1])[None, :] < counts[:, None])
    return batch


class SimulatedWindows(IterableDataset):
    """Items made without end, each worker of its own random numbers (seed, worker), and of its own front (or fronts)
    on the CPU, of a thread (unet v5: some 20 ms a window of 2048 points)."""

    def __init__(self, length: int, seed: int = 0, config: Config = Config(),
                 errors: CandidateErrors = CandidateErrors(), front: Front | list[Front] | None = None,
                 rough: float = 0.0):
        self.length, self.seed, self.config, self.errors = length, seed, config, errors
        self.front, self.rough = front, rough

    def __iter__(self):
        info = get_worker_info()
        if info is not None:
            torch.set_num_threads(1)
        rng = np.random.default_rng([self.seed, 0 if info is None else info.id])
        while True:
            yield make_item(simulate(rng, self.config), self.length, rng, self.errors, self.front, self.rough)[0]


def validation_set(count: int, length: int, config: Config = Config(), errors: CandidateErrors = CandidateErrors(),
                   front: Front | list[Front] | None = None) -> list[tuple[dict, dict, list[dict]]]:
    """The same items every time: (arrays, window, candidates) each; the front's candidates as they are (of fronts,
    of each in turn)."""
    out = []
    for i in range(count):
        rng = np.random.default_rng([VALIDATION_SEED, i])
        given = front[i % len(front)] if isinstance(front, list) else front
        out.append(make_item(simulate(rng, config), length, rng, errors, given))
    return out
