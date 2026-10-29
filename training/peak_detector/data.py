"""Training data made as it is used, from the simulator: samples cut or padded to one length, their channels
(features.py) and their targets (targets.py). torch only for the dataset."""

from dataclasses import replace

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from features import features, scaling_of
from simulator import Config, simulate
from targets import make_targets

VALIDATION_SEED = 1_000_003  # of the validation samples, apart from the training ones


def cut(sample, length: int, rng: np.random.Generator) -> dict:
    """A window of length points of the sample, at random where it is longer, padded at its end with its last value
    where shorter (valid False there); its peaks moved into it. A peak whose top is near an end of the window, or
    just past it, is not scored there. Of a sample not measured over a part, the window is where it has that part,
    or, of a part longer than it, where the signal comes back in its middle."""
    n = len(sample.signal)
    start = 0
    if n >= length:
        low, high = 0, n - length
        if sample.meta.get("zeros"):
            first, end = sample.meta["zeros"]
            low, high = max(low, end - length), min(high, first)
            if low > high:
                low = high = min(max(end - length // 2, 0), n - length)
        start = int(rng.integers(low, high + 1))
    shown = min(length, n)

    def window(values, fill):
        part = values[start:start + shown]
        return part if shown == length else np.concatenate([part, np.full(length - shown, fill, dtype=part.dtype)])

    peaks = []
    for peak in sample.peaks:
        apex = peak.apex - start
        if -peak.fwhm / 2 <= apex <= shown - 1 + peak.fwhm / 2:
            label = peak.label
            if label == "positive" and (apex < peak.fwhm / 2 or apex > shown - 1 - peak.fwhm / 2):
                label = "ignore"
            peaks.append(replace(peak, apex=float(np.clip(apex, 0, shown - 1)), label=label))
    valid = np.zeros(length, dtype=bool)
    valid[:shown] = True
    return {"signal": window(sample.signal, sample.signal[-1]), "clean": window(sample.clean, sample.clean[-1]),
            "missing": window(sample.missing, False), "valid": valid, "peaks": peaks, "meta": sample.meta}


def make_item(sample, length: int, stride: int, rng: np.random.Generator) -> tuple[dict, dict]:
    """The arrays of a training item, and its window (with its peaks), of a sample."""
    window = cut(sample, length, rng)
    valid = window["valid"]
    scaling = scaling_of(window["signal"][valid], window["missing"][valid])
    x, _ = features(window["signal"], window["missing"], scaling)
    clean, _ = features(window["clean"], window["missing"], scaling)
    item = {"x": x, "clean": clean[:2], "valid": valid.astype(np.float32)}
    item |= make_targets(window["peaks"], length, stride, valid)
    return item, window


def to_tensors(item: dict) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in item.items()}


class SimulatedPeaks(IterableDataset):
    """Items made without end, each worker of its own random numbers (seed, worker)."""

    def __init__(self, length: int, stride: int, seed: int = 0, config: Config = Config()):
        self.length, self.stride, self.seed, self.config = length, stride, seed, config

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng([self.seed, 0 if info is None else info.id])
        while True:
            yield to_tensors(make_item(simulate(rng, self.config), self.length, self.stride, rng)[0])


def validation_set(count: int, length: int, stride: int, config: Config = Config()) -> list[tuple[dict, dict]]:
    """The same items every time: (tensors, window) each."""
    items = []
    for i in range(count):
        rng = np.random.default_rng([VALIDATION_SEED, i])
        item, window = make_item(simulate(rng, config), length, stride, rng)
        items.append((to_tensors(item), window))
    return items
