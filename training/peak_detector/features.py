"""The input of the peak detector from a signal, the three channels of SPEC.md 4. numpy only, so that the app makes
them as training does.

    linear: the signal less a low percentile of it, over its highest point so
    log:    the same over the noise, sign(v) log1p(|v|), over log1p(1e6): an SNR of 1e6 is 1
    missing: 1 where a point was not measured (interpolated across)
"""

from dataclasses import dataclass

import numpy as np

BASELINE_PERCENTILE = 5
LOG_SCALE = float(np.log1p(1e6))
CHANNELS = 3


@dataclass(frozen=True)
class Scaling:
    """How a signal was brought to the channels, to bring another (its clean version) to the same."""
    baseline: float
    top: float
    noise: float


def noise_level(signal: np.ndarray, missing: np.ndarray | None = None) -> float:
    """The standard deviation of the noise, of the median absolute difference of a point from the next: little
    moved by peaks and by a slow baseline."""
    values = np.asarray(signal, dtype=np.float64)
    if missing is not None and (~missing).sum() >= 3:
        values = values[~missing]
    if len(values) < 3:
        return 1.0
    differences = np.diff(values)
    sigma = 1.4826 * np.median(np.abs(differences - np.median(differences))) / np.sqrt(2)
    if sigma <= 0:
        sigma = np.std(differences) / np.sqrt(2)
    return float(sigma) if sigma > 0 else 1.0


def scaling_of(signal: np.ndarray, missing: np.ndarray | None = None) -> Scaling:
    values = np.asarray(signal, dtype=np.float64)
    measured = values if missing is None or missing.all() else values[~missing]
    baseline = float(np.percentile(measured, BASELINE_PERCENTILE))
    noise = noise_level(values, missing)
    top = max(float(measured.max()) - baseline, noise)
    return Scaling(baseline, top, noise)


def features(signal: np.ndarray, missing: np.ndarray | None = None, scaling: Scaling | None = None) -> tuple[np.ndarray, Scaling]:
    """The channels (3, length), float32, and the scaling used; that of the signal itself unless given."""
    values = np.asarray(signal, dtype=np.float64)
    missing = np.zeros(len(values), dtype=bool) if missing is None else np.asarray(missing, dtype=bool)
    scaling = scaling or scaling_of(values, missing)
    above = values - scaling.baseline
    linear = above / scaling.top
    log = np.sign(above) * np.log1p(np.abs(above) / scaling.noise) / LOG_SCALE
    return np.stack([linear, log, missing.astype(np.float64)]).astype(np.float32), scaling
