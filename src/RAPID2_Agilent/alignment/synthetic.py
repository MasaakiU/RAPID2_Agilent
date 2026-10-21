"""Reproducible scan spectra with known smooth drift and condition-specific ions."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq


@dataclass
class ArrayScanSource:
    rt_seconds: np.ndarray
    mz: np.ndarray
    intensities: np.ndarray
    metadata: dict

    @property
    def scan_ids(self) -> np.ndarray:
        return np.arange(len(self.rt_seconds), dtype=np.int64)

    def read_scan(self, position: int) -> tuple[np.ndarray, np.ndarray]:
        mz = self.mz if self.mz.ndim == 1 else self.mz[position]
        return mz, self.intensities[position]


@dataclass
class SyntheticPair:
    reference: ArrayScanSource
    query: ArrayScanSource
    query_to_reference_truth: np.ndarray
    drift_amplitude: float
    duration: float


def generate_synthetic_pair(
    *, n_reference: int = 101, n_query: int = 117, n_shared_ions: int = 140,
    duration: float = 300.0, drift_amplitude: float = 10.0,
    strong_amplitude: float = 1e6, seed: int = 7,
) -> SyntheticPair:
    """t_query = t_latent + amplitude*sin(2*pi*t_latent/duration).

    Different scan schedules sample shared Gaussian elution profiles. Each ion
    contributes three raw profile samples, with scan-specific small m/z jitter.
    These are synthetic m/z spectra, not assumptions about Agilent's TOF grid.
    An A-only ion at m/z 1000 and a B-only ion at m/z 1200 dwarf the shared ions.
    """
    if min(n_reference, n_query, n_shared_ions) < 2 or duration <= 0:
        raise ValueError("Require positive duration and at least two scans/ions")
    if not np.isfinite(drift_amplitude) or abs(drift_amplitude) * 2 * np.pi >= duration:
        raise ValueError("Drift must preserve strict RT monotonicity")
    if not np.isfinite(strong_amplitude) or strong_amplitude < 0:
        raise ValueError("strong_amplitude must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    reference_rt = np.linspace(0, duration, n_reference)
    query_rt = np.linspace(0, duration, n_query)

    def forward(t: float) -> float:
        return t + drift_amplitude * np.sin(2 * np.pi * t / duration)

    query_latent = np.array([brentq(lambda t: forward(t) - q, 0, duration) for q in query_rt])
    centers = rng.uniform(0, duration, n_shared_ions)
    widths = rng.uniform(duration / 50, duration / 20, n_shared_ions)
    amplitudes = rng.uniform(2, 12, n_shared_ions)
    ion_mz = np.r_[np.geomspace(80, 850, n_shared_ions), 1000.0, 1200.0]
    sample_offsets = np.array([-1.5, 0, 1.5]) * 1e-6
    sample_weights = np.array([0.25, 1.0, 0.25])
    sources = []
    for run, (rt, latent) in enumerate(((reference_rt, reference_rt), (query_rt, query_latent))):
        shared = amplitudes[None, :] * np.exp(-0.5 * ((latent[:, None] - centers) / widths) ** 2)
        features = np.pad(shared, ((0, 0), (0, 2)))
        strong_center = duration * (0.34 if run == 0 else 0.67)
        features[:, n_shared_ions + run] = strong_amplitude * np.exp(-0.5 * ((latent - strong_center) / (duration / 30)) ** 2)
        intensity = (features[:, :, None] * sample_weights).reshape(len(rt), -1)
        intensity += rng.uniform(0, 0.005, intensity.shape)
        jitter = rng.normal(0, 0.15e-6, size=(len(rt), 1, 1))
        mz = (ion_mz[None, :, None] * (1 + sample_offsets[None, None, :] + jitter)).reshape(len(rt), -1)
        metadata = {
            "kind": "synthetic", "name": "reference" if run == 0 else "query", "seed": seed,
            "ms_level": 1, "polarity": 0, "rt_unit": "seconds", "n_shared_ions": n_shared_ions,
            "duration": duration, "drift_amplitude": drift_amplitude,
            "strong_amplitude": strong_amplitude,
        }
        sources.append(ArrayScanSource(rt, mz, intensity.astype(np.float32), metadata))
    return SyntheticPair(sources[0], sources[1], query_latent, drift_amplitude, duration)


def mapping_errors(estimated: np.ndarray, truth: np.ndarray) -> dict:
    estimated, truth = np.asarray(estimated), np.asarray(truth)
    if estimated.shape != truth.shape or not np.all(np.isfinite(estimated)):
        raise ValueError("Mapping estimates must be finite and match truth shape")
    error = estimated - truth
    return {"mae_seconds": float(np.mean(np.abs(error))),
            "rmse_seconds": float(np.sqrt(np.mean(error ** 2))),
            "max_abs_seconds": float(np.max(np.abs(error)))}
