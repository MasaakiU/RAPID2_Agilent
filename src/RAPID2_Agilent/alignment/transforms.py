"""Intensity transformations applied only to already aggregated bins."""

import numpy as np

from .backend import Backend, process_rows, resolve_backend
from .config import TransformConfig


def transform_intensity(
    values: np.ndarray, config: TransformConfig, *, out: np.ndarray | None = None,
    backend: Backend | None = None, chunk_size: int = 64,
) -> np.ndarray:
    """Transform vectors or matrices; out=values avoids an extra full matrix.

    Only asinh uses scale. sqrt and log1p mean sqrt(I) and log(1+I).
    """
    config.validate_for_use()
    values = np.asarray(values)
    if values.ndim not in {1, 2} or values.size == 0:
        raise ValueError("Expected a nonempty intensity vector or matrix")
    if out is None:
        out = np.empty(values.shape, dtype=np.float32)
    if out.shape != values.shape or out.dtype != np.float32:
        raise ValueError("out must match the input shape and be float32")
    backend = backend or resolve_backend("cpu")
    xp = np if backend.torch is None else backend.torch

    def transform(block):
        raw = backend.numpy(block) if backend.torch is None else None
        if raw is not None and (not np.all(np.isfinite(raw)) or np.any(raw < 0)):
            raise ValueError("Binned intensity must be finite and nonnegative")
        if backend.torch is not None:
            if not bool((xp.isfinite(block) & (block >= 0)).all()):
                raise ValueError("Binned intensity must be finite and nonnegative")
        if config.method == "none":
            return block
        if config.method == "sqrt":
            return xp.sqrt(block)
        if config.method == "log1p":
            return xp.log1p(block)
        # A float64 temporary keeps I/scale finite for ordinary float32 inputs.
        wide = block.astype(np.float64) if backend.torch is None else block.to(xp.float64)
        result = xp.asinh(wide / config.scale)
        return result.astype(np.float32) if backend.torch is None else result.to(xp.float32)

    process_rows(values.reshape(-1, values.shape[-1]), out.reshape(-1, values.shape[-1]),
                 transform, backend, chunk_size)
    return out


def suggest_scales(
    values: np.ndarray, *, percentiles: tuple[float, ...] = (10, 25, 75, 90),
    max_samples: int = 100_000, seed: int = 0,
) -> dict:
    """Suggest, never select, scales from nonzero *binned* intensities.

    Reservoir sampling bounds storage and samples all scan rows. The lower-tail
    median is only a noise-level proxy: positive ions alone cannot identify the
    instrument noise floor. The caller must explicitly choose an analysis scale.
    """
    values = np.asarray(values)
    if values.ndim not in {1, 2} or max_samples < 1:
        raise ValueError("Expected a vector/matrix and positive max_samples")
    if any(not 0 <= p <= 100 for p in percentiles):
        raise ValueError("Percentiles must be in [0, 100]")
    rng = np.random.default_rng(seed)
    sample = np.empty(0, dtype=np.float64)
    seen = 0
    for row in values.reshape(-1, values.shape[-1]):
        positive = row[np.isfinite(row) & (row > 0)]
        count = len(positive)
        if not count:
            continue
        total = seen + count
        size = min(max_samples, total)
        take_new = int(rng.hypergeometric(count, seen, size))
        old = sample[rng.choice(len(sample), size - take_new, replace=False)]
        new = positive[rng.choice(count, take_new, replace=False)]
        sample = np.concatenate((old, new))
        seen = total
    if not len(sample):
        raise ValueError("No finite nonzero binned intensities for scale suggestions")
    lower_tail = sample[sample <= np.percentile(sample, 25)]
    return {
        "median": float(np.median(sample)),
        "percentiles": {str(p): float(np.percentile(sample, p)) for p in percentiles},
        "noise_lower_tail_proxy": float(np.median(lower_tail)),
        "nonzero_count": seen, "sample_count": len(sample), "seed": seed,
        "note": "Candidates only; lower-tail proxy is not a measured instrument noise floor.",
    }
