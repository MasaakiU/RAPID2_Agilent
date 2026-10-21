"""Constant-ppm half-open bins, followed by aggregation of raw intensities."""

from dataclasses import dataclass, field
import numpy as np

from .config import BinningConfig


@dataclass(frozen=True)
class PPMGrid:
    config: BinningConfig
    edges: np.ndarray = field(init=False, repr=False)
    log_step: float = field(init=False)

    def __post_init__(self) -> None:
        c = self.config
        step = float(np.log1p(c.ppm * 1e-6))
        n = int(np.ceil(np.log(c.mz_max / c.mz_min) / step))
        edges = c.mz_min * np.exp(np.arange(n + 1, dtype=np.float64) * step)
        edges[-1] = c.mz_max  # last bin can be narrower
        if not np.all(np.diff(edges) > 0):
            raise ValueError("ppm is too small for a strictly increasing floating-point grid")
        edges.flags.writeable = False
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "log_step", step)

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    def indices(self, mz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return an in-range mask and bin indices for [mz_min, mz_max).

        Logarithms supply O(N) indices. Edge comparisons correct one-bin
        roundoff at exact geometric boundaries, retaining half-open semantics.
        """
        mz = np.asarray(mz)
        valid = (mz >= self.config.mz_min) & (mz < self.config.mz_max)
        selected = mz[valid]
        indices = np.floor(np.log(selected / self.config.mz_min) / self.log_step).astype(np.int64)
        np.clip(indices, 0, self.n_bins - 1, out=indices)
        indices -= selected < self.edges[indices]
        indices += selected >= self.edges[indices + 1]
        return valid, indices


def aggregate_scan(
    mz: np.ndarray, intensity: np.ndarray, grid: PPMGrid,
    method: str | None = None, *, out: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate *before* transformation. Mean counts zero-valued samples too.

    Sum preserves total recorded sample intensity in range, not a calibrated
    integral over m/z. Max and mean deliberately define different representations.
    Empty bins are zero for all methods; input spectra need not share m/z grids.
    """
    mz, intensity = np.asarray(mz), np.asarray(intensity)
    if mz.ndim != 1 or intensity.shape != mz.shape:
        raise ValueError("m/z and intensity must be matching one-dimensional arrays")
    if not np.all(np.isfinite(mz)) or not np.all(np.isfinite(intensity)) or np.any(intensity < 0):
        raise ValueError("Raw samples must be finite with nonnegative intensities")
    method = method or grid.config.aggregation
    if method not in {"sum", "mean", "max"}:
        raise ValueError("Unknown aggregation method")
    if out is None:
        out = np.zeros(grid.n_bins, dtype=np.float32)
    elif out.shape != (grid.n_bins,) or out.dtype != np.float32:
        raise ValueError("out must be a float32 vector matching the grid")
    else:
        out.fill(0)
    mask, indices = grid.indices(mz)
    values = intensity[mask]
    if len(indices) == 0:
        return out
    if method == "max":
        np.maximum.at(out, indices, values)
    else:
        sums = np.bincount(indices, weights=values, minlength=grid.n_bins)
        if method == "mean":
            counts = np.bincount(indices, minlength=grid.n_bins)
            np.divide(sums, counts, out=sums, where=counts != 0)
        out[:] = sums
    if not np.all(np.isfinite(out)):
        raise ValueError("Binned intensities exceed float32 range")
    return out
