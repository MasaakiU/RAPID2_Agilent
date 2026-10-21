"""m/z x RT intensity map of the MS1 scans of one file, at three resolutions.

overview and mid are computed once in a single pass over the file; detail is
computed on demand for an m/z range. Every level keeps the max intensity of
each (scan, m/z bin). Bins are constant-ppm and nested: a mid bin is
MID_FACTOR detail bins, an overview bin is several mid bins. Detail and mid
bins of every file lie on one lattice, so that two files can be compared bin
by bin.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .rpd_model import BaseMSDataModel


DETAIL_PPM = 10.0
MID_FACTOR = 10             # mid bins ~ 100 ppm
MAX_OVERVIEW_BINS = 4096


@dataclass(frozen=True)
class PPMGrid:
    """Bins [edges[i], edges[i+1]) with edges[i] = mz_min * exp(i * log_step)."""

    mz_min: float
    log_step: float
    n_bins: int

    @classmethod
    def spanning(cls, mz_min: float, mz_max: float, ppm: float, snap_bins: int = 1) -> "PPMGrid":
        """Covers [mz_min, mz_max]. The first edge is exp(k * snap_bins * log_step) for an integer k,
        so grids of any ranges share their edges, also after coarsening by snap_bins."""
        log_step = math.log1p(ppm * 1e-6)
        snap_step = log_step * snap_bins
        start = math.exp(math.floor(math.log(mz_min) / snap_step) * snap_step)
        return cls(start, log_step, math.ceil(math.log(mz_max / start) / log_step))

    def coarsened(self, factor: int) -> "PPMGrid":
        return PPMGrid(self.mz_min, self.log_step * factor, math.ceil(self.n_bins / factor))

    @property
    def ppm(self) -> float:
        return math.expm1(self.log_step) * 1e6

    @property
    def lattice_index(self) -> int:
        """Index of the first bin on the lattice of all grids with this step."""
        return round(math.log(self.mz_min) / self.log_step)

    def edges(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Edges of the bins [start, stop)."""
        stop = self.n_bins if stop is None else stop
        return self.mz_min * np.exp(np.arange(start, stop + 1) * self.log_step)

    def bin_range(self, mz_min: float, mz_max: float) -> tuple[int, int]:
        """Bins overlapping [mz_min, mz_max], clipped to the grid."""
        start = math.floor(math.log(mz_min / self.mz_min) / self.log_step)
        stop = math.ceil(math.log(mz_max / self.mz_min) / self.log_step)
        start = min(max(start, 0), self.n_bins)
        stop = min(max(stop, start), self.n_bins)
        if stop == start:
            raise ValueError(f"m/z range {mz_min}-{mz_max} is outside the grid")
        return start, stop


@dataclass(frozen=True)
class MapLayer:
    rt: np.ndarray          # (n_scans,), increasing
    mz_edges: np.ndarray    # (n_bins + 1,)
    intensity: np.ndarray   # (n_scans, n_bins) float32, max per bin
    bin_ppm: float


class MapLevel(Enum):
    OVERVIEW = "overview"
    MID = "mid"
    DETAIL = "detail"


class MzRtMap:
    ms_level = 1

    def __init__(self, ms_data_model: BaseMSDataModel):
        self.ms_data_model = ms_data_model
        self.scan_indices = ms_data_model.scan_indices_for_ms_level(self.ms_level)
        self.rt = ms_data_model.RT_list[self.scan_indices]
        mz_min, mz_max = ms_data_model.profile_mz_range(self.scan_indices)
        self.detail_grid = PPMGrid.spanning(mz_min, mz_max, DETAIL_PPM, snap_bins=MID_FACTOR)
        self.mid_grid = self.detail_grid.coarsened(MID_FACTOR)
        self.overview_factor = math.ceil(self.mid_grid.n_bins / MAX_OVERVIEW_BINS)
        self.overview_grid = self.mid_grid.coarsened(self.overview_factor)
        self.mid: MapLayer | None = None
        self.overview: MapLayer | None = None

    @property
    def rt_range(self) -> tuple[float, float]:
        return float(self.rt[0]), float(self.rt[-1])

    @property
    def mz_range(self) -> tuple[float, float]:
        grid = self.mid_grid
        return grid.mz_min, grid.mz_min * math.exp(grid.n_bins * grid.log_step)

    def build(self, progress: Callable[[int, int], None], is_cancelled: Callable[[], bool]) -> bool:
        """Compute overview and mid; False if cancelled."""
        edges = self.mid_grid.edges()
        intensity = self._max_binned(edges, progress, is_cancelled)
        if intensity is None:
            return False
        self.mid = MapLayer(self.rt, edges, intensity, self.mid_grid.ppm)
        group_starts = np.arange(0, self.mid_grid.n_bins, self.overview_factor)
        self.overview = MapLayer(
            self.rt,
            self.overview_grid.edges(),
            np.maximum.reduceat(intensity, group_starts, axis=1),
            self.overview_grid.ppm,
        )
        return True

    def detail(self, mz_min: float, mz_max: float, is_cancelled: Callable[[], bool]) -> MapLayer | None:
        """Detail level for [mz_min, mz_max]; None if cancelled."""
        start, stop = self.detail_grid.bin_range(mz_min, mz_max)
        edges = self.detail_grid.edges(start, stop)
        intensity = self._max_binned(edges, lambda done, total: None, is_cancelled)
        if intensity is None:
            return None
        return MapLayer(self.rt, edges, intensity, self.detail_grid.ppm)

    def level_for(self, pixel_ppm: float) -> MapLevel:
        """Coarsest level whose bins are not coarser than a pixel."""
        if pixel_ppm >= self.overview_grid.ppm:
            return MapLevel.OVERVIEW
        if pixel_ppm >= self.mid_grid.ppm:
            return MapLevel.MID
        return MapLevel.DETAIL

    def default_scale(self) -> float:
        """Median of the nonzero overview intensities."""
        values = self.overview.intensity[self.overview.intensity > 0]
        if len(values) == 0:
            raise ValueError("The map has no signal")
        return float(np.median(values))

    def scale_range(self) -> tuple[float, float]:
        """Smallest nonzero and largest overview intensity: asinh(x / s) is log-like at
        the former and nearly linear at the latter."""
        values = self.overview.intensity[self.overview.intensity > 0]
        if len(values) == 0:
            raise ValueError("The map has no signal")
        return float(values.min()), float(values.max())

    def max_intensity(self) -> float:
        return float(self.overview.intensity.max())

    def _max_binned(self, edges: np.ndarray, progress, is_cancelled) -> np.ndarray | None:
        intensity = np.zeros((len(self.scan_indices), len(edges) - 1), dtype=np.float32)
        for row, scan_index in enumerate(self.scan_indices):
            if is_cancelled():
                return None
            self.ms_data_model.max_into_bins(int(scan_index), edges, intensity[row])
            progress(row + 1, len(self.scan_indices))
        return intensity


def rasterize(
    layer: MapLayer,
    rt_range: tuple[float, float],
    mz_range: tuple[float, float],
    shape: tuple[int, int],
) -> np.ndarray:
    """Max intensity per pixel of a (n_rt_pixels, n_mz_pixels) image on linear RT and m/z axes.

    A scan covers RT up to the next scan. A pixel smaller than a cell shows the
    cell under its left edge.
    """
    n_rt_pixels, n_mz_pixels = shape
    rt_step = float(np.median(np.diff(layer.rt))) if len(layer.rt) > 1 else 0.0
    rt_cells = _pixel_cells(layer.rt, float(layer.rt[-1]) + rt_step, rt_range, n_rt_pixels)
    mz_cells = _pixel_cells(layer.mz_edges[:-1], float(layer.mz_edges[-1]), mz_range, n_mz_pixels)
    image = np.zeros(shape, dtype=np.float32)
    if rt_cells is None or mz_cells is None:
        return image
    rt_slice, rt_starts, rt_inside = rt_cells
    mz_slice, mz_starts, mz_inside = mz_cells
    data = layer.intensity[rt_slice, mz_slice]
    pooled = np.maximum.reduceat(np.maximum.reduceat(data, rt_starts, axis=0), mz_starts, axis=1)
    image[np.ix_(rt_inside, mz_inside)] = pooled[np.ix_(rt_inside, mz_inside)]
    return image


def _pixel_cells(cell_starts: np.ndarray, cell_end: float, value_range: tuple[float, float], n_pixels: int):
    """Cells overlapping the range, and for each pixel the first cell under it (relative to the slice)."""
    low, high = value_range
    first = max(int(np.searchsorted(cell_starts, low, side="right")) - 1, 0)
    stop = int(np.searchsorted(cell_starts, high, side="left"))
    if stop <= first or high <= cell_starts[0] or cell_end <= low:
        return None
    pixel_edges = np.linspace(low, high, n_pixels + 1)
    starts = np.searchsorted(cell_starts, pixel_edges[:-1], side="right") - 1
    starts = np.clip(starts, first, stop - 1) - first
    centers = (pixel_edges[:-1] + pixel_edges[1:]) / 2
    inside = (cell_starts[0] <= centers) & (centers < cell_end)
    return slice(first, stop), starts, inside
