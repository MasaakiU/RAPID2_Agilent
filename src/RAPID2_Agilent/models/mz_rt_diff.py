"""Signed difference of two m/z x RT maps: B (query) aligned in RT onto A (reference).

D = asinh(c B / s) - asinh(A / s) on A's scans and m/z bins. Where both are well
above s, D is about ln(c B / A), the log fold change; below s it is about
(c B - A) / s, so that differences within the noise floor stay small. c
optionally normalizes the total intensity of B to that of A.

So that a slightly shifted peak does not show as a red/blue pair, each side is
compared with the maximum of the other within a tolerance window:

    D = max(0, T(c B) - max_window T(A)) - max(0, T(A) - max_window T(c B))

At most one of the two parts is nonzero. They are kept as separate nonnegative
layers, so that max pooling keeps the strongest difference of either sign.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .mz_rt_map import MapLayer, MzRtMap, rasterize


SECONDS_PER_MINUTE = 60.0
OVERVIEW_BLOCK_GROUPS = 256  # overview bins computed at once


@dataclass(frozen=True)
class RTAlignment:
    """B's scans on A's RT axis.

    B at A's scan i is (1 - weights[i]) * B[rows[i]] + weights[i] * B[rows[i] + 1].
    """

    warped_rt: np.ndarray   # (n_B,) B's scan RTs mapped onto A's RT axis
    rows: np.ndarray        # (n_A,) int
    weights: np.ndarray     # (n_A,) float32
    normalization: float    # median of A / B over the mid bins where both are strong
    # for diagnostics; scan indices are (A, B)
    similarity: np.ndarray  # (n_A, n_B) float32, of the mid profiles
    path: np.ndarray        # (n_steps, 2) DTW path
    anchors: np.ndarray     # (n_anchors, 2) one-to-one points of the path that the RT mapping goes through

    def max_shift(self, rt_b: np.ndarray) -> float:
        return float(np.max(np.abs(self.warped_rt - rt_b)))


@dataclass(frozen=True)
class DiffParams:
    scale: float              # s
    b_factor: float           # c
    rt_tolerance: int         # scans on each side
    mz_tolerance_ppm: float   # on each side, rounded to whole bins of each level


@dataclass(frozen=True)
class DiffLayers:
    higher_in_b: MapLayer     # max(0, D)
    higher_in_a: MapLayer     # max(0, -D)


@dataclass(frozen=True)
class DiffOverview:
    layers: DiffLayers
    # per scan of A, sums over the mid bins
    squared_higher_in_b: np.ndarray  # (n_A,) of max(0, D)^2
    squared_higher_in_a: np.ndarray  # (n_A,) of max(0, -D)^2


def align_maps(map_a: MzRtMap, map_b: MzRtMap, is_cancelled: Callable[[], bool]) -> RTAlignment | None:
    """OBI-Warp-like alignment of B to A on the mid layers; None if cancelled.

    NOTE: deliberately not `alignment.align_sources`. That one bins the raw scans
    itself; here the map already holds them binned, and binning again would read
    both files a second time. So only the similarity is computed here, from the
    mid layers with the map's own asinh scale, and everything after it is the
    shared `alignment.map_from_similarity`.

    The alignment package needs scipy, so it is imported only here.
    """
    from dataclasses import replace

    from RAPID2_Agilent.alignment import (
        AlignmentConfig,
        ComputeConfig,
        compute_similarity,
        map_from_similarity,
    )

    config = replace(AlignmentConfig(), extrapolation="linear")
    scale = map_a.default_scale()
    offset = _bin_offset(map_a.mid, map_b.mid)
    start, stop = max(0, -offset), min(map_a.mid.intensity.shape[1], map_b.mid.intensity.shape[1] - offset)
    if stop <= start:
        raise ValueError("A and B have no m/z range in common")
    profiles = []
    for layer, columns in ((map_a.mid, slice(start, stop)), (map_b.mid, slice(start + offset, stop + offset))):
        values = np.arcsinh(layer.intensity[:, columns] / np.float32(scale))
        profiles.append(np.ascontiguousarray(values, dtype=np.float32))
    if is_cancelled():
        return None
    similarity = compute_similarity(*profiles, ComputeConfig(device="cpu"), inplace=True)
    del profiles
    if is_cancelled():
        return None
    dtw, anchors, mapping = map_from_similarity(
        similarity, map_a.rt * SECONDS_PER_MINUTE, map_b.rt * SECONDS_PER_MINUTE, config,
    )
    path = dtw.path
    warped_rt = mapping.transform(map_b.rt * SECONDS_PER_MINUTE) / SECONDS_PER_MINUTE
    rows, weights = _resampling(warped_rt, map_a.rt)
    normalization = _normalization(map_a.mid, map_b.mid, rows, weights, threshold=scale)
    return RTAlignment(warped_rt, rows, weights, normalization, similarity, path, anchors)


class MzRtDiff:
    def __init__(self, map_a: MzRtMap, map_b: MzRtMap, alignment: RTAlignment):
        self.map_a = map_a
        self.map_b = map_b
        self.alignment = alignment

    def overview(self, params: DiffParams, is_cancelled: Callable[[], bool]) -> DiffOverview | None:
        """Parts on A's overview bins, from the mid bins; None if cancelled."""
        mid = self.map_a.mid
        n_rows, n_bins = mid.intensity.shape
        factor = self.map_a.overview_factor
        n_groups = self.map_a.overview.intensity.shape[1]
        pooled = [np.zeros((n_rows, n_groups), dtype=np.float32) for _ in range(2)]
        squared = [np.zeros(n_rows) for _ in range(2)]
        block = OVERVIEW_BLOCK_GROUPS * factor
        for start in range(0, n_bins, block):
            if is_cancelled():
                return None
            stop = min(start + block, n_bins)
            parts = self._parts(mid, self.map_b.mid, slice(0, n_rows), slice(start, stop), params)
            group_starts = np.arange(0, stop - start, factor)
            for out, sums, part in zip(pooled, squared, parts):
                out[:, start // factor:start // factor + len(group_starts)] = np.maximum.reduceat(part, group_starts, axis=1)
                sums += np.square(part, dtype=np.float64).sum(axis=1)
        overview = self.map_a.overview
        layers = DiffLayers(*(MapLayer(overview.rt, overview.mz_edges, part, overview.bin_ppm) for part in pooled))
        return DiffOverview(layers, *squared)

    def parts(
        self,
        a: MapLayer,
        b: MapLayer | None,
        rt_range: tuple[float, float],
        mz_range: tuple[float, float],
        params: DiffParams,
    ) -> DiffLayers | None:
        """Parts on the bins of a (mid or detail of A) covering the range; b is the same level of B,
        None where B has no bins. None if a does not overlap the range."""
        first_row = max(int(np.searchsorted(a.rt, rt_range[0], side="right")) - 1, 0)
        stop_row = min(int(np.searchsorted(a.rt, rt_range[1], side="left")) + 1, len(a.rt))
        first_bin = max(int(np.searchsorted(a.mz_edges, mz_range[0], side="right")) - 1, 0)
        stop_bin = min(int(np.searchsorted(a.mz_edges, mz_range[1], side="left")), len(a.mz_edges) - 1)
        if stop_row <= first_row or stop_bin <= first_bin:
            return None
        rows, bins = slice(first_row, stop_row), slice(first_bin, stop_bin)
        parts = self._parts(a, b, rows, bins, params)
        return DiffLayers(*(MapLayer(a.rt[rows], a.mz_edges[first_bin:stop_bin + 1], part, a.bin_ppm) for part in parts))

    def _parts(self, a: MapLayer, b: MapLayer | None, rows: slice, bins: slice, params: DiffParams):
        """(max(0, D), max(0, -D)) on A's scans rows and a's bins, reading the tolerance windows around them."""
        rt_margin = params.rt_tolerance
        mz_margin = round(params.mz_tolerance_ppm / a.bin_ppm)
        n_rows, n_bins = a.intensity.shape
        row_start, row_stop = max(rows.start - rt_margin, 0), min(rows.stop + rt_margin, n_rows)
        bin_start, bin_stop = max(bins.start - mz_margin, 0), min(bins.stop + mz_margin, n_bins)
        scale = np.float32(params.scale)
        t_a = np.arcsinh(a.intensity[row_start:row_stop, bin_start:bin_stop] / scale)
        b_on_a = _resampled(a, b, self.alignment.rows[row_start:row_stop], self.alignment.weights[row_start:row_stop], bin_start, bin_stop)
        t_b = np.arcsinh(b_on_a * np.float32(params.b_factor / params.scale))
        inner = (slice(rows.start - row_start, rows.stop - row_start), slice(bins.start - bin_start, bins.stop - bin_start))
        higher_in_b = np.maximum(t_b - _dilate(t_a, rt_margin, mz_margin), 0)[inner]
        higher_in_a = np.maximum(t_a - _dilate(t_b, rt_margin, mz_margin), 0)[inner]
        return higher_in_b, higher_in_a


def rasterize_diff(
    layers: DiffLayers,
    rt_range: tuple[float, float],
    mz_range: tuple[float, float],
    shape: tuple[int, int],
) -> np.ndarray:
    """D per pixel: the stronger of the two parts within the pixel, with its sign."""
    higher_in_b = rasterize(layers.higher_in_b, rt_range, mz_range, shape)
    higher_in_a = rasterize(layers.higher_in_a, rt_range, mz_range, shape)
    return np.where(higher_in_b >= higher_in_a, higher_in_b, -higher_in_a)


def _resampled(
    a: MapLayer, b: MapLayer | None, b_rows: np.ndarray, weights: np.ndarray, bin_start: int, bin_stop: int,
) -> np.ndarray:
    """B interpolated between scans b_rows and b_rows + 1 (see RTAlignment), on a's bins
    [bin_start, bin_stop); zero where B has no bins."""
    out = np.zeros((len(b_rows), bin_stop - bin_start), dtype=np.float32)
    if b is None:
        return out
    offset = _bin_offset(a, b)
    b_start, b_stop = max(bin_start + offset, 0), min(bin_stop + offset, b.intensity.shape[1])
    if b_stop <= b_start:
        return out
    first = int(b_rows.min())
    values = b.intensity[first:int(b_rows.max()) + 2, b_start:b_stop]
    local, weights = b_rows - first, weights[:, None]
    out[:, b_start - offset - bin_start:b_stop - offset - bin_start] = values[local] * (1 - weights) + values[local + 1] * weights
    return out


def _bin_offset(a: MapLayer, b: MapLayer) -> int:
    """j such that bin i of a is bin i + j of b."""
    log_step = np.log(a.mz_edges[1] / a.mz_edges[0])
    offset = np.log(a.mz_edges[0] / b.mz_edges[0]) / log_step
    if abs(offset - round(offset)) > 1e-3 or not np.isclose(a.bin_ppm, b.bin_ppm):
        raise ValueError("The layers are not on one m/z lattice")
    return round(offset)


def _resampling(warped_rt: np.ndarray, rt_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of B's scans, placed at warped_rt, at A's scans."""
    rows = np.clip(np.searchsorted(warped_rt, rt_a, side="right") - 1, 0, len(warped_rt) - 2)
    span = warped_rt[rows + 1] - warped_rt[rows]
    weights = np.clip((rt_a - warped_rt[rows]) / np.where(span > 0, span, 1), 0, 1)
    return rows, weights.astype(np.float32)


def _normalization(a: MapLayer, b: MapLayer, b_rows: np.ndarray, weights: np.ndarray, threshold: float) -> float:
    """Median of A / B over the bins where both exceed threshold; 1 if there are none."""
    n_bins = a.intensity.shape[1]
    ratios = []
    block = 4096
    for start in range(0, n_bins, block):
        stop = min(start + block, n_bins)
        values_a = a.intensity[:, start:stop]
        values_b = _resampled(a, b, b_rows, weights, start, stop)
        strong = (values_a > threshold) & (values_b > threshold)
        ratios.append(values_a[strong] / values_b[strong])
    ratios = np.concatenate(ratios)
    return float(np.median(ratios)) if len(ratios) else 1.0


def _dilate(values: np.ndarray, rt_margin: int, mz_margin: int) -> np.ndarray:
    """Max within +-rt_margin rows and +-mz_margin columns."""
    out = values
    for axis, margin in ((0, rt_margin), (1, mz_margin)):
        if margin == 0:
            continue
        source, out = out, out.copy()
        n = source.shape[axis]
        for shift in range(1, min(margin, n - 1) + 1):
            head, tail = [slice(None)] * 2, [slice(None)] * 2
            head[axis], tail[axis] = slice(None, n - shift), slice(shift, None)
            np.maximum(out[tuple(tail)], source[tuple(head)], out=out[tuple(tail)])
            np.maximum(out[tuple(head)], source[tuple(tail)], out=out[tuple(head)])
    return out
