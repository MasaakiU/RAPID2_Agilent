"""Where a run's m/z x RT map is strongest, and where the maps of two runs differ most, as lists. No Qt.

The maps and their B - A difference are those of the m/z-RT map window (`models.mz_rt_map`, `models.mz_rt_diff`), on
the mid level -- the MS1 scans, in m/z bins of about 100 ppm --, with the window's default settings of the
difference. A spot is a cell that no one of its 8 neighbours exceeds and that lies apart from every stronger spot
listed (SPOT_RT_SEPARATION, a bin); its m/z is then the profile point of highest intensity within the cell, in the
scan of the spot, since a bin is too wide to name an ion by.
"""

from dataclasses import dataclass

import numpy as np

from ..io.ms_data import MSData
from ..models.mz_rt_diff import DiffParams, MzRtDiff, align_maps
from ..models.mz_rt_map import MapLayer, MzRtMap


# the window's defaults: B normalized to A, and each side compared with the other within 1 scan and 10 ppm
DIFF_RT_TOLERANCE_SCANS = 1
DIFF_MZ_TOLERANCE_PPM = 10.0
BLOCK_BINS = 4096  # m/z bins of the difference computed at once, as the window's overview does
# a spot listed hides the weaker ones of its m/z bin and the bins next to it within this many minutes: one
# chromatographic peak is a ridge of local maxima along RT
SPOT_RT_SEPARATION = 0.25
# the strongest local maxima kept for the choice of each spot listed; the rest cannot be reached
CANDIDATES_PER_SPOT = 50


@dataclass(frozen=True)
class Spot:
    scan: int                    # index in the run
    rt: float                    # of the scan, in minutes
    mz: float                    # the cell's highest profile point in the scan; the cell's centre if it has none
    mz_bin: tuple[float, float]  # the cell's m/z range
    intensity: float             # the cell's: the highest of the scan in the bin


@dataclass(frozen=True)
class Difference:
    spot: Spot                   # in the run that is higher: the reference's scan, or the query's nearest one
    rt: float                    # on the reference's RT axis
    query_rt: float              # the query's own RT there, before alignment
    difference: float            # D, about ln(c B / A) where both are well above the scale
    reference_intensity: float   # A in the cell
    query_intensity: float       # c B, aligned onto the cell


@dataclass(frozen=True)
class MapComparison:
    scale: float                 # s of D
    normalization: float         # c, by which the query is multiplied
    max_rt_shift: float          # minutes, of the query's scans onto the reference's RT axis
    higher_in_query: list[Difference]
    higher_in_reference: list[Difference]


def built_map(data: MSData) -> MzRtMap:
    """The map of a run's MS1 scans; ValueError if it has none."""
    mz_rt_map = MzRtMap(data)
    mz_rt_map.build(lambda done, total: None, lambda: False)
    return mz_rt_map


def strongest_spots(
    mz_rt_map: MzRtMap,
    *,
    top_n: int,
    rt_range: tuple[float, float] | None = None,
    mz_range: tuple[float, float] | None = None,
) -> list[Spot]:
    """The most intense spots of the map, most intense first."""
    layer = mz_rt_map.mid
    rows, bins = _local_maxima(layer.intensity)
    rows, bins = _within(layer, rows, bins, rt_range, mz_range)
    chosen = _apart(layer, _strongest(layer.intensity[rows, bins], rows, bins, top_n), top_n)
    return [_spot(mz_rt_map, row, bin_) for _value, row, bin_ in chosen]


def compare_maps(
    map_a: MzRtMap,
    map_b: MzRtMap,
    *,
    top_n: int,
    min_difference: float,
    rt_range: tuple[float, float] | None = None,
    mz_range: tuple[float, float] | None = None,
) -> MapComparison:
    """B (query) aligned in RT onto A (reference), and the spots of D of either sign, the largest first; of |D| at
    least min_difference, as two runs that are the same differ by rounding."""
    alignment = align_maps(map_a, map_b, lambda: False)
    params = DiffParams(
        scale=map_a.default_scale(),
        b_factor=alignment.normalization,
        rt_tolerance=DIFF_RT_TOLERANCE_SCANS,
        mz_tolerance_ppm=DIFF_MZ_TOLERANCE_PPM,
    )
    diff = MzRtDiff(map_a, map_b, alignment)
    a = map_a.mid
    n_bins = a.intensity.shape[1]
    found = {"higher_in_b": [], "higher_in_a": []}  # (D, row, bin) of each part
    for start in range(0, n_bins, BLOCK_BINS):
        stop = min(start + BLOCK_BINS, n_bins)
        # a bin more on each side, so that a spot at the block's edge sees its neighbours
        low, high = max(start - 1, 0), min(stop + 1, n_bins)
        layers = diff.parts(a, map_b.mid, map_a.rt_range, (float(a.mz_edges[low]), float(a.mz_edges[high])), params)
        for name in found:
            part = getattr(layers, name).intensity
            rows, bins = _local_maxima(part)
            bins = bins + low
            inner = (start <= bins) & (bins < stop)
            rows, bins = _within(a, rows[inner], bins[inner], rt_range, mz_range)
            values = part[rows, bins - low]
            strong = values >= min_difference
            found[name] += _strongest(values[strong], rows[strong], bins[strong], top_n)
    lattice_offset = map_a.mid_grid.lattice_index - map_b.mid_grid.lattice_index

    def differences(name: str, sign: int) -> list[Difference]:
        chosen = _apart(a, found[name], top_n)
        return [_difference(map_a, map_b, alignment, lattice_offset, sign * value, row, bin_) for value, row, bin_ in chosen]

    return MapComparison(
        scale=params.scale,
        normalization=alignment.normalization,
        max_rt_shift=alignment.max_shift(map_b.rt),
        higher_in_query=differences("higher_in_b", 1),
        higher_in_reference=differences("higher_in_a", -1),
    )


def _local_maxima(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rows, columns) of the positive cells that no one of their 8 neighbours exceeds."""
    padded = np.pad(values, 1, constant_values=-np.inf)
    n_rows, n_columns = values.shape
    is_maximum = values > 0
    for row_shift in (-1, 0, 1):
        for column_shift in (-1, 0, 1):
            if row_shift or column_shift:
                neighbours = padded[1 + row_shift:1 + row_shift + n_rows, 1 + column_shift:1 + column_shift + n_columns]
                is_maximum &= values >= neighbours
    return np.nonzero(is_maximum)


def _strongest(values: np.ndarray, rows: np.ndarray, bins: np.ndarray, top_n: int) -> list[tuple[float, int, int]]:
    """(value, row, bin) of the candidates for top_n spots: the CANDIDATES_PER_SPOT x top_n highest."""
    keep = top_n * CANDIDATES_PER_SPOT
    if len(values) > keep:
        best = np.argpartition(values, len(values) - keep)[-keep:]
        values, rows, bins = values[best], rows[best], bins[best]
    return list(zip(values.tolist(), rows.tolist(), bins.tolist()))


def _apart(layer: MapLayer, candidates: list[tuple[float, int, int]], top_n: int) -> list[tuple[float, int, int]]:
    """Of (value, row, bin) candidates, up to top_n of the highest, each apart from the higher ones chosen."""
    chosen = []
    for value, row, bin_ in sorted(candidates, key=lambda candidate: candidate[0], reverse=True):
        if len(chosen) == top_n:
            break
        rt = layer.rt[row]
        if all(abs(bin_ - other_bin) > 1 or abs(rt - layer.rt[other_row]) > SPOT_RT_SEPARATION for _, other_row, other_bin in chosen):
            chosen.append((value, row, bin_))
    return chosen


def _within(layer: MapLayer, rows: np.ndarray, bins: np.ndarray, rt_range, mz_range) -> tuple[np.ndarray, np.ndarray]:
    """The cells whose scan lies in rt_range and whose bin overlaps mz_range."""
    keep = np.ones(len(rows), dtype=bool)
    if rt_range is not None:
        rt = layer.rt[rows]
        keep &= (rt_range[0] <= rt) & (rt <= rt_range[1])
    if mz_range is not None:
        keep &= (layer.mz_edges[bins + 1] > mz_range[0]) & (layer.mz_edges[bins] < mz_range[1])
    return rows[keep], bins[keep]


def _spot(mz_rt_map: MzRtMap, row: int, bin_: int) -> Spot:
    layer = mz_rt_map.mid
    low, high = float(layer.mz_edges[bin_]), float(layer.mz_edges[bin_ + 1])
    scan = int(mz_rt_map.scan_indices[row])
    mz, intensity = mz_rt_map.ms_data_model.extract_averaged_sub_spectrum(low, high, scan_indices=np.array([scan]))
    peak_mz = float(mz[np.argmax(intensity)]) if len(intensity) and intensity.max() > 0 else float(np.sqrt(low * high))
    return Spot(scan, float(layer.rt[row]), peak_mz, (low, high), float(layer.intensity[row, bin_]))


def _difference(map_a: MzRtMap, map_b: MzRtMap, alignment, lattice_offset: int, value: float, row: int, bin_: int) -> Difference:
    """D at A's scan row and bin, with what A and B (interpolated between two scans, as MzRtDiff does) hold there."""
    b_row, weight = int(alignment.rows[row]), float(alignment.weights[row])
    b_bin = bin_ + lattice_offset
    query = 0.0
    in_b = 0 <= b_bin < map_b.mid.intensity.shape[1]
    if in_b:
        query = (1 - weight) * map_b.mid.intensity[b_row, b_bin] + weight * map_b.mid.intensity[b_row + 1, b_bin]
    if value > 0 and in_b:
        # the query's nearest scan, on its own bin of the same m/z
        spot = _spot(map_b, b_row + (weight >= 0.5), b_bin)
    else:
        spot = _spot(map_a, row, bin_)
    return Difference(
        spot=spot,
        rt=float(map_a.rt[row]),
        query_rt=float((1 - weight) * map_b.rt[b_row] + weight * map_b.rt[b_row + 1]),
        difference=value,
        reference_intensity=float(map_a.mid.intensity[row, bin_]),
        query_intensity=float(alignment.normalization * query),
    )
