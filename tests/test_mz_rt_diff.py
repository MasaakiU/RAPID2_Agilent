import math
from types import SimpleNamespace

import numpy as np
import pytest

from RAPID2_Agilent.models import mz_rt_diff
from RAPID2_Agilent.models.mz_rt_diff import DiffParams, MzRtDiff, RTAlignment, _resampling, rasterize_diff
from RAPID2_Agilent.models.mz_rt_map import DETAIL_PPM, MID_FACTOR, MapLayer, PPMGrid
from RAPID2_Agilent.views.mz_rt_map.mz_rt_map_presenter import fold_change_ticks


LOG_STEP = math.log1p(1e-4)  # 100 ppm
RT = np.arange(8, dtype=float)


def layer(first_bin: int, n_bins: int, peaks: dict[tuple[int, int], float]) -> MapLayer:
    """A layer on the 100 ppm lattice starting at lattice bin first_bin; peaks: (row, bin): intensity."""
    intensity = np.zeros((len(RT), n_bins), dtype=np.float32)
    for (row, column), value in peaks.items():
        intensity[row, column] = value
    edges = np.exp((first_bin + np.arange(n_bins + 1)) * LOG_STEP)
    return MapLayer(RT, edges, intensity, 100.0)


def overview_of(mid: MapLayer, factor: int) -> MapLayer:
    starts = np.arange(0, mid.intensity.shape[1], factor)
    edges = np.append(mid.mz_edges[starts], mid.mz_edges[-1])
    return MapLayer(mid.rt, edges, np.maximum.reduceat(mid.intensity, starts, axis=1), 300.0)


def identity_alignment() -> RTAlignment:
    rows, weights = _resampling(RT, RT)
    diagonal = np.repeat(np.arange(len(RT))[:, None], 2, axis=1)
    return RTAlignment(RT.copy(), rows, weights, 1.0, np.eye(len(RT), dtype=np.float32), diagonal, diagonal)


def make_diff(a: MapLayer, b: MapLayer, factor: int = 3) -> MzRtDiff:
    map_a = SimpleNamespace(mid=a, overview=overview_of(a, factor), overview_factor=factor, rt=RT)
    map_b = SimpleNamespace(mid=b, rt=RT)
    return MzRtDiff(map_a, map_b, identity_alignment())


def full_parts(diff: MzRtDiff, params: DiffParams):
    a = diff.map_a.mid
    parts = diff.parts(a, diff.map_b.mid, (RT[0], RT[-1] + 1), (a.mz_edges[0], a.mz_edges[-1]), params)
    return parts.higher_in_b.intensity, parts.higher_in_a.intensity


# A bins 0..9 are lattice bins 1000..1009; B starts two bins later, so B's bin 2 is A's bin 4
A = layer(1000, 10, {(3, 4): 1000.0, (6, 1): 500.0})
B = layer(1002, 10, {(4, 2): 1000.0, (1, 5): 800.0})


def test_tolerance_hides_a_one_scan_shift_and_keeps_peaks_of_one_file():
    higher_in_b, higher_in_a = full_parts(make_diff(A, B), DiffParams(10.0, 1.0, rt_tolerance=1, mz_tolerance_ppm=0))

    assert higher_in_b[1, 7] == pytest.approx(np.arcsinh(80.0))  # only in B
    assert higher_in_a[6, 1] == pytest.approx(np.arcsinh(50.0))  # only in A
    higher_in_b[1, 7] = higher_in_a[6, 1] = 0
    assert not higher_in_b.any() and not higher_in_a.any()  # the shifted peak cancels


def test_without_tolerance_a_shifted_peak_shows_as_a_pair():
    higher_in_b, higher_in_a = full_parts(make_diff(A, B), DiffParams(10.0, 1.0, rt_tolerance=0, mz_tolerance_ppm=0))

    assert higher_in_a[3, 4] == pytest.approx(np.arcsinh(100.0))
    assert higher_in_b[4, 4] == pytest.approx(np.arcsinh(100.0))
    assert not (higher_in_b * higher_in_a).any()  # at most one part per bin


def test_mz_tolerance_hides_a_one_bin_shift():
    b = layer(1002, 10, {(3, 3): 1000.0})  # A's bin 5
    params = DiffParams(10.0, 1.0, rt_tolerance=0, mz_tolerance_ppm=100)

    higher_in_b, higher_in_a = full_parts(make_diff(layer(1000, 10, {(3, 4): 1000.0}), b), params)

    assert not higher_in_b.any() and not higher_in_a.any()


def test_b_factor_scales_b_to_a_fold_change():
    a = layer(1000, 10, {(3, 4): 1000.0})
    b = layer(1000, 10, {(3, 4): 1000.0})

    higher_in_b, higher_in_a = full_parts(make_diff(a, b), DiffParams(1.0, 2.0, rt_tolerance=1, mz_tolerance_ppm=0))

    assert higher_in_b[3, 4] == pytest.approx(math.log(2), abs=1e-6)  # asinh(x) ~ ln(2x) for large x
    assert not higher_in_a.any()


def test_overview_is_the_max_of_mid_parts_over_each_overview_bin(monkeypatch):
    monkeypatch.setattr(mz_rt_diff, "OVERVIEW_BLOCK_GROUPS", 1)  # one overview bin per block
    diff = make_diff(A, B, factor=3)
    params = DiffParams(10.0, 1.0, rt_tolerance=0, mz_tolerance_ppm=100)

    overview = diff.overview(params, lambda: False)

    starts = np.arange(0, 10, 3)
    pooled_parts = (overview.layers.higher_in_b, overview.layers.higher_in_a)
    squared_parts = (overview.squared_higher_in_b, overview.squared_higher_in_a)
    for part, pooled, squared in zip(full_parts(diff, params), pooled_parts, squared_parts):
        np.testing.assert_array_equal(pooled.intensity, np.maximum.reduceat(part, starts, axis=1))
        np.testing.assert_array_equal(pooled.mz_edges, diff.map_a.overview.mz_edges)
        np.testing.assert_allclose(squared, np.square(part.astype(float)).sum(axis=1))


def test_rasterized_difference_keeps_the_stronger_sign_per_pixel():
    diff = make_diff(A, B)
    layers = diff.parts(A, B, (0.0, 8.0), (A.mz_edges[0], A.mz_edges[-1]), DiffParams(10.0, 1.0, 1, 0))

    image = rasterize_diff(layers, (0.0, 8.0), (A.mz_edges[0], A.mz_edges[-1]), (8, 10))

    assert image[1, 7] > 0 and image[6, 1] < 0
    assert np.count_nonzero(image) == 2


def test_resampling_interpolates_b_between_warped_scans():
    rows, weights = _resampling(np.array([0.0, 2.0, 4.0]), np.array([0.0, 1.0, 3.5, 4.0]))

    np.testing.assert_array_equal(rows, [0, 0, 1, 1])
    np.testing.assert_allclose(weights, [0.0, 0.5, 0.75, 1.0])


def test_grids_of_different_ranges_share_edges():
    a = PPMGrid.spanning(50.000462, 1699.4, DETAIL_PPM, snap_bins=MID_FACTOR)
    b = PPMGrid.spanning(49.999992, 1699.5, DETAIL_PPM, snap_bins=MID_FACTOR)

    for grid_a, grid_b in ((a, b), (a.coarsened(MID_FACTOR), b.coarsened(MID_FACTOR))):
        offset = grid_a.lattice_index - grid_b.lattice_index
        assert offset >= 0
        np.testing.assert_array_equal(grid_a.edges()[:50], grid_b.edges()[offset:offset + 50])
    assert a.mz_min <= 50.000462 and b.mz_min <= 49.999992


def test_fold_change_ticks():
    labels = [label for _position, label in fold_change_ticks(8.0)]
    assert labels == ["×1/8", "×1/4", "×1/2", "×1", "×2", "×4", "×8"]
    assert [label for _position, label in fold_change_ticks(1000.0)] == [
        "×1/1000", "×1/100", "×1/10", "×1", "×10", "×100", "×1000",
    ]
    positions = [position for position, _label in fold_change_ticks(1.5)]
    assert positions == pytest.approx([-math.log(1.5), 0.0, math.log(1.5)])


def test_diagnostics_of_an_identity_alignment():
    from RAPID2_Agilent.views.mz_rt_map.diagnostics import dtw_path, path_similarity, rt_shift, squared_difference

    diff = make_diff(A, B)
    diff.map_b.rt = diff.map_a.rt = RT

    path = dtw_path(diff)
    assert path.image.values.shape == (len(RT), len(RT))
    np.testing.assert_array_equal(path.image.values, np.eye(len(RT)))  # uniform RTs: scan i at pixel i
    assert path.image.rect == pytest.approx((-0.5, -0.5, 8.0, 8.0))
    curves = {curve.role: curve for curve in path.curves}
    np.testing.assert_array_equal(curves["path"].x, curves["path"].y)

    shift = {curve.role: curve for curve in rt_shift(diff).curves}
    assert not shift["main"].y.any() and not shift["anchors"].y.any()

    similarity = {curve.role: curve for curve in path_similarity(diff).curves}
    np.testing.assert_array_equal(similarity["main"].y, 1.0)

    overview = diff.overview(DiffParams(10.0, 1.0, rt_tolerance=1, mz_tolerance_ppm=0), lambda: False)
    squared = {curve.name: curve for curve in squared_difference(diff, overview).curves}
    assert squared["B higher"].y[1] == pytest.approx(np.arcsinh(80.0) ** 2, rel=1e-6)
    assert squared["A higher"].y[6] == pytest.approx(np.arcsinh(50.0) ** 2, rel=1e-6)


def test_similarity_image_takes_the_nearest_scan_on_an_even_rt_grid():
    from RAPID2_Agilent.views.mz_rt_map.diagnostics import _uniform

    grid, nearest = _uniform(np.array([0.0, 0.1, 0.2, 3.0]))

    np.testing.assert_allclose(grid, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(nearest, [0, 2, 3, 3])
