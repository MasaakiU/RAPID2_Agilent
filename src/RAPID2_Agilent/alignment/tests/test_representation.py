from dataclasses import replace
import numpy as np
import pytest

from RAPID2_Agilent.alignment.binning import PPMGrid, aggregate_scan
from RAPID2_Agilent.alignment.config import BinningConfig, TransformConfig
from RAPID2_Agilent.alignment.representation import build_profile_matrix
from RAPID2_Agilent.alignment.synthetic import ArrayScanSource
from RAPID2_Agilent.alignment.transforms import suggest_scales, transform_intensity


@pytest.mark.parametrize("method, expected", [
    ("sum", [6, 12, 9]), ("mean", [2, 6, 9]), ("max", [4, 7, 9]),
])
def test_aggregation_counts_zero_samples_and_excludes_upper_bound(method, expected):
    grid = PPMGrid(BinningConfig(1, 8, 1e6, method))
    mz = np.array([1, 1.1, 1.2, 2, 2.5, 7.9, 8, 0.9])
    intensity = np.array([0, 2, 4, 5, 7, 9, 100, 200], dtype=np.uint32)
    values = aggregate_scan(mz, intensity, grid)
    np.testing.assert_array_equal(values, expected)
    assert values.dtype == np.float32


def test_geometric_boundaries_and_adjacent_floats():
    grid = PPMGrid(BinningConfig(ppm=10))
    k = np.array([1, 9, 300, 20000])
    edges = grid.edges[k]
    _, exact = grid.indices(edges)
    _, below = grid.indices(np.nextafter(edges, -np.inf))
    _, above = grid.indices(np.nextafter(edges, np.inf))
    np.testing.assert_array_equal(exact, k)
    np.testing.assert_array_equal(below, k - 1)
    np.testing.assert_array_equal(above, k)
    assert grid.edges[-1] == 1700
    np.testing.assert_allclose(grid.edges[1:-1] / grid.edges[:-2], 1 + 10e-6, rtol=1e-12)


def test_uint32_sums_do_not_overflow_and_empty_bins_are_zero():
    grid = PPMGrid(BinningConfig(1, 8, 1e6))
    result = aggregate_scan(np.array([1.1, 1.2]), np.full(2, 2**32 - 1, dtype=np.uint32), grid)
    np.testing.assert_allclose(result, [2 * (2**32 - 1), 0, 0], rtol=1e-7)


@pytest.mark.parametrize("method", ["sum", "mean", "max"])
def test_empty_spectrum(method):
    grid = PPMGrid(BinningConfig(1, 8, 1e6, method))
    np.testing.assert_array_equal(aggregate_scan(np.array([]), np.array([]), grid), [0, 0, 0])


@pytest.mark.parametrize("method, operation", [
    ("none", lambda x: x), ("sqrt", np.sqrt), ("log1p", np.log1p),
    ("asinh", lambda x: np.arcsinh(x / 2)),
])
def test_transform_and_inplace_chunking(method, operation):
    values = np.arange(30, dtype=np.float32).reshape(5, 6)
    expected = operation(values.copy())
    original = values.copy()
    separate = transform_intensity(values, TransformConfig(method, 2), chunk_size=2)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_allclose(separate, expected, rtol=1e-6)
    result = transform_intensity(values, TransformConfig(method, 2), out=values, chunk_size=2)
    assert result is values
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_transform_must_follow_aggregation():
    config = BinningConfig(1, 8, 1e6)
    source = ArrayScanSource(np.array([0., 1.]), np.array([[1.1, 1.2], [2.1, 2.2]]),
                             np.array([[4., 5.], [4., 5.]], dtype=np.float32), {})
    profile = build_profile_matrix(source, config)
    transformed = transform_intensity(profile.values, TransformConfig("sqrt"))
    np.testing.assert_array_equal(transformed, [[3, 0, 0], [0, 3, 0]])


def test_asinh_requires_explicit_scale_and_candidates_do_not_modify_data():
    values = np.arange(101, dtype=np.float32)
    with pytest.raises(ValueError, match="explicit scale"):
        transform_intensity(values, TransformConfig())
    candidates = suggest_scales(values, max_samples=1000)
    assert candidates["median"] == 50.5
    assert candidates["nonzero_count"] == 100
    assert candidates["noise_lower_tail_proxy"] == 13
    sampled = suggest_scales(values.reshape(1, -1), max_samples=10)
    assert sampled == suggest_scales(values.reshape(1, -1), max_samples=10)
    assert sampled["sample_count"] == 10
    np.testing.assert_array_equal(values, np.arange(101))
    with pytest.raises(ValueError, match="No finite nonzero"):
        suggest_scales(np.zeros(10))


@pytest.mark.parametrize("config", [
    {"ppm": 0}, {"ppm": float("nan")}, {"mz_min": 0}, {"mz_max": 1}, {"aggregation": "median"},
])
def test_invalid_binning_config(config):
    with pytest.raises(ValueError):
        BinningConfig(**config)
