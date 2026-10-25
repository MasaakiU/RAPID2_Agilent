import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulator import SHAPES, Config, _labels, _measure, _shape, _shape_params, simulate  # noqa: E402


@pytest.mark.parametrize("shape", SHAPES)
def test_each_shape_has_its_top_at_1_and_its_fwhm(shape):
    rng = np.random.default_rng(0)
    for _ in range(5):
        component = {"center": 500.0, "fwhm": 20.0, "shape": shape, "params": _shape_params(rng, shape), "height": 1.0}
        component |= _measure(component)
        x = np.arange(0, 1500, 0.01)
        values = _shape(component, x)
        above = x[values >= 0.5]
        assert values.max() == pytest.approx(1.0, abs=1e-3)
        assert above[-1] - above[0] == pytest.approx(20.0, rel=0.01)
        assert x[np.argmax(values)] == pytest.approx(component["apex"], abs=0.05)


def test_a_sample_is_the_same_of_the_same_seed_and_finite():
    first, second = simulate(np.random.default_rng(7)), simulate(np.random.default_rng(7))
    np.testing.assert_array_equal(first.signal, second.signal)
    assert first.peaks == second.peaks
    assert np.isfinite(first.signal).all() and np.isfinite(first.clean).all()


def test_a_reversed_sample_is_the_same_one_mirrored():
    plain = simulate(np.random.default_rng(3), replace(Config(), reverse=0.0))
    mirrored = simulate(np.random.default_rng(3), replace(Config(), reverse=1.0))
    np.testing.assert_array_equal(mirrored.signal, plain.signal[::-1])
    last = len(plain.signal) - 1
    assert [last - peak.apex for peak in reversed(mirrored.peaks)] == pytest.approx([peak.apex for peak in plain.peaks])
    assert [peak.left for peak in reversed(mirrored.peaks)] == [peak.right for peak in plain.peaks]


def test_the_labels_follow_the_snr_and_the_ends():
    config = Config()
    for seed in range(30):
        sample = simulate(np.random.default_rng(seed), config)
        for peak in sample.peaks:
            if peak.label == "positive":
                assert peak.saturated or peak.snr >= config.snr_positive
                assert min(peak.apex, len(sample.signal) - 1 - peak.apex) >= peak.fwhm / 2
            elif peak.label == "negative":
                assert peak.snr <= config.snr_negative


def _pair(apart: float, smaller: float):
    """Two Gaussians of a FWHM of 20 points, of heights 1000 and smaller, apart by that many points, on white noise
    of 1."""
    components = []
    for center, height in ((500.0, 1000.0), (500.0 + apart, smaller)):
        component = {"center": center, "fwhm": 20.0, "shape": "gauss", "params": {}, "height": height}
        components.append(component | _measure(component))
    x = np.arange(1000, dtype=float)
    return _labels(Config(), x, components, np.random.default_rng(0).normal(0, 1, 1000), np.zeros(1000, bool), None,
                   1000)


def test_of_a_pair_between_merge_and_split_only_the_smaller_is_not_scored():
    sigma = 20.0 / 2.3548
    # Rs 0.35: the larger a peak either way, the smaller perhaps a shoulder
    larger, smaller = sorted(_pair(0.35 * 4 * sigma, 100.0), key=lambda peak: -peak.height)
    assert (larger.label, smaller.label) == ("positive", "ignore")
    # lost in the noise, it makes no pair
    assert sorted(peak.label for peak in _pair(0.35 * 4 * sigma, 1.0)) == ["negative", "positive"]
    # Rs 0.1: one peak, at the top of the two
    (merged,) = _pair(0.1 * 4 * sigma, 100.0)
    assert (merged.label, merged.components) == ("positive", 2)
    # Rs 1: two peaks
    assert [peak.label for peak in _pair(4 * sigma, 100.0)] == ["positive", "positive"]


def test_a_coarse_sample_is_its_grid_interpolated_linearly():
    config = replace(Config(), coarse=1.0, reverse=0.0)
    for seed in range(10):
        sample = simulate(np.random.default_rng(seed), config)
        factor = sample.meta["coarse"]
        assert factor >= 2 and (len(sample.signal) - 1) % factor == 0
        grid = sample.signal[::factor]
        between = np.interp(np.arange(len(sample.signal)) / factor, np.arange(len(grid)), grid)
        measured = ~sample.missing
        np.testing.assert_allclose(sample.signal[measured], between[measured], rtol=1e-5, atol=1e-3)
        # of W_min in points of the grid; that of merged peaks measured in steps of 0.05 of them
        assert all(peak.fwhm >= (2 - 0.1) * factor for peak in sample.peaks)


def test_points_missing_are_interpolated_across():
    for seed in range(40):
        sample = simulate(np.random.default_rng(seed), replace(Config(), coarse=0.0))
        if sample.missing.any() and not sample.missing.all():
            x = np.arange(len(sample.signal))
            across = np.interp(x, x[~sample.missing], sample.signal[~sample.missing])
            np.testing.assert_allclose(sample.signal[sample.missing], across[sample.missing], rtol=1e-5)
            return
    pytest.fail("no sample with missing points")


def _narrow_on_wide(apart: float):
    components = []
    for center, fwhm, height in ((500.0, 150.0, 1000.0), (500.0 + apart, 10.0, 500.0)):
        component = {"center": center, "fwhm": fwhm, "shape": "gauss", "params": {}, "height": height}
        components.append(component | _measure(component))
    x = np.arange(1000, dtype=float)
    return _labels(Config(), x, components, np.random.default_rng(0).normal(0, 1, 1000), np.zeros(1000, bool), None,
                   1000)


def test_a_narrow_peak_on_a_wide_one_is_a_peak_of_its_own():
    # on its top: the wide one's top hidden under it, not scored
    wide, narrow = sorted(_narrow_on_wide(3.0), key=lambda peak: -peak.fwhm)
    assert (wide.label, narrow.label, narrow.components) == ("ignore", "positive", 1)
    # on its side: both peaks
    assert [peak.label for peak in _narrow_on_wide(60.0)] == ["positive", "positive"]
