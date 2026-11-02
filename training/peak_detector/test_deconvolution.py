import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deconvolve import (ADDED, CandidateErrors, clusters, corrections, heights, oracle_candidates,  # noqa: E402
                        perturbed_candidates, roughened, score, shaped, whole)
from evaluate_deconvolution import app, emg, fit_emg, truth  # noqa: E402
from simulator import Config, Peak, Sample, simulate  # noqa: E402


def _gauss(x, center, fwhm):
    return np.exp(-0.5 * ((x - center) / (fwhm / 2.3548)) ** 2)


def _window(heights_of, centers, fwhm=20.0, length=600, noise=0.0, seed=0):
    """A window of Gaussians of a FWHM on a flat baseline of 10, with white noise, and its parts."""
    x = np.arange(length, dtype=float)
    peaks, clean = [], np.full(length, 10.0)
    for height, center in zip(heights_of, centers):
        profile = height * _gauss(x, center, fwhm)
        clean += profile
        peaks.append(Peak(center, height, fwhm, 2.45 * fwhm / 2.3548, 2.45 * fwhm / 2.3548, "gauss", height / max(noise, 1),
                          "positive", start=0, profile=profile.astype(np.float32)))
    signal = clean + np.random.default_rng(seed).normal(0, noise, length)
    sample = Sample(signal.astype(np.float32), clean.astype(np.float32), np.zeros(length, bool), peaks, None,
                    {"coarse": 1, "baseline": []}, np.full(length, 10.0, np.float32), np.zeros(length, np.float32),
                    np.full(length, max(noise, 1e-3), np.float32))
    return whole(sample)


def test_heights_of_shapes_make_up_the_signal():
    x = np.arange(300, dtype=float)
    shapes = np.array([_gauss(x, 140, 20), _gauss(x, 160, 20)])
    found, covariance, _ = heights(3 * shapes[0] + 5 * shapes[1], shapes, np.ones(300))
    np.testing.assert_allclose(found, [3, 5], rtol=1e-9)
    # of noise of sigma 1, the heights' covariance as least squares has it
    np.testing.assert_allclose(covariance, np.linalg.inv(shapes @ shapes.T), rtol=1e-9)
    # none below 0; and a prior holds them
    assert heights(-shapes[0], shapes, np.ones(300))[0].min() == 0
    held, _, _ = heights(3 * shapes[0] + 5 * shapes[1], shapes, np.ones(300), prior=[4, 4], strength=1e9)
    np.testing.assert_allclose(held, [4, 4], atol=1e-3)
    # of a strength each: the one of 0 free, the other held
    held, _, _ = heights(3 * shapes[0] + 5 * shapes[1], shapes, np.ones(300), prior=[4, 4], strength=[1e9, 0])
    assert abs(held[0] - 4) < 1e-3 and abs(held[1] - 4) > 0.1


def test_heights_come_through_a_baseline_off_by_an_offset_and_a_tilt_under_each_cluster():
    x = np.arange(600, dtype=float)
    shapes = np.array([_gauss(x, 140, 20), _gauss(x, 160, 20), _gauss(x, 450, 30)])
    candidates = [{"apex": c, "left": w, "right": w} for c, w in ((140, 20.0), (160, 20.0), (450, 25.0))]
    spans = clusters(candidates, 600)
    assert spans == [(60.0, 240.0), (350.0, 550.0)]
    off = corrections(spans, 600).T @ [5.0, 2.0, -3.0, 1.0]
    found, covariance, amounts = heights([3, 5, 2] @ shapes + off, shapes, np.ones(600), free=corrections(spans, 600))
    np.testing.assert_allclose(found, [3, 5, 2], rtol=1e-9)
    np.testing.assert_allclose(amounts, [5, 2, -3, 1], rtol=1e-9)
    # less told of the heights, so more uncertain, than of a baseline known
    assert (np.diag(covariance) > np.diag(heights([3, 5, 2] @ shapes, shapes, np.ones(600))[1])).all()


@pytest.mark.parametrize("tau", [0.0, 2.0, -2.0, 15.0])
def test_an_emg_is_of_its_area_and_mirrored_of_a_negative_tau(tau):
    t = np.arange(-500, 500, 0.05)
    values = emg(t, 3.0, 10.0, 4.0, tau)
    assert np.isfinite(values).all() and values.min() >= 0
    assert values.sum() * 0.05 == pytest.approx(3.0, rel=1e-6)
    if tau:
        np.testing.assert_allclose(emg(t, 3.0, 10.0, 4.0, -tau), emg(20.0 - t, 3.0, 10.0, 4.0, tau), atol=1e-12)
    else:  # a Gaussian, of a tail of 1e-3 sigma
        np.testing.assert_allclose(values, 3.0 * np.exp(-0.5 * ((t - 10) / 4) ** 2) / (4 * np.sqrt(2 * np.pi)),
                                   atol=1e-3 * values.max())


def test_the_true_shapes_give_the_areas_within_their_noise():
    errors, z = [], []
    for seed in range(40):
        window = whole(simulate(np.random.default_rng(seed)))
        for estimate, peak in zip(truth(window), [p for p in shaped(window) if len(p.profile) and p.profile.max() > 0]):
            area = peak.profile.sum()
            if estimate["sigma"] and peak.snr > 100 and not peak.saturated:
                errors.append(abs(estimate["area"] - area) / area)
                z.append((estimate["area"] - area) / estimate["sigma"])
    assert len(errors) > 100 and np.median(errors) < 0.01
    assert 0.7 < 1.4826 * np.median(np.abs(np.array(z) - np.median(z))) < 1.5


def test_scores_count_errors_misses_and_areas_on_nothing():
    window = _window([1000, 500], [200, 400])
    true_areas = [peak.profile.sum() for peak in window["peaks"]]
    right = [{"apex": 200.0, "confidence": 1.0, "area": true_areas[0], "sigma": 1.0, "shape": window["peaks"][0].profile},
             {"apex": 401.0, "confidence": 1.0, "area": 1.1 * true_areas[1]}]
    summary = score([(window, right + [{"apex": 550.0, "confidence": 0.2, "area": 0.5 * true_areas[1]}])])
    every = summary["by category"]["all"]
    assert every["peaks"] == 2 and every["missed"] == 0
    assert every["area |error| median"] == pytest.approx(0.05, rel=1e-5)
    assert every["shape error median"] == pytest.approx(0, abs=1e-6)
    assert summary["false area"] == pytest.approx(0.5 * true_areas[1] / sum(true_areas))
    assert summary["false peaks a window"] == 1 and every["area 0"] == 0
    assert score([(window, right[:1])])["by category"]["all"]["missed"] == pytest.approx(0.5)
    assert score([(window, [right[0], right[1] | {"area": 0.0}])])["by category"]["all"]["area 0"] == 0.5


def test_the_emgs_fit_two_overlapped_gaussians_and_the_app_cuts_at_the_valley():
    window = _window([1000, 500], [280, 318], noise=2.0)
    true_areas = np.array([peak.profile.sum() for peak in window["peaks"]])
    fitted = np.array([estimate["area"] for estimate in fit_emg(window, oracle_candidates(window))])
    np.testing.assert_allclose(fitted, true_areas, rtol=0.03)
    # the app: both cut at the valley, and less a flat baseline at their lowest (the valley, of the smaller), short
    split = np.array([estimate["area"] for estimate in app(window, oracle_candidates(window))])
    assert (0 < split).all() and (split < 0.7 * true_areas).all()


def test_perturbed_candidates_miss_shift_and_add_some():
    window = whole(simulate(np.random.default_rng(3), replace(Config(), length=(4000, 4096))))
    rng = np.random.default_rng(0)
    counts = [len(perturbed_candidates(window, rng)) for _ in range(50)]
    peaks = len(shaped(window))
    errors = CandidateErrors()
    expected = peaks * (1 - errors.missed) * (1 + errors.top_duplicates + errors.tail_duplicates) + errors.false_rate * 4000
    assert abs(np.mean(counts) - expected) < 0.15 * expected + 1
    assert all(0 <= c["apex"] <= len(window["signal"]) - 1 for c in perturbed_candidates(window, rng))


def test_the_fronts_candidates_roughened_keep_their_confidences_and_those_added_are_low():
    candidates = [{"apex": 100.0, "confidence": 0.9, "left": 10.0, "right": 20.0},
                  {"apex": 300.0, "confidence": 0.12, "left": 5.0, "right": 5.0}]
    rng = np.random.default_rng(0)
    none = CandidateErrors(missed=0, shift=0, width_spread=0, top_duplicates=0, tail_duplicates=0, false_rate=0)
    assert roughened(candidates, 400, rng, none) == candidates
    doubled = roughened(candidates, 400, rng, replace(none, top_duplicates=1.0))
    assert doubled[0::2] == candidates
    for candidate, second in zip(candidates, doubled[1::2]):
        fwhm = 0.48 * (candidate["left"] + candidate["right"])
        assert 0.15 * fwhm <= abs(second["apex"] - candidate["apex"]) <= 0.5 * fwhm
        assert ADDED[0] <= second["confidence"] <= ADDED[1]
    assert roughened(candidates, 400, rng, replace(none, missed=1.0)) == []
    added = roughened([], 400_000, rng, replace(none, false_rate=1e-3))
    assert abs(len(added) - 400) < 80 and all(ADDED[0] <= c["confidence"] <= ADDED[1] for c in added)
