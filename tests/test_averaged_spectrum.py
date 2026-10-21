from pathlib import Path

import numpy as np

from RAPID2_Agilent.models.rpd_model import AgilentDModel


DATA_PATH = Path("tests/data/dummy_RP_2min_neg.d")


def load_dummy_model():
    model = AgilentDModel()
    model.load(DATA_PATH)
    return model


def single_scan_rt_window(rt_list: np.ndarray, scan_index: int) -> tuple[float, float]:
    return (
        float((rt_list[scan_index - 1] + rt_list[scan_index]) / 2),
        float((rt_list[scan_index] + rt_list[scan_index + 1]) / 2),
    )


def test_extract_averaged_spectrum_single_scan_matches_extract_spectrum():
    model = load_dummy_model()
    scan_index = 5
    rt_min, rt_max = single_scan_rt_window(model.RT_list, scan_index)

    mz_list, inten_list, representative_scan_index = model.extract_averaged_spectrum(rt_min, rt_max)
    expected_mz, expected_inten = model.extract_spectrum(scan_index)

    assert representative_scan_index == scan_index
    np.testing.assert_allclose(mz_list, expected_mz)
    np.testing.assert_allclose(inten_list, expected_inten)


def test_extract_averaged_spectrum_multiple_scans_sum_and_mean_are_consistent():
    model = load_dummy_model()
    scan_indices = np.arange(4, 7)
    rt_min = float((model.RT_list[3] + model.RT_list[4]) / 2)
    rt_max = float((model.RT_list[6] + model.RT_list[7]) / 2)

    mz_mean, inten_mean, representative_scan_index = model.extract_averaged_spectrum(rt_min, rt_max, mode="mean")
    mz_sum, inten_sum, _ = model.extract_averaged_spectrum(rt_min, rt_max, mode="sum")
    representative_mz, _ = model.extract_spectrum(representative_scan_index)

    assert representative_scan_index in scan_indices
    np.testing.assert_allclose(mz_mean, representative_mz)
    np.testing.assert_allclose(mz_sum, mz_mean)
    assert len(inten_mean) == len(representative_mz)
    assert np.all(np.isfinite(inten_mean))
    np.testing.assert_allclose(inten_sum, inten_mean * len(scan_indices))


def test_extract_averaged_spectrum_native_versions_match_python_implementation():
    model = load_dummy_model()
    rt_min = float((model.RT_list[3] + model.RT_list[4]) / 2)
    rt_max = float((model.RT_list[10] + model.RT_list[11]) / 2)

    mz_fused, inten_fused, scan_fused = model.extract_averaged_spectrum(rt_min, rt_max, mode="mean")
    mz_native_bins, inten_native_bins, scan_native_bins = model.extract_averaged_spectrum_native_bins(rt_min, rt_max, mode="mean")
    mz_python, inten_python, scan_python = model.extract_averaged_spectrum_python(rt_min, rt_max, mode="mean")

    assert scan_fused == scan_python
    assert scan_native_bins == scan_python
    np.testing.assert_allclose(mz_fused, mz_python)
    np.testing.assert_allclose(mz_native_bins, mz_python)
    np.testing.assert_allclose(inten_fused, inten_python)
    np.testing.assert_allclose(inten_native_bins, inten_python)


def test_extract_averaged_spectrum_empty_rt_window_falls_back_to_nearest_scan():
    model = load_dummy_model()
    rt_min = float(model.RT_list[-1] + 10.0)
    rt_max = float(model.RT_list[-1] + 11.0)

    mz_list, inten_list, representative_scan_index = model.extract_averaged_spectrum(rt_min, rt_max)
    expected_mz, expected_inten = model.extract_spectrum(model.N_scans - 1)

    assert representative_scan_index == model.N_scans - 1
    np.testing.assert_allclose(mz_list, expected_mz)
    np.testing.assert_allclose(inten_list, expected_inten)


def test_extract_averaged_spectrum_rejects_unknown_mode():
    model = load_dummy_model()

    try:
        model.extract_averaged_spectrum(float(model.RT_list[0]), float(model.RT_list[1]), mode="median")
    except ValueError as exc:
        assert "mode must be" in str(exc)
    else:
        raise AssertionError("extract_averaged_spectrum should reject unknown modes")
