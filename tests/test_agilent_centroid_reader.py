"""The centroid spectra of .d runs (MSPeak.bin), read without the profile, and the choice of them on opening."""

import numpy as np
import pytest

from RAPID2_Agilent.io import AgilentCentroidReader, AgilentReader
from RAPID2_Agilent.io.agilent_centroid_reader import stored_spectra
from tests.data_paths import DATA_DIR, PEAK_EVAL_DIR, require

DATA_PATH = DATA_DIR / "dummy_RP_2min_neg.d"
COPY_DATA_PATH = DATA_DIR / "dummy_RP_2min_neg_copy.d"
MSMS_DATA_PATH = DATA_DIR / "MSMS_data" / "9_PRJ_0-1-2_PA34-1_1uM_n1.d"
LISTS = ("RT_list", "MS_level_list", "polarity_list", "ion_mode_list", "scan_type_list", "precursor_mz_list",
         "isolation_width_list", "collision_energy_list")


@pytest.mark.parametrize("path", [DATA_PATH, MSMS_DATA_PATH], ids=["ms1", "msms"])
def test_the_centroids_have_the_scan_records_the_profile_reader_has(path):
    with AgilentReader() as profile, AgilentCentroidReader() as centroid:
        profile.load(require(path))
        centroid.load(path)
        assert centroid.N_scans == profile.N_scans and centroid.file_path == path
        for name in LISTS:
            assert np.array_equal(getattr(centroid, name), getattr(profile, name)), name
        assert centroid.get_scan_type_summary() == profile.get_scan_type_summary()
        assert np.array_equal(centroid.extract_tic()[1], profile.extract_tic()[1])
        assert np.array_equal(centroid.extract_bpc()[1], profile.extract_bpc()[1])
        assert np.array_equal(centroid.scan_indices_for_ms_level(1), profile.scan_indices_for_ms_level(1))
    assert centroid.file_path is None and centroid.RT_list is None


def test_a_scans_centroids_are_its_record_in_mspeak_calibrated_as_the_profile():
    with AgilentReader() as profile, AgilentCentroidReader() as centroid:
        profile.load(require(DATA_PATH))
        centroid.load(DATA_PATH)
        records = profile.ms_scan_info.scan_info
        # the first scan of a run has none
        assert records[0]["SpectrumParamValues"][1]["PointCount"] == 0 and len(centroid.extract_spectrum(0)[0]) == 0
        spectrum = records[50]["SpectrumParamValues"][1]
        assert spectrum["SpectrumFormatID"] == 2
        count, offset = spectrum["PointCount"], spectrum["SpectrumOffset"]
        peaks = (DATA_PATH / "AcqData" / "MSPeak.bin").read_bytes()
        mz, inten = centroid.extract_spectrum(50)
        assert len(mz) == count and np.all(np.diff(mz) > 0)
        assert inten.tolist() == np.frombuffer(peaks, "<f4", count, offset + 8 * count).tolist()
        np.testing.assert_allclose(mz[[0, -1]], [spectrum["MinX"], spectrum["MaxX"]], rtol=10e-6)
        # the highest centroid, of the reference ion: at the profile's highest point, as high
        profile_mz, profile_inten = profile.extract_spectrum(50)
        top, profile_top = np.argmax(inten), np.argmax(profile_inten)
        assert mz[top] == pytest.approx(profile_mz[profile_top], rel=5e-6)
        assert inten[top] == pytest.approx(profile_inten[profile_top], rel=0.01)
        # 12 bytes a centroid, all of them in one array
        byte_counts = sum(record["SpectrumParamValues"][1]["ByteCount"] for record in records)
        assert centroid._mz.nbytes + centroid._inten.nbytes == byte_counts


def test_an_xic_of_centroids_follows_the_profiles_and_leaves_no_scan_out():
    with AgilentReader() as profile, AgilentCentroidReader() as centroid:
        profile.load(require(DATA_PATH))
        centroid.load(DATA_PATH)
        low, high = 965.9998 * (1 - 10e-6), 965.9998 * (1 + 10e-6)
        rt, xic = centroid.extract_chromatogram(low, high)
        assert np.array_equal(rt, profile.RT_list) and xic.dtype == np.float64
        assert np.corrcoef(xic[1:], profile.extract_chromatogram(low, high)[1][1:])[0, 1] > 0.9
        assert centroid.mz_window_gaps(low, high).tolist() == [0.0] * centroid.N_scans
        mz, intensity, representative = centroid.extract_averaged_spectrum(0.5, 0.6)
        assert len(mz) == len(intensity) == len(centroid.extract_spectrum(representative)[0])
        # on the first scan, of none (the window shown on opening): on the nearest with any, or none if none
        rt0 = centroid.RT_list[0]
        mz, intensity, representative = centroid.extract_averaged_spectrum(rt0 - 0.5, rt0 + 0.5)
        assert representative == 1 and len(mz) == len(centroid.extract_spectrum(1)[0]) and intensity.any()
        mz, intensity, representative = centroid.extract_averaged_spectrum(rt0 - 1e-4, rt0 + 1e-4)
        assert representative == 0 and len(mz) == len(intensity) == 0
        out = np.zeros(10, dtype=np.float32)
        centroid.max_into_bins(50, np.linspace(960, 970, 11), out)
        assert out[6] == centroid.extract_spectrum(50)[1].max()


def test_the_spectra_a_run_holds_are_its_files_not_empty(tmp_path):
    acq_data = tmp_path / "x.d" / "AcqData"
    acq_data.mkdir(parents=True)
    (acq_data / "MSProfile.bin").write_bytes(b"x" * 10)
    (acq_data / "MSPeak.bin").write_bytes(b"")
    assert stored_spectra(tmp_path / "x.d") == {"profile": 10}
    (acq_data / "MSPeak.bin").write_bytes(b"x" * 3)
    assert stored_spectra(tmp_path / "x.d") == {"profile": 10, "centroid": 3}
    assert stored_spectra(require(DATA_PATH)).keys() == {"profile", "centroid"}


def test_a_run_without_centroid_spectra_is_refused():
    path = require(PEAK_EVAL_DIR / "20220801_HILIC_LipidStd" / "1_blank__p1.d")
    assert stored_spectra(path).keys() == {"profile"}
    with pytest.raises(ValueError, match="no centroid spectra"):
        AgilentCentroidReader().load(path)


# -- opening in the main window ------------------------------------------------

def _answering(monkeypatch, button_text, *, apply_to_all=False):
    """QMessageBox.exec answering with the button of that text; the texts asked of, kept."""
    from PyQt6.QtWidgets import QMessageBox

    asked = []

    def exec_(box):
        asked.append((box.text(), box.checkBox() is not None))
        if box.checkBox() is not None:
            box.checkBox().setChecked(apply_to_all)
        next(button for button in box.buttons() if button.text() == button_text).click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", exec_)
    return asked


def test_folders_chosen_are_read_as_answered_and_apply_to_all_answers_for_the_rest(monkeypatch):
    from RAPID2_Agilent.models.rpd_model import AgilentCentroidModel, AgilentDModel
    from tests.test_target_panel import build_app

    _app, _view, _model, presenter = build_app(())
    paths = [str(require(DATA_PATH)), str(require(COPY_DATA_PATH))]
    monkeypatch.setattr(presenter, "open_native_file", lambda *args, **kwargs: paths)
    asked = _answering(monkeypatch, "Centroid", apply_to_all=True)
    presenter.open_d_folders_clicked()
    # asked once, of the first, offered to apply to the second
    assert len(asked) == 1 and asked[0][0].startswith("dummy_RP_2min_neg.d holds centroid spectra") and asked[0][1]
    models = presenter.data_stack_presenter.models
    assert [type(model) for model in models] == [AgilentCentroidModel] * 2
    assert [model.file_path for model in models] == [DATA_PATH, COPY_DATA_PATH]

    _app, _view, _model, presenter = build_app(())
    monkeypatch.setattr(presenter, "open_native_file", lambda *args, **kwargs: paths)
    asked = _answering(monkeypatch, "Profile")
    presenter.open_d_folders_clicked()
    # asked of each
    assert len(asked) == 2 and not asked[1][1]
    assert [type(model) for model in presenter.data_stack_presenter.models] == [AgilentDModel] * 2


def test_a_cancel_opens_none_and_folders_given_are_read_of_their_profile_unasked(monkeypatch):
    from RAPID2_Agilent.models.rpd_model import AgilentDModel
    from tests.test_target_panel import build_app

    _app, _view, _model, presenter = build_app(())
    monkeypatch.setattr(presenter, "open_native_file", lambda *args, **kwargs: [str(require(DATA_PATH))])
    asked = _answering(monkeypatch, "Cancel")
    presenter.open_d_folders_clicked()
    assert len(asked) == 1 and presenter.data_stack_presenter.models == []

    presenter.open_d_folders_clicked([str(DATA_PATH)])
    assert len(asked) == 1 and [type(model) for model in presenter.data_stack_presenter.models] == [AgilentDModel]


def test_folders_given_with_centroid_are_read_of_their_centroid_spectra_unasked(monkeypatch):
    from RAPID2_Agilent.main import parse_args
    from RAPID2_Agilent.models.rpd_model import AgilentCentroidModel
    from tests.test_target_panel import build_app

    args = parse_args([str(require(DATA_PATH)), "--centroid"])
    assert args.centroid and not parse_args([str(DATA_PATH)]).centroid
    _app, _view, _model, presenter = build_app(())
    asked = _answering(monkeypatch, "Profile")
    presenter.open_d_folders_clicked([str(path) for path in args.folders], centroid=args.centroid)
    assert asked == [] and [type(model) for model in presenter.data_stack_presenter.models] == [AgilentCentroidModel]
    assert presenter.data_stack_presenter.models[0].spectra == "centroid"
