import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import Trace, candidate_tops, read_dataset, write_dataset  # noqa: E402


def _trace(id_, rt, intensity, **fields):
    values = dict(run="/x.d", kind="spot", name="m/z 100", mz_window=(100.0, 100.002), polarity=1, ms_level=1)
    return Trace(id_, rt=np.asarray(rt, dtype=float), intensity=np.asarray(intensity, dtype=float), **(values | fields))


def _gaussians(rt, *peaks):
    return sum(height * np.exp(-((rt - center) / width) ** 2 / 2) for center, width, height in peaks)


def test_a_dataset_reads_back_as_written(tmp_path):
    traces = [_trace("0000", [0, 1, 2], [1, 3, 1]), _trace("0001", [0, 1, 2, 3], [0, 5, 2, 1], kind="random",
                                                           mz_window=None, polarity=None)]
    write_dataset(tmp_path / "x.npz", traces)

    read = read_dataset(tmp_path / "x.npz")

    assert [trace.id for trace in read] == ["0000", "0001"]
    assert read[1].kind == "random" and read[1].mz_window is None and read[0].mz_window == (100.0, 100.002)
    np.testing.assert_array_equal(read[1].intensity, [0, 5, 2, 1])


def test_the_tops_offered_are_the_cnns_with_their_probability_and_the_clear_ones_of_scipy():
    rt = np.linspace(0, 10, 601)
    rng = np.random.default_rng(0)

    candidates = candidate_tops(rt, _gaussians(rt, (3.0, 0.08, 100.0), (6.0, 0.1, 50.0)) + rng.random(len(rt)))

    kensert = [candidate for candidate in candidates if candidate["source"] == "kensert"]
    assert [round(candidate["rt"], 1) for candidate in kensert if candidate["confidence"] >= 0.9] == [3.0, 6.0]
    assert [candidate["rt"] for candidate in candidates] == sorted(candidate["rt"] for candidate in candidates)
    # none of scipy's where one of the CNN's is
    assert all(candidate["source"] == "kensert" for candidate in candidates if abs(candidate["rt"] - 3.0) < 0.05)


def test_labels_are_saved_at_each_change_and_done_leaves_the_undecided(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from label_peaks import LabelWindow

    app = QApplication.instance() or QApplication([])
    rt = np.linspace(0, 10, 601)
    write_dataset(tmp_path / "x.npz", [_trace("0000", rt, _gaussians(rt, (3.0, 0.08, 100.0)) + 1.0),
                                       _trace("0001", rt, _gaussians(rt, (5.0, 0.08, 100.0)) + 1.0)])
    window = LabelWindow(str(tmp_path / "x.npz"), tmp_path / "x.labels.json")

    window.take_all()
    window.current["added"].append(7.0)
    window.changed()
    window.done()

    saved = json.loads((tmp_path / "x.labels.json").read_text())["traces"]["0000"]
    assert saved["done"] and saved["added"] == [7.0]
    assert {candidate["state"] for candidate in saved["candidates"]} <= {"taken", "left"}
    assert any(candidate["state"] == "taken" and abs(candidate["rt"] - 3.0) < 0.05 for candidate in saved["candidates"])
    # on to the next not done
    assert window.index == 1
    window.close()
    app.processEvents()


class _Point:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


def test_tops_offered_and_added_can_be_unsure_and_older_labels_read(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from label_peaks import LabelWindow

    app = QApplication.instance() or QApplication([])
    rt = np.linspace(0, 10, 601)
    write_dataset(tmp_path / "x.npz", [_trace("0000", rt, _gaussians(rt, (3.0, 0.08, 100.0)) + 1.0)])
    # saved before there was unsure: no added_unsure
    (tmp_path / "x.labels.json").write_text(json.dumps({"dataset": "x.npz", "traces": {"0000": {
        "candidates": [{"rt": 3.0, "source": "kensert", "confidence": 1.0, "state": "undecided"}],
        "added": [5.0], "done": False, "unusable": False, "note": ""}}}))
    window = LabelWindow(str(tmp_path / "x.npz"), tmp_path / "x.labels.json")

    for _ in range(3):  # taken, left, unsure
        window.on_candidate_clicked(None, [_Point(0)], None)
    window.on_added_clicked(None, [_Point(("added", 0))], None)  # unsure

    saved = json.loads((tmp_path / "x.labels.json").read_text())["traces"]["0000"]
    assert saved["candidates"][0]["state"] == "unsure"
    assert saved["added"] == [] and saved["added_unsure"] == [5.0]
    window.on_added_clicked(None, [_Point(("added_unsure", 0))], None)  # removed
    window.on_candidate_clicked(None, [_Point(0)], None)  # undecided again
    saved = json.loads((tmp_path / "x.labels.json").read_text())["traces"]["0000"]
    assert saved["added_unsure"] == [] and saved["candidates"][0]["state"] == "undecided"
    window.close()
    app.processEvents()


def test_shift_click_takes_the_top_there_and_what_is_near_it_off_it_gives_way(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from label_peaks import LabelWindow

    app = QApplication.instance() or QApplication([])
    rt = np.linspace(0, 10, 601)  # 1 s a scan
    write_dataset(tmp_path / "x.npz", [_trace("0000", rt, _gaussians(rt, (3.0, 0.08, 100.0), (6.0, 0.08, 100.0)) + 1.0)])
    step = rt[1]
    # offered: one 2 scans off the top at 3, one at the top at 6; added: one 1 scan off the top at 3
    (tmp_path / "x.labels.json").write_text(json.dumps({"dataset": "x.npz", "traces": {"0000": {
        "candidates": [{"rt": 3.0 - 2 * step, "source": "kensert", "confidence": 0.95, "state": "unsure"},
                       {"rt": 6.0, "source": "scipy", "confidence": None, "state": "left"}],
        "added": [], "added_unsure": [3.0 + step], "done": False, "unusable": False, "note": ""}}}))
    window = LabelWindow(str(tmp_path / "x.npz"), tmp_path / "x.labels.json")

    window.take_top_at(3.0 + 3 * step)  # near the top at 3: the top there taken, the others near it gone
    window.take_top_at(6.0 - step)  # the one offered there taken

    saved = json.loads((tmp_path / "x.labels.json").read_text())["traces"]["0000"]
    assert [c["state"] for c in saved["candidates"]] == ["left", "taken"]
    assert saved["added"] == [pytest.approx(3.0)] and saved["added_unsure"] == []
    window.undo()
    window.undo()
    saved = json.loads((tmp_path / "x.labels.json").read_text())["traces"]["0000"]
    assert saved["candidates"][0]["state"] == "unsure" and saved["added_unsure"] == [pytest.approx(3.0 + step)]
    window.show_help()
    window.close()
    app.processEvents()


def test_undo_puts_back_the_last_changes_and_saves(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from label_peaks import LabelWindow

    app = QApplication.instance() or QApplication([])
    rt = np.linspace(0, 10, 601)
    write_dataset(tmp_path / "x.npz", [_trace("0000", rt, _gaussians(rt, (3.0, 0.08, 100.0)) + 1.0),
                                       _trace("0001", rt, _gaussians(rt, (5.0, 0.08, 100.0)) + 1.0)])
    window = LabelWindow(str(tmp_path / "x.npz"), tmp_path / "x.labels.json")
    window.on_candidate_clicked(None, [_Point(0)], None)  # taken
    window.done()  # and on to the next

    window.undo()  # not done, back at the first
    saved = json.loads((tmp_path / "x.labels.json").read_text())["traces"]["0000"]
    assert window.index == 0 and not saved["done"] and saved["candidates"][0]["state"] == "taken"
    window.undo()
    saved = json.loads((tmp_path / "x.labels.json").read_text())["traces"]["0000"]
    assert saved["candidates"][0]["state"] == "undecided"
    window.undo()  # nothing more to undo
    window.close()
    app.processEvents()


RAW = Path(__file__).resolve().parents[2] / "data" / "peak_eval" / "raw"
JUMPING = RAW / "20260520_Z-HILIC-PA_test" / "16_PRJ_0-1-2_LipidStd_dilx2inMeOH_n3.d"
STEADY = RAW / "20220801_RP_LipidStd" / "5_Std_100ng-200ul-dil20_p0.d"


def test_scans_of_a_calibration_off_are_put_back_in_line_and_those_of_one_steady_not():
    if not (JUMPING.is_dir() and STEADY.is_dir()):
        pytest.skip("data/peak_eval/raw not found")
    from calibration import calibration_offsets, repair_calibration_jumps
    from RAPID2_Agilent.io import AgilentReader

    with AgilentReader() as reader:
        reader.load(STEADY)
        assert repair_calibration_jumps(reader) == []
    with AgilentReader() as reader:
        reader.load(JUMPING)
        stored = reader.ms_calib_info.calib_info[262].copy()
        repaired = repair_calibration_jumps(reader)
        # scan 262 had POPtdEtOH [M-H]- at 701.5706, +81 ppm, and the lock mass at -30 ppm
        assert 262 in repaired and 0.05 < len(repaired) / reader.N_scans < 0.15
        assert np.abs(calibration_offsets(reader)).max() <= 10
        mz, intensity = reader.extract_spectrum(262)
        near = (mz > 701.4) & (mz < 701.65)
        assert abs(mz[near][np.argmax(intensity[near])] - 701.5127) / 701.5127 * 1e6 < 5
        # coeff and base only
        np.testing.assert_array_equal(reader.ms_calib_info.calib_info[262][2:], stored[2:])
