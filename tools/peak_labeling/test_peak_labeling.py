import json
import os
import sys
from pathlib import Path

import numpy as np

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
