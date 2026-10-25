"""Label the peaks of the XICs of a dataset (extract_xics.py): of the tops offered -- the CNN's of Kensert et al.,
with their probability, and the scipy stand-in's -- take those of peaks and leave the others, and add those missed.

    python tools/peak_labeling/label_peaks.py xics.npz [--labels xics.labels.json]

Click a top offered to take it, again to leave it, again to leave it undecided. Shift-click the trace to add a top
there, at its highest point within 3 scans; click one added to remove it. A takes the tops of 0.9 or more still
undecided; Done (Enter) marks the trace looked at whole, every peak of it labeled, and leaves those still undecided.
N and P go to the next trace and the one before. The labels are saved at each change, beside the dataset by default.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SAME_TOP_SCANS, candidate_tops, read_dataset  # noqa: E402

TAKE_ALL_CONFIDENCE = 0.9
STATES = ("undecided", "taken", "left")
STYLES = {  # symbol, brush, pen of a top offered
    "undecided": ("o", None, pg.mkPen("#808080", width=1.5)),
    "taken": ("t1", pg.mkBrush("#00A000"), pg.mkPen("#00A000")),
    "left": ("x", pg.mkBrush("#D00000"), pg.mkPen("#D00000")),
}


def default_labels_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    return path.with_name(path.name.removesuffix(".npz") + ".labels.json")


class Labels:
    """The labels of the traces of a dataset, by trace id, saved at each change: of each, the tops offered with their
    state, the tops added, whether it is done (looked at whole), whether it is of no use, and a note."""

    def __init__(self, path: Path, dataset_path: str):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text())
        else:
            self.data = {"dataset": str(dataset_path), "traces": {}}

    def of(self, trace) -> dict:
        """The labels of a trace; the tops offered found the first time, and kept so after."""
        labels = self.data["traces"].get(trace.id)
        if labels is None:
            candidates = [candidate | {"state": "undecided"} for candidate in candidate_tops(trace.rt, trace.intensity)]
            labels = {"candidates": candidates, "added": [], "done": False, "unusable": False, "note": ""}
            self.data["traces"][trace.id] = labels
            self.save()
        return labels

    def save(self) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=1))
        os.replace(temporary, self.path)


class LabelWindow(QMainWindow):
    def __init__(self, dataset_path: str, labels_path: Path):
        super().__init__()
        self.setWindowTitle("Peak labeling")
        self.traces = read_dataset(dataset_path)
        self.labels = Labels(labels_path, dataset_path)
        self.index = 0

        self.trace_list = QListWidget()
        self.plot = pg.PlotWidget(background="w")
        self.plot.setLabel("bottom", "RT [min]")
        self.curve = self.plot.plot(pen=pg.mkPen("#1f77b4", width=1))
        self.candidate_items = pg.ScatterPlotItem(size=12, hoverable=True)
        self.added_items = pg.ScatterPlotItem(size=12, symbol="t1", brush=pg.mkBrush("#2060FF"), pen=pg.mkPen("#2060FF"))
        for item in (self.candidate_items, self.added_items):
            item.setZValue(5)
            self.plot.addItem(item)
        self.confidence_labels: list[pg.TextItem] = []
        self.title_label = QLabel()
        self.status_label = QLabel()
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("Note")
        self.unusable_box = QCheckBox("Of no use")
        take_all_button = QPushButton(f"Take {TAKE_ALL_CONFIDENCE} or more (A)")
        done_button = QPushButton("Done (Enter)")
        previous_button = QPushButton("Previous (P)")
        next_button = QPushButton("Next (N)")

        buttons = QHBoxLayout()
        for widget in (take_all_button, done_button, self.unusable_box, previous_button, next_button):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.title_label)
        right_layout.addWidget(self.plot, 1)
        right_layout.addLayout(buttons)
        right_layout.addWidget(self.note_edit)
        right_layout.addWidget(self.status_label)
        splitter = QSplitter()
        splitter.addWidget(self.trace_list)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.resize(1400, 700)

        self.trace_list.currentRowChanged.connect(self.show_trace)
        self.candidate_items.sigClicked.connect(self.on_candidate_clicked)
        self.added_items.sigClicked.connect(self.on_added_clicked)
        self.plot.scene().sigMouseClicked.connect(self.on_plot_clicked)
        take_all_button.clicked.connect(self.take_all)
        done_button.clicked.connect(self.done)
        previous_button.clicked.connect(lambda: self.go(-1))
        next_button.clicked.connect(lambda: self.go(1))
        self.note_edit.editingFinished.connect(self.on_note_edited)
        self.unusable_box.toggled.connect(self.on_unusable_toggled)
        for keys, action in (("A", self.take_all), ("Return", self.done), ("Enter", self.done),
                             ("N", lambda: self.go(1)), ("P", lambda: self.go(-1))):
            QShortcut(QKeySequence(keys), self, action)

        for trace in self.traces:
            self.trace_list.addItem("")
        self.refresh_list()
        undone = [i for i, trace in enumerate(self.traces) if not self.labels.data["traces"].get(trace.id, {}).get("done")]
        self.trace_list.setCurrentRow(undone[0] if undone else 0)

    ### the trace shown ###
    @property
    def trace(self):
        return self.traces[self.index]

    @property
    def current(self) -> dict:
        return self.labels.of(self.trace)

    def show_trace(self, index: int):
        if index < 0:
            return
        self.index = index
        trace, labels = self.trace, self.current
        self.title_label.setText(f"{trace.id}  {trace.kind}: {trace.name}  -  {Path(trace.run).name}")
        self.curve.setData(trace.rt, trace.intensity)
        self.note_edit.setText(labels["note"])
        self.unusable_box.blockSignals(True)
        self.unusable_box.setChecked(labels["unusable"])
        self.unusable_box.blockSignals(False)
        self.plot.enableAutoRange()
        self.draw_tops()

    def draw_tops(self):
        trace, labels = self.trace, self.current
        spots = []
        for i, candidate in enumerate(labels["candidates"]):
            symbol, brush, pen = STYLES[candidate["state"]]
            spots.append({"pos": (candidate["rt"], self._height(candidate["rt"])), "symbol": symbol, "brush": brush,
                          "pen": pen, "data": i})
        self.candidate_items.setData(spots)
        self.added_items.setData([{"pos": (rt, self._height(rt)), "data": i} for i, rt in enumerate(labels["added"])])
        for item in self.confidence_labels:
            self.plot.removeItem(item)
        self.confidence_labels = []
        for candidate in labels["candidates"]:
            text = "scipy" if candidate["confidence"] is None else f"{candidate['confidence']:.2f}"
            label = pg.TextItem(text, color="#606060", anchor=(0.5, 1.4))
            label.setPos(candidate["rt"], self._height(candidate["rt"]))
            self.plot.addItem(label)
            self.confidence_labels.append(label)
        states = [candidate["state"] for candidate in labels["candidates"]]
        done = sum(self.labels.data["traces"].get(t.id, {}).get("done", False) for t in self.traces)
        self.status_label.setText(
            f"Trace {self.index + 1} of {len(self.traces)}, {done} done.  This one: "
            f"{states.count('taken')} taken, {states.count('left')} left, {states.count('undecided')} undecided, "
            f"{len(labels['added'])} added" + ("  (done)" if labels["done"] else "")
        )

    def _height(self, rt: float) -> float:
        return float(self.trace.intensity[int(np.argmin(np.abs(self.trace.rt - rt)))])

    ### editing ###
    def on_candidate_clicked(self, _item, points, _event):
        candidate = self.current["candidates"][points[0].data()]
        candidate["state"] = STATES[(STATES.index(candidate["state"]) + 1) % len(STATES)]
        self.changed()

    def on_added_clicked(self, _item, points, _event):
        del self.current["added"][points[0].data()]
        self.changed()

    def on_plot_clicked(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            return
        rt = self.plot.getViewBox().mapSceneToView(event.scenePos()).x()
        trace, labels = self.trace, self.current
        nearest = int(np.argmin(np.abs(trace.rt - rt)))
        near = np.arange(max(nearest - SAME_TOP_SCANS, 0), min(nearest + SAME_TOP_SCANS + 1, len(trace.rt)))
        top = float(trace.rt[near[np.argmax(trace.intensity[near])]])
        step = float(np.median(np.diff(trace.rt)))
        offered = [c for c in labels["candidates"] if abs(c["rt"] - top) <= SAME_TOP_SCANS * step]
        if offered:
            offered[0]["state"] = "taken"  # one offered there: taken rather than added again
        elif all(abs(added - top) > SAME_TOP_SCANS * step for added in labels["added"]):
            labels["added"].append(top)
            labels["added"].sort()
        self.changed()

    def take_all(self):
        for candidate in self.current["candidates"]:
            if candidate["state"] == "undecided" and (candidate["confidence"] or 0) >= TAKE_ALL_CONFIDENCE:
                candidate["state"] = "taken"
        self.changed()

    def done(self):
        labels = self.current
        for candidate in labels["candidates"]:
            if candidate["state"] == "undecided":
                candidate["state"] = "left"
        labels["done"] = True
        self.changed()
        undone = [i for i, trace in enumerate(self.traces) if not self.labels.data["traces"].get(trace.id, {}).get("done")]
        later = [i for i in undone if i > self.index]
        if later or undone:
            self.trace_list.setCurrentRow((later or undone)[0])

    def on_note_edited(self):
        self.current["note"] = self.note_edit.text()
        self.labels.save()

    def on_unusable_toggled(self, checked: bool):
        self.current["unusable"] = checked
        self.changed()

    def go(self, step: int):
        self.trace_list.setCurrentRow(min(max(self.index + step, 0), len(self.traces) - 1))

    def changed(self):
        self.labels.save()
        self.draw_tops()
        self.refresh_list()

    def refresh_list(self):
        for i, trace in enumerate(self.traces):
            labels = self.labels.data["traces"].get(trace.id, {})
            mark = "x " if labels.get("unusable") else ("✓ " if labels.get("done") else "   ")
            self.trace_list.item(i).setText(f"{mark}{trace.id} {trace.kind}: {trace.name} - {Path(trace.run).name}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="a dataset of extract_xics.py, .npz")
    parser.add_argument("--labels", help="the labels, .json; the dataset's name .labels.json beside it by default")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    window = LabelWindow(args.dataset, Path(args.labels) if args.labels else default_labels_path(args.dataset))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
