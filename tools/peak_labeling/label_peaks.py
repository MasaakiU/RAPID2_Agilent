"""Label the peaks of the XICs of a dataset (extract_xics.py): of the tops offered -- the CNN's of Kensert et al.,
with their probability, and the scipy stand-in's -- take those of peaks and leave the others, and add those missed.

    python tools/peak_labeling/label_peaks.py xics.npz [--labels xics.labels.json]

Click a top offered to take it, again to leave it, again to mark it unsure (not scored: a peak or not, either may be
right), again to leave it undecided. Shift-click the trace to take the top there, at its highest point within 3
scans: the one offered there if any, else added, and those offered or added near it, off it, left or removed; click
one added to mark it unsure, again to remove it. A takes the tops of 0.9 or more still undecided; Done (Enter)
marks the trace looked at whole, every peak of it labeled, and leaves those still undecided. N and P go to the next
trace and the one before; Ctrl+Z (Cmd+Z too on a Mac) undoes the last change, of whichever trace. Help shows all this
and the rules of labeling. The labels are saved at each change, beside the dataset by default.
"""

import argparse
import copy
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
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SAME_TOP_SCANS, candidate_tops, read_dataset  # noqa: E402

TAKE_ALL_CONFIDENCE = 0.9
UNDO_STEPS = 200
STATES = ("undecided", "taken", "left", "unsure")
STYLES = {  # symbol, brush, pen of a top offered
    "undecided": ("o", None, pg.mkPen("#808080", width=1.5)),
    "taken": ("t1", pg.mkBrush("#00A000"), pg.mkPen("#00A000")),
    "left": ("x", pg.mkBrush("#D00000"), pg.mkPen("#D00000")),
    "unsure": ("d", pg.mkBrush("#E69F00"), pg.mkPen("#E69F00")),
}
ADDED_STYLES = {  # of a top added, taken or unsure
    "added": ("t1", pg.mkBrush("#2060FF"), pg.mkPen("#2060FF")),
    "added_unsure": ("d", pg.mkBrush("#E69F00"), pg.mkPen("#2060FF", width=1.5)),
}
HELP = """\
操作
・候補の頂点をクリック：採用（緑の▲）→ 却下（赤の×）→ 保留（橙の◆）→ 未決（灰色の○）
　数字は CNN（Kensert ら）の確率、「scipy」は簡易版の候補
・Shift + クリック：そこを頂点にして採用（近く 3 スキャン以内のいちばん高い点に合わせる）
　候補がちょうどそこにあればそれを採用。ずれた候補や足した頂点が近くにあれば、却下・削除して付け直す
・足した頂点（青の▲）をクリック：保留（橙の◆、青の縁）→ もう一度で削除
・A：確率 0.9 以上の未決をまとめて採用
・Done（Enter）：このトレースを見終わったとして確定（未決は却下）、次の未確定へ
・N / P：次・前のトレース
・Ctrl + Z（Mac では ⌘ + Z も）：直前の変更を戻す
・Of no use：使えないトレースの印。Note：メモ
・グラフ：ホイールで拡大・縮小、ドラッグで移動。左下の「A」か右クリック → View All で全体
・変更はその場で保存される（保存ボタンはない）

付け方
・ターゲットかどうか、blank かどうかに関係なく、トレースのピークはすべて付ける（void 付近も）
・割れたピークは頂点をそれぞれ採用。頂点を持たない肩、ノイズと区別しにくい（S/N 2〜5 ほど）小さな山は保留
・平らな頭、ギザギザの頭は、その中央
・1 スキャンだけの落ち込み・飛び出しは、ピークでもピークの割れでもない
・Done で確定すると、印のない所は「ピークなし」として採点される
"""


def default_labels_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    return path.with_name(path.name.removesuffix(".npz") + ".labels.json")


class Labels:
    """The labels of the traces of a dataset, by trace id, saved at each change: of each, the tops offered with their
    state, the tops added (taken, and unsure), whether it is done (looked at whole), whether it is of no use, and a
    note."""

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
            labels = {"candidates": candidates, "added": [], "added_unsure": [], "done": False, "unusable": False,
                      "note": ""}
            self.data["traces"][trace.id] = labels
            self.save()
        labels.setdefault("added_unsure", [])  # of labels saved before there was unsure
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
        self.undo_stack: list[tuple[str, dict]] = []  # (trace id, its labels before a change), the last last

        self.trace_list = QListWidget()
        self.plot = pg.PlotWidget(background="w")
        self.plot.setLabel("bottom", "RT [min]")
        self.curve = self.plot.plot(pen=pg.mkPen("#1f77b4", width=1))
        self.candidate_items = pg.ScatterPlotItem(size=12, hoverable=True)
        self.added_items = pg.ScatterPlotItem(size=12)
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
        help_button = QPushButton("Help")

        buttons = QHBoxLayout()
        for widget in (take_all_button, done_button, self.unusable_box, previous_button, next_button, help_button):
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
        help_button.clicked.connect(self.show_help)
        self.note_edit.editingFinished.connect(self.on_note_edited)
        self.unusable_box.toggled.connect(self.on_unusable_toggled)
        for keys, action in (("A", self.take_all), ("Return", self.done), ("Enter", self.done),
                             ("N", lambda: self.go(1)), ("P", lambda: self.go(-1)),
                             # Ctrl+Z, Cmd+Z on a Mac; and Meta+Z, the Control key on a Mac
                             (QKeySequence.StandardKey.Undo, self.undo), ("Meta+Z", self.undo)):
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
        added = []
        for key, (symbol, brush, pen) in ADDED_STYLES.items():
            added += [{"pos": (rt, self._height(rt)), "symbol": symbol, "brush": brush, "pen": pen, "data": (key, i)}
                      for i, rt in enumerate(labels[key])]
        self.added_items.setData(added)
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
            f"{states.count('taken')} taken, {states.count('left')} left, {states.count('unsure')} unsure, "
            f"{states.count('undecided')} undecided, {len(labels['added'])} added, "
            f"{len(labels['added_unsure'])} added unsure" + ("  (done)" if labels["done"] else "")
        )

    def _height(self, rt: float) -> float:
        return float(self.trace.intensity[int(np.argmin(np.abs(self.trace.rt - rt)))])

    ### editing ###
    def on_candidate_clicked(self, _item, points, _event):
        self.remember()
        candidate = self.current["candidates"][points[0].data()]
        candidate["state"] = STATES[(STATES.index(candidate["state"]) + 1) % len(STATES)]
        self.changed()

    def on_added_clicked(self, _item, points, _event):
        key, i = points[0].data()
        self.remember()
        labels = self.current
        rt = labels[key].pop(i)
        if key == "added":  # taken, to unsure; unsure, removed
            labels["added_unsure"].append(rt)
            labels["added_unsure"].sort()
        self.changed()

    def on_plot_clicked(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            return
        self.take_top_at(self.plot.getViewBox().mapSceneToView(event.scenePos()).x())

    def take_top_at(self, rt: float):
        """Take the top at the highest scan within SAME_TOP_SCANS of rt: the one offered there, or one added there;
        those offered near it, off it, left, and those added near it removed, as it is the top there."""
        self.remember()
        trace, labels = self.trace, self.current
        nearest = int(np.argmin(np.abs(trace.rt - rt)))
        near = np.arange(max(nearest - SAME_TOP_SCANS, 0), min(nearest + SAME_TOP_SCANS + 1, len(trace.rt)))
        top = float(trace.rt[near[np.argmax(trace.intensity[near])]])
        step = float(np.median(np.diff(trace.rt)))
        offered = [c for c in labels["candidates"] if abs(c["rt"] - top) <= SAME_TOP_SCANS * step]
        at_top = min(offered, key=lambda c: abs(c["rt"] - top), default=None)
        if at_top is not None and abs(at_top["rt"] - top) >= step / 2:
            at_top = None
        for candidate in offered:
            candidate["state"] = "taken" if candidate is at_top else "left"
        for key in ("added", "added_unsure"):
            labels[key] = [added for added in labels[key] if abs(added - top) > SAME_TOP_SCANS * step]
        if at_top is None:
            labels["added"] = sorted(labels["added"] + [top])
        self.changed()

    def take_all(self):
        self.remember()
        for candidate in self.current["candidates"]:
            if candidate["state"] == "undecided" and (candidate["confidence"] or 0) >= TAKE_ALL_CONFIDENCE:
                candidate["state"] = "taken"
        self.changed()

    def done(self):
        self.remember()
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
        if self.note_edit.text() == self.current["note"]:
            return
        self.remember()
        self.current["note"] = self.note_edit.text()
        self.labels.save()

    def on_unusable_toggled(self, checked: bool):
        self.remember()
        self.current["unusable"] = checked
        self.changed()

    def show_help(self):
        box = QMessageBox(QMessageBox.Icon.NoIcon, "How to label", HELP, parent=self)
        box.setModal(False)  # to keep it beside, labeling on
        box.show()

    def remember(self):
        """Keep the trace's labels as they are before a change, to undo it."""
        self.undo_stack.append((self.trace.id, copy.deepcopy(self.current)))
        del self.undo_stack[:-UNDO_STEPS]

    def undo(self):
        """Put back the labels of the last change, and show the trace they are of."""
        if not self.undo_stack:
            return
        trace_id, labels = self.undo_stack.pop()
        self.labels.data["traces"][trace_id] = labels
        self.labels.save()
        index = next(i for i, trace in enumerate(self.traces) if trace.id == trace_id)
        if index != self.index:
            self.trace_list.setCurrentRow(index)
        else:  # as it is zoomed
            self.note_edit.setText(labels["note"])
            self.unusable_box.blockSignals(True)
            self.unusable_box.setChecked(labels["unusable"])
            self.unusable_box.blockSignals(False)
            self.draw_tops()
        self.refresh_list()

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
