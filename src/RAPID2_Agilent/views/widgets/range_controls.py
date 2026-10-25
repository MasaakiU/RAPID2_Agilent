from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QPushButton,
    QStackedWidget,
    QWidget,
)
from RAPID2_Agilent.views.layout_metrics import TINY


class _BlankableSpinBox:
    """Of MyDoubleSpinBox and MySpinBox: set_blank() shows no value until one is set or typed, for a field whose
    targets have different values.

    Blank, the box holds a value a step below its minimum, shown as a blank special value text: any value typed in
    is a change (valueChanged), and leaving the box untouched is none. The minimum comes back with the next value.
    """

    _minimum_before_blank = None

    def set_blank(self):
        if self._minimum_before_blank is None:
            self._minimum_before_blank = self.minimum()
            self.setMinimum(self._minimum_before_blank - self.singleStep())
            self.setSpecialValueText(" ")
        super().setValue(self.minimum())

    def is_blank(self) -> bool:
        return self._minimum_before_blank is not None

    def setValue(self, value):
        self._end_blank()
        super().setValue(value)

    def stepEnabled(self):
        # a step from no value would be a value nobody typed
        if self.is_blank():
            return QAbstractSpinBox.StepEnabledFlag.StepNone
        return super().stepEnabled()

    def _on_value_changed(self, value):
        if self.is_blank() and value != self.minimum():
            self._end_blank()  # typed in

    def _end_blank(self):
        if self._minimum_before_blank is not None:
            minimum, self._minimum_before_blank = self._minimum_before_blank, None
            self.setSpecialValueText("")
            self.setMinimum(minimum)


class MyDoubleSpinBox(_BlankableSpinBox, QDoubleSpinBox):
    def __init__(self, *keys, **kwargs):
        super().__init__(*keys, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setKeyboardTracking(False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.valueChanged.connect(self._on_value_changed)

    def wheelEvent(self, event):
        event.ignore()


class MySpinBox(_BlankableSpinBox, QSpinBox):
    def __init__(self, *keys, **kwargs):
        super().__init__(*keys, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setKeyboardTracking(False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.valueChanged.connect(self._on_value_changed)

    def wheelEvent(self, event):
        event.ignore()


def typed_value(spin_box: QDoubleSpinBox | QSpinBox) -> float | int | None:
    """The value typed into ``spin_box`` so far, which it takes only once left (no keyboard tracking); None while
    the text is not a value it accepts."""
    if not spin_box.hasAcceptableInput():
        return None
    return spin_box.valueFromText(spin_box.cleanText())


def _set_or_blank(spin_box: MyDoubleSpinBox | MySpinBox, value) -> None:
    if value is None:
        spin_box.set_blank()
    else:
        spin_box.setValue(value)


def _fix_spin_box_width(*spin_boxes: QAbstractSpinBox) -> None:
    """Give every spin box the same width so RT and m/z rows line up."""
    for spin_box in spin_boxes:
        width = spin_box.fontMetrics().horizontalAdvance("00000.0000") + 16
        spin_box.setFixedWidth(width)


class RtRangeControl(QWidget):
    def __init__(
        self,
        parent=None,
        *,
        decimals: int,
        single_step: float,
        center_minimum: float,
        center_maximum: float,
        tolerance_minimum: float,
        tolerance_maximum: float,
    ):
        super().__init__(parent)
        self.setObjectName("RtRangeControl")

        self.label = QLabel("RT:")
        self.center_spin_box = MyDoubleSpinBox()
        self.tolerance_spin_box = MyDoubleSpinBox()

        self.center_spin_box.setDecimals(decimals)
        self.center_spin_box.setSingleStep(single_step)
        self.center_spin_box.setRange(center_minimum, center_maximum)

        self.tolerance_spin_box.setDecimals(decimals)
        self.tolerance_spin_box.setSingleStep(single_step)
        self.tolerance_spin_box.setRange(tolerance_minimum, tolerance_maximum)

        _fix_spin_box_width(self.center_spin_box, self.tolerance_spin_box)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(TINY)
        layout.addWidget(self.label)
        layout.addWidget(self.center_spin_box)
        layout.addWidget(QLabel("±"))
        layout.addWidget(self.tolerance_spin_box)
        layout.addStretch(1)

    def value(self) -> float:
        return self.center_spin_box.value()

    def tolerance(self) -> float:
        return self.tolerance_spin_box.value()

    def set_value(self, value: float | None) -> None:
        """None: blank, see _BlankableSpinBox; also for the setters below."""
        _set_or_blank(self.center_spin_box, value)

    def set_tolerance(self, tolerance: float | None) -> None:
        _set_or_blank(self.tolerance_spin_box, tolerance)


class MzRangeControl(QWidget):
    mode_changed = pyqtSignal(str)

    MZ_MODE = "m/z"
    PPM_MODE = "ppm"
    # Text of the mode button (used as the row label); it shows the current mode.
    MODE_LABELS = {MZ_MODE: "m/z (abs)", PPM_MODE: "m/z (ppm)"}

    def __init__(
        self,
        parent=None,
        *,
        decimals: int,
        single_step: float,
        center_minimum: float,
        center_maximum: float,
        tolerance_minimum: float,
        tolerance_maximum: float,
        ppm_minimum: int,
        ppm_maximum: int,
        ppm_single_step: int,
    ):
        super().__init__(parent)
        self.setObjectName("MzRangeControl")

        self.label = QLabel("mz:")
        self.center_spin_box = MyDoubleSpinBox()
        self._mode = self.MZ_MODE
        self.mode_button = QPushButton()
        self.mode_button.clicked.connect(self._toggle_tolerance_mode)
        # Below Fusion's 80px minimum: the longer text plus Fusion's margins (6px) and frame (1px) on each side.
        metrics = self.mode_button.fontMetrics()
        self.mode_button.setFixedWidth(max(metrics.horizontalAdvance(t) for t in self.MODE_LABELS.values()) + 5)
        self.tolerance_spin_box = MyDoubleSpinBox()
        self.ppm_spin_box = MySpinBox()
        self.tolerance_stack = QStackedWidget()

        self.center_spin_box.setDecimals(decimals)
        self.center_spin_box.setSingleStep(single_step)
        self.center_spin_box.setRange(center_minimum, center_maximum)

        self.tolerance_spin_box.setDecimals(decimals)
        self.tolerance_spin_box.setSingleStep(single_step)
        self.tolerance_spin_box.setRange(tolerance_minimum, tolerance_maximum)

        self.ppm_spin_box.setSingleStep(ppm_single_step)
        self.ppm_spin_box.setRange(ppm_minimum, ppm_maximum)

        self.tolerance_stack.addWidget(self.tolerance_spin_box)
        self.tolerance_stack.addWidget(self.ppm_spin_box)

        _fix_spin_box_width(self.center_spin_box, self.tolerance_spin_box, self.ppm_spin_box)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(TINY)
        layout.addWidget(self.label)
        layout.addWidget(self.center_spin_box)
        layout.addWidget(QLabel("±"))
        layout.addWidget(self.tolerance_stack)
        layout.addStretch(1)

        self._update_tolerance_mode(self.tolerance_mode())

    def value(self) -> float:
        return self.center_spin_box.value()

    def tolerance(self) -> float:
        return self.tolerance_spin_box.value()

    def ppm_tolerance(self) -> int:
        return self.ppm_spin_box.value()

    def tolerance_mode(self) -> str:
        return self._mode

    def set_value(self, value: float | None) -> None:
        _set_or_blank(self.center_spin_box, value)

    def set_tolerance(self, tolerance: float | None) -> None:
        _set_or_blank(self.tolerance_spin_box, tolerance)

    def set_ppm_tolerance(self, ppm_tolerance: int | None) -> None:
        _set_or_blank(self.ppm_spin_box, ppm_tolerance)

    def set_tolerance_mode(self, mode: str) -> None:
        self._update_tolerance_mode(mode)

    def _toggle_tolerance_mode(self) -> None:
        self._update_tolerance_mode(self.MZ_MODE if self._mode == self.PPM_MODE else self.PPM_MODE)

    def _update_tolerance_mode(self, mode: str) -> None:
        changed = mode != self._mode
        self._mode = mode
        self.mode_button.setText(self.MODE_LABELS[mode])
        stack_index = 1 if mode == self.PPM_MODE else 0
        self.tolerance_stack.setCurrentIndex(stack_index)
        if changed:
            self.mode_changed.emit(mode)
