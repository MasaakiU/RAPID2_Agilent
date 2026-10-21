from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QStackedWidget,
    QWidget,
)


class MyDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *keys, **kwargs):
        super().__init__(*keys, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setKeyboardTracking(False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)

    def wheelEvent(self, event):
        event.ignore()


class MySpinBox(QSpinBox):
    def __init__(self, *keys, **kwargs):
        super().__init__(*keys, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setKeyboardTracking(False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)

    def wheelEvent(self, event):
        event.ignore()


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

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.label)
        layout.addWidget(self.center_spin_box)
        layout.addWidget(QLabel("±"))
        layout.addWidget(self.tolerance_spin_box)

    def value(self) -> float:
        return self.center_spin_box.value()

    def tolerance(self) -> float:
        return self.tolerance_spin_box.value()

    def set_value(self, value: float) -> None:
        self.center_spin_box.setValue(value)

    def set_tolerance(self, tolerance: float) -> None:
        self.tolerance_spin_box.setValue(tolerance)


class MzRangeControl(QWidget):
    MZ_MODE = "m/z"
    PPM_MODE = "ppm"

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
        self.mode_combo_box = QComboBox()
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

        self.mode_combo_box.addItems([self.MZ_MODE, self.PPM_MODE])
        self.mode_combo_box.currentTextChanged.connect(self._update_tolerance_mode)

        self.tolerance_stack.addWidget(self.tolerance_spin_box)
        self.tolerance_stack.addWidget(self.ppm_spin_box)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.label)
        layout.addWidget(self.center_spin_box)
        layout.addWidget(QLabel("±"))
        layout.addWidget(self.tolerance_stack)
        layout.addWidget(self.mode_combo_box)

        self._update_tolerance_mode(self.mode_combo_box.currentText())

    def value(self) -> float:
        return self.center_spin_box.value()

    def tolerance(self) -> float:
        return self.tolerance_spin_box.value()

    def ppm_tolerance(self) -> int:
        return self.ppm_spin_box.value()

    def tolerance_mode(self) -> str:
        return self.mode_combo_box.currentText()

    def set_value(self, value: float) -> None:
        self.center_spin_box.setValue(value)

    def set_tolerance(self, tolerance: float) -> None:
        self.tolerance_spin_box.setValue(tolerance)

    def set_ppm_tolerance(self, ppm_tolerance: int) -> None:
        self.ppm_spin_box.setValue(ppm_tolerance)

    def set_tolerance_mode(self, mode: str) -> None:
        self.mode_combo_box.setCurrentText(mode)

    def _update_tolerance_mode(self, mode: str) -> None:
        stack_index = 1 if mode == self.PPM_MODE else 0
        self.tolerance_stack.setCurrentIndex(stack_index)
