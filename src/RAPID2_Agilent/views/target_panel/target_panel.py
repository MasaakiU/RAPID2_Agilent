from contextlib import contextmanager

from PyQt6.QtCore import QSortFilterProxyModel, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from RAPID2_Agilent.models.target_model import MSLevel, Target, ToleranceUnit, TraceType
from RAPID2_Agilent.views.view_constants import (
    FIELD_ERROR_COLOR,
    MZ_MISMATCH_COLOR,
    MZ_RANGE_CONTROL,
    RT_RANGE_CONTROL,
)
from RAPID2_Agilent.views.widgets import MzRangeControl, RtRangeControl

from .target_table_model import TargetTableModel


class TargetPanel(QWidget):
    """Detail of the current target (top) and the target list (bottom).

    Trace type, MS level, RT and m/z are view parameters and stay editable
    without a target (viewer mode); the other fields belong to the current
    target. The list can be searched and sorted; both only affect the display.
    """

    selection_changed = pyqtSignal()
    rt_edited = pyqtSignal()
    mz_edited = pyqtSignal()
    trace_type_edited = pyqtSignal()
    ms_level_edited = pyqtSignal()
    chemistry_text_changed = pyqtSignal()  # formula or adduct is being typed
    field_edited = pyqtSignal(str, str)  # field name, raw text
    add_requested = pyqtSignal()
    duplicate_requested = pyqtSignal()
    remove_requested = pyqtSignal()
    reset_index_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TargetPanel")
        self._updating = False
        self._table_model: TargetTableModel | None = None
        self._proxy_model: QSortFilterProxyModel | None = None

        self.name_edit = QLineEdit()
        self.formula_edit = QLineEdit()
        self.adduct_edit = QLineEdit()
        self.adduct_edit.setPlaceholderText("[M+H]+")
        self.formula_mass_label = self._make_mass_label()
        self.calculated_mz_label = self._make_mass_label()
        self.trace_type_combo = QComboBox()
        self.trace_type_combo.addItems([trace_type.value for trace_type in TraceType])
        self.ms_level_combo = QComboBox()
        self.ms_level_combo.addItems([ms_level.value for ms_level in MSLevel])
        self.rt_range_control = RtRangeControl(**RT_RANGE_CONTROL.__dict__)
        self.mz_range_control = MzRangeControl(**MZ_RANGE_CONTROL.__dict__)
        self.rt_range_control.label.hide()
        self.mz_range_control.label.hide()
        self.product_ion_edit = QLineEdit()
        self.collision_energy_edit = QLineEdit()
        self.note_edit = QLineEdit()
        self.field_edits: dict[str, QLineEdit] = {
            "name": self.name_edit,
            "formula": self.formula_edit,
            "adduct": self.adduct_edit,
            "product_ion": self.product_ion_edit,
            "collision_energy": self.collision_energy_edit,
            "note": self.note_edit,
        }

        self.add_button = QPushButton("Add")
        self.add_button.setToolTip("Add the current view as a new target")
        self.duplicate_button = QPushButton("Duplicate")
        self.remove_button = QPushButton("Remove")
        self.reset_index_button = QPushButton("Reset index")
        self.reset_index_button.setToolTip("Renumber the targets 1, 2, ... in the displayed order")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search")
        self.search_edit.setClearButtonEnabled(True)
        self.table_view = QTableView()
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_view.verticalHeader().hide()

        detail_widget = QWidget()
        detail_layout = QFormLayout(detail_widget)
        detail_layout.setContentsMargins(6, 2, 6, 2)
        detail_layout.setVerticalSpacing(0)
        detail_layout.addRow("Name", self.name_edit)
        detail_layout.addRow("Formula", self._with_label(self.formula_edit, self.formula_mass_label))
        detail_layout.addRow("Adduct", self._with_label(self.adduct_edit, self.calculated_mz_label))
        detail_layout.addRow("Trace", self.trace_type_combo)
        detail_layout.addRow("RT", self.rt_range_control)
        detail_layout.addRow("m/z", self.mz_range_control)
        detail_layout.addRow("MS level", self.ms_level_combo)
        detail_layout.addRow("Prod. ion", self.product_ion_edit)
        detail_layout.addRow("CE", self.collision_energy_edit)
        detail_layout.addRow("Note", self.note_edit)

        list_widget = QWidget()
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.duplicate_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.reset_index_button)
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(6, 0, 6, 6)
        list_layout.addLayout(button_layout)
        list_layout.addWidget(self.search_edit)
        list_layout.addWidget(self.table_view)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(detail_widget)
        splitter.addWidget(list_widget)
        splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.trace_type_combo.currentTextChanged.connect(self._update_mz_enabled)
        self.trace_type_combo.currentTextChanged.connect(lambda _text: self._emit(self.trace_type_edited))
        self.formula_edit.textChanged.connect(lambda _text: self._emit(self.chemistry_text_changed))
        self.adduct_edit.textChanged.connect(lambda _text: self._emit(self.chemistry_text_changed))
        self.ms_level_combo.currentTextChanged.connect(lambda _text: self._emit(self.ms_level_edited))
        for spin_box in (self.rt_range_control.center_spin_box, self.rt_range_control.tolerance_spin_box):
            spin_box.valueChanged.connect(lambda _value: self._emit(self.rt_edited))
        for spin_box in (
            self.mz_range_control.center_spin_box,
            self.mz_range_control.tolerance_spin_box,
            self.mz_range_control.ppm_spin_box,
        ):
            spin_box.valueChanged.connect(lambda _value: self._emit(self.mz_edited))
        self.mz_range_control.mode_combo_box.currentTextChanged.connect(lambda _text: self._emit(self.mz_edited))
        for field_name, line_edit in self.field_edits.items():
            line_edit.editingFinished.connect(
                lambda field_name=field_name, line_edit=line_edit: self._emit(self.field_edited, field_name, line_edit.text())
            )
        self.add_button.clicked.connect(self.add_requested.emit)
        self.duplicate_button.clicked.connect(self.duplicate_requested.emit)
        self.remove_button.clicked.connect(self.remove_requested.emit)
        self.reset_index_button.clicked.connect(self.reset_index_requested.emit)
        self.search_edit.textChanged.connect(self._on_search_text_changed)

    @staticmethod
    def _make_mass_label() -> QLabel:
        label = QLabel()
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        label.setMinimumWidth(label.fontMetrics().horizontalAdvance("00000.000000"))
        return label

    @staticmethod
    def _with_label(line_edit: QLineEdit, label: QLabel) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        layout.addWidget(label)
        return widget

    def _update_mz_enabled(self, _text: str = ""):
        # the m/z window is used only for XIC
        self.mz_range_control.setEnabled(self.trace_type() is TraceType.XIC)

    def _emit(self, signal, *args):
        if not self._updating:
            signal.emit(*args)

    @contextmanager
    def _suppress_signals(self):
        """Programmatic updates inside this block do not emit the edit signals."""
        previous = self._updating
        self._updating = True
        try:
            yield
        finally:
            self._updating = previous

    ### list ###
    def set_table_model(self, table_model: TargetTableModel):
        self._table_model = table_model
        self._proxy_model = QSortFilterProxyModel(self)
        self._proxy_model.setSourceModel(table_model)
        self._proxy_model.setSortRole(TargetTableModel.SORT_ROLE)
        self._proxy_model.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._proxy_model.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._proxy_model.setFilterKeyColumn(-1)  # search all displayed columns
        # keep the row in place while its target is being edited; re-sorted on reset
        self._proxy_model.setDynamicSortFilter(False)
        self.table_view.setModel(self._proxy_model)
        # all columns at their content width; the table scrolls horizontally
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_view.horizontalHeader().setSortIndicator(TargetTableModel.INDEX_COLUMN, Qt.SortOrder.AscendingOrder)
        self.table_view.setSortingEnabled(True)
        self.table_view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self._update_list_buttons()

    def selected_target_ids(self) -> list[str]:
        source_rows = sorted(
            self._proxy_model.mapToSource(index).row()
            for index in self.table_view.selectionModel().selectedRows()
        )
        return [self._table_model.target_at(row).id for row in source_rows]

    def displayed_target_ids(self) -> list[str]:
        """All targets in the order of the current sort (the search is cleared)."""
        with self._suppress_signals():
            self.search_edit.clear()
        header = self.table_view.horizontalHeader()
        # rows are not re-sorted while being edited; sort with the current values
        self._proxy_model.sort(header.sortIndicatorSection(), header.sortIndicatorOrder())
        source_rows = [
            self._proxy_model.mapToSource(self._proxy_model.index(row, 0)).row()
            for row in range(self._proxy_model.rowCount())
        ]
        return [self._table_model.target_at(row).id for row in source_rows]

    def select_target(self, target_id: str | None):
        with self._suppress_signals():
            if target_id is None:
                self.table_view.clearSelection()
            else:
                proxy_index = self._proxy_index_of(target_id)
                if not proxy_index.isValid():
                    # hidden by the search: show all rows
                    self.search_edit.clear()
                    proxy_index = self._proxy_index_of(target_id)
                self.table_view.selectRow(proxy_index.row())
        self._update_list_buttons()

    def _proxy_index_of(self, target_id: str):
        return self._proxy_model.mapFromSource(self._table_model.index(self._table_model.row_of(target_id), 0))

    def _on_search_text_changed(self, text: str):
        self._proxy_model.setFilterFixedString(text)
        # rows hidden by the search are deselected
        self._on_selection_changed()

    def _on_selection_changed(self, *_args):
        self._update_list_buttons()
        self._emit(self.selection_changed)

    def _update_list_buttons(self):
        n_selected = len(self.table_view.selectionModel().selectedRows())
        self.duplicate_button.setEnabled(n_selected == 1)
        self.remove_button.setEnabled(n_selected > 0)

    ### detail ###
    def show_target(self, target: Target | None, *, mz_decimals: int):
        """Fill the target fields; None clears and disables them (viewer mode)."""
        with self._suppress_signals():
            for line_edit in self.field_edits.values():
                line_edit.setEnabled(target is not None)
            msms_enabled = target is not None and target.ms_level is MSLevel.MS2
            self.product_ion_edit.setEnabled(msms_enabled)
            self.collision_energy_edit.setEnabled(msms_enabled)
            self.name_edit.setText("" if target is None else target.name)
            self.formula_edit.setText("" if target is None else target.formula)
            self.adduct_edit.setText("" if target is None else target.adduct)
            self.product_ion_edit.setText(self._optional_float_text(None if target is None else target.product_ion))
            self.collision_energy_edit.setText(self._optional_float_text(None if target is None else target.collision_energy))
            self.note_edit.setText("" if target is None else target.note)
            self.set_mz_mismatch(target is not None and target.mz_deviates_from_formula(mz_decimals))
            for field_name in self.field_edits:
                self.set_field_error(field_name, None)
            if target is not None:
                self.set_trace_type(target.trace_type)
                self.set_ms_level(target.ms_level)
                self.set_rt_values(target.rt, target.rt_tolerance)
                self.set_mz_values(target.mz, target.mz_tolerance, target.mz_tolerance_unit)

    def chemistry_texts(self) -> tuple[str, str]:
        return self.formula_edit.text().strip(), self.adduct_edit.text().strip()

    def set_chemistry_preview(self, formula_mass: float | None, ion_mz: float | None, *, decimals: int):
        """Neutral mass of the formula and m/z of the ion; None when they cannot be calculated."""
        self.formula_mass_label.setText("" if formula_mass is None else f"{formula_mass:.{decimals}f}")
        self.calculated_mz_label.setText("" if ion_mz is None else f"{ion_mz:.{decimals}f}")

    @staticmethod
    def _optional_float_text(value: float | None) -> str:
        return "" if value is None else f"{value:g}"

    def set_field_error(self, field_name: str, message: str | None):
        line_edit = self.field_edits[field_name]
        line_edit.setStyleSheet("" if message is None else f"QLineEdit {{ border: 1px solid {FIELD_ERROR_COLOR}; }}")
        line_edit.setToolTip(message or "")

    def set_mz_mismatch(self, mismatch: bool):
        spin_box = self.mz_range_control.center_spin_box
        spin_box.setStyleSheet(f"QDoubleSpinBox {{ background: {MZ_MISMATCH_COLOR}; }}" if mismatch else "")
        spin_box.setToolTip("m/z differs from the value calculated from formula and adduct" if mismatch else "")

    ### view parameters ###
    def trace_type(self) -> TraceType:
        return TraceType(self.trace_type_combo.currentText())

    def set_trace_type(self, trace_type: TraceType):
        with self._suppress_signals():
            self.trace_type_combo.setCurrentText(trace_type.value)

    def ms_level(self) -> MSLevel:
        return MSLevel(self.ms_level_combo.currentText())

    def set_ms_level(self, ms_level: MSLevel):
        with self._suppress_signals():
            self.ms_level_combo.setCurrentText(ms_level.value)

    def rt_values(self) -> tuple[float, float]:
        return self.rt_range_control.value(), self.rt_range_control.tolerance()

    def set_rt_values(self, center: float, tolerance: float):
        with self._suppress_signals():
            self.rt_range_control.set_value(center)
            self.rt_range_control.set_tolerance(tolerance)

    def mz_values(self) -> tuple[float, float, ToleranceUnit]:
        unit = ToleranceUnit(self.mz_range_control.tolerance_mode())
        tolerance = self.mz_range_control.ppm_tolerance() if unit is ToleranceUnit.PPM else self.mz_range_control.tolerance()
        return self.mz_range_control.value(), tolerance, unit

    def set_mz_values(self, center: float, tolerance: float, unit: ToleranceUnit):
        with self._suppress_signals():
            self.mz_range_control.set_value(center)
            self.mz_range_control.set_tolerance_mode(unit.value)
            if unit is ToleranceUnit.PPM:
                self.mz_range_control.set_ppm_tolerance(round(tolerance))
            else:
                self.mz_range_control.set_tolerance(tolerance)
