from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtGui import QColor

from RAPID2_Agilent.models.target_model import Target, TargetListModel, ToleranceUnit
from RAPID2_Agilent.views.view_constants import MZ_MISMATCH_COLOR


def _optional_float_text(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def _mz_tolerance_text(target: Target) -> str:
    if target.mz_tolerance_unit is ToleranceUnit.PPM:
        return f"{target.mz_tolerance:g} ppm"
    return f"{target.mz_tolerance:.4f}"


@dataclass(frozen=True)
class _Column:
    header: str
    text: Callable[[Target], str]
    sort_value: Callable[[Target], object]  # raw values, so that numbers sort numerically


def _text_column(header: str, text: Callable[[Target], str]) -> _Column:
    return _Column(header, text, text)


def _optional_float_column(header: str, value: Callable[[Target], float | None]) -> _Column:
    # None sorts before any number
    return _Column(header, lambda t: _optional_float_text(value(t)), lambda t: -float("inf") if value(t) is None else value(t))


class TargetTableModel(QAbstractTableModel):
    """Read-only Qt adapter showing all attributes of each target."""

    COLUMNS = (
        _Column("Index", lambda t: str(t.index), lambda t: t.index),
        _text_column("Name", lambda t: t.name),
        _text_column("Formula", lambda t: t.formula),
        _text_column("Adduct", lambda t: t.adduct),
        _Column("m/z", lambda t: f"{t.mz:.4f}", lambda t: t.mz),
        _Column("m/z tol.", _mz_tolerance_text, lambda t: t.mz_tolerance),
        _Column("RT", lambda t: f"{t.rt:.2f}", lambda t: t.rt),
        _Column("RT tol.", lambda t: f"{t.rt_tolerance:.2f}", lambda t: t.rt_tolerance),
        _text_column("Trace", lambda t: t.trace_type.value),
        _text_column("MS", lambda t: t.ms_level.value),
        _optional_float_column("Prod. ion", lambda t: t.product_ion),
        _optional_float_column("CE", lambda t: t.collision_energy),
        _text_column("Note", lambda t: t.note),
    )
    INDEX_COLUMN = 0
    MZ_COLUMN = 4
    SORT_ROLE = Qt.ItemDataRole.UserRole

    def __init__(self, target_list_model: TargetListModel, *, mz_decimals: int, parent=None):
        super().__init__(parent)
        self.target_list_model = target_list_model
        self.mz_decimals = mz_decimals
        self.target_list_model.targets_reset.connect(self._on_targets_reset)
        self.target_list_model.target_changed.connect(self._on_target_changed)

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.target_list_model.targets)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.COLUMNS[section].header
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        target = self.target_at(index.row())
        column = self.COLUMNS[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return column.text(target)
        if role == self.SORT_ROLE:
            return column.sort_value(target)
        if (
            role == Qt.ItemDataRole.BackgroundRole
            and index.column() == self.MZ_COLUMN
            and target.mz_deviates_from_formula(self.mz_decimals)
        ):
            return QColor(MZ_MISMATCH_COLOR)
        return None

    def target_at(self, row: int) -> Target:
        return self.target_list_model.targets[row]

    def row_of(self, target_id: str) -> int:
        return self.target_list_model.index_of(target_id)

    def _on_targets_reset(self):
        self.beginResetModel()
        self.endResetModel()

    def _on_target_changed(self, target_id: str):
        row = self.row_of(target_id)
        self.dataChanged.emit(self.index(row, 0), self.index(row, len(self.COLUMNS) - 1))
