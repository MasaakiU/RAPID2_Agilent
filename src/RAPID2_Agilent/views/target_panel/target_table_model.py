from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    QRectF,
    QSize,
    QSortFilterProxyModel,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QPalette, QPen
from PyQt6.QtWidgets import (
    QAbstractItemDelegate,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionComboBox,
)

from RAPID2_Agilent.models.target import MSLevel, Target, ToleranceUnit, TraceType
from RAPID2_Agilent.models.target_model import TargetListModel
from RAPID2_Agilent.models.target_search import Matcher
from RAPID2_Agilent.services.operations import Precursor
from RAPID2_Agilent.views.view_constants import MZ_MISMATCH_COLOR, MZ_RANGE_CONTROL, RT_RANGE_CONTROL
from RAPID2_Agilent.views.widgets.one_line_validator import OneLineValidator
from RAPID2_Agilent.views.widgets.range_controls import MyDoubleSpinBox, MySpinBox, typed_value


def _optional_float_text(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def _optional_mz_text(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def precursor_text(precursor_mz: float, collision_energy: float) -> str:
    return f"{precursor_mz:.4f}, CE {collision_energy:g}"


def _target_precursor_text(target: Target) -> str:
    return "" if target.precursor_mz is None else precursor_text(target.precursor_mz, target.collision_energy)


# as on the mode button of the m/z control
UNIT_TEXTS = {ToleranceUnit.MZ: "abs", ToleranceUnit.PPM: "ppm"}


def _mz_tolerance_text(target: Target) -> str:
    if target.mz_tolerance_unit is ToleranceUnit.PPM:
        return f"{target.mz_tolerance:g}"
    return f"{target.mz_tolerance:.4f}"


def _halfway(a: QColor, b: QColor) -> QColor:
    return QColor.fromRgbF(*((x + y) / 2 for x, y in zip(a.getRgbF(), b.getRgbF())))


def _over(bottom: QColor, top: QColor) -> QColor:
    """``top`` laid over the opaque ``bottom``."""
    alpha = top.alphaF()
    return QColor.fromRgbF(*(b * (1 - alpha) + t * alpha for b, t in zip(bottom.getRgbF()[:3], top.getRgbF()[:3])))


@dataclass(frozen=True)
class _Column:
    header: str
    text: Callable[[Target], str]
    sort_value: Callable[[Target], object]  # raw values, so that numbers sort numerically
    field: str | None = None  # Target field shown in this column; None for the index


def _text_column(header: str, text: Callable[[Target], str], field: str) -> _Column:
    return _Column(header, text, text, field)


def _spin_box_range(spec, *, tolerance: bool) -> tuple[int, float, float, float]:
    if tolerance:
        return spec.decimals, spec.single_step, spec.tolerance_minimum, spec.tolerance_maximum
    return spec.decimals, spec.single_step, spec.center_minimum, spec.center_maximum


# edited with the same ranges as the controls above the table: decimals, single step, minimum, maximum;
# a ppm m/z tolerance takes an integer spin box instead
DOUBLE_SPIN_BOX_RANGES = {
    "rt": _spin_box_range(RT_RANGE_CONTROL, tolerance=False),
    "rt_tolerance": _spin_box_range(RT_RANGE_CONTROL, tolerance=True),
    "mz": _spin_box_range(MZ_RANGE_CONTROL, tolerance=False),
    "mz_tolerance": _spin_box_range(MZ_RANGE_CONTROL, tolerance=True),
}
COMBO_BOX_ITEMS = {
    "trace_type": [trace_type.value for trace_type in TraceType],
    "mz_tolerance_unit": list(UNIT_TEXTS.values()),
    "ms_level": [ms_level.value for ms_level in MSLevel],
}
MSMS_FIELDS = ("precursor_mz", "collision_energy")
XIC_FIELDS = ("mz", "mz_tolerance", "mz_tolerance_unit")


class TargetTableModel(QAbstractTableModel):
    """Qt adapter showing the targets; edits are passed on as edit_requested."""

    edit_requested = pyqtSignal(str, str, object)  # target id, field name, edited value

    COLUMNS = (
        _Column("#", lambda t: str(t.index), lambda t: t.index),  # hidden: sorts; the index is in the row header
        _text_column("Name", lambda t: t.name, "name"),
        _text_column("Formula", lambda t: t.formula, "formula"),
        _text_column("Adduct", lambda t: t.adduct, "adduct"),
        _text_column("Trace", lambda t: t.trace_type.value, "trace_type"),
        _Column("m/z", lambda t: f"{t.mz:.4f}", lambda t: t.mz, "mz"),
        _Column("m/z ±", _mz_tolerance_text, lambda t: t.mz_tolerance, "mz_tolerance"),
        _text_column("Unit", lambda t: UNIT_TEXTS[t.mz_tolerance_unit], "mz_tolerance_unit"),
        _Column("RT", lambda t: f"{t.rt:.2f}", lambda t: t.rt, "rt"),
        _Column("RT ±", lambda t: f"{t.rt_tolerance:.2f}", lambda t: t.rt_tolerance, "rt_tolerance"),
        _text_column("MS level", lambda t: t.ms_level.value, "ms_level"),
        # the precursor and CE, chosen from those of the files' MS2 scans as in the controls above; sorted by the m/z,
        # None before any
        _Column(
            "Precursor",
            _target_precursor_text,
            lambda t: -float("inf") if t.precursor_mz is None else t.precursor_mz,
            "precursor_mz",
        ),
        _text_column("Note", lambda t: t.note, "note"),
    )
    INDEX_COLUMN = 0
    MZ_COLUMN = next(column for column, spec in enumerate(COLUMNS) if spec.field == "mz")
    # the columns shown when not all of them are; the index column always is
    DEFAULT_SHOWN_FIELDS = (
        "name", "formula", "adduct", "trace_type", "mz", "rt", "ms_level", "precursor_mz", "note",
    )
    SORT_ROLE = Qt.ItemDataRole.UserRole
    UNUSED_ROLE = Qt.ItemDataRole.UserRole + 1  # whether the target uses the field, see is_unused()
    PRECURSORS_ROLE = Qt.ItemDataRole.UserRole + 2  # of the precursor column: the texts of those to choose from

    def __init__(self, target_list_model: TargetListModel, *, mz_decimals: int, parent=None):
        super().__init__(parent)
        self.target_list_model = target_list_model
        self.mz_decimals = mz_decimals
        # the precursors to choose from, and the index of a target's among them (see set_precursors)
        self._precursor_texts: list[str] = []
        self._listed_index: Callable[[float, float], int | None] = lambda _mz, _ce: None
        self._previews: dict[str, Target] = {}  # shown in place of the targets with their ids
        # the ids of the rows the views show: the list model's own are new already when it tells of a reset, and
        # these are still those of the rows until the reset is through (see shown_id)
        self._shown_ids = tuple(target.id for target in target_list_model.targets)
        self.target_list_model.targets_reset.connect(self._on_targets_reset)
        self.target_list_model.target_changed.connect(self._on_target_changed)

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.target_list_model.targets)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.COLUMNS[section].header
        return str(self.target_at(section).index)

    def flags(self, index: QModelIndex):
        flags = super().flags(index)
        column = self.COLUMNS[index.column()]
        if column.field is None or self.is_unused(self.target_at(index.row()), column.field):
            return flags
        return flags | Qt.ItemFlag.ItemIsEditable

    @staticmethod
    def is_unused(target: Target, field: str) -> bool:
        """A field the target does not use: MS/MS settings without MS2, the m/z window without XIC, as in the controls above."""
        return (field in MSMS_FIELDS and target.ms_level is not MSLevel.MS2) or (
            field in XIC_FIELDS and target.trace_type is not TraceType.XIC
        )

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        target = self.target_at(index.row())
        column = self.COLUMNS[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return column.text(target)
        if role == self.PRECURSORS_ROLE and column.field == "precursor_mz":
            return self._precursor_texts
        if role == Qt.ItemDataRole.EditRole and column.field is not None:
            # numbers go to spin boxes; the precursor is the index of the listed one, -1 if not listed; the others
            # are edited as the displayed text
            if column.field == "precursor_mz":
                index = None if target.precursor_mz is None else self._listed_index(target.precursor_mz, target.collision_energy)
                return -1 if index is None else index
            if column.field not in DOUBLE_SPIN_BOX_RANGES:
                return column.text(target)
            value = getattr(target, column.field)
            if column.field == "mz_tolerance" and target.mz_tolerance_unit is ToleranceUnit.PPM:
                return round(value)
            return value
        if role == self.SORT_ROLE:
            return column.sort_value(target)
        if role == self.UNUSED_ROLE:
            return column.field is not None and self.is_unused(target, column.field)
        # what differs from formula + adduct: the m/z, or the precursor of an MS2 target
        if role == Qt.ItemDataRole.BackgroundRole and (
            (column.field == "mz" and target.mz_deviates_from_formula(self.mz_decimals))
            or (column.field == "precursor_mz" and target.precursor_deviates_from_formula())
        ):
            return QColor(MZ_MISMATCH_COLOR)
        return None

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole) -> bool:
        if role != Qt.ItemDataRole.EditRole:
            return False
        field = self.COLUMNS[index.column()].field
        if field == "mz_tolerance_unit":
            value = next(unit for unit, text in UNIT_TEXTS.items() if text == value)
        # the presenter checks the value and changes the target; a rejected edit leaves the cell as it was
        self.edit_requested.emit(self.target_at(index.row()).id, field, value)
        return True

    def set_precursors(self, precursors: list[Precursor], listed_index: Callable[[float, float], int | None]):
        """The precursors a target's is chosen from; listed_index(precursor_mz, collision_energy) is the index of
        one among them, None if it is not listed."""
        self._precursor_texts = [precursor_text(p.precursor_mz, p.collision_energy) for p in precursors]
        self._listed_index = listed_index

    def target_at(self, row: int) -> Target:
        target = self.target_list_model.targets[row]
        return self._previews.get(target.id, target)

    def set_preview(self, target: Target):
        """Show ``target`` in place of the one with its id until clear_preview(), e.g. while its window is dragged."""
        self.set_previews([target])

    def set_previews(self, targets: list[Target]):
        """As set_preview(), for several targets: those selected, edited all at once."""
        changed_ids = set(self._previews) | {target.id for target in targets}
        self._previews = {target.id: target for target in targets}
        for target_id in changed_ids:
            self._on_target_changed(target_id)

    def clear_preview(self):
        self.set_previews([])

    def row_of(self, target_id: str) -> int:
        return self.target_list_model.index_of(target_id)

    @classmethod
    def column_of(cls, field: str) -> int:
        return next(column for column, spec in enumerate(cls.COLUMNS) if spec.field == field)

    def shown_id(self, row: int) -> str:
        """The id of the target in ``row`` as the views show it, also while they are told of a reset."""
        return self._shown_ids[row]

    def _on_targets_reset(self):
        self._previews = {}
        self.beginResetModel()
        self._shown_ids = tuple(target.id for target in self.target_list_model.targets)
        self.endResetModel()

    def _on_target_changed(self, target_id: str):
        row = self.row_of(target_id)
        self.dataChanged.emit(self.index(row, 0), self.index(row, len(self.COLUMNS) - 1))


class _SearchRow:
    """A target as the search sees it (see target_search.SearchRow): its fields as the list shows them, and its
    numbers as kept."""

    # the precursor and the CE, which share a column, each as it would be shown alone
    _TEXTS = {column.field or "index": column.text for column in TargetTableModel.COLUMNS} | {
        "precursor_mz": lambda t: _optional_mz_text(t.precursor_mz),
        "collision_energy": lambda t: _optional_float_text(t.collision_energy),
    }

    def __init__(self, target: Target):
        self._target = target

    def texts(self):
        return (column.text(self._target) for column in TargetTableModel.COLUMNS)

    def text(self, attribute: str) -> str:
        return self._TEXTS[attribute](self._target)

    def number(self, attribute: str) -> float | None:
        return getattr(self._target, attribute)


class TargetFilterProxyModel(QSortFilterProxyModel):
    """The rows the search lets through (see target_search), over a TargetTableModel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._matches: Matcher | None = None

    def set_search(self, matches: Matcher):
        self.beginFilterChange()
        self._matches = matches
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        if self._matches is None:
            return True
        return self._matches(_SearchRow(self.sourceModel().target_at(source_row)))


class TargetTableDelegate(QStyledItemDelegate):
    """Edits RT and m/z with spin boxes like the controls above, and trace, unit, MS level and precursor with combo
    boxes.

    A cell's own background (the m/z mismatch) also shows on a selected row, and
    the current cell of a focused table is framed. A column edited with a combo
    box is as wide as the box, so that it shows whole.

    What is typed into an editor is passed on as it is typed (value_typed), to be
    shown before it is committed; a choice in a combo box is committed at once,
    as in the controls above.
    """

    value_typed = pyqtSignal(QModelIndex, object)  # the cell, the text or the number so far

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if option.state & QStyle.StateFlag.State_HasFocus:
            # in the text colour, so that it reads on the selection in Light and Dark alike; on an unused field,
            # halfway to the disabled text: grey, and still easy to find
            color = option.palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Text)
            if self._is_unused(index):
                color = _halfway(color, option.palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text))
            painter.save()
            painter.setPen(QPen(color, 2))
            painter.drawRect(QRectF(option.rect).adjusted(1, 1, -1, -1))
            painter.restore()

    @staticmethod
    def _is_unused(index) -> bool:
        """A field the target does not use, and so cannot be edited: m/z without XIC, MS/MS settings without MS2."""
        return bool(index.data(TargetTableModel.UNUSED_ROLE))

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        # the frame drawn in paint() takes the place of the style's faint focus rectangle
        option.state &= ~QStyle.StateFlag.State_HasFocus
        if self._is_unused(index):
            # greyed as a disabled control is: the theme's disabled text, translucent, reads on any row
            palette = QPalette(option.palette)
            disabled_text = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
            for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
                palette.setColor(group, QPalette.ColorRole.Text, disabled_text)
                palette.setColor(group, QPalette.ColorRole.HighlightedText, disabled_text)
            option.palette = palette
        background = index.data(Qt.ItemDataRole.BackgroundRole)
        if background is None or not option.state & QStyle.StateFlag.State_Selected:
            return
        # the style fills a selected cell with the highlight alone: lay the cell's colour over it
        palette = QPalette(option.palette)
        for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
            palette.setColor(group, QPalette.ColorRole.Highlight, _over(palette.color(group, QPalette.ColorRole.Highlight), background))
        option.palette = palette

    @staticmethod
    def _combo_box_texts(index) -> list[str] | None:
        """The items of the combo box ``index`` is edited with; None if it is edited otherwise."""
        field = TargetTableModel.COLUMNS[index.column()].field
        if field == "precursor_mz":
            return index.data(TargetTableModel.PRECURSORS_ROLE)
        return COMBO_BOX_ITEMS.get(field)

    @staticmethod
    def _combo_box_width(texts: list[str], option) -> int:
        """The width QComboBox.sizeHint() gives a box of these items, in the cell's font."""
        metrics = option.fontMetrics
        contents = QSize(max(metrics.boundingRect(text).width() for text in texts), max(metrics.height(), 14) + 2)
        widget = option.widget
        style = widget.style() if widget is not None else QApplication.style()
        return style.sizeFromContents(QStyle.ContentsType.CT_ComboBox, QStyleOptionComboBox(), contents, widget).width()

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        texts = self._combo_box_texts(index)
        if texts and index.flags() & Qt.ItemFlag.ItemIsEditable:
            size.setWidth(max(size.width(), self._combo_box_width(texts, option)))
        return size

    def createEditor(self, parent, option, index):
        editor = self._make_editor(parent, option, index)
        cell = QPersistentModelIndex(index)
        if isinstance(editor, QComboBox):
            editor.activated.connect(lambda _row, editor=editor: self._commit_and_close(editor))
        elif isinstance(editor, QAbstractSpinBox):
            editor.lineEdit().textEdited.connect(lambda _text, editor=editor: self._pass_on_typed(cell, typed_value(editor)))
        elif isinstance(editor, QLineEdit):
            editor.textEdited.connect(lambda text: self._pass_on_typed(cell, text))
        return editor

    def _pass_on_typed(self, cell: QPersistentModelIndex, value):
        if value is not None and cell.isValid():
            self.value_typed.emit(QModelIndex(cell), value)

    def _commit_and_close(self, editor):
        self.commitData.emit(editor)
        self.closeEditor.emit(editor, QAbstractItemDelegate.EndEditHint.NoHint)

    def _make_editor(self, parent, option, index):
        field = TargetTableModel.COLUMNS[index.column()].field
        unit_index = index.siblingAtColumn(TargetTableModel.column_of("mz_tolerance_unit"))
        if field == "mz_tolerance" and unit_index.data() == UNIT_TEXTS[ToleranceUnit.PPM]:
            spin_box = MySpinBox(parent)
            spin_box.setSingleStep(MZ_RANGE_CONTROL.ppm_single_step)
            spin_box.setRange(MZ_RANGE_CONTROL.ppm_minimum, MZ_RANGE_CONTROL.ppm_maximum)
            return spin_box
        if field in DOUBLE_SPIN_BOX_RANGES:
            decimals, single_step, minimum, maximum = DOUBLE_SPIN_BOX_RANGES[field]
            spin_box = MyDoubleSpinBox(parent)
            spin_box.setDecimals(decimals)
            spin_box.setSingleStep(single_step)
            spin_box.setRange(minimum, maximum)
            return spin_box
        texts = self._combo_box_texts(index)
        if field == "precursor_mz" and not texts:
            return None  # no MS2 scans in the files to choose from
        if texts:
            combo_box = QComboBox(parent)
            combo_box.addItems(texts)
            return combo_box
        editor = super().createEditor(parent, option, index)
        if field in ("name", "note") and isinstance(editor, QLineEdit):
            editor.setValidator(OneLineValidator(editor))
        return editor

    def setEditorData(self, editor, index):
        if TargetTableModel.COLUMNS[index.column()].field == "precursor_mz":
            # the listed one the target's is; none if it is not listed
            editor.setCurrentIndex(index.data(Qt.ItemDataRole.EditRole))
            return
        super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if TargetTableModel.COLUMNS[index.column()].field == "precursor_mz":
            # another one only: the same would set its listed m/z over the target's own
            chosen = editor.currentIndex()
            if chosen >= 0 and chosen != index.data(Qt.ItemDataRole.EditRole):
                model.setData(index, chosen)
            return
        super().setModelData(editor, model, index)
