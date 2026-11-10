from contextlib import contextmanager

from PyQt6.QtCore import QEvent, QItemSelection, QItemSelectionModel, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QActionGroup, QColor, QPalette
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSpinBox,
    QStyle,
    QStyleOptionHeader,
    QStylePainter,
    QTableView,
    QToolTip,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from RAPID2_Agilent.models.target import MSLevel, Target, ToleranceUnit, TraceType
from RAPID2_Agilent.models.target_search import SearchError, parse_search
from RAPID2_Agilent.services.operations import Precursor
from RAPID2_Agilent.views.view_constants import (
    MZ_MISMATCH_COLOR,
    MZ_RANGE_CONTROL,
    RT_RANGE_CONTROL,
)
from RAPID2_Agilent.views.widgets import (
    InfoIconButton,
    MzRangeControl,
    OneLineValidator,
    RowChangeOverlay,
    RtRangeControl,
    make_icon_button,
    typed_value,
)

from .target_table_model import TargetFilterProxyModel, TargetTableDelegate, TargetTableModel, _over, precursor_text
from RAPID2_Agilent.views.layout_metrics import TINY, SMALL, MEDIUM


# m/z with a word joiner, which a label breaks a line at the slash of otherwise (<nobr> and nowrap do not keep it)
_MZ = "m/\u2060z"
_SEARCH_INFO_HTML = (
    "<b>Search</b><br>"
    "Text typed as it is is looked for in all the fields, also in the columns not shown; upper and lower case "
    "are the same. With a condition, the text is a query:"
    "<ul>"
    '<li><code>name:"glu"</code> has, <code>name="glucose"</code> is, <code>name~"^glu"</code> matches the '
    'regular expression; <code>"glu"</code> and <code>~"^glu"</code>: any field.</li>'
    f"<li><code>{_MZ}:150.5</code> as shown starts so (150.5 &le; {_MZ} &lt; 150.6), <code>{_MZ}=150.5</code> "
    f"is so; <code>200 &lt; {_MZ} &lt; 300</code>, <code>rt &gt;= 1.5</code>.</li>"
    "<li><code>AND</code>, <code>OR</code> and <code>NOT</code>, in capitals, and brackets; conditions next to "
    "each other are AND.</li>"
    "<li>In a query, text is quoted, in &quot;...&quot; or '...' (<code>\\&quot;</code>, <code>\\'</code> and "
    "<code>\\\\</code> for the characters themselves); numbers are not.</li>"
    "</ul>"
    '<code>(name~"^glu" OR formula:"C6") AND 200 &lt; mz &lt; 300</code><br><br>'
    f"Fields: name, formula, adduct, trace, mz ({_MZ}), mztol, unit, rt, rttol, ms, precursor, ce, note, index (#). "
    "A query that cannot be read is marked red, with why in its tooltip; the list stays as it was meanwhile."
    "<ul>"
    "<li>A selected target the search hides is no longer selected.</li>"
    "<li>Selecting a target the search hides &ndash; one just added, say &ndash; clears the search, to show it.</li>"
    "<li>A target edited so that it no longer matches stays in the list until the search or the list changes.</li>"
    "</ul>"
)
# the solvents as in chemistry.SOLVENT_FORMULAS
_ADDUCT_INFO_HTML = (
    "<b>Adduct</b><br>"
    "How the ion is made from M, the molecule of the formula: <code>[nM&plusmn;group&plusmn;group...]z&plusmn;</code>"
    "<ul>"
    "<li><code>nM</code>: n molecules, as in <code>[2M+Na]+</code>; M alone for one.</li>"
    "<li>A group is added (+) or taken off (&minus;), with a count before it for more than one: <code>+H</code>, "
    "<code>+2H</code>, <code>-H2O</code>, <code>+NH4</code>, <code>+HCOO</code>. It is a formula, or one of the "
    "solvents ACN (C2H3N), IsoProp (C3H8O), DMSO (C2H6OS), FA (CH2O2), Hac (C2H4O2) and TFA (C2HF3O2).</li>"
    "<li>The charge, its number before the sign, left out for one: <code>+</code>, <code>-</code>, "
    "<code>2+</code>.</li>"
    "</ul>"
    "<code>[M+H]+</code>, <code>[M+Na]+</code>, <code>[M+NH4]+</code>, <code>[M+H-H2O]+</code>, "
    "<code>[M+2H]2+</code>, <code>[M-H]-</code>, <code>[M+Cl]-</code>, <code>[M+FA-H]-</code>; <code>[M]+</code> "
    "for a molecule charged itself, as choline.<br><br>"
    f"{_MZ} = (n &times; M + groups &minus; z &times; electron) / |z|, of monoisotopic masses. Spaces, a count or "
    "charge of 0, <code>[M+2H]+2</code> and <code>M+H</code> are not read."
)


class _AllColumnsCheckBox(QCheckBox):
    """Shows all / none / some of the columns; the panel decides what a click makes."""

    def __init__(self, text: str):
        super().__init__(text)
        self.setTristate(True)

    def nextCheckState(self):
        pass  # the state follows the column check boxes the click changes


class TargetPanel(QWidget):
    """The current target (top) and the target list (bottom).

    The top shows all of one target: the selected one, which it edits, or with
    none selected a draft, which only moves the view until it is added. Leaving a
    target starts a new draft with the same view and adduct. With several
    selected it shows what they have in common, blank where they differ, and a
    field changed changes that field of them all. Framed, the top takes the
    list's selection colour while it edits targets. The list edits any
    target too. It can be searched and sorted, and shows the chosen
    columns; all of these only affect the display.

    The index of each target is in the row header, which stays in view when
    the table scrolls sideways; the corner above it sorts by index.

    A click on a row, its header or a cell, selects it; another click on the
    selected row's header or its current cell deselects it. A double click
    selects the row again if its first click deselected it, and on a cell edits it.
    """

    selection_changed = pyqtSignal()
    rt_edited = pyqtSignal(str)  # the field: "rt" or "rt_tolerance"
    mz_edited = pyqtSignal(str)  # the field: "mz", "mz_tolerance" or "mz_tolerance_unit"
    trace_type_edited = pyqtSignal()
    ms_level_edited = pyqtSignal()
    field_edited = pyqtSignal(str, str)  # field name, raw text
    # what is typed in a field, or in a cell of the list, as it is typed: committed only once it is left, or dropped
    field_typed = pyqtSignal(str, object)  # field name, the text or the number so far
    typing_finished = pyqtSignal()  # the field typed in is left
    cell_typed = pyqtSignal(str, str, object)  # the cell's target id and field name, the text or the number so far
    cell_editing_finished = pyqtSignal()  # the cell's editor is closed
    precursor_chosen = pyqtSignal(int)  # its index in the list set by set_precursors()
    merge_ppm_edited = pyqtSignal(int)
    chemistry_text_changed = pyqtSignal()  # formula or adduct is being typed
    add_requested = pyqtSignal()
    remove_requested = pyqtSignal()
    ADD_TOOLTIP = "Add the current view as a new target"
    DUPLICATE_TOOLTIP = "Duplicate the selected target"
    DUPLICATE_ALL_TOOLTIP = "Duplicate the selected targets"
    reset_index_requested = pyqtSignal()
    columns_changed = pyqtSignal()  # the columns shown
    import_requested = pyqtSignal()
    export_requested = pyqtSignal()
    apply_mz_requested = pyqtSignal()  # the m/z of formula + adduct to the selected targets
    recalculate_all_requested = pyqtSignal()  # the same to every target of the list
    auto_requested = pyqtSignal()  # find the peaks of the selected targets, of all if none is

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TargetPanel")
        self._updating = False
        self._table_model: TargetTableModel | None = None
        self._proxy_model: TargetFilterProxyModel | None = None
        self._rows_before_reset = None  # (rows, pictures) of the list as it was, while a removal waits to be shown
        self._showing_target = False
        self._mz_mismatch = False
        self._precursor_mismatch = False
        self._chosen_precursor_text = "None"  # shown on the precursor button for MS2
        self._marked_edits: set[QLineEdit] = set()  # those _mark_error marks
        # of several targets shown (show_targets), the fields whose values differ, left blank
        self.mixed_fields: set[str] = set()

        # what the controls below edit: the selected target, or a draft
        self.state_label = QLabel()
        font = self.state_label.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self.state_label.setFont(font)
        self.list_label = QLabel("Targets")
        self.auto_button = QPushButton("Auto")
        self.list_label.setFont(font)
        self.name_edit = QLineEdit()
        self.name_edit.setValidator(OneLineValidator(self.name_edit))
        self.name_edit.setPlaceholderText("New target")
        self.formula_edit = QLineEdit()
        self.adduct_edit = QLineEdit()
        self.adduct_edit.setPlaceholderText("[M+H]+")
        self.formula_mass_label = self._make_mass_label()
        self.calculated_mz_label = self._make_mass_label()
        self.apply_mz_button = make_icon_button("arrow-turn-left-solid", "Set the m/z to the one of formula + adduct")
        self.apply_mz_button.setEnabled(False)
        self.trace_type_combo = QComboBox()
        self.trace_type_combo.addItems([trace_type.value for trace_type in TraceType])
        self.trace_type_combo.setToolTip("Trace type")
        self.ms_level_combo = QComboBox()
        self.ms_level_combo.addItems([ms_level.value for ms_level in MSLevel])
        self.ms_level_combo.setToolTip("MS level of the scans the trace is made from")
        self.rt_range_control = RtRangeControl(**RT_RANGE_CONTROL.__dict__)
        self.mz_range_control = MzRangeControl(**MZ_RANGE_CONTROL.__dict__)
        self.rt_range_control.label.hide()
        self.mz_range_control.label.hide()
        # the MS2 scans used, by their precursor and collision energy, chosen from those of the files; for MS2 the
        # m/z control above holds the product ion
        self.precursor_button = QPushButton()
        self.precursor_button.setToolTip("Precursor and collision energy of the MS2 scans used")
        self.precursor_menu = QMenu(self)
        # precursors this close are listed as one, for data-dependent scans that record slightly different values
        self.merge_ppm_spin_box = QSpinBox()
        self.merge_ppm_spin_box.setRange(0, 1000)
        self.merge_ppm_spin_box.setSuffix(" ppm")
        self.merge_ppm_spin_box.setToolTip("List the precursors within this many ppm of each other as one; 0 lists each value")
        merge_widget = QWidget()
        merge_layout = QHBoxLayout(merge_widget)
        merge_layout.setContentsMargins(SMALL, TINY, SMALL, TINY)
        merge_layout.addWidget(QLabel("Merge within"))
        merge_layout.addWidget(self.merge_ppm_spin_box)
        merge_action = QWidgetAction(self.precursor_menu)
        merge_action.setDefaultWidget(merge_widget)
        self.precursor_menu.addAction(merge_action)
        self.precursor_menu.addSeparator()
        self._precursor_actions = QActionGroup(self)
        self.precursor_button.setMenu(self.precursor_menu)
        self.note_edit = QLineEdit()
        self.note_edit.setValidator(OneLineValidator(self.note_edit))
        self.field_edits: dict[str, QLineEdit] = {
            "name": self.name_edit,
            "formula": self.formula_edit,
            "adduct": self.adduct_edit,
            "note": self.note_edit,
        }
        # blank where several targets differ, rather than seeming to have the default
        self._placeholders = {field_name: line_edit.placeholderText() for field_name, line_edit in self.field_edits.items()}

        # with one target selected, it duplicates that target (see _update_list_buttons)
        self.add_button = make_icon_button("plus-solid", self.ADD_TOOLTIP)
        self.remove_button = make_icon_button("minus-solid", "Remove the selected targets")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search")
        self.search_edit.setClearButtonEnabled(True)
        self.search_info_button = InfoIconButton(_SEARCH_INFO_HTML, popup_width=420, accessible_name="About the search")
        self.adduct_info_button = InfoIconButton(_ADDUCT_INFO_HTML, popup_width=420, accessible_name="About the adduct")
        self.columns_button = make_icon_button("table-list-solid", "Columns, and renumbering by the displayed order")
        # the columns shown and the renumbering by the displayed order; also opened from the column headers
        self.columns_menu = QMenu(self)
        # all / none / some of the columns; clicks go some -> all -> none -> some, the last "some" chosen
        self.all_check_box = _AllColumnsCheckBox("All")
        self._partial_fields = list(TargetTableModel.DEFAULT_SHOWN_FIELDS)
        all_action = QWidgetAction(self.columns_menu)
        all_action.setDefaultWidget(self.all_check_box)
        self.columns_menu.addAction(all_action)
        self.columns_menu.addSeparator()
        self.column_check_boxes: dict[str, QCheckBox] = {}
        for column in TargetTableModel.COLUMNS:
            if column.field is None:
                continue  # the index is shown in the row header
            check_box = QCheckBox(column.header)
            action = QWidgetAction(self.columns_menu)
            action.setDefaultWidget(check_box)
            self.columns_menu.addAction(action)
            self.column_check_boxes[column.field] = check_box
        self.columns_menu.addSeparator()
        self.default_columns_action = self.columns_menu.addAction("Restore default columns")
        self.columns_menu.addSeparator()
        self.reset_index_action = self.columns_menu.addAction("Renumber targets in displayed order")
        self.columns_button.setMenu(self.columns_menu)
        self.more_button = make_icon_button("ellipsis-solid", "Import and export of the targets")
        self.more_menu = QMenu(self)
        self.recalculate_all_action = self.more_menu.addAction("Recalculate all m/z from formula + adduct")
        self.more_menu.addSeparator()
        self.import_action = self.more_menu.addAction("Import targets…")
        self.export_action = self.more_menu.addAction("Export targets…")
        self.more_button.setMenu(self.more_menu)
        self.more_button.set_menu_arrow_shown(False)
        self.table_view = QTableView()
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table_view.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        # No hover highlight on the rows, for now -- deferred, not ruled out, and open to change. Unlike the file info
        # panel (FileInfoPanel), which looks like a plain label and needs its hover to show that it can be clicked, a
        # table shows that its rows can be selected already; and the rows carry several colours already (the
        # selection, the greyer selection of an unfocused table, the m/z mismatch, the greyed unused fields, the
        # focus frame), which a hover has to stay clear of. Where it would help is following a row across the wide
        # table. If added: the whole row (the selection is by rows, while the style's State_MouseOver is per cell, so
        # the view has to track the row under the mouse), in the info panel's faint text colour (alpha 25), drawn in
        # TargetTableDelegate.paint() and over selected rows as well.
        delegate = TargetTableDelegate(self.table_view)
        self.table_view.setItemDelegate(delegate)
        delegate.value_typed.connect(self._on_cell_typed)
        delegate.closeEditor.connect(lambda *_args: self.cell_editing_finished.emit())
        self.table_view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table_view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        row_header = self.table_view.verticalHeader()
        row_header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        # lower rows than the style's: the text with a small margin, still room for the combo box editors
        row_header.setDefaultSectionSize(self.table_view.fontMetrics().height() + 2 * SMALL)
        style = self.table_view.style()
        # room for the sort indicator of the corner above it
        row_header.setMinimumWidth(
            style.pixelMetric(QStyle.PixelMetric.PM_HeaderMarkSize) + 2 * style.pixelMetric(QStyle.PixelMetric.PM_HeaderMargin)
        )
        # the corner button selects all by default; here it sorts by index, as a column header sorts by its column
        self._corner_button = self.table_view.findChild(QAbstractButton)
        self._corner_button.clicked.disconnect()
        self.table_view.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        for watched in (self.table_view.viewport(), row_header.viewport(), self._corner_button):
            watched.installEventFilter(self)
        self._deselect_on_release = False

        # framed in both states; filled with a faint grey without a target, with the list's selection colour while it
        # edits one
        self.detail_frame = QFrame()
        self.detail_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.detail_frame.setAutoFillBackground(True)
        detail_layout = QFormLayout(self.detail_frame)
        detail_layout.setContentsMargins(SMALL, TINY, SMALL, TINY)
        detail_layout.setVerticalSpacing(TINY)
        detail_widget = QWidget()
        detail_widget_layout = QVBoxLayout(detail_widget)
        detail_widget_layout.setContentsMargins(SMALL, SMALL, SMALL, SMALL)
        detail_widget_layout.addWidget(self.state_label)
        detail_widget_layout.addWidget(self.detail_frame)
        detail_layout.addRow("Name", self.name_edit)
        detail_layout.addRow("Formula", self._with_label(self.formula_edit, self.formula_mass_label))
        # the info button by the label, so that the field ends where the formula's does
        adduct_label = QWidget()
        adduct_label_layout = QHBoxLayout(adduct_label)
        adduct_label_layout.setContentsMargins(0, 0, 0, 0)
        adduct_label_layout.setSpacing(TINY)
        adduct_label_layout.addWidget(QLabel("Adduct"))
        adduct_label_layout.addWidget(self.adduct_info_button)
        adduct_label_layout.addStretch(1)
        adduct_widget = self._with_label(self.adduct_edit, self.calculated_mz_label)
        adduct_widget.layout().insertWidget(1, self.apply_mz_button)
        detail_layout.addRow(adduct_label, adduct_widget)
        # the trace type as the row's label, as the m/z mode button is; the MS level of its scans and the precursor
        # of the MS2 ones, side by side
        trace_widget = QWidget()
        trace_layout = QHBoxLayout(trace_widget)
        trace_layout.setContentsMargins(0, 0, 0, 0)
        trace_layout.setSpacing(TINY)
        trace_layout.addWidget(self.ms_level_combo)
        trace_layout.addWidget(self.precursor_button)
        trace_layout.addStretch(1)
        detail_layout.addRow(self.trace_type_combo, trace_widget)
        detail_layout.addRow("RT [min]", self.rt_range_control)
        detail_layout.addRow(self.mz_range_control.mode_button, self.mz_range_control)
        detail_layout.addRow("Note", self.note_edit)

        list_widget = QWidget()
        # the label on the left, the list's buttons on the right; the search box under the list
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(self.list_label)
        header_layout.addWidget(self.auto_button)
        header_layout.addStretch(1)
        header_layout.addWidget(self.add_button)
        header_layout.addWidget(self.remove_button)
        header_layout.addWidget(self.columns_button)
        header_layout.addWidget(self.more_button)
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(SMALL, MEDIUM, SMALL, SMALL)
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.addWidget(self.search_edit, 1)
        search_layout.addWidget(self.search_info_button)
        list_layout.addLayout(header_layout)
        list_layout.addWidget(self.table_view)
        list_layout.addLayout(search_layout)

        # the top at the height of its fields, the list below it in the rest
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(detail_widget)
        layout.addWidget(list_widget, 1)

        self.trace_type_combo.currentTextChanged.connect(self._update_mz_enabled)
        self.trace_type_combo.currentTextChanged.connect(lambda _text: self._emit(self.trace_type_edited))
        self.ms_level_combo.currentTextChanged.connect(self._update_msms_enabled)
        self.ms_level_combo.currentTextChanged.connect(lambda _text: self._emit(self.ms_level_edited))
        self.formula_edit.textChanged.connect(lambda _text: self._emit(self.chemistry_text_changed))
        self.adduct_edit.textChanged.connect(lambda _text: self._emit(self.chemistry_text_changed))
        for spin_box, field_name in (
            (self.rt_range_control.center_spin_box, "rt"),
            (self.rt_range_control.tolerance_spin_box, "rt_tolerance"),
        ):
            spin_box.valueChanged.connect(lambda _value, field_name=field_name: self._emit(self.rt_edited, field_name))
        for spin_box, field_name in (
            (self.mz_range_control.center_spin_box, "mz"),
            (self.mz_range_control.tolerance_spin_box, "mz_tolerance"),
            (self.mz_range_control.ppm_spin_box, "mz_tolerance"),
        ):
            spin_box.valueChanged.connect(lambda _value, field_name=field_name: self._emit(self.mz_edited, field_name))
        self.mz_range_control.mode_changed.connect(lambda _text: self._emit(self.mz_edited, "mz_tolerance_unit"))
        self._precursor_actions.triggered.connect(lambda action: self._emit(self.precursor_chosen, action.data()))
        self.merge_ppm_spin_box.valueChanged.connect(lambda value: self._emit(self.merge_ppm_edited, value))
        for field_name in self.field_edits:
            self.field_edits[field_name].editingFinished.connect(
                lambda field_name=field_name: self._on_field_editing_finished(field_name)
            )
            self.field_edits[field_name].textEdited.connect(
                lambda text, field_name=field_name: self._emit(self.field_typed, field_name, text)
            )
            self.field_edits[field_name].editingFinished.connect(lambda: self._emit(self.typing_finished))
        for spin_box, field_name in (
            (self.rt_range_control.center_spin_box, "rt"),
            (self.rt_range_control.tolerance_spin_box, "rt_tolerance"),
            (self.mz_range_control.center_spin_box, "mz"),
            (self.mz_range_control.tolerance_spin_box, "mz_tolerance"),
            (self.mz_range_control.ppm_spin_box, "mz_tolerance"),
        ):
            spin_box.lineEdit().textEdited.connect(
                lambda _text, spin_box=spin_box, field_name=field_name: self._on_spin_box_typed(spin_box, field_name)
            )
            spin_box.editingFinished.connect(lambda: self._emit(self.typing_finished))
        self.add_button.clicked.connect(self.add_requested.emit)
        self.remove_button.clicked.connect(self.remove_requested.emit)
        self.auto_button.clicked.connect(self.auto_requested.emit)
        self.reset_index_action.triggered.connect(self.reset_index_requested.emit)
        self.apply_mz_button.clicked.connect(self.apply_mz_requested.emit)
        self.recalculate_all_action.triggered.connect(self.recalculate_all_requested.emit)
        self.import_action.triggered.connect(self.import_requested.emit)
        self.export_action.triggered.connect(self.export_requested.emit)
        self.search_edit.textChanged.connect(self._on_search_text_changed)
        self.all_check_box.clicked.connect(self._on_all_clicked)
        self.default_columns_action.triggered.connect(self._on_default_columns)
        for check_box in self.column_check_boxes.values():
            check_box.toggled.connect(self._on_columns_edited)
        self.table_view.horizontalHeader().customContextMenuRequested.connect(self._show_columns_menu_at_header)
        self._corner_button.clicked.connect(self._sort_by_index)
        self.table_view.horizontalHeader().sortIndicatorChanged.connect(lambda *_args: self._corner_button.update())
        self._update_msms_enabled()

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
        # the m/z window is used only for XIC; of several targets whose traces differ, by some of them
        self.mz_range_control.setEnabled(self.trace_type() in (TraceType.XIC, None))

    def _update_msms_enabled(self, _text: str = ""):
        # MS/MS settings exist only for MS2, the target's or the draft's
        is_ms2 = self.ms_level() is MSLevel.MS2
        self.precursor_button.setEnabled(is_ms2)
        self.precursor_button.setText(self._chosen_precursor_text if is_ms2 else "Precursor: n.a.")

    def _on_field_editing_finished(self, field_name: str):
        # leaving a field that was not typed in changes nothing: a draft's m/z set by hand stays
        line_edit = self.field_edits[field_name]
        if not line_edit.isModified():
            return
        line_edit.setModified(False)
        if field_name in self.mixed_fields and not line_edit.text().strip():
            return  # left blank, the targets keep their values
        self._emit(self.field_edited, field_name, line_edit.text())

    def _on_spin_box_typed(self, spin_box, field_name: str):
        value = typed_value(spin_box)
        if value is not None:
            self._emit(self.field_typed, field_name, value)

    def _on_cell_typed(self, proxy_index, value):
        source_index = self._proxy_model.mapToSource(proxy_index)
        target_id = self._table_model.target_at(source_index.row()).id
        self.cell_typed.emit(target_id, TargetTableModel.COLUMNS[source_index.column()].field, value)

    def commit_pending_fields(self):
        """Commit the fields typed in but not yet left."""
        for field_name in self.field_edits:
            self._on_field_editing_finished(field_name)

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
        self._proxy_model = TargetFilterProxyModel(self)
        self._proxy_model.setSourceModel(table_model)
        self._proxy_model.setSortRole(TargetTableModel.SORT_ROLE)
        self._proxy_model.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        # keep the row in place while its target is being edited; re-sorted on reset
        self._proxy_model.setDynamicSortFilter(False)
        self.table_view.setModel(self._proxy_model)
        # all columns at their content width; the table scrolls horizontally
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_view.horizontalHeader().setSortIndicator(TargetTableModel.INDEX_COLUMN, Qt.SortOrder.AscendingOrder)
        self.table_view.setSortingEnabled(True)
        self.table_view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.table_view.selectionModel().currentRowChanged.connect(self._on_current_row_changed)
        self._proxy_model.modelAboutToBeReset.connect(self._on_rows_about_to_reset)
        self._proxy_model.modelReset.connect(self._on_rows_reset)
        self._update_list_buttons()

    ### removed targets shown going ###
    # Every change to the list resets the table; one that only takes rows away -- a removal, or the undo of an
    # add -- is played over it (see RowChangeOverlay), the rows and the row header alike.
    def _row_viewports(self) -> tuple[QWidget, QWidget]:
        return self.table_view.viewport(), self.table_view.verticalHeader().viewport()

    def _row_spans(self) -> dict[str, tuple[int, int]]:
        """The top in the viewport and the height of each shown target's row, by id."""
        spans = {}
        for row in range(self._proxy_model.rowCount()):
            source_row = self._proxy_model.mapToSource(self._proxy_model.index(row, 0)).row()
            spans[self._table_model.shown_id(source_row)] = (
                self.table_view.rowViewportPosition(row),
                self.table_view.rowHeight(row),
            )
        return spans

    def _on_rows_about_to_reset(self):
        self._rows_before_reset = None
        if not self.table_view.isVisible():
            return
        for viewport in self._row_viewports():
            RowChangeOverlay.finish_all(viewport)
        self._rows_before_reset = (
            self._row_spans(),
            [RowChangeOverlay.picture_of_rows(viewport) for viewport in self._row_viewports()],
        )

    def _on_rows_reset(self):
        if self._rows_before_reset is not None:
            # once the next selection and the scroll are set, and before the rows are painted
            QTimer.singleShot(0, self._play_row_removal)

    def _play_row_removal(self):
        before, self._rows_before_reset = self._rows_before_reset, None
        if before is None:
            return
        rows, pictures = before
        rows_now = self._row_spans()
        if not rows_now.keys() < rows.keys():
            return  # rows added, or the list made anew: shown at once
        for viewport, picture in zip(self._row_viewports(), pictures):
            picture_now = RowChangeOverlay.picture_of_rows(viewport)
            height = viewport.height()

            def in_view(span):
                return span[0] + span[1] > 0 and span[0] < height

            def picture_of(target_id):
                # with its new look (selected, say) where it is in view now; one gone out of view, as it was
                span = rows_now[target_id]
                if in_view(span):
                    return RowChangeOverlay.cut_row(picture_now, span)
                return RowChangeOverlay.cut_row(picture, rows[target_id])

            removed = {
                target_id: RowChangeOverlay.cut_row(picture, span)
                for target_id, span in rows.items()
                if target_id not in rows_now and in_view(span)
            }
            background = viewport.palette().color(viewport.backgroundRole())
            RowChangeOverlay.play(viewport, rows, rows_now, removed, picture_of, background)

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

    def target_id_after_removal(self, removed_ids: list[str]) -> str | None:
        """The target that takes the place of the removed ones: the one then at the
        top removed row, or the last one when that row is past the end."""
        shown = [
            self._table_model.target_at(self._proxy_model.mapToSource(self._proxy_model.index(row, 0)).row()).id
            for row in range(self._proxy_model.rowCount())
        ]
        removed = set(removed_ids)
        remaining = [target_id for target_id in shown if target_id not in removed]
        top_row = next((row for row, target_id in enumerate(shown) if target_id in removed), None)
        if top_row is None or not remaining:
            return None
        return remaining[min(top_row, len(remaining) - 1)]

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

    def select_targets(self, target_ids: list[str], current_id: str | None = None):
        """Select several targets, current_id's row, if given, the current one."""
        with self._suppress_signals():
            if not all(self._proxy_index_of(target_id).isValid() for target_id in target_ids):
                # some hidden by the search: show all rows
                self.search_edit.clear()
            selection = QItemSelection()
            for target_id in target_ids:
                proxy_index = self._proxy_index_of(target_id)
                selection.select(proxy_index, proxy_index)
            selection_model = self.table_view.selectionModel()
            selection_model.select(
                selection, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows
            )
            if current_id is not None:
                selection_model.setCurrentIndex(self._proxy_index_of(current_id), QItemSelectionModel.SelectionFlag.NoUpdate)
        self._update_list_buttons()

    def current_target_id(self) -> str | None:
        """The target of the current row, the one last clicked, if it is selected; None otherwise."""
        index = self.table_view.currentIndex()
        if not index.isValid() or not self.table_view.selectionModel().isRowSelected(index.row()):
            return None
        return self._table_model.target_at(self._proxy_model.mapToSource(index).row()).id

    def _proxy_index_of(self, target_id: str, column: int = 0):
        return self._proxy_model.mapFromSource(self._table_model.index(self._table_model.row_of(target_id), column))

    def show_table_edit_error(self, target_id: str, field_name: str, message: str):
        """Point at the cell whose edit was rejected; the cell keeps its value."""
        proxy_index = self._proxy_index_of(target_id, TargetTableModel.column_of(field_name))
        cell_rect = self.table_view.visualRect(proxy_index)
        QToolTip.showText(self.table_view.viewport().mapToGlobal(cell_rect.bottomLeft()), message, self.table_view)

    def _on_search_text_changed(self, text: str):
        try:
            matches = parse_search(text)
        except SearchError as error:
            # the list stays as the last search that could be read left it: a query being typed often cannot be
            self._mark_error(self.search_edit, str(error))
            return
        self._mark_error(self.search_edit, None)
        self._proxy_model.set_search(matches)
        # rows hidden by the search are deselected
        self._on_selection_changed()

    ### clicks on the list ###
    def eventFilter(self, watched, event):
        if watched is self._corner_button and event.type() == QEvent.Type.Paint:
            self._paint_corner()
            return True
        mouse_events = (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease, QEvent.Type.MouseButtonDblClick)
        if event.type() in mouse_events:
            if watched is self.table_view.viewport():
                self._on_cell_mouse_event(event)
            elif watched is self.table_view.verticalHeader().viewport():
                self._on_row_header_mouse_event(event)
        return super().eventFilter(watched, event)

    def _on_cell_mouse_event(self, event):
        index = self.table_view.indexAt(event.position().toPoint())
        if event.type() == QEvent.Type.MouseButtonPress:
            self._deselect_on_release = (
                index.isValid() and index == self.table_view.currentIndex() and self._clicks_only_selected_row(event, index.row())
            )
            if not index.isValid() and event.modifiers() == Qt.KeyboardModifier.NoModifier:
                # off the rows the view deselects but keeps the current cell, and its frame
                self._deselect_row()
        elif event.type() == QEvent.Type.MouseButtonRelease:
            self._end_click()
        else:
            # the first click of a double click on the selected row deselected it: the double click
            # selects it again, and edits the cell
            self._deselect_on_release = False
            if index.isValid() and not self.table_view.selectionModel().isRowSelected(index.row()):
                self.table_view.selectionModel().setCurrentIndex(
                    index, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows
                )

    def _on_row_header_mouse_event(self, event):
        row = self.table_view.verticalHeader().logicalIndexAt(event.position().toPoint())
        if event.type() == QEvent.Type.MouseButtonPress:
            self._deselect_on_release = row >= 0 and self._clicks_only_selected_row(event, row)
        elif event.type() == QEvent.Type.MouseButtonRelease:
            self._end_click()
        else:
            # as on a cell, without the edit
            self._deselect_on_release = False
            if row >= 0 and not self.table_view.selectionModel().isRowSelected(row):
                self.table_view.selectRow(row)

    def _clicks_only_selected_row(self, event, row: int) -> bool:
        if event.button() != Qt.MouseButton.LeftButton or event.modifiers() != Qt.KeyboardModifier.NoModifier:
            return False
        return [selected.row() for selected in self.table_view.selectionModel().selectedRows()] == [row]

    def _end_click(self):
        if self._deselect_on_release:
            self._deselect_row()
        self._deselect_on_release = False

    def _deselect_row(self):
        # the frame of the current cell goes with the selection
        self.table_view.selectionModel().clearCurrentIndex()
        self.table_view.clearSelection()

    ### the corner: sort by index ###
    def _sort_by_index(self):
        header = self.table_view.horizontalHeader()
        # a click on the sorted corner turns the order around, as on a column header
        if header.sortIndicatorSection() == TargetTableModel.INDEX_COLUMN and header.sortIndicatorOrder() == Qt.SortOrder.AscendingOrder:
            order = Qt.SortOrder.DescendingOrder
        else:
            order = Qt.SortOrder.AscendingOrder
        self.table_view.sortByColumn(TargetTableModel.INDEX_COLUMN, order)

    def _paint_corner(self):
        # as Qt paints the corner, with the sort indicator of a header section sorted by index
        button = self._corner_button
        option = QStyleOptionHeader()
        option.initFrom(button)
        if button.isDown():
            option.state |= QStyle.StateFlag.State_Sunken
        option.rect = button.rect()
        # not OnlyOneSection: that is the last section, which gets no separator on its right
        option.position = QStyleOptionHeader.SectionPosition.Beginning
        header = self.table_view.horizontalHeader()
        if header.sortIndicatorSection() == TargetTableModel.INDEX_COLUMN:
            ascending = header.sortIndicatorOrder() == Qt.SortOrder.AscendingOrder
            option.sortIndicator = QStyleOptionHeader.SortIndicator.SortDown if ascending else QStyleOptionHeader.SortIndicator.SortUp
        painter = QStylePainter(button)
        painter.drawControl(QStyle.ControlElement.CE_Header, option)
        painter.end()

    ### shown columns ###
    def column_settings(self) -> list[str]:
        """The fields of the columns shown."""
        return [field for field, check_box in self.column_check_boxes.items() if check_box.isChecked()]

    def partial_column_settings(self) -> list[str]:
        """The fields of the last choice of some columns, which the All check box returns to from none."""
        return list(self._partial_fields)

    def set_column_settings(self, shown_fields, partial_fields=None):
        """Show the columns of shown_fields; partial_fields, if some, is the choice All returns to."""
        if partial_fields is not None:
            partial_fields = [field for field in self.column_check_boxes if field in partial_fields]
            if 0 < len(partial_fields) < len(self.column_check_boxes):
                self._partial_fields = partial_fields
        with self._suppress_signals():
            for field, check_box in self.column_check_boxes.items():
                check_box.setChecked(field in shown_fields)
        self._update_shown_columns()

    def _on_all_clicked(self, _checked: bool):
        state = self.all_check_box.checkState()
        if state == Qt.CheckState.PartiallyChecked:
            shown_fields = list(self.column_check_boxes)
        elif state == Qt.CheckState.Checked:
            shown_fields = []
        else:
            shown_fields = self._partial_fields
        with self._suppress_signals():
            for field, check_box in self.column_check_boxes.items():
                check_box.setChecked(field in shown_fields)
        self._on_columns_edited()

    def _on_default_columns(self):
        self.set_column_settings(TargetTableModel.DEFAULT_SHOWN_FIELDS)
        self._emit(self.columns_changed)

    def _on_columns_edited(self, *_args):
        self._update_shown_columns()
        self._emit(self.columns_changed)

    def _show_columns_menu_at_header(self, pos):
        self.columns_menu.popup(self.table_view.horizontalHeader().mapToGlobal(pos))

    def _update_shown_columns(self):
        shown_fields = self.column_settings()
        n_shown = len(shown_fields)
        if n_shown == 0:
            state = Qt.CheckState.Unchecked
        elif n_shown == len(self.column_check_boxes):
            state = Qt.CheckState.Checked
        else:
            state = Qt.CheckState.PartiallyChecked
            if not self._updating:  # not halfway through setting all the check boxes
                self._partial_fields = shown_fields
        self.all_check_box.setCheckState(state)
        for column, spec in enumerate(TargetTableModel.COLUMNS):
            # the index column only sorts: its values are in the row header
            shown = spec.field is not None and spec.field in shown_fields
            self.table_view.setColumnHidden(column, not shown)

    def _on_selection_changed(self, *_args):
        self._update_list_buttons()
        self._emit(self.selection_changed)

    def _on_current_row_changed(self, current, _previous):
        # of several selected, the plots show the current row's target (current_target_id), which a key makes current
        # after it changed the selection
        selection_model = self.table_view.selectionModel()
        if current.isValid() and len(selection_model.selectedRows()) > 1 and selection_model.isRowSelected(current.row()):
            self._emit(self.selection_changed)

    def _update_list_buttons(self):
        n_selected = len(self.table_view.selectionModel().selectedRows())
        self.add_button.setToolTip(
            self.ADD_TOOLTIP if n_selected == 0 else self.DUPLICATE_TOOLTIP if n_selected == 1 else self.DUPLICATE_ALL_TOOLTIP
        )
        self.remove_button.setEnabled(n_selected > 0)

    ### detail ###
    def show_target(self, target: Target | None):
        """Fill the fields with ``target``; None keeps the draft, or starts one on leaving a target.

        A new draft keeps the view and the adduct; name, formula, note and MS/MS settings start empty.
        """
        self.state_label.setText("New target (not added)" if target is None else f"Target #{target.index}: {target.name}")
        with self._suppress_signals():
            for field_name, line_edit in self.field_edits.items():
                line_edit.setPlaceholderText(self._placeholders[field_name])
            if target is not None:
                self.name_edit.setText(target.name)
                self.formula_edit.setText(target.formula)
                self.adduct_edit.setText(target.adduct)
                self.note_edit.setText(target.note)
                self.set_trace_type(target.trace_type)
                self.set_ms_level(target.ms_level)
                self.set_rt_values(target.rt, target.rt_tolerance)
                self.set_mz_values(target.mz, target.mz_tolerance, target.mz_tolerance_unit)
            elif self._showing_target:
                for field_name in ("name", "formula", "note"):
                    self.field_edits[field_name].clear()
            for field_name in self.field_edits:
                self.set_field_error(field_name, None)
        self.mixed_fields = set()
        self._showing_target = target is not None
        self._update_detail_fill()
        self._update_msms_enabled()

    def show_targets(self, targets: list[Target], shown: Target):
        """Fill the fields with what ``targets``, several selected, have in common, blank where they differ
        (mixed_fields), to edit them all at once. The m/z tolerance goes with its unit: where the units differ, the
        tolerance is blank, and the unit is that of ``shown``, the target the plots show."""
        mixed = set()

        def common(field_name: str):
            values = {getattr(target, field_name) for target in targets}
            if len(values) > 1:
                mixed.add(field_name)
                return None
            return values.pop()

        self.state_label.setText(f"{len(targets)} targets selected")
        with self._suppress_signals():
            for field_name, line_edit in self.field_edits.items():
                value = common(field_name)
                line_edit.setText(value or "")
                line_edit.setPlaceholderText("" if value is None else self._placeholders[field_name])
                self.set_field_error(field_name, None)
            self.set_trace_type(common("trace_type"))
            self.set_ms_level(common("ms_level"))
            self.set_rt_values(common("rt"), common("rt_tolerance"))
            unit = common("mz_tolerance_unit")
            if unit is None:
                mixed.add("mz_tolerance")
                self.set_mz_values(common("mz"), None, shown.mz_tolerance_unit)
            else:
                self.set_mz_values(common("mz"), common("mz_tolerance"), unit)
        self.mixed_fields = mixed
        self._showing_target = True
        self._update_detail_fill()
        self._update_msms_enabled()

    def _update_detail_fill(self):
        palette = QPalette()
        if not self._showing_target:
            # the text colour, very faint: a slightly darker grey in Light mode, a slightly lighter one in Dark
            text = self.palette().color(QPalette.ColorRole.WindowText)
            text.setAlpha(25)
            palette.setColor(QPalette.ColorRole.Window, _over(self.palette().color(QPalette.ColorRole.Window), text))
        else:
            # the pale (unfocused) selection colour: the table shows its selected row so while the panel has focus
            palette.setColor(QPalette.ColorRole.Window, self.palette().color(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Highlight))
        self.detail_frame.setPalette(palette)

    def set_mz_mismatch(self, mismatch: bool):
        """Mark the m/z when it differs from the one of formula + adduct, as the list does."""
        self._mz_mismatch = mismatch
        self._update_mz_mismatch_fill()

    def set_apply_mz_enabled(self, enabled: bool):
        self.apply_mz_button.setEnabled(enabled)

    def focus_mz(self):
        self.mz_range_control.center_spin_box.setFocus()

    def set_precursor_mismatch(self, mismatch: bool):
        """Mark the precursor when its isolation window misses the ion of formula + adduct, as the list does."""
        self._precursor_mismatch = mismatch
        self._update_mz_mismatch_fill()

    def _update_mz_mismatch_fill(self):
        # the list's translucent red, laid over the widget's own colour; else the panel's own colours
        for widget, role, mismatch in (
            (self.mz_range_control.center_spin_box, QPalette.ColorRole.Base, self._mz_mismatch),
            (self.precursor_button, QPalette.ColorRole.Button, self._precursor_mismatch),
        ):
            palette = QPalette()
            if mismatch:
                palette.setColor(role, _over(self.palette().color(role), QColor(MZ_MISMATCH_COLOR)))
            widget.setPalette(palette)

    ### MS2 scans ###
    def set_precursors(self, precursors: list[Precursor]):
        """List the precursors to choose from; none is chosen until set_chosen_precursor()."""
        for action in self._precursor_actions.actions():
            self._precursor_actions.removeAction(action)
            self.precursor_menu.removeAction(action)
            action.deleteLater()
        for index, precursor in enumerate(precursors):
            action = self.precursor_menu.addAction(
                f"{precursor_text(precursor.precursor_mz, precursor.collision_energy)}  ({precursor.n_scans} scans)"
            )
            action.setCheckable(True)
            action.setData(index)
            self._precursor_actions.addAction(action)
        if not precursors:
            action = self.precursor_menu.addAction("No MS2 scans in the files")
            action.setEnabled(False)
            self._precursor_actions.addAction(action)

    def set_chosen_precursor(
        self, index: int | None, precursor_mz: float | None, collision_energy: float | None, *, mixed: bool = False
    ):
        """Show the precursor and CE in use; index is theirs in the list, None when they are not in it. mixed: of
        several targets, theirs differ, shown blank."""
        for action in self._precursor_actions.actions():
            action.setChecked(action.data() is not None and action.data() == index)
        if mixed:
            self._chosen_precursor_text = ""
        else:
            self._chosen_precursor_text = "None" if precursor_mz is None else precursor_text(precursor_mz, collision_energy)
        self._update_msms_enabled()

    def set_merge_ppm(self, ppm: int):
        with self._suppress_signals():
            self.merge_ppm_spin_box.setValue(ppm)

    def changeEvent(self, event):
        super().changeEvent(event)
        # a Light/Dark switch changes the selection colour and the field colour
        if event.type() == QEvent.Type.PaletteChange:
            self._update_detail_fill()
            self._update_mz_mismatch_fill()
            for line_edit in self._marked_edits:
                self._update_error_fill(line_edit)

    def chemistry_texts(self) -> tuple[str, str]:
        return self.formula_edit.text().strip(), self.adduct_edit.text().strip()

    def draft_texts(self) -> dict[str, str]:
        """The stripped text of each field, by its target field name."""
        return {field_name: line_edit.text().strip() for field_name, line_edit in self.field_edits.items()}

    def set_chemistry_preview(self, formula_mass: float | None, ion_mz: float | None, *, decimals: int):
        """Neutral mass of the formula and m/z of the ion; None when they cannot be calculated."""
        self.formula_mass_label.setText("" if formula_mass is None else f"{formula_mass:.{decimals}f}")
        self.calculated_mz_label.setText("" if ion_mz is None else f"{ion_mz:.{decimals}f}")

    def set_field_error(self, field_name: str, message: str | None):
        self._mark_error(self.field_edits[field_name], message)

    def _mark_error(self, line_edit: QLineEdit, message: str | None):
        """Fill ``line_edit`` with the list's translucent red, with ``message`` in its tooltip; None takes both away.

        Not a red frame: a style sheet's border, drawn in place of the style's frame, changed the field's height.
        """
        if message is None:
            self._marked_edits.discard(line_edit)
        else:
            self._marked_edits.add(line_edit)
        line_edit.setToolTip(message or "")
        self._update_error_fill(line_edit)

    def _update_error_fill(self, line_edit: QLineEdit):
        # as the m/z that differs from the formula's
        palette = QPalette()
        if line_edit in self._marked_edits:
            palette.setColor(QPalette.ColorRole.Base, _over(self.palette().color(QPalette.ColorRole.Base), QColor(MZ_MISMATCH_COLOR)))
        line_edit.setPalette(palette)

    ### view parameters ###
    # None, of the getters below, and to the setters: blank, the values of the targets selected differing (see
    # show_targets); the getters of the numbers give a blank one's below its minimum then
    def trace_type(self) -> TraceType | None:
        text = self.trace_type_combo.currentText()
        return TraceType(text) if text else None

    def set_trace_type(self, trace_type: TraceType | None):
        with self._suppress_signals():
            if trace_type is None:
                self.trace_type_combo.setCurrentIndex(-1)
            else:
                self.trace_type_combo.setCurrentText(trace_type.value)

    def ms_level(self) -> MSLevel | None:
        text = self.ms_level_combo.currentText()
        return MSLevel(text) if text else None

    def set_ms_level(self, ms_level: MSLevel | None):
        with self._suppress_signals():
            if ms_level is None:
                self.ms_level_combo.setCurrentIndex(-1)
            else:
                self.ms_level_combo.setCurrentText(ms_level.value)

    def rt_values(self) -> tuple[float, float]:
        return self.rt_range_control.value(), self.rt_range_control.tolerance()

    def set_rt_values(self, center: float | None, tolerance: float | None):
        with self._suppress_signals():
            self.rt_range_control.set_value(center)
            self.rt_range_control.set_tolerance(tolerance)

    def mz_values(self) -> tuple[float, float, ToleranceUnit]:
        unit = ToleranceUnit(self.mz_range_control.tolerance_mode())
        tolerance = self.mz_range_control.ppm_tolerance() if unit is ToleranceUnit.PPM else self.mz_range_control.tolerance()
        return self.mz_range_control.value(), tolerance, unit

    def set_mz_values(self, center: float | None, tolerance: float | None, unit: ToleranceUnit):
        with self._suppress_signals():
            self.mz_range_control.set_value(center)
            self.mz_range_control.set_tolerance_mode(unit.value)
            if unit is ToleranceUnit.PPM:
                self.mz_range_control.set_ppm_tolerance(None if tolerance is None else round(tolerance))
            else:
                self.mz_range_control.set_tolerance(tolerance)
