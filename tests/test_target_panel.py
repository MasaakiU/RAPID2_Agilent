from dataclasses import replace
import os
import sys

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QAbstractButton, QAbstractItemDelegate, QApplication, QStyleOptionViewItem, QToolTip

from RAPID2_Agilent.models.chemistry import ion_mz
from RAPID2_Agilent.models.main_model import MainModel
from RAPID2_Agilent.models.run_peaks import Peak, run_peaks_for
from RAPID2_Agilent.models.target import MSLevel, ToleranceUnit, TraceType
from RAPID2_Agilent.presenters.main_presenter import MainPresenter
from RAPID2_Agilent.views.main_window import MainWindow
from RAPID2_Agilent.views.target_panel import TargetTableModel
from RAPID2_Agilent.views.view_constants import BROWSING_REGION_COLOR, OWN_RT_REGION_COLOR, TARGET_REGION_COLOR
from tests.data_paths import DATA_DIR, require


DATA_PATH = DATA_DIR / "dummy_RP_2min_neg.d"
COPY_DATA_PATH = DATA_DIR / "dummy_RP_2min_neg_copy.d"  # the same data, another run to the app
MSMS_DATA_PATH = DATA_DIR / "MSMS_data" / "9_PRJ_0-1-2_PA34-1_1uM_n1.d"
CHOLINE_DATA_PATH = DATA_DIR / "9_P2-8-15_spl1_p0.d"  # choline, C5H14NO+, near RT 3.3 min
OTHER_DATA_PATH = DATA_DIR / "4_P2-10-12_BD_sv_MeOH100_ACN0_p0.d"  # no peak of choline near 3.3 min


def build_app(data_paths=(DATA_PATH, COPY_DATA_PATH), *, keep_settings=False):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)
    view = MainWindow()
    model = MainModel()
    if not keep_settings:
        # each test starts from the default settings (a temporary file, see conftest)
        model.settings_model.settings.clear()
    presenter = MainPresenter(view, model)
    presenter.open_d_folders_clicked([str(require(path)) for path in data_paths])
    return app, view, model, presenter


def type_into(panel, field_name: str, text: str, *, leave=True):
    """As the user types ``text`` into a field of the panel, and leaves it."""
    line_edit = panel.field_edits[field_name]
    line_edit.setText(text)
    line_edit.setModified(True)
    if leave:
        line_edit.editingFinished.emit()


def edit_cell(panel, target_id: str, field_name: str, value):
    proxy_index = panel._proxy_index_of(target_id, TargetTableModel.column_of(field_name))
    assert panel.table_view.model().setData(proxy_index, value)


def test_controls_follow_plots_in_viewer_mode():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    stack = presenter.data_stack_presenter

    stack.child_presenters[0].set_mz_window((200.0, 201.0), emit_signal=True)

    assert panel.mz_values() == pytest.approx((200.5, 0.5, ToleranceUnit.MZ))
    assert model.target_list_model.targets == ()


def test_controls_move_plots_in_viewer_mode():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    stack = presenter.data_stack_presenter

    panel.rt_range_control.set_value(0.5)
    panel.rt_range_control.set_tolerance(0.05)

    np.testing.assert_allclose(stack.current_rt_window, (0.45, 0.55))
    for child in stack.child_presenters:
        np.testing.assert_allclose(child.current_rt_window, (0.45, 0.55))


def test_add_target_takes_current_view():
    app, view, model, presenter = build_app()
    stack = presenter.data_stack_presenter

    target = presenter.target_panel_presenter.add_target_from_view()

    assert presenter.target_panel_presenter.current_id == target.id
    assert view.target_panel.selected_target_ids() == [target.id]
    np.testing.assert_allclose(target.rt_window(), stack.current_rt_window)
    np.testing.assert_allclose(target.mz_window(), stack.current_mz_window)


def test_plot_change_edits_current_target():
    app, view, model, presenter = build_app()
    target = presenter.target_panel_presenter.add_target_from_view()

    presenter.data_stack_presenter.child_presenters[1].set_mz_window((300.0, 300.2), emit_signal=True)

    updated = model.target_list_model.get(target.id)
    assert (updated.mz, updated.mz_tolerance) == pytest.approx((300.1, 0.1))


def drag_region(window, region):
    """As the user drags the window's region to ``region``, without dropping it."""
    window.region0.blockSignals(True)
    window.region0.setRegion(region)
    window.region0.blockSignals(False)
    window.region0.sigRegionChanged.emit(window.region0)


def test_dragged_window_shows_in_panel_and_list_and_edits_target_once_dropped():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target = presenter.target_panel_presenter.add_target_from_view()
    table_model = presenter.target_panel_presenter.table_model
    mz_cell = table_model.index(table_model.row_of(target.id), TargetTableModel.column_of("mz"))
    child = presenter.data_stack_presenter.child_presenters[1]
    mz_window = child.view.spectrum_window

    drag_region(mz_window, (300.0, 300.2))

    assert panel.mz_values() == pytest.approx((300.1, 0.1, ToleranceUnit.MZ))
    assert mz_cell.data() == "300.1000"
    assert model.target_list_model.get(target.id) == target

    mz_window.region0.sigRegionChangeFinished.emit(mz_window.region0)
    edited = model.target_list_model.get(target.id)
    assert edited.mz_window() == pytest.approx((300.0, 300.2))
    assert mz_cell.data() == "300.1000"
    drag_region(mz_window, (310.0, 310.2))
    mz_window.region0.sigRegionChangeFinished.emit(mz_window.region0)
    # one edit for each drop
    model.target_list_model.undo()
    assert model.target_list_model.get(target.id) == edited
    model.target_list_model.undo()
    assert model.target_list_model.get(target.id) == target


def test_dragged_rt_window_edits_the_target_while_the_files_windows_are_linked():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target = presenter.target_panel_presenter.add_target_from_view()
    table_model = presenter.target_panel_presenter.table_model
    rt_cell = table_model.index(table_model.row_of(target.id), TargetTableModel.column_of("rt"))
    first, second = presenter.data_stack_presenter.child_presenters
    window = second.view.chromatogram_window

    drag_region(window, (0.4, 0.6))

    assert panel.rt_values() == pytest.approx((0.5, 0.1))
    assert rt_cell.data() == "0.50"
    assert model.target_list_model.get(target.id) == target
    np.testing.assert_allclose(first.view.chromatogram_window.get_region(), (0.4, 0.6))

    window.region0.sigRegionChangeFinished.emit(window.region0)
    assert model.target_list_model.get(target.id).rt_window() == pytest.approx((0.4, 0.6))
    np.testing.assert_allclose(first.current_rt_window, (0.4, 0.6))
    assert model.run_peaks.get(target.id, COPY_DATA_PATH) is None
    model.target_list_model.undo()
    assert model.target_list_model.get(target.id) == target


def test_a_targets_rt_window_set_on_a_file_is_its_peak_there_and_comes_back_with_the_target():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target_presenter = presenter.target_panel_presenter
    stack = presenter.data_stack_presenter
    check_box = view.link_rt_windows_check_box
    assert not check_box.isEnabled() and check_box.isChecked()
    target = target_presenter.add_target_from_view()
    assert check_box.isEnabled()
    check_box.setChecked(False)
    first, second = stack.child_presenters
    window = second.view.chromatogram_window

    drag_region(window, (0.4, 0.6))
    window.region0.sigRegionChangeFinished.emit(window.region0)

    # the target is as it was, its window the one to look for the peak in; the other file keeps it
    assert model.target_list_model.get(target.id) == target
    assert panel.rt_values() == pytest.approx((target.rt, target.rt_tolerance))
    np.testing.assert_allclose(second.current_rt_window, (0.4, 0.6))
    np.testing.assert_allclose(first.current_rt_window, target.rt_window())
    run_peaks = model.run_peaks.get(target.id, COPY_DATA_PATH)
    assert (run_peaks.mz_window, run_peaks.rt_window, run_peaks.manual) == (
        target.mz_window(), pytest.approx((0.4, 0.6)), True
    )
    assert model.run_peaks.get(target.id, DATA_PATH) is None
    # a click on a file's chromatogram moves its window there
    first.on_chromatogram_clicked(1.0, 0.0)
    assert np.mean(model.run_peaks.get(target.id, DATA_PATH).rt_window) == pytest.approx(1.0)

    # without the target, one window for all; the target brings back its peak in each
    target_presenter.select_target(None)
    np.testing.assert_allclose(first.current_rt_window, target.rt_window())
    np.testing.assert_allclose(second.current_rt_window, target.rt_window())
    target_presenter.select_target(target.id)
    np.testing.assert_allclose(second.current_rt_window, (0.4, 0.6))
    assert np.mean(first.current_rt_window) == pytest.approx(1.0)
    # and so does a file opened again
    stack.view.set_selected_items([second.view])
    stack.close_selected()
    presenter.open_d_folders_clicked([str(COPY_DATA_PATH)])
    second = stack.child_presenters[1]
    np.testing.assert_allclose(second.current_rt_window, (0.4, 0.6))

    # the target's RT moves where to look, not the peaks; its m/z is the files' too
    model.target_list_model.update(target.id, rt=1.5, mz=300.1)
    edited = model.target_list_model.get(target.id)
    np.testing.assert_allclose(stack.current_rt_window, edited.rt_window())
    np.testing.assert_allclose(second.current_rt_window, (0.4, 0.6))
    run_peaks = model.run_peaks.get(target.id, COPY_DATA_PATH)
    assert (run_peaks.mz_window, run_peaks.rt_window) == (edited.mz_window(), pytest.approx((0.4, 0.6)))

    # linked again, the target's window is every file's, and exported; unlinked, the peaks come back
    check_box.setChecked(True)
    np.testing.assert_allclose(second.current_rt_window, edited.rt_window())
    assert presenter.run_peaks_to_export() is None
    check_box.setChecked(False)
    np.testing.assert_allclose(second.current_rt_window, (0.4, 0.6))
    assert presenter.run_peaks_to_export() is model.run_peaks
    target_presenter.select_target(None)
    assert not check_box.isEnabled()


def test_the_peaks_found_in_a_file_are_shaded_while_its_rt_window_is_its_own():
    app, view, model, presenter = build_app()
    target_presenter = presenter.target_panel_presenter
    stack = presenter.data_stack_presenter
    type_into(view.target_panel, "adduct", "[M-H]-")  # the files' polarity
    target = target_presenter.add_target_from_view()
    first, second = stack.child_presenters
    rt = [float(value) for value in first.model.RT_list]
    peaks = (Peak(rt[2], rt[8], rt[5]), Peak(rt[20], rt[30], rt[25]))
    model.run_peaks.set(target.id, COPY_DATA_PATH, run_peaks_for(target, (rt[2], rt[8]), peaks))
    areas = lambda: [len(child.view.chromatogram_window.peak_area_items) for child in stack.child_presenters]
    region_color = lambda child: child.view.chromatogram_window.region0.brush.color().getRgb()
    target_presenter.select_target(None)
    target_presenter.select_target(target.id)
    # linked, as at first: the target's window in each file, nothing shaded
    assert areas() == [0, 0] and region_color(second) == TARGET_REGION_COLOR

    view.link_rt_windows_check_box.setChecked(False)

    assert areas() == [0, 2]
    assert {region_color(child) for child in stack.child_presenters} == {OWN_RT_REGION_COLOR}
    np.testing.assert_allclose(second.current_rt_window, (rt[2], rt[8]))
    # the target's m/z changed on the plots: what was found in the former XIC is gone, the window taken too
    second.set_mz_window((150.4, 150.6), emit_signal=True)
    assert areas() == [0, 0]
    assert model.run_peaks.get(target.id, COPY_DATA_PATH) is None
    np.testing.assert_allclose(second.current_rt_window, target.rt_window())


def test_auto_finds_the_peaks_and_asks_before_replacing_windows_set_by_hand(monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    app, view, model, presenter = build_app((CHOLINE_DATA_PATH, OTHER_DATA_PATH))
    panel = view.target_panel
    type_into(panel, "adduct", "[M]+")
    type_into(panel, "formula", "C5H14NO")
    panel.rt_range_control.set_value(3.3)
    panel.rt_range_control.set_tolerance(0.5)
    target = presenter.target_panel_presenter.add_target_from_view()
    choline, other = presenter.data_stack_presenter.child_presenters
    asked, answers = [], []

    def answer(box):
        asked.append(box.text())
        text = answers.pop(0)
        next(button for button in box.buttons() if button.text() == text).click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", answer)

    panel.auto_button.click()

    # nothing set by hand, nothing asked; each file's window its own, the target's peak in it
    assert asked == []
    assert not view.link_rt_windows_check_box.isChecked()
    start, end = choline.current_rt_window
    assert (start, end) == model.run_peaks.get(target.id, CHOLINE_DATA_PATH).rt_window and start < 3.29 < end
    assert choline.view.chromatogram_window.peak_area_items
    # no peak near 3.3 min in the other: its window hidden, and that said
    other_window = other.view.chromatogram_window
    assert not other_window.region0.isVisible()
    assert "No peak in the target's RT window" in other_window.warning_label.textItem.toPlainText()
    np.testing.assert_allclose(other.current_rt_window, target.rt_window())

    # a click on its chromatogram sets one there by hand
    other.on_chromatogram_clicked(3.3, 0.0)
    assert other_window.region0.isVisible()
    assert "No peak" not in other_window.warning_label.textItem.toPlainText()
    hand_set = model.run_peaks.get(target.id, OTHER_DATA_PATH)
    assert hand_set.manual and np.mean(hand_set.rt_window) == pytest.approx(3.3)

    # found again: asked about it first, and kept
    answers[:] = ["Keep hand-set"]
    panel.auto_button.click()
    assert "1 of the RT windows were set by hand" in asked[0]
    assert model.run_peaks.get(target.id, OTHER_DATA_PATH).rt_window == hand_set.rt_window
    # or not
    answers[:] = ["Overwrite all"]
    panel.auto_button.click()
    assert model.run_peaks.get(target.id, OTHER_DATA_PATH).rt_window is None
    assert not other_window.region0.isVisible()
    # undone as one edit
    view.undo_target_edit_action.trigger()
    assert model.run_peaks.get(target.id, OTHER_DATA_PATH).rt_window == hand_set.rt_window
    assert other_window.region0.isVisible()


class _RightClick:
    """A right click, as pyqtgraph hands it to an item."""

    def button(self):
        return Qt.MouseButton.RightButton

    def screenPos(self):
        return QPointF(10, 10)

    def accept(self):
        pass


def test_a_files_rt_window_set_is_undone_and_its_menu_takes_the_peak_away():
    app, view, model, presenter = build_app()
    target = presenter.target_panel_presenter.add_target_from_view()
    first, second = presenter.data_stack_presenter.child_presenters
    window = second.view.chromatogram_window
    # no menu of pyqtgraph's; the RT window's only while it is the file's own
    assert not window.plotItem.menuEnabled() and not window.vb().menuEnabled()
    assert not window.rt_window_menu_enabled
    view.link_rt_windows_check_box.setChecked(False)
    assert window.rt_window_menu_enabled

    drag_region(window, (0.4, 0.6))
    window.region0.sigRegionChangeFinished.emit(window.region0)
    view.undo_target_edit_action.trigger()
    assert model.run_peaks.get(target.id, COPY_DATA_PATH) is None
    np.testing.assert_allclose(second.current_rt_window, target.rt_window())
    view.redo_target_edit_action.trigger()
    np.testing.assert_allclose(second.current_rt_window, (0.4, 0.6))

    # right-clicked, the window asks for its menu, whose No peak takes the target's peak in the file away
    asked = []
    window.region0.menu_requested.connect(asked.append)
    window.region0.mouseClickEvent(_RightClick())
    assert asked == [QPointF(10, 10)]
    window.rt_window_menu.close()
    window.no_peak_action.trigger()
    run_peaks = model.run_peaks.get(target.id, COPY_DATA_PATH)
    assert run_peaks.manual and run_peaks.rt_window is None
    assert not window.region0.isVisible()
    np.testing.assert_allclose(first.current_rt_window, target.rt_window())
    view.undo_target_edit_action.trigger()
    assert window.region0.isVisible()
    np.testing.assert_allclose(second.current_rt_window, (0.4, 0.6))


def test_a_folder_open_already_is_not_opened_again(monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda parent, title, text: shown.append(text))
    app, view, model, presenter = build_app((DATA_PATH,))

    presenter.open_d_folders_clicked([str(DATA_PATH), str(require(COPY_DATA_PATH)), str(COPY_DATA_PATH)])

    names = [child.model.file_path.name for child in presenter.data_stack_presenter.child_presenters]
    assert names == [DATA_PATH.name, COPY_DATA_PATH.name]
    assert shown == [f"Open already:\n\n{DATA_PATH.name}\n{COPY_DATA_PATH.name}"]


def test_ppm_plot_change_snaps_to_integer_ppm():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target = presenter.target_panel_presenter.add_target_from_view()
    panel.mz_range_control.set_tolerance_mode(ToleranceUnit.PPM.value)

    presenter.data_stack_presenter.child_presenters[0].set_mz_window((299.9996, 300.0004), emit_signal=True)

    updated = model.target_list_model.get(target.id)
    assert updated.mz_tolerance_unit is ToleranceUnit.PPM
    assert updated.mz_tolerance == 1
    np.testing.assert_allclose(presenter.data_stack_presenter.current_mz_window, updated.mz_window())


def test_selecting_target_moves_plots():
    app, view, model, presenter = build_app()
    target_presenter = presenter.target_panel_presenter
    stack = presenter.data_stack_presenter
    first = target_presenter.add_target_from_view()
    target_presenter.select_target(None)
    stack.child_presenters[0].set_mz_window((250.0, 251.0), emit_signal=True)
    second = target_presenter.add_target_from_view()
    assert model.target_list_model.get(first.id) == first
    assert second.mz_window() == pytest.approx((250.0, 251.0))

    view.target_panel.table_view.selectRow(model.target_list_model.index_of(first.id))

    assert target_presenter.current_id == first.id
    np.testing.assert_allclose(stack.current_mz_window, first.mz_window())
    np.testing.assert_allclose(stack.child_presenters[1].current_mz_window, first.mz_window())


def test_formula_and_adduct_set_mz_and_mismatch_is_highlighted():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target = presenter.target_panel_presenter.add_target_from_view()

    edit_cell(panel, target.id, "adduct", "[M-H]-")
    edit_cell(panel, target.id, "formula", "C6H12O6")

    updated = model.target_list_model.get(target.id)
    mz_index = panel._proxy_index_of(target.id, TargetTableModel.MZ_COLUMN)
    field_color = panel.palette().color(QPalette.ColorRole.Base)

    def panel_mz_color():
        return panel.mz_range_control.center_spin_box.palette().color(QPalette.ColorRole.Base)

    assert updated.mz == pytest.approx(179.056112, abs=1e-6)
    assert mz_index.data(Qt.ItemDataRole.BackgroundRole) is None
    assert panel_mz_color() == field_color
    np.testing.assert_allclose(presenter.data_stack_presenter.current_mz_window, updated.mz_window())

    panel.mz_range_control.set_value(179.06)

    # only the m/z is marked, in the list and in the panel; formula and adduct are not
    assert mz_index.data(Qt.ItemDataRole.BackgroundRole) is not None
    for field in ("formula", "adduct"):
        assert panel._proxy_index_of(target.id, TargetTableModel.column_of(field)).data(Qt.ItemDataRole.BackgroundRole) is None
    assert panel_mz_color() != field_color

    # a draft is not marked
    presenter.target_panel_presenter.select_target(None)
    assert panel_mz_color() == field_color


def test_invalid_table_edit_is_not_committed():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target = presenter.target_panel_presenter.add_target_from_view()
    view.show()

    edit_cell(panel, target.id, "adduct", "M+H")

    assert model.target_list_model.get(target.id) == target
    assert "Invalid adduct" in QToolTip.text()


def test_table_edits_view_parameters_of_target():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    stack = presenter.data_stack_presenter
    target = presenter.target_panel_presenter.add_target_from_view()

    edit_cell(panel, target.id, "rt", 0.75)
    edit_cell(panel, target.id, "mz", 200.0)

    updated = model.target_list_model.get(target.id)
    assert (updated.rt, updated.mz) == pytest.approx((0.75, 200.0))
    assert panel.rt_values()[0] == pytest.approx(0.75)
    assert panel.mz_values()[0] == pytest.approx(200.0)
    np.testing.assert_allclose(stack.current_rt_window, updated.rt_window())
    np.testing.assert_allclose(stack.current_mz_window, updated.mz_window())


def test_table_spin_box_editor_commits_typed_value():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target = presenter.target_panel_presenter.add_target_from_view()
    view.show()
    proxy_index = panel._proxy_index_of(target.id, TargetTableModel.column_of("rt"))

    panel.table_view.edit(proxy_index)
    editor = panel.table_view.indexWidget(proxy_index)
    editor.selectAll()
    QTest.keyClicks(editor, "0.75")
    QTest.keyClick(editor, Qt.Key.Key_Return)
    app.processEvents()  # the delegate commits on Return through the event loop

    assert model.target_list_model.get(target.id).rt == pytest.approx(0.75)


def test_typed_in_the_panel_shows_in_the_list_before_it_is_committed():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target = presenter.target_panel_presenter.add_target_from_view()
    view.show()
    name_index = panel._proxy_index_of(target.id, TargetTableModel.column_of("name"))
    rt_index = panel._proxy_index_of(target.id, TargetTableModel.column_of("rt"))
    assert target.rt != pytest.approx(0.75)

    panel.name_edit.selectAll()
    QTest.keyClicks(panel.name_edit, "glucose")

    assert name_index.data() == "glucose"
    assert model.target_list_model.get(target.id).name == target.name
    panel.name_edit.editingFinished.emit()  # left
    assert model.target_list_model.get(target.id).name == name_index.data() == "glucose"

    spin_box = panel.rt_range_control.center_spin_box
    spin_box.lineEdit().selectAll()
    QTest.keyClicks(spin_box, "0.75")

    assert rt_index.data() == "0.75"
    assert model.target_list_model.get(target.id).rt == pytest.approx(target.rt)
    QTest.keyClick(spin_box, Qt.Key.Key_Return)
    assert model.target_list_model.get(target.id).rt == pytest.approx(0.75)
    assert rt_index.data() == "0.75"


def test_typed_in_the_list_shows_in_the_panel_before_it_is_committed():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target = presenter.target_panel_presenter.add_target_from_view()
    view.show()
    name_index = panel._proxy_index_of(target.id, TargetTableModel.column_of("name"))
    rt_index = panel._proxy_index_of(target.id, TargetTableModel.column_of("rt"))
    assert target.rt != pytest.approx(0.75)

    panel.table_view.edit(name_index)
    editor = panel.table_view.indexWidget(name_index)
    editor.selectAll()
    QTest.keyClicks(editor, "glucose")

    assert panel.name_edit.text() == "glucose"
    assert model.target_list_model.get(target.id).name == target.name
    # dropped: the panel shows the target as it is
    QTest.keyClick(editor, Qt.Key.Key_Escape)
    assert panel.name_edit.text() == target.name

    panel.table_view.edit(rt_index)
    editor = panel.table_view.indexWidget(rt_index)
    editor.selectAll()
    QTest.keyClicks(editor, "0.75")

    assert panel.rt_values()[0] == pytest.approx(0.75)
    assert model.target_list_model.get(target.id).rt == pytest.approx(target.rt)
    QTest.keyClick(editor, Qt.Key.Key_Return)
    app.processEvents()  # the delegate commits on Return through the event loop
    assert model.target_list_model.get(target.id).rt == pytest.approx(0.75)
    assert panel.rt_values()[0] == pytest.approx(0.75)


def test_a_choice_in_a_list_combo_box_is_committed_at_once():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target = presenter.target_panel_presenter.add_target_from_view()
    view.show()
    trace_index = panel._proxy_index_of(target.id, TargetTableModel.column_of("trace_type"))

    panel.table_view.edit(trace_index)
    editor = panel.table_view.indexWidget(trace_index)
    editor.setCurrentText(TraceType.TIC.value)
    editor.activated.emit(editor.currentIndex())
    app.processEvents()

    assert model.target_list_model.get(target.id).trace_type is TraceType.TIC
    assert panel.trace_type() is TraceType.TIC
    assert panel.table_view.indexWidget(trace_index) is None


def test_row_header_shows_index_and_corner_sorts_by_it():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target_presenter = presenter.target_panel_presenter
    for _ in range(2):
        target_presenter.select_target(None)
        target_presenter.add_target_from_view()
    table = panel.table_view
    row_header_texts = lambda: [table.model().headerData(row, Qt.Orientation.Vertical) for row in range(2)]
    corner = table.findChild(QAbstractButton)
    assert table.isColumnHidden(TargetTableModel.INDEX_COLUMN)
    assert row_header_texts() == ["1", "2"]

    corner.click()  # sorted by index already: the order turns around
    assert row_header_texts() == ["2", "1"]

    table.sortByColumn(TargetTableModel.column_of("name"), Qt.SortOrder.DescendingOrder)
    corner.click()
    assert row_header_texts() == ["1", "2"]
    assert table.horizontalHeader().sortIndicatorSection() == TargetTableModel.INDEX_COLUMN


def test_mz_cells_are_editable_only_for_xic():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target = presenter.target_panel_presenter.add_target_from_view()
    proxy = panel.table_view.model()
    mz_fields = ("mz", "mz_tolerance", "mz_tolerance_unit")

    edit_cell(panel, target.id, "trace_type", TraceType.TIC.value)

    assert model.target_list_model.get(target.id).trace_type is TraceType.TIC
    assert panel.trace_type() is TraceType.TIC
    for field in mz_fields:
        index = panel._proxy_index_of(target.id, TargetTableModel.column_of(field))
        assert not proxy.flags(index) & Qt.ItemFlag.ItemIsEditable

    edit_cell(panel, target.id, "trace_type", TraceType.XIC.value)

    for field in mz_fields:
        index = panel._proxy_index_of(target.id, TargetTableModel.column_of(field))
        assert proxy.flags(index) & Qt.ItemFlag.ItemIsEditable


def test_name_and_note_take_a_tab_or_line_break_as_a_space():
    # the .rpd.csv export writes them tab-separated, a line a row
    app, view, model, presenter = build_app()
    panel = view.target_panel
    panel.name_edit.insert("PC 30:0\tsn-1\nnote")
    assert panel.name_edit.text() == "PC 30:0 sn-1 note"
    panel.note_edit.insert("a\r\nb")
    assert panel.note_edit.text() == "a  b"
    target = presenter.target_panel_presenter.add_target_from_view()
    assert (target.name, target.note) == ("PC 30:0 sn-1 note", "a  b")

    # the list's editors too
    delegate = panel.table_view.itemDelegate()
    for field in ("name", "note"):
        index = panel._proxy_index_of(target.id, TargetTableModel.column_of(field))
        editor = delegate.createEditor(panel.table_view.viewport(), QStyleOptionViewItem(), index)
        editor.insert("x\ty")
        assert editor.text() == "x y"


def test_unused_cells_are_greyed():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target = presenter.target_panel_presenter.add_target_from_view()
    edit_cell(panel, target.id, "trace_type", TraceType.TIC.value)
    delegate = panel.table_view.itemDelegate()

    def text_color(field):
        option = QStyleOptionViewItem()
        delegate.initStyleOption(option, panel._proxy_index_of(target.id, TargetTableModel.column_of(field)))
        return option.palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Text)

    disabled_text = panel.table_view.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
    assert text_color("mz") == disabled_text
    assert text_color("name") != disabled_text


def test_table_tolerance_unit_switch_keeps_window():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target = presenter.target_panel_presenter.add_target_from_view()
    edit_cell(panel, target.id, "mz", 200.0)
    edit_cell(panel, target.id, "mz_tolerance", 0.01)
    edit_cell(panel, target.id, "rt_tolerance", 0.2)

    edit_cell(panel, target.id, "mz_tolerance_unit", "ppm")

    ppm_target = model.target_list_model.get(target.id)
    assert (ppm_target.mz_tolerance_unit, ppm_target.mz_tolerance) == (ToleranceUnit.PPM, 50)
    assert ppm_target.rt_tolerance == pytest.approx(0.2)
    assert panel.mz_values() == pytest.approx((200.0, 50, ToleranceUnit.PPM))

    edit_cell(panel, target.id, "mz_tolerance_unit", "abs")

    abs_target = model.target_list_model.get(target.id)
    assert abs_target.mz_tolerance_unit is ToleranceUnit.MZ
    assert abs_target.mz_tolerance == pytest.approx(0.01)


def test_table_shows_chosen_columns_and_keeps_the_choice():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    hidden = lambda field: panel.table_view.isColumnHidden(TargetTableModel.column_of(field))
    assert panel.column_settings() == list(TargetTableModel.DEFAULT_SHOWN_FIELDS)
    assert panel.all_check_box.checkState() == Qt.CheckState.PartiallyChecked
    assert hidden("mz_tolerance") and not hidden("trace_type")
    assert panel.table_view.isColumnHidden(TargetTableModel.INDEX_COLUMN)  # in the row header

    panel.all_check_box.click()

    assert [column for column in range(len(TargetTableModel.COLUMNS)) if panel.table_view.isColumnHidden(column)] == [
        TargetTableModel.INDEX_COLUMN
    ]
    assert panel.all_check_box.checkState() == Qt.CheckState.Checked

    panel.all_check_box.click()
    assert panel.column_settings() == []
    assert panel.all_check_box.checkState() == Qt.CheckState.Unchecked

    # from none back to the columns chosen before all
    panel.all_check_box.click()
    assert panel.column_settings() == list(TargetTableModel.DEFAULT_SHOWN_FIELDS)
    assert panel.all_check_box.checkState() == Qt.CheckState.PartiallyChecked

    panel.column_check_boxes["note"].click()
    panel.column_check_boxes["rt_tolerance"].click()
    chosen = panel.column_settings()
    assert hidden("note") and not hidden("rt_tolerance")

    for _ in range(3):  # all, none, and the chosen ones again
        panel.all_check_box.click()
    assert panel.column_settings() == chosen
    assert panel.all_check_box.checkState() == Qt.CheckState.PartiallyChecked

    _app, view, _model, _presenter = build_app(keep_settings=True)
    assert view.target_panel.column_settings() == panel.column_settings()
    assert view.target_panel.table_view.isColumnHidden(TargetTableModel.column_of("note"))

    # the chosen columns are remembered across restarts even while all or none are shown
    view.target_panel.all_check_box.click()  # all
    _app, view, _model, _presenter = build_app(keep_settings=True)
    assert view.target_panel.all_check_box.checkState() == Qt.CheckState.Checked
    view.target_panel.all_check_box.click()  # none
    view.target_panel.all_check_box.click()  # the chosen ones
    assert view.target_panel.column_settings() == chosen

    view.target_panel.default_columns_action.trigger()
    assert view.target_panel.column_settings() == list(TargetTableModel.DEFAULT_SHOWN_FIELDS)
    _app, view, _model, _presenter = build_app(keep_settings=True)
    assert view.target_panel.column_settings() == list(TargetTableModel.DEFAULT_SHOWN_FIELDS)
    assert view.target_panel.partial_column_settings() == list(TargetTableModel.DEFAULT_SHOWN_FIELDS)


ROW_HEADER = None


def _click(panel, row, column, double=False):
    """On a cell, or on the row header (column ROW_HEADER)."""
    table = panel.table_view
    if column is ROW_HEADER:
        header = table.verticalHeader()
        widget = header.viewport()
        pos = QPoint(header.width() // 2, header.sectionViewportPosition(row) + header.sectionSize(row) // 2)
    else:
        widget = table.viewport()
        pos = table.visualRect(table.model().index(row, column)).center()
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=pos)
    if double:
        QTest.mouseDClick(widget, Qt.MouseButton.LeftButton, pos=pos)


def test_click_selects_and_second_click_deselects():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target_presenter = presenter.target_panel_presenter
    ids = []
    for _ in range(2):
        target_presenter.select_target(None)
        ids.append(target_presenter.add_target_from_view().id)
    target_presenter.select_target(None)
    view.show()
    name_column, rt_column = TargetTableModel.column_of("name"), TargetTableModel.column_of("rt")

    _click(panel, 1, ROW_HEADER)
    assert target_presenter.current_id == ids[1]
    assert panel.table_view.currentIndex().column() == name_column
    _click(panel, 1, ROW_HEADER)
    assert target_presenter.current_id is None
    assert not panel.table_view.currentIndex().isValid()

    _click(panel, 0, name_column)
    _click(panel, 0, rt_column)  # another cell of the selected row
    assert target_presenter.current_id == ids[0]
    assert panel.table_view.currentIndex().column() == rt_column
    _click(panel, 0, rt_column)
    assert target_presenter.current_id is None

    _click(panel, 0, name_column)
    below_rows = panel.table_view.visualRect(panel.table_view.model().index(1, name_column)).center()
    below_rows.setY(below_rows.y() + 3 * panel.table_view.rowHeight(1))
    QTest.mouseClick(panel.table_view.viewport(), Qt.MouseButton.LeftButton, pos=below_rows)
    assert target_presenter.current_id is None
    assert not panel.table_view.currentIndex().isValid()  # no frame left behind


def test_keys_do_not_leave_the_row_for_its_header():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target_presenter = presenter.target_panel_presenter
    target = target_presenter.add_target_from_view()
    view.show()
    panel.table_view.setFocus()
    name_index = panel._proxy_index_of(target.id, TargetTableModel.column_of("name"))
    assert panel.table_view.currentIndex() == name_index

    for key in (Qt.Key.Key_Left, Qt.Key.Key_Home):
        QTest.keyClick(panel.table_view, key)

        assert target_presenter.current_id == target.id
        assert panel.table_view.currentIndex() == name_index

    QTest.keyClick(panel.table_view, Qt.Key.Key_Right)
    QTest.keyClick(panel.table_view, Qt.Key.Key_Home)

    assert panel.table_view.currentIndex() == name_index
    assert target_presenter.current_id == target.id


def test_double_click_selects_and_edits():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target_presenter = presenter.target_panel_presenter
    target = target_presenter.add_target_from_view()
    target_presenter.select_target(None)
    view.show()
    name_index = panel._proxy_index_of(target.id, TargetTableModel.column_of("name"))

    _click(panel, 0, name_index.column(), double=True)  # from no selection
    assert target_presenter.current_id == target.id
    assert panel.table_view.indexWidget(name_index) is not None
    panel.table_view.closeEditor(panel.table_view.indexWidget(name_index), QAbstractItemDelegate.EndEditHint.NoHint)

    _click(panel, 0, name_index.column(), double=True)  # on the selected row's current cell
    assert target_presenter.current_id == target.id
    assert panel.table_view.indexWidget(name_index) is not None
    panel.table_view.closeEditor(panel.table_view.indexWidget(name_index), QAbstractItemDelegate.EndEditHint.NoHint)

    _click(panel, 0, ROW_HEADER, double=True)  # on the row header: no edit
    assert target_presenter.current_id == target.id
    assert panel.table_view.indexWidget(name_index) is None


def test_table_ms_level_edit_and_msms_cells():
    app, view, model, presenter = build_app((MSMS_DATA_PATH,))
    panel = view.target_panel
    type_into(panel, "adduct", "[M-H]-")  # the file's scans are negative ions
    target = presenter.target_panel_presenter.add_target_from_view()
    proxy = panel.table_view.model()
    precursor_index = panel._proxy_index_of(target.id, TargetTableModel.column_of("precursor_mz"))
    assert precursor_index.data(TargetTableModel.UNUSED_ROLE)

    edit_cell(panel, target.id, "ms_level", MSLevel.MS2.value)

    # the file's first precursor and CE; shown in the list, and chosen there from those of the file as above it
    ms2_target = model.target_list_model.get(target.id)
    assert (round(ms2_target.precursor_mz, 4), ms2_target.collision_energy) == (673.4814, 0.0)
    assert precursor_index.data() == "673.4814, CE 0"
    assert not precursor_index.data(TargetTableModel.UNUSED_ROLE)
    assert proxy.flags(precursor_index) & Qt.ItemFlag.ItemIsEditable
    ce20 = precursor_index.data(TargetTableModel.PRECURSORS_ROLE).index("673.4814, CE 20")

    edit_cell(panel, target.id, "precursor_mz", ce20)

    ms2_target = model.target_list_model.get(target.id)
    assert (round(ms2_target.precursor_mz, 4), ms2_target.collision_energy) == (673.4814, 20.0)
    assert panel.precursor_button.text() == "673.4814, CE 20"
    # the editor opens on it; closed on it, it changes nothing
    delegate = panel.table_view.itemDelegate()
    editor = delegate.createEditor(panel.table_view.viewport(), QStyleOptionViewItem(), precursor_index)
    delegate.setEditorData(editor, precursor_index)
    assert editor.currentText() == "673.4814, CE 20"
    edits = []
    presenter.target_panel_presenter.table_model.edit_requested.connect(lambda *args: edits.append(args))
    delegate.setModelData(editor, proxy, precursor_index)
    assert edits == []

    edit_cell(panel, target.id, "ms_level", MSLevel.MS1.value)

    ms1_target = model.target_list_model.get(target.id)
    assert (ms1_target.ms_level, ms1_target.precursor_mz, ms1_target.collision_energy) == (MSLevel.MS1, None, None)
    assert panel.ms_level() is MSLevel.MS1


def test_undo_target_edit_restores_plots():
    app, view, model, presenter = build_app()
    stack = presenter.data_stack_presenter
    target = presenter.target_panel_presenter.add_target_from_view()
    view.target_panel.rt_range_control.set_value(0.5)
    assert model.target_list_model.get(target.id).rt == pytest.approx(0.5)

    view.undo_target_edit_action.trigger()

    restored = model.target_list_model.get(target.id)
    assert restored == target
    assert presenter.target_panel_presenter.current_id == target.id
    np.testing.assert_allclose(stack.current_rt_window, target.rt_window())
    assert view.redo_target_edit_action.isEnabled()


def test_state_label_names_what_the_controls_edit():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target_presenter = presenter.target_panel_presenter
    assert panel.state_label.text() == "New target (not added)"

    first = target_presenter.add_target_from_view()
    assert panel.state_label.text() == f"Target #{first.index}: {first.name}"
    edit_cell(panel, first.id, "name", "glucose")
    assert panel.state_label.text() == f"Target #{first.index}: glucose"

    target_presenter.add_target_from_view()
    panel.table_view.selectAll()
    assert panel.state_label.text() == "2 targets selected"
    target_presenter.select_target(None)
    assert panel.state_label.text() == "New target (not added)"


def test_draft_formula_sets_mz_without_adding_a_target():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    stack = presenter.data_stack_presenter

    type_into(panel, "adduct", "[M-H]-")
    type_into(panel, "formula", "C6H12O6")

    assert panel.calculated_mz_label.text() == "179.056112"
    assert panel.mz_values()[0] == pytest.approx(179.056112, abs=1e-6)
    assert np.mean(stack.current_mz_window) == pytest.approx(179.056112, abs=1e-6)
    assert model.target_list_model.targets == ()

    panel.mz_range_control.set_value(179.06)  # by hand
    panel.formula_edit.editingFinished.emit()  # left without typing in it
    assert panel.mz_values()[0] == pytest.approx(179.06)


def test_chemistry_preview_updates_while_typing():
    app, view, model, presenter = build_app()
    panel = view.target_panel

    panel.adduct_edit.setText("[M-H]-")
    panel.formula_edit.setText("C6H12O6")

    assert panel.formula_mass_label.text() == "180.063388"
    assert panel.calculated_mz_label.text() == "179.056112"

    panel.formula_edit.setText("C6H12O6X")

    assert panel.formula_mass_label.text() == ""
    assert panel.calculated_mz_label.text() == ""


def test_add_takes_the_draft_and_leaving_starts_a_new_one():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target_presenter = presenter.target_panel_presenter
    type_into(panel, "name", "glucose")
    type_into(panel, "adduct", "[M-H]-")
    type_into(panel, "formula", "C6H12O6", leave=False)  # still being typed

    target = target_presenter.add_target_from_view()

    assert (target.name, target.formula, target.adduct) == ("glucose", "C6H12O6", "[M-H]-")
    assert target.mz == pytest.approx(179.056112, abs=1e-6)
    assert panel.state_label.text() == f"Target #{target.index}: glucose"

    rt_values = panel.rt_values()
    target_presenter.select_target(None)

    assert panel.draft_texts() == {"name": "", "formula": "", "adduct": "[M-H]-", "note": ""}
    assert panel.rt_values() == pytest.approx(rt_values)
    assert panel.state_label.text() == "New target (not added)"


def test_invalid_draft_is_marked_and_not_added():
    app, view, model, presenter = build_app()
    panel = view.target_panel

    type_into(panel, "formula", "C6H12O6X")

    assert panel.formula_edit.toolTip() != ""
    assert panel.formula_edit.palette().color(QPalette.ColorRole.Base) != panel.palette().color(QPalette.ColorRole.Base)
    assert presenter.target_panel_presenter.add_target_from_view() is None
    assert model.target_list_model.targets == ()


def test_adduct_is_marked_as_it_is_typed():
    app, view, model, presenter = build_app()
    panel = view.target_panel

    def marked():
        return panel.adduct_edit.palette().color(QPalette.ColorRole.Base) != panel.palette().color(QPalette.ColorRole.Base)

    # unfinished, not yet left
    type_into(panel, "adduct", "[M+Na", leave=False)
    assert marked() and panel.adduct_edit.toolTip() != ""
    type_into(panel, "adduct", "[M+Na]+", leave=False)
    assert not marked() and panel.adduct_edit.toolTip() == ""
    # an empty one is the default
    type_into(panel, "adduct", "[M+2H]0+", leave=False)
    assert marked()
    type_into(panel, "adduct", "  ", leave=False)
    assert not marked()


def test_plots_use_the_scans_of_the_adducts_polarity():
    app, view, model, presenter = build_app((DATA_PATH,))  # negative ions only
    panel = view.target_panel
    chromatogram = presenter.data_stack_presenter.child_presenters[0].view.chromatogram_window
    warning = chromatogram.warning_label.textItem.toPlainText

    def points():
        x = chromatogram.plot_items[0].xData
        return 0 if x is None else len(x)

    # none typed: all the scans
    assert warning() == "" and points() > 0
    type_into(panel, "adduct", "[M+H]+")
    assert warning() == "No positive scans in this file, for the adduct's charge"
    assert points() == 0
    type_into(panel, "adduct", "[M-H]-")
    assert warning() == "" and points() > 0
    # a target uses its own
    target = presenter.target_panel_presenter.add_target_from_view()
    model.target_list_model.update(target.id, adduct="[M+Na]+")
    assert points() == 0


def test_polarity_follows_the_adduct_as_it_is_typed():
    app, view, model, presenter = build_app((DATA_PATH,))  # negative ions only
    panel = view.target_panel
    chromatogram = presenter.data_stack_presenter.child_presenters[0].view.chromatogram_window

    def points():
        x = chromatogram.plot_items[0].xData
        return 0 if x is None else len(x)

    drawn = points()
    # not yet left: the sign typed switches the scans; an unfinished adduct keeps them
    type_into(panel, "adduct", "[M+H]+", leave=False)
    assert points() == 0
    type_into(panel, "adduct", "[M+H]", leave=False)
    assert points() == 0
    type_into(panel, "adduct", "[M+H]-", leave=False)
    assert points() == drawn
    # a selected target's typed adduct too, ahead of its commit; its own while the field is empty
    target = presenter.target_panel_presenter.add_target_from_view()
    type_into(panel, "adduct", "[M+Na]+", leave=False)
    assert points() == 0 and model.target_list_model.get(target.id).adduct == "[M+H]-"
    type_into(panel, "adduct", "", leave=False)
    assert points() == drawn


def test_panel_edits_selected_target_identity():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target = presenter.target_panel_presenter.add_target_from_view()
    assert target.name == "New target"

    type_into(panel, "name", "citrate")
    type_into(panel, "formula", "C6H8O7")

    updated = model.target_list_model.get(target.id)
    assert (updated.name, updated.formula) == ("citrate", "C6H8O7")
    assert updated.mz == pytest.approx(ion_mz("C6H8O7", updated.adduct))
    assert panel._proxy_index_of(target.id, TargetTableModel.column_of("name")).data() == "citrate"


def test_panel_takes_the_selection_colour_while_editing_a_target():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    fill = lambda: panel.detail_frame.palette().color(QPalette.ColorRole.Window)
    selection_colour = panel.palette().color(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Highlight)
    # without a target, the text colour, very faint (alpha 25), over the panel's own colour
    window = panel.palette().color(QPalette.ColorRole.Window)
    text = panel.palette().color(QPalette.ColorRole.WindowText)
    no_target_fill = fill()
    assert no_target_fill != window
    expected_lightness = window.lightness() + (text.lightness() - window.lightness()) * 25 / 255
    assert abs(no_target_fill.lightness() - expected_lightness) <= 1

    target = presenter.target_panel_presenter.add_target_from_view()
    assert fill() == selection_colour

    # a Light/Dark switch hands the application a new palette
    original = QApplication.palette()
    switched = QPalette(original)
    switched.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Highlight, QColor("#3a4a62"))
    QApplication.setPalette(switched)
    try:
        app.processEvents()  # the change reaches the widgets through the event loop
        assert fill() == QColor("#3a4a62")
    finally:
        QApplication.setPalette(original)
        app.processEvents()

    presenter.target_panel_presenter.select_target(None)
    assert fill() == no_target_fill


def test_plot_windows_take_the_selection_colour_while_editing_a_target():
    app, view, model, presenter = build_app()
    target_presenter = presenter.target_panel_presenter
    windows = lambda: [
        window
        for child in presenter.data_stack_presenter.child_presenters
        for window in (child.view.chromatogram_window, child.view.spectrum_window)
    ]
    colors = lambda: {window.region0.brush.color().getRgb() for window in windows()}
    assert colors() == {BROWSING_REGION_COLOR}

    target_presenter.add_target_from_view()
    assert colors() == {TARGET_REGION_COLOR}
    assert {line.pen.color().getRgb() for window in windows() for line in window.region0.lines} == {TARGET_REGION_COLOR}
    # under the mouse an edge is drawn opaque, as pyqtgraph drew the former red one
    assert {line.hoverPen.color().getRgb() for window in windows() for line in window.region0.lines} == {
        (*TARGET_REGION_COLOR[:3], 255)
    }

    # a file opened while a target is edited
    presenter.open_d_folders_clicked([str(require(DATA_DIR / "dummy_RP_2min_neg_copy_2.d"))])
    assert colors() == {TARGET_REGION_COLOR}

    target_presenter.select_target(None)
    assert colors() == {BROWSING_REGION_COLOR}


def test_add_with_one_target_selected_duplicates_it():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target_presenter = presenter.target_panel_presenter
    assert panel.add_button.toolTip() == panel.ADD_TOOLTIP
    first = target_presenter.add_target_from_view()
    edit_cell(panel, first.id, "name", "glucose")
    edit_cell(panel, first.id, "note", "a note")
    first = model.target_list_model.get(first.id)
    assert panel.add_button.toolTip() == panel.DUPLICATE_TOOLTIP

    panel.add_button.click()

    copy = model.target_list_model.targets[-1]
    assert copy.id != first.id
    assert copy.index == first.index + 1
    assert replace(copy, id=first.id, index=first.index) == first
    # the copy is selected, and the next add copies it in turn
    assert target_presenter.current_id == copy.id


def test_multiple_selection_edits_them_all():
    app, view, model, presenter = build_app()
    target_presenter = presenter.target_panel_presenter
    target_presenter.add_target_from_view()
    target_presenter.add_target_from_view()

    view.target_panel.table_view.selectAll()

    assert target_presenter.current_id is None
    assert len(target_presenter.batch_ids) == 2
    # copies them all (see test_target_batch_edit)
    assert view.target_panel.add_button.toolTip() == view.target_panel.DUPLICATE_ALL_TOOLTIP
    assert view.target_panel.remove_button.isEnabled()

    target_presenter.remove_selected_targets()

    assert model.target_list_model.targets == ()


def test_remove_selects_target_now_in_its_row_or_the_one_above_at_the_end():
    app, view, model, presenter = build_app()
    target_presenter = presenter.target_panel_presenter
    ids = [target_presenter.add_target_from_view().id for _ in range(3)]

    target_presenter.select_target(ids[0])
    target_presenter.remove_selected_targets()
    assert target_presenter.current_id == ids[1]

    target_presenter.select_target(ids[2])
    target_presenter.remove_selected_targets()
    assert target_presenter.current_id == ids[1]

    target_presenter.remove_selected_targets()
    assert target_presenter.current_id is None


def test_tic_trace_type():
    app, view, model, presenter = build_app()
    stack = presenter.data_stack_presenter

    view.target_panel.trace_type_combo.setCurrentText(TraceType.TIC.value)

    child = stack.child_presenters[0]
    _rt_list, tic = child.model.extract_tic()
    np.testing.assert_allclose(child.view.chromatogram_window.plot_items[0].yData, tic)


def test_precursor_is_chosen_only_for_ms2():
    app, view, model, presenter = build_app((MSMS_DATA_PATH,))
    panel = view.target_panel
    type_into(panel, "adduct", "[M-H]-")  # the file's scans are negative ions
    target = presenter.target_panel_presenter.add_target_from_view()
    assert target.ms_level is MSLevel.ALL
    assert not panel.precursor_button.isEnabled()

    panel.ms_level_combo.setCurrentText(MSLevel.MS2.value)

    assert panel.precursor_button.isEnabled()
    assert model.target_list_model.get(target.id).collision_energy == 0.0

    panel.ms_level_combo.setCurrentText(MSLevel.MS1.value)

    ms1_target = model.target_list_model.get(target.id)
    assert (ms1_target.ms_level, ms1_target.precursor_mz, ms1_target.collision_energy) == (MSLevel.MS1, None, None)
    assert not panel.precursor_button.isEnabled()


def _choose_precursor(panel, collision_energy: float):
    next(action for action in panel._precursor_actions.actions() if f", CE {collision_energy:g} " in action.text()).trigger()


def _n_points(window) -> int:
    x_data = window.plot_items[0].xData
    return 0 if x_data is None else len(x_data)


def test_ms2_plots_the_scans_of_the_chosen_precursor_and_ce():
    app, view, model, presenter = build_app((MSMS_DATA_PATH, DATA_PATH))
    panel = view.target_panel
    msms_child, ms1_child = presenter.data_stack_presenter.child_presenters
    data = msms_child.model

    # the first precursor and CE of the files, until another is chosen
    panel.ms_level_combo.setCurrentText(MSLevel.MS2.value)
    assert panel.precursor_button.text() == "673.4814, CE 0"
    _choose_precursor(panel, 20)

    ce20 = np.flatnonzero((data.MS_level_list == 2) & (data.collision_energy_list == 20.0))
    window = msms_child.view.chromatogram_window
    assert panel.precursor_button.text() == "673.4814, CE 20"
    np.testing.assert_allclose(window.plot_items[0].xData, data.RT_list[ce20])
    assert msms_child.current_scan_index in set(ce20.tolist())
    assert window.top_right_label.toPlainText().startswith("673.4814 / CE 20 > ")
    # a file without these scans shows nothing
    assert _n_points(ms1_child.view.chromatogram_window) == _n_points(ms1_child.view.spectrum_window) == 0


def test_ms2_target_keeps_its_precursor_and_traces_the_product_ion():
    app, view, model, presenter = build_app((MSMS_DATA_PATH,))
    panel = view.target_panel
    type_into(panel, "adduct", "[M-H]-")  # the file's scans are negative ions
    target_presenter = presenter.target_panel_presenter
    stack = presenter.data_stack_presenter
    panel.ms_level_combo.setCurrentText(MSLevel.MS2.value)
    _choose_precursor(panel, 30)
    stack.child_presenters[0].set_mz_window((200.0, 201.0), emit_signal=True)

    target = target_presenter.add_target_from_view()

    assert (target.mz, round(target.precursor_mz, 4), target.collision_energy) == (200.5, 673.4814, 30.0)
    assert target.isolation_width == pytest.approx(1.3)
    target_presenter.select_target(None)
    panel.ms_level_combo.setCurrentText(MSLevel.MS1.value)
    target_presenter.select_target(target.id)
    assert (stack.ms_level, stack.ms2_selection.collision_energy) == (2, 30.0)
    assert panel.precursor_button.text() == "673.4814, CE 30"
    assert panel.mz_values()[0] == pytest.approx(200.5)

    _choose_precursor(panel, 10)

    assert model.target_list_model.get(target.id).collision_energy == 10.0
    assert stack.ms2_selection.collision_energy == 10.0


def test_ms2_formula_chooses_the_isolating_precursor_and_marks_one_that_is_not():
    app, view, model, presenter = build_app((MSMS_DATA_PATH,))
    panel = view.target_panel
    type_into(panel, "adduct", "[M-H]-")  # the file's scans are negative ions
    target_presenter = presenter.target_panel_presenter
    panel.ms_level_combo.setCurrentText(MSLevel.MS2.value)
    _choose_precursor(panel, 20)
    target = target_presenter.add_target_from_view()

    # the isolation window of 673.4814 is 1.3 wide; the CE is kept
    assert target_presenter._chemistry_changes(target, 674.0)["collision_energy"] == 20.0
    assert target_presenter._chemistry_changes(target, 675.0) == {}
    edit_cell(panel, target.id, "formula", "C6H12O6")  # [M-H]- 179.06: outside, the precursor stays

    edited = model.target_list_model.get(target.id)
    assert (edited.formula, edited.collision_energy) == ("C6H12O6", 20.0)
    assert panel._proxy_index_of(target.id, TargetTableModel.column_of("precursor_mz")).data(Qt.ItemDataRole.BackgroundRole) is not None
    assert panel._proxy_index_of(target.id, TargetTableModel.MZ_COLUMN).data(Qt.ItemDataRole.BackgroundRole) is None
    assert panel.precursor_button.palette().color(QPalette.ColorRole.Button) != panel.palette().color(QPalette.ColorRole.Button)


def test_merging_precursors_is_remembered():
    app, view, model, presenter = build_app((MSMS_DATA_PATH,))
    view.target_panel.merge_ppm_spin_box.setValue(20)

    assert model.settings_model.ms2_merge_ppm == 20
    _app, view, model, presenter = build_app((MSMS_DATA_PATH,), keep_settings=True)
    assert view.target_panel.merge_ppm_spin_box.value() == 20
    # one precursor value in this targeted file: nothing to merge
    assert len(view.target_panel._precursor_actions.actions()) == 5


def test_ms_level_filters_scans():
    app, view, model, presenter = build_app((MSMS_DATA_PATH,))
    child = presenter.data_stack_presenter.child_presenters[0]
    ms1_indices = child.model.scan_indices_for_ms_level(1)
    assert len(ms1_indices) == 209
    assert len(child.view.chromatogram_window.plot_items[0].xData) == child.model.N_scans

    view.target_panel.ms_level_combo.setCurrentText(MSLevel.MS1.value)
    view.target_panel.trace_type_combo.setCurrentText(TraceType.TIC.value)

    _rt_list, all_tic = child.model.extract_tic()
    np.testing.assert_allclose(child.view.chromatogram_window.plot_items[0].xData, child.model.RT_list[ms1_indices])
    np.testing.assert_allclose(child.view.chromatogram_window.plot_items[0].yData, all_tic[ms1_indices])
    assert child.current_scan_index in set(ms1_indices.tolist())


def test_search_filters_rows_and_deselects_hidden_target():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target_presenter = presenter.target_panel_presenter
    glucose = target_presenter.add_target_from_view()
    edit_cell(panel, glucose.id, "name", "glucose")
    other = target_presenter.add_target_from_view()
    edit_cell(panel, other.id, "name", "citrate")

    panel.search_edit.setText("GLU")

    assert panel.table_view.model().rowCount() == 1
    assert target_presenter.current_id is None
    panel.table_view.selectRow(0)
    assert target_presenter.current_id == glucose.id

    target_presenter.select_target(other.id)

    assert panel.search_edit.text() == ""
    assert panel.selected_target_ids() == [other.id]


def test_sort_by_mz_is_numeric_and_keeps_model_order():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target_presenter = presenter.target_panel_presenter
    stack = presenter.data_stack_presenter
    ids = []
    for mz_window in ((999.0, 1001.0), (99.0, 101.0), (199.0, 201.0)):
        target_presenter.select_target(None)
        stack.child_presenters[0].set_mz_window(mz_window, emit_signal=True)
        ids.append(target_presenter.add_target_from_view().id)

    mz_column = TargetTableModel.MZ_COLUMN
    panel.table_view.sortByColumn(mz_column, Qt.SortOrder.AscendingOrder)

    proxy = panel.table_view.model()
    assert [proxy.index(row, mz_column).data() for row in range(3)] == ["100.0000", "200.0000", "1000.0000"]
    assert [target.id for target in model.target_list_model.targets] == ids

    panel.table_view.selectRow(0)
    assert target_presenter.current_id == ids[1]

    # back to the order of addition
    panel.table_view.sortByColumn(TargetTableModel.INDEX_COLUMN, Qt.SortOrder.AscendingOrder)
    assert panel.displayed_target_ids() == ids


def test_reset_index_uses_displayed_order():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target_presenter = presenter.target_panel_presenter
    ids = [target_presenter.add_target_from_view().id for _ in range(3)]
    target_presenter.select_target(ids[1])
    target_presenter.remove_selected_targets()
    assert target_presenter.current_id == ids[2]
    assert [target.index for target in model.target_list_model.targets] == [1, 3]

    panel.table_view.sortByColumn(TargetTableModel.INDEX_COLUMN, Qt.SortOrder.DescendingOrder)
    panel.search_edit.setText("no such target")
    panel.reset_index_action.trigger()

    assert panel.search_edit.text() == ""
    assert [model.target_list_model.get(target_id).index for target_id in (ids[0], ids[2])] == [2, 1]
    assert model.target_list_model.next_index() == 3


def test_table_shows_all_columns_and_scrolls_horizontally():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    presenter.target_panel_presenter.add_target_from_view()
    panel.all_check_box.click()
    view.show()
    app.processEvents()

    # all but the index column, whose values are in the row header
    assert [column for column in range(len(TargetTableModel.COLUMNS)) if panel.table_view.isColumnHidden(column)] == [
        TargetTableModel.INDEX_COLUMN
    ]
    proxy = panel.table_view.model()
    headers = [proxy.headerData(column, Qt.Orientation.Horizontal) for column in range(proxy.columnCount())]
    assert headers == [column.header for column in TargetTableModel.COLUMNS]
    assert panel.table_view.horizontalScrollBar().maximum() > 0


def test_view_range_undo_redo():
    app, view, model, presenter = build_app()
    stack = presenter.data_stack_presenter
    stack.record_view_range()
    chromatogram_window = stack.child_presenters[0].view.chromatogram_window
    before = tuple(chromatogram_window.plotItem.viewRange_x())

    chromatogram_window.plotItem.setXRange(0.2, 0.4, padding=0)
    stack.record_view_range()
    assert view.undo_view_range_action.isEnabled()

    view.undo_view_range_action.trigger()
    np.testing.assert_allclose(chromatogram_window.plotItem.viewRange_x(), before)
    np.testing.assert_allclose(stack.child_presenters[1].view.chromatogram_window.plotItem.viewRange_x(), before)

    view.redo_view_range_action.trigger()
    np.testing.assert_allclose(chromatogram_window.plotItem.viewRange_x(), (0.2, 0.4))


def test_mz_is_not_editable_without_xic():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    child = presenter.data_stack_presenter.child_presenters[0]
    before_mz_window = child.current_mz_window

    panel.trace_type_combo.setCurrentText(TraceType.BPC.value)

    assert not panel.mz_range_control.isEnabled()
    assert not child.view.spectrum_window.region0.isVisible()
    child.on_spectrum_clicked(300.0, 0.0)
    assert child.current_mz_window == before_mz_window

    panel.trace_type_combo.setCurrentText(TraceType.XIC.value)

    assert panel.mz_range_control.isEnabled()


def test_default_tolerances():
    app, view, model, presenter = build_app()

    assert view.target_panel.rt_values()[1] == pytest.approx(0.5)
    assert view.target_panel.mz_values() == pytest.approx((150.5, 0.05, ToleranceUnit.MZ))


def test_search_info_button_shows_its_text_until_clicked_again():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    button = panel.search_info_button

    button.click()

    popup = button._popup
    assert popup.isVisible()
    assert popup.label.text().startswith("<b>Search</b>")
    # the whole text, wrapped to the popup's width
    assert popup.label.height() == popup.label.heightForWidth(popup.label.width())
    button.click()
    assert not popup.isVisible()


def test_search_takes_a_query_and_keeps_the_list_while_one_cannot_be_read():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target_presenter = presenter.target_panel_presenter
    stack = presenter.data_stack_presenter
    ids = {}
    for name, mz in (("glucose", 181.07), ("citrate", 191.02), ("PC(16:0/18:1)", 760.59)):
        target_presenter.select_target(None)
        stack.child_presenters[0].set_mz_window((mz - 0.5, mz + 0.5), emit_signal=True)
        ids[name] = target_presenter.add_target_from_view().id
        edit_cell(panel, ids[name], "name", name)

    def shown():
        proxy = panel.table_view.model()
        return [panel._table_model.shown_id(proxy.mapToSource(proxy.index(row, 0)).row()) for row in range(proxy.rowCount())]

    panel.search_edit.setText("PC(16:0/18:1)")
    assert shown() == [ids["PC(16:0/18:1)"]]
    panel.search_edit.setText('name~"^g" OR mz > 700')
    assert shown() == [ids["glucose"], ids["PC(16:0/18:1)"]]
    assert panel.search_edit.toolTip() == ""

    # being typed: the list stays, and the box says why it cannot be read
    panel.search_edit.setText('name~"^g" OR')
    assert shown() == [ids["glucose"], ids["PC(16:0/18:1)"]]
    assert panel.search_edit.toolTip() == "The query ends too soon"

    panel.search_edit.setText('name~"^g" OR name:"cit"')
    assert shown() == [ids["glucose"], ids["citrate"]]
    assert panel.search_edit.toolTip() == ""
    panel.search_edit.clear()
    assert len(shown()) == 3
