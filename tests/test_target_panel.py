import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from RAPID2_Agilent.models.main_model import MainModel
from RAPID2_Agilent.models.target_model import MSLevel, ToleranceUnit, TraceType
from RAPID2_Agilent.presenters.main_presenter import MainPresenter
from RAPID2_Agilent.views.main_window import MainWindow
from RAPID2_Agilent.views.target_panel import TargetTableModel


DATA_PATH = Path("tests/data/dummy_RP_2min_neg.d")
MSMS_DATA_PATH = Path("tests/data/MSMS_data/9_PRJ_0-1-2_PA34-1_1uM_n1.d")


def build_app(data_paths=(DATA_PATH, DATA_PATH)):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)
    view = MainWindow()
    model = MainModel()
    presenter = MainPresenter(view, model)
    presenter.open_d_folders_clicked([str(path) for path in data_paths])
    return app, view, model, presenter


def edit_field(panel, field_name: str, text: str):
    line_edit = panel.field_edits[field_name]
    line_edit.setText(text)
    line_edit.editingFinished.emit()


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

    edit_field(panel, "adduct", "[M-H]-")
    edit_field(panel, "formula", "C6H12O6")

    updated = model.target_list_model.get(target.id)
    assert updated.mz == pytest.approx(179.056112, abs=1e-6)
    assert panel.calculated_mz_label.text() == "179.056112"
    assert panel.mz_range_control.center_spin_box.styleSheet() == ""
    np.testing.assert_allclose(presenter.data_stack_presenter.current_mz_window, updated.mz_window())

    panel.mz_range_control.set_value(179.06)

    assert panel.mz_range_control.center_spin_box.styleSheet() != ""


def test_invalid_field_is_marked_and_not_committed():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target = presenter.target_panel_presenter.add_target_from_view()

    edit_field(panel, "adduct", "M+H")

    assert model.target_list_model.get(target.id) == target
    assert panel.adduct_edit.styleSheet() != ""
    assert "Invalid adduct" in panel.adduct_edit.toolTip()


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


def test_multiple_selection_is_viewer_mode():
    app, view, model, presenter = build_app()
    target_presenter = presenter.target_panel_presenter
    target_presenter.add_target_from_view()
    target_presenter.add_target_from_view()

    view.target_panel.table_view.selectAll()

    assert target_presenter.current_id is None
    assert not view.target_panel.name_edit.isEnabled()
    assert not view.target_panel.duplicate_button.isEnabled()
    assert view.target_panel.remove_button.isEnabled()

    target_presenter.remove_selected_targets()

    assert model.target_list_model.targets == ()


def test_tic_trace_type():
    app, view, model, presenter = build_app()
    stack = presenter.data_stack_presenter

    view.target_panel.trace_type_combo.setCurrentText(TraceType.TIC.value)

    child = stack.child_presenters[0]
    _rt_list, tic = child.model.extract_tic()
    np.testing.assert_allclose(child.view.chromatogram_window.plot_items[0].yData, tic)


def test_msms_fields_only_for_ms2_targets():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target_presenter = presenter.target_panel_presenter
    target = target_presenter.add_target_from_view()
    assert target.ms_level is MSLevel.ALL
    assert not panel.product_ion_edit.isEnabled()

    ms2_target = model.target_list_model.update(target.id, ms_level=MSLevel.MS2, product_ion=89.0, collision_energy=20.0)
    assert panel.product_ion_edit.isEnabled()
    with pytest.raises(NotImplementedError):
        target_presenter._apply_target_to_plots(ms2_target)

    panel.ms_level_combo.setCurrentText(MSLevel.MS1.value)

    ms1_target = model.target_list_model.get(target.id)
    assert (ms1_target.ms_level, ms1_target.product_ion, ms1_target.collision_energy) == (MSLevel.MS1, None, None)
    assert not panel.product_ion_edit.isEnabled()


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
    edit_field(panel, "name", "glucose")
    other = target_presenter.add_target_from_view()
    edit_field(panel, "name", "citrate")

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
    assert target_presenter.current_id is None
    assert [target.index for target in model.target_list_model.targets] == [1, 3]

    panel.table_view.sortByColumn(TargetTableModel.INDEX_COLUMN, Qt.SortOrder.DescendingOrder)
    panel.search_edit.setText("no such target")
    panel.reset_index_button.click()

    assert panel.search_edit.text() == ""
    assert [model.target_list_model.get(target_id).index for target_id in (ids[0], ids[2])] == [2, 1]
    assert model.target_list_model.next_index() == 3


def test_table_shows_all_columns_and_scrolls_horizontally():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    presenter.target_panel_presenter.add_target_from_view()
    view.show()
    app.processEvents()

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


def test_chemistry_preview_updates_while_typing():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target = presenter.target_panel_presenter.add_target_from_view()

    panel.adduct_edit.setText("[M-H]-")
    panel.formula_edit.setText("C6H12O6")

    assert panel.formula_mass_label.text() == "180.063388"
    assert panel.calculated_mz_label.text() == "179.056112"
    assert model.target_list_model.get(target.id) == target  # not committed yet

    panel.formula_edit.setText("C6H12O6X")

    assert panel.formula_mass_label.text() == ""
    assert panel.calculated_mz_label.text() == ""


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
