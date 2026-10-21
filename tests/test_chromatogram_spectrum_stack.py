import os
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication

from RAPID2_Agilent.models.rpd_model import AgilentDModel
from RAPID2_Agilent.views.data_stack import (
    ChromatogramSpectrumStackPresenter,
    ChromatogramSpectrumStackView,
)


DATA_PATH = Path("tests/data/dummy_RP_2min_neg.d")
MSMS_DATA_PATH = Path("tests/data/MSMS_data/9_PRJ_0-1-2_PA34-1_1uM_n1.d")


def load_model(data_path: Path = DATA_PATH):
    model = AgilentDModel()
    model.load(data_path)
    return model


def build_stack():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)
    view = ChromatogramSpectrumStackView()
    presenter = ChromatogramSpectrumStackPresenter(view)
    child0 = presenter.add_model(load_model(), scan_index=5, mz_window=(150.0, 151.0))
    child1 = presenter.add_model(load_model(), scan_index=15, mz_window=(150.0, 151.0))
    return app, view, presenter, child0, child1


def test_stack_adds_child_windows_in_rows():
    app, view, presenter, child0, child1 = build_stack()

    assert len(view.stack_items) == 2
    assert len(presenter.child_presenters) == 2
    assert view.row_layout.count() == 2
    assert view.stack_items[0].chromatogram_window is child0.view.chromatogram_window
    assert view.stack_items[0].spectrum_window is child0.view.spectrum_window
    assert view.stack_items[0].file_info_panel is child0.view.file_info_panel


def test_stack_shows_file_name_in_info_panels():
    app, view, presenter, child0, child1 = build_stack()

    assert view.stack_items[0].file_info_panel.toPlainText() == f"{DATA_PATH.name}\nNegative MS1 (118)"
    assert view.stack_items[1].file_info_panel.toPlainText() == f"{DATA_PATH.name}\nNegative MS1 (118)"
    assert view.stack_items[0].file_info_panel.file_name_edit.toPlainText() == DATA_PATH.name
    assert view.stack_items[0].file_info_panel.scan_type_edit.toPlainText() == "Negative MS1 (118)"
    assert not view.stack_items[0].file_info_panel.move_to_top_button.icon().isNull()
    assert not view.stack_items[0].file_info_panel.move_up_button.icon().isNull()
    assert not view.stack_items[0].file_info_panel.move_down_button.icon().isNull()
    assert not view.stack_items[0].file_info_panel.move_to_bottom_button.icon().isNull()
    assert view.stack_items[0].file_info_panel.close_button.objectName() == "CloseButton"
    assert view.stack_items[0].file_info_panel.close_button.width() == 27
    assert not view.stack_items[0].file_info_panel.close_button.icon().isNull()
    assert view.stack_items[0].file_info_panel.toolTip() == ""
    assert view.stack_items[0].file_info_panel.file_name_edit.toolTip() == str(DATA_PATH)
    assert view.stack_items[0].file_info_panel.scan_type_edit.toolTip() == ""


def test_agilent_model_returns_scan_type_summary():
    model = load_model()

    assert model.get_scan_type_summary() == [("Negative MS1", 118)]


def test_agilent_model_returns_msms_scan_type_summary():
    model = load_model(MSMS_DATA_PATH)

    assert model.get_scan_type_summary() == [
        ("Negative MS1", 209),
        ("Negative MS2", 1024),
    ]


def test_stack_syncs_initial_windows_when_adding_children():
    app, view, presenter, child0, child1 = build_stack()

    np.testing.assert_allclose(child1.current_rt_window, child0.current_rt_window)
    np.testing.assert_allclose(child1.view.chromatogram_window.get_region(), child0.current_rt_window)
    np.testing.assert_allclose(child1.current_mz_window, child0.current_mz_window)
    np.testing.assert_allclose(child1.view.spectrum_window.get_region(), child0.current_mz_window)


def test_stack_syncs_rt_window_between_children():
    app, view, presenter, child0, child1 = build_stack()
    rt_window = (float(child0.model.RT_list[4]), float(child0.model.RT_list[8]))

    child0.set_rt_window(rt_window, emit_signal=True)

    np.testing.assert_allclose(child1.current_rt_window, rt_window)
    np.testing.assert_allclose(child1.view.chromatogram_window.get_region(), rt_window)


def test_stack_syncs_mz_window_between_children():
    app, view, presenter, child0, child1 = build_stack()
    mz_window = (200.0, 203.0)

    child0.set_mz_window(mz_window, emit_signal=True)

    np.testing.assert_allclose(child1.current_mz_window, mz_window)
    np.testing.assert_allclose(child1.view.spectrum_window.get_region(), mz_window)


def test_stack_syncs_chromatogram_axis_ranges_between_children():
    app, view, presenter, child0, child1 = build_stack()
    x_range = (float(child0.model.RT_list[10]), float(child0.model.RT_list[20]))
    y_range = (0.0, 123.0)
    child0.view.chromatogram_window.set_auto_y_range_enabled(False)
    child1.view.chromatogram_window.set_auto_y_range_enabled(False)

    child0.view.chromatogram_window.plotItem.setRange(xRange=x_range, yRange=y_range, padding=0)

    np.testing.assert_allclose(child1.view.chromatogram_window.plotItem.viewRange_x(), x_range)
    np.testing.assert_allclose(child1.view.chromatogram_window.plotItem.viewRange_y(), y_range)


def test_stack_syncs_spectrum_axis_ranges_between_children():
    app, view, presenter, child0, child1 = build_stack()
    x_range = (100.0, 300.0)
    y_range = (0.0, 456.0)
    child0.view.spectrum_window.set_auto_y_range_enabled(False)
    child1.view.spectrum_window.set_auto_y_range_enabled(False)

    child0.view.spectrum_window.plotItem.setRange(xRange=x_range, yRange=y_range, padding=0)

    np.testing.assert_allclose(child1.view.spectrum_window.plotItem.viewRange_x(), x_range)
    np.testing.assert_allclose(child1.view.spectrum_window.plotItem.viewRange_y(), y_range)


def test_stack_syncs_chromatogram_region_during_drag_without_data_update():
    app, view, presenter, child0, child1 = build_stack()
    rt_region = (float(child0.model.RT_list[7]), float(child0.model.RT_list[12]))
    before_child1_rt_window = child1.current_rt_window

    child0.view.chromatogram_window.region0.blockSignals(True)
    child0.view.chromatogram_window.region0.setRegion(rt_region)
    child0.view.chromatogram_window.region0.blockSignals(False)
    presenter.on_child_chromatogram_region_dragged(child0)

    np.testing.assert_allclose(child1.view.chromatogram_window.get_region(), rt_region)
    np.testing.assert_allclose(child1.current_rt_window, before_child1_rt_window)


def test_stack_syncs_spectrum_region_during_drag_without_data_update():
    app, view, presenter, child0, child1 = build_stack()
    mz_region = (210.0, 214.0)
    before_child1_mz_window = child1.current_mz_window

    child0.view.spectrum_window.region0.blockSignals(True)
    child0.view.spectrum_window.region0.setRegion(mz_region)
    child0.view.spectrum_window.region0.blockSignals(False)
    presenter.on_child_spectrum_region_dragged(child0)

    np.testing.assert_allclose(child1.view.spectrum_window.get_region(), mz_region)
    np.testing.assert_allclose(child1.current_mz_window, before_child1_mz_window)
