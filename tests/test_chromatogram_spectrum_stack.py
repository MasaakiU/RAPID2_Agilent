import os
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from RAPID2_Agilent.models.rpd_model import AgilentDModel
from RAPID2_Agilent.views.data_stack import (
    ChromatogramSpectrumStackPresenter,
    ChromatogramSpectrumStackView,
)
from RAPID2_Agilent.views.data_view.chromatogram_spectrum_view import ChromatogramWindow
from RAPID2_Agilent.views.view_constants import OWN_RT_REGION_COLOR, TARGET_REGION_COLOR
from tests.data_paths import DATA_DIR, require


DATA_PATH = DATA_DIR / "dummy_RP_2min_neg.d"
MSMS_DATA_PATH = DATA_DIR / "MSMS_data" / "9_PRJ_0-1-2_PA34-1_1uM_n1.d"


def load_model(data_path: Path = DATA_PATH):
    model = AgilentDModel()
    model.load(require(data_path))
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


def test_chromatogram_warns_of_an_xic_window_narrower_than_the_data_spacing():
    app, view, presenter, child0, child1 = build_stack()
    warning = child0.view.chromatogram_window.warning_label.textItem.toPlainText

    assert warning() == ""
    presenter.set_mz_window((150.5, 150.5))  # between two data points in every scan
    assert warning().startswith("m/z window narrower than the data spacing")
    assert warning().endswith(f"{len(child0.scan_indices())}/{len(child0.scan_indices())} scans have no point in it")
    presenter.set_mz_window((150.0, 151.0))
    assert warning() == ""


def test_xic_leaves_out_scans_whose_data_points_miss_the_window_and_joins_across_them_dashed():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)
    presenter = ChromatogramSpectrumStackPresenter(ChromatogramSpectrumStackView())
    model = load_model(DATA_DIR / "9_P2-8-15_spl1_p0.d")
    mz = 104.10699  # choline, where the data points lie 7.65 ppm apart
    child = presenter.add_model(model, mz_window=(mz * (1 - 1e-6), mz * (1 + 1e-6)))
    window = child.view.chromatogram_window
    y = window.plot_items[0].yData
    missing = np.isnan(y)

    assert 0 < np.count_nonzero(missing) < len(y)
    # the dashed line joins the measured points on either side of each run of missing ones; none past the ends
    bridge_x, bridge_y = window.bridge_item.getData()
    edges = np.diff(np.concatenate(([0], missing.astype(int), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    assert len(bridge_x) == 2 * np.count_nonzero((starts > 0) & (stops < len(y)))
    assert np.isin(bridge_x, window.plot_items[0].xData[~missing]).all()
    assert np.isfinite(bridge_y).all()
    # the y range from the measured points
    assert window.get_displayed_y_bounds(*window.viewRange()[0])[0] is not None

    presenter.set_mz_window((104.0, 104.2))
    assert not np.isnan(window.plot_items[0].yData).any()
    bridge_x = window.bridge_item.getData()[0]
    assert bridge_x is None or len(bridge_x) == 0

    presenter.set_mz_window((5000.0, 5000.001))
    assert np.isnan(window.plot_items[0].yData).all()
    assert window.warning_label.textItem.toPlainText() == "m/z window beyond the scanned m/z range"


def test_turning_back_to_the_polarity_draws_the_last_plots_again_without_extracting(monkeypatch):
    app, view, presenter, child0, child1 = build_stack()  # negative ions only
    chromatogram, spectrum = child0.view.chromatogram_window, child0.view.spectrum_window
    drawn = chromatogram.plot_items[0].yData.copy(), spectrum.plot_items[0].yData.copy()
    extractions = []
    monkeypatch.setattr(child0.model, "extract_chromatogram", lambda *a, **k: extractions.append(a) or (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(child0.model, "extract_averaged_spectrum", lambda *a, **k: extractions.append(a) or (_ for _ in ()).throw(AssertionError))

    presenter.set_polarity(-1)  # its own: the same scans
    presenter.set_polarity(1)
    assert chromatogram.plot_items[0].yData is None or len(chromatogram.plot_items[0].yData) == 0
    presenter.set_polarity(-1)

    assert extractions == []
    assert np.array_equal(chromatogram.plot_items[0].yData, drawn[0])
    assert np.array_equal(spectrum.plot_items[0].yData, drawn[1])


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


def drag_region(window, region):
    """As the user drags the window's region to ``region``, without dropping it."""
    window.region0.blockSignals(True)
    window.region0.setRegion(region)
    window.region0.blockSignals(False)
    window.region0.sigRegionChanged.emit(window.region0)


def test_stack_shows_dragged_windows_in_labels_without_data_update():
    app, view, presenter, child0, child1 = build_stack()
    rt_region = (float(child0.model.RT_list[7]), float(child0.model.RT_list[12]))
    mz_region = (210.0, 214.0)
    before_windows = [(child.current_rt_window, child.current_mz_window) for child in (child0, child1)]
    dragged = []
    presenter.rt_window_dragged.connect(lambda *window: dragged.append(("rt", window)))
    presenter.mz_window_dragged.connect(lambda *window: dragged.append(("mz", window)))

    drag_region(child0.view.chromatogram_window, rt_region)
    drag_region(child1.view.spectrum_window, mz_region)

    assert dragged == [("rt", rt_region), ("mz", mz_region)]
    for child in (child0, child1):
        assert child.view.spectrum_window.range_label_text == f"{rt_region[0]:.4f} - {rt_region[1]:.4f}"
        assert child.view.chromatogram_window.range_label_text == "210.0000 - 214.0000"
    assert [(child.current_rt_window, child.current_mz_window) for child in (child0, child1)] == before_windows


def test_stack_does_not_signal_a_drag_for_windows_set_by_code():
    app, view, presenter, child0, child1 = build_stack()
    dragged = []
    presenter.rt_window_dragged.connect(lambda *window: dragged.append(window))
    presenter.mz_window_dragged.connect(lambda *window: dragged.append(window))

    presenter.set_rt_window((float(child0.model.RT_list[7]), float(child0.model.RT_list[12])))
    presenter.set_mz_window((210.0, 214.0))
    child1.on_chromatogram_clicked(float(child1.model.RT_list[20]), 0.0)
    child1.on_spectrum_clicked(220.0, 0.0)

    assert dragged == []


def test_files_can_be_given_rt_windows_of_their_own():
    app, view, presenter, child0, child1 = build_stack()
    rt_window = (float(child0.model.RT_list[7]), float(child0.model.RT_list[12]))
    own = (float(child1.model.RT_list[20]), float(child1.model.RT_list[25]))

    presenter.set_rt_window(rt_window, {child1.model: own})

    assert presenter.current_rt_window == rt_window
    np.testing.assert_allclose(child0.current_rt_window, rt_window)
    np.testing.assert_allclose(child1.view.chromatogram_window.get_region(), own)


def test_while_a_target_is_edited_unlinked_each_file_keeps_its_own_rt_window():
    app, view, presenter, child0, child1 = build_stack()
    shared = presenter.current_rt_window
    own = (float(child0.model.RT_list[4]), float(child0.model.RT_list[8]))
    signalled = []
    presenter.rt_window_changed.connect(lambda *window: signalled.append(("all", window)))
    presenter.rt_window_dragged.connect(lambda *window: signalled.append(("dragged", window)))
    presenter.run_rt_window_changed.connect(lambda model, *window: signalled.append((model, window)))
    presenter.set_link_rt_windows(False)
    presenter.set_editing_target(True)
    window = child0.view.chromatogram_window

    drag_region(window, own)
    window.region0.sigRegionChangeFinished.emit(window.region0)

    assert signalled == [(child0.model, own)]
    np.testing.assert_allclose(child0.current_rt_window, own)
    np.testing.assert_allclose(child1.current_rt_window, shared)
    np.testing.assert_allclose(child1.view.chromatogram_window.get_region(), shared)
    assert presenter.current_rt_window == shared
    # each spectrum is of its own file's window
    assert child0.view.spectrum_window.range_label_text == f"{own[0]:.4f} - {own[1]:.4f}"
    assert child1.view.spectrum_window.range_label_text == f"{shared[0]:.4f} - {shared[1]:.4f}"
    # the files' own RT windows red, the m/z windows as a target's; the peaks found in a file shaded in it
    assert child0.view.chromatogram_window.region0.brush.color().getRgb() == OWN_RT_REGION_COLOR
    assert child0.view.spectrum_window.region0.brush.color().getRgb() == TARGET_REGION_COLOR
    presenter.set_peak_areas({child1.model: [own]})
    assert [len(child.view.chromatogram_window.peak_area_items) for child in (child0, child1)] == [0, 1]

    presenter.set_link_rt_windows(True)

    np.testing.assert_allclose(child0.current_rt_window, shared)
    assert child0.view.chromatogram_window.region0.brush.color().getRgb() == TARGET_REGION_COLOR
    assert child1.view.chromatogram_window.peak_area_items == []


def test_a_peaks_area_is_shaded_under_the_points_measured_from_its_start_to_its_end():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)
    window = ChromatogramWindow()
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    window.set_primary_plot_data(x, np.array([0.0, 2, 5, 9, 4, 1]), missing=np.array([False, False, True, False, False, False]))

    window.set_peak_areas([(1.0, 4.0), (4.5, 5.0)])

    # the point not measured left out, as the area leaves it out; none for a peak of fewer than 2 points
    assert len(window.peak_area_items) == 1
    np.testing.assert_allclose(window.peak_area_items[0].xData, [1, 3, 4])
    assert window.peak_area_items[0].opts["fillLevel"] == 0
    # under a trace drawn again
    window.set_primary_plot_data(x, np.ones(6))
    np.testing.assert_allclose(window.peak_area_items[0].xData, [1, 2, 3, 4])
    np.testing.assert_allclose(window.peak_area_items[0].yData, [1, 1, 1, 1])
    window.set_peak_areas([])
    assert window.peak_area_items == []


def test_the_fit_of_a_peak_is_drawn_with_its_label_at_its_top():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)
    window = ChromatogramWindow()
    window.set_primary_plot_data(np.arange(6.0), np.array([0.0, 2, 5, 9, 4, 1]))

    window.set_peak_fits([((1.0, 2.0, 3.0), (0.0, 5.0, 1.0), "12 ± 1"), ((4.0,), (1.0,), "of one point")])

    line, label = window.peak_fit_items  # none of a fit of one point
    np.testing.assert_allclose(line.xData, [1, 2, 3])
    assert (label.pos().x(), label.pos().y(), label.textItem.toPlainText()) == (2.0, 5.0, "12 ± 1")
    window.set_peak_fits([])
    assert window.peak_fit_items == []


def build_stack_of(count):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)
    view = ChromatogramSpectrumStackView()
    presenter = ChromatogramSpectrumStackPresenter(view)
    children = [presenter.add_model(load_model()) for _ in range(count)]
    return app, view, presenter, children


def order_of(presenter, children):
    """The row of each child, as its index in children; checks that the view keeps the presenter's order."""
    assert [child.view for child in presenter.child_presenters] == presenter.view.stack_items
    layout = presenter.view.row_layout
    assert [layout.itemAt(row).widget() for row in range(layout.count())] == presenter.view.stack_items
    return [children.index(child) for child in presenter.child_presenters]


def select(view, children, indices):
    view.set_selected_items([children[index].view for index in indices])


def test_stack_moves_selected_files_up_and_down():
    app, view, presenter, children = build_stack_of(5)
    select(view, children, [1, 3])
    models_changed = []
    presenter.models_changed.connect(lambda: models_changed.append(True))

    presenter.move_selected_up()
    assert order_of(presenter, children) == [1, 0, 3, 2, 4]
    presenter.move_selected_up()
    # the one at the top stays, the other goes on up to right below it
    assert order_of(presenter, children) == [1, 3, 0, 2, 4]
    presenter.move_selected_up()
    assert order_of(presenter, children) == [1, 3, 0, 2, 4]
    assert len(models_changed) == 2

    presenter.move_selected_down()
    assert order_of(presenter, children) == [0, 1, 3, 2, 4]
    assert presenter.models == [children[index].model for index in [0, 1, 3, 2, 4]]
    # the selection moves with the files
    assert view.selected_items == [children[1].view, children[3].view]


def test_stack_moves_selected_files_to_top_and_bottom():
    app, view, presenter, children = build_stack_of(4)
    select(view, children, [2, 0])

    presenter.move_selected_to_bottom()
    assert order_of(presenter, children) == [1, 3, 0, 2]
    presenter.move_selected_to_top()
    assert order_of(presenter, children) == [0, 2, 1, 3]


def test_stack_closes_selected_files_and_selects_the_next():
    app, view, presenter, children = build_stack_of(5)
    rt_window = presenter.current_rt_window
    select(view, children, [1, 2])

    closed = presenter.close_selected()

    assert closed == [children[1].model, children[2].model]
    assert order_of(presenter, children) == [0, 3, 4]
    # the one right below the closed ones, from which a Shift+click selects
    assert view.selected_items == [children[3].view]
    view.on_item_clicked(children[0].view, Qt.KeyboardModifier.ShiftModifier)
    assert view.selected_items == [children[0].view, children[3].view]
    view.select_item(children[3].view)

    presenter.close_selected()
    assert order_of(presenter, children) == [0, 4]
    assert view.selected_items == [children[4].view]
    presenter.close_selected()
    # none below: the bottom one
    assert order_of(presenter, children) == [0]
    assert view.selected_items == [children[0].view]

    presenter.close_selected()
    assert presenter.child_presenters == []
    assert view.stack_items == []
    assert view.selected_items == []
    assert presenter.close_selected() == []
    assert not presenter.can_undo_view_range()
    # the windows stay for the next file
    assert presenter.current_rt_window == rt_window
    child = presenter.add_model(load_model())
    np.testing.assert_allclose(child.current_rt_window, rt_window)
