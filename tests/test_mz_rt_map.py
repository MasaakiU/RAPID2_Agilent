import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QThreadPool, Qt
from PyQt6.QtWidgets import QApplication

from RAPID2_Agilent.models.main_model import MainModel
from RAPID2_Agilent.models.mz_rt_map import MID_FACTOR, MapLayer, MzRtMap, rasterize
from RAPID2_Agilent.models.rpd_model import AgilentDModel
from RAPID2_Agilent.models.target_model import ToleranceUnit, TraceType
from RAPID2_Agilent.presenters.main_presenter import MainPresenter
from RAPID2_Agilent.views.main_window import MainWindow
from RAPID2_Agilent.views.mz_rt_map.mz_rt_map_presenter import intensity_ticks


DATA_PATH = Path("tests/data/dummy_RP_2min_neg.d")


def load_model() -> AgilentDModel:
    model = AgilentDModel()
    model.load(DATA_PATH)
    return model


def reference_max_bins(mz, intensity, edges):
    bins = np.searchsorted(edges, mz, side="right") - 1
    inside = (bins >= 0) & (bins < len(edges) - 1)
    out = np.zeros(len(edges) - 1, dtype=np.float32)
    np.maximum.at(out, bins[inside], intensity[inside].astype(np.float32))
    return out


def test_native_max_into_bins_matches_numpy():
    model = load_model()
    scan_index = 10
    mz, intensity = model.extract_spectrum(scan_index)
    edges = 100.0 * np.exp(np.arange(20000) * np.log1p(1e-4))
    reference = reference_max_bins(mz, intensity, edges)

    out = np.zeros(len(edges) - 1, dtype=np.float32)
    model.max_into_bins(scan_index, edges, out)
    np.testing.assert_array_equal(out, reference)

    sub_edges = edges[5000:5101]
    sub_out = np.zeros(100, dtype=np.float32)
    model.max_into_bins(scan_index, sub_edges, sub_out)
    np.testing.assert_array_equal(sub_out, reference[5000:5100])


def build_map() -> MzRtMap:
    mz_rt_map = MzRtMap(load_model())
    assert mz_rt_map.build(lambda done, total: None, lambda: False)
    return mz_rt_map


def test_levels_are_nested_max():
    mz_rt_map = build_map()
    mid = mz_rt_map.mid.intensity
    starts = np.arange(0, mid.shape[1], mz_rt_map.overview_factor)
    np.testing.assert_array_equal(mz_rt_map.overview.intensity, np.maximum.reduceat(mid, starts, axis=1))

    mid_start, mid_stop = 3000, 3010
    detail = mz_rt_map.detail(
        mz_rt_map.mid_grid.edges(mid_start, mid_stop)[0] * (1 + 1e-9),
        mz_rt_map.mid_grid.edges(mid_start, mid_stop)[-1] * (1 - 1e-9),
        lambda: False,
    )
    pooled = np.maximum.reduceat(detail.intensity, np.arange(0, detail.intensity.shape[1], MID_FACTOR), axis=1)
    np.testing.assert_array_equal(pooled, mid[:, mid_start:mid_stop])


def test_cancelled_build_returns_false():
    mz_rt_map = MzRtMap(load_model())

    assert not mz_rt_map.build(lambda done, total: None, lambda: True)
    assert mz_rt_map.mid is None


def test_rasterize_takes_max_per_pixel():
    layer = MapLayer(
        rt=np.array([0.0, 1.0, 2.0, 3.0]),
        mz_edges=np.array([100.0, 101.0, 102.0, 103.0]),
        intensity=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32),
        bin_ppm=0.0,
    )

    np.testing.assert_array_equal(rasterize(layer, (0.0, 4.0), (100.0, 103.0), (2, 1)), [[6], [12]])
    # pixels smaller than cells show the cell under them; pixels outside the data are empty
    np.testing.assert_array_equal(rasterize(layer, (-1.0, 1.0), (100.0, 101.0), (2, 2)), [[0, 0], [1, 1]])
    assert not rasterize(layer, (10.0, 11.0), (100.0, 103.0), (3, 3)).any()


@pytest.mark.parametrize("scale", [1.0, 543.0, 2e5, 1e8])
def test_intensity_ticks_are_round_spaced_and_on_the_bar(scale):
    max_intensity = 487606.0
    top = np.arcsinh(max_intensity / scale)

    ticks = intensity_ticks(scale, max_intensity)

    positions = np.array([position for position, _label in ticks])
    labels = [label for _position, label in ticks]
    assert labels[0] == "0" and positions[0] == 0.0
    assert (np.diff(positions) >= top * 0.08 - 1e-12).all()
    assert positions[-1] <= top
    assert len(ticks) >= 4
    np.testing.assert_allclose(positions, np.arcsinh(np.array([float(label) for label in labels]) / scale))


def test_intensity_ticks_labels():
    labels = [label for _position, label in intensity_ticks(543.0, 487606.0)]

    assert labels == ["0", "1000", "2000", "5000", "1e4", "2e4", "5e4", "1e5", "2e5"]
    # nearly linear: the other multiples fill in
    assert [label for _position, label in intensity_ticks(2e5, 487606.0)] == ["0", "5e4", "1e5", "2e5", "3e5", "4e5"]


def build_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)
    view = MainWindow()
    model = MainModel()
    presenter = MainPresenter(view, model)
    presenter.open_d_folders_clicked([str(DATA_PATH), str(DATA_PATH)])
    return app, view, model, presenter


def wait_for_tasks(app):
    for _ in range(3):
        while QThreadPool.globalInstance().activeThreadCount():
            app.processEvents()
            time.sleep(0.01)
        for _ in range(20):  # queued signals and the range timer
            app.processEvents()
            time.sleep(0.01)


def test_map_window_builds_and_computes_detail_on_zoom():
    app, view, model, presenter = build_app()
    map_presenter = presenter.mz_rt_map_presenter
    window = view.mz_rt_map_window

    view.mz_rt_map_requested.emit()
    wait_for_tasks(app)

    assert window.file_combo.count() == 2
    assert window.file_combo.currentText() == f"1: {DATA_PATH.name}"
    assert window.status_label.text().endswith("ppm bins")
    assert window.scale_spin_box.value() > 0

    window.view_box.setRange(xRange=(0.5, 1.0), yRange=(200.0, 201.0), padding=0)
    wait_for_tasks(app)

    assert window.status_label.text() == "10 ppm bins"
    assert map_presenter._detail_layers[0].mz_edges[0] <= 200.0


def test_click_jumps_without_editing_selected_target():
    app, view, model, presenter = build_app()
    stack = presenter.data_stack_presenter
    target = presenter.target_panel_presenter.add_target_from_view()

    view.mz_rt_map_window.clicked.emit(1.2, 300.0)

    assert presenter.target_panel_presenter.current_id is None
    assert model.target_list_model.get(target.id) == target
    np.testing.assert_allclose(stack.current_rt_window, (0.7, 1.7))
    np.testing.assert_allclose(stack.current_mz_window, (299.95, 300.05))
    x_min, x_max = stack.child_presenters[0].view.spectrum_window.plotItem.viewRange_x()
    assert x_min <= 300.0 <= x_max


def test_click_keeps_mz_without_xic():
    app, view, model, presenter = build_app()
    stack = presenter.data_stack_presenter
    view.target_panel.trace_type_combo.setCurrentText(TraceType.TIC.value)
    before_mz_window = stack.current_mz_window

    view.mz_rt_map_window.clicked.emit(1.2, 300.0)

    assert stack.current_mz_window == before_mz_window
    np.testing.assert_allclose(stack.current_rt_window, (0.7, 1.7))


def test_marker_follows_current_windows_and_can_be_hidden():
    app, view, model, presenter = build_app()
    window = view.mz_rt_map_window
    presenter.mz_rt_map_presenter.show()

    view.target_panel.mz_range_control.set_value(250.0)

    rt_window = presenter.data_stack_presenter.current_rt_window
    mz_window = presenter.data_stack_presenter.current_mz_window
    np.testing.assert_allclose(window.marker.pos(), (rt_window[0], mz_window[0]))
    np.testing.assert_allclose(window.marker.size(), (rt_window[1] - rt_window[0], mz_window[1] - mz_window[0]))
    assert len(window.marker.getHandles()) == 8
    assert window.marker.isVisible()
    window.show_marker_check_box.setChecked(False)
    assert not window.marker.isVisible()
    wait_for_tasks(app)


def test_dropped_marker_jumps_inside_the_view():
    app, view, model, presenter = build_app()
    stack = presenter.data_stack_presenter
    window = view.mz_rt_map_window
    view.mz_rt_map_requested.emit()
    wait_for_tasks(app)
    (rt_min, rt_max), (mz_min, mz_max) = window.visible_range()

    window.marker.setPos((0.7, 299.95))

    np.testing.assert_allclose(stack.current_rt_window, (0.7, 1.7))
    np.testing.assert_allclose(stack.current_mz_window, (299.95, 300.05))

    window.marker.setPos((rt_max + 10.0, mz_min - 10.0))  # dropped outside the plot

    assert sum(stack.current_rt_window) / 2 == pytest.approx(rt_max)
    assert sum(stack.current_mz_window) / 2 == pytest.approx(mz_min)


def test_scale_slider_and_spin_box_stay_in_sync():
    app, view, model, presenter = build_app()
    map_presenter = presenter.mz_rt_map_presenter
    window = view.mz_rt_map_window
    view.mz_rt_map_requested.emit()
    wait_for_tasks(app)
    mz_rt_map = map_presenter._maps[map_presenter.current_model]
    scale_min, scale_max = mz_rt_map.scale_range()

    window.scale_slider.setValue(window.scale_slider.minimum())
    assert window.scale_spin_box.value() == pytest.approx(max(scale_min, 0.1), rel=1e-2, abs=0.1)
    window.scale_slider.setValue(window.scale_slider.maximum())
    assert window.scale_spin_box.value() == pytest.approx(scale_max, rel=1e-2)
    assert map_presenter._scales[map_presenter.current_model] == window.scale_spin_box.value()
    expected = np.arcsinh(map_presenter._raster[0] / window.scale_spin_box.value())
    np.testing.assert_allclose(window.image_item.image, expected, rtol=1e-6)

    window.scale_spin_box.setValue(scale_max * 100)  # beyond the slider
    assert window.scale_slider.value() == window.scale_slider.maximum()
    assert map_presenter._scales[map_presenter.current_model] == pytest.approx(scale_max * 100)


def test_color_map_can_be_changed_and_has_previews():
    app, view, model, presenter = build_app()
    window = view.mz_rt_map_window
    combo = window.color_map_combo
    before = window.image_item.lut

    assert all(not combo.itemIcon(index).isNull() for index in range(combo.count()))
    combo.setCurrentText("gray")

    lut = window.image_item.lut.astype(int)
    assert not np.array_equal(lut, before)
    assert (lut.max(axis=1) - lut.min(axis=1) <= 1).all()  # gray up to rounding in the color map data
    assert window.color_bar.colorMap().name == "CET-L1"


def test_color_bar_follows_levels_and_scale():
    app, view, model, presenter = build_app()
    map_presenter = presenter.mz_rt_map_presenter
    window = view.mz_rt_map_window
    view.mz_rt_map_requested.emit()
    wait_for_tasks(app)
    max_intensity = map_presenter._maps[map_presenter.current_model].max_intensity()

    for scale in (window.scale_spin_box.value(), 2e5):
        window.scale_spin_box.setValue(scale)
        top = np.arcsinh(max_intensity / scale)
        np.testing.assert_allclose(window.color_bar.levels(), (0.0, top))
        np.testing.assert_allclose(window.image_item.getLevels(), (0.0, top))
        assert window.color_bar.axis._tickLevels == [intensity_ticks(scale, max_intensity)]


def test_control_panel_can_be_collapsed_but_not_the_map():
    app, view, model, presenter = build_app()
    window = view.mz_rt_map_window
    splitter = window.splitter

    assert splitter.widget(0) is window.control_panel and splitter.widget(1) is window.plot_splitter
    assert splitter.isCollapsible(0) and not splitter.isCollapsible(1)
    splitter.setSizes([0, 900])
    assert splitter.sizes()[0] == 0


def test_rt_alignment_of_a_file_with_itself_is_the_identity():
    pytest.importorskip("scipy")
    from RAPID2_Agilent.models.mz_rt_diff import align_maps

    mz_rt_map = build_map()
    alignment = align_maps(mz_rt_map, build_map(), lambda: False)

    assert alignment.max_shift(mz_rt_map.rt) < 1e-9
    assert alignment.normalization == 1.0
    # each scan of A falls on the same scan of B
    np.testing.assert_allclose(alignment.rows + alignment.weights, np.arange(len(mz_rt_map.rt)), atol=1e-6)


def test_difference_of_identical_files_is_zero_and_switches_back():
    pytest.importorskip("scipy")
    app, view, model, presenter = build_app()
    map_presenter = presenter.mz_rt_map_presenter
    window = view.mz_rt_map_window
    view.mz_rt_map_requested.emit()
    wait_for_tasks(app)

    window.compare_combo.setCurrentIndex(2)  # "2: ...", the same data as A
    wait_for_tasks(app)

    assert map_presenter.compare_model is presenter.data_stack_presenter.models[1]
    assert window.diff_group.isVisibleTo(window)
    assert window.color_bar.axis.labelText == "B / A"
    assert window.color_bar.colorMap().name == "CET-D1A"
    assert "B ×1" in window.normalize_check_box.text()
    assert "ppm bins, B shifted by up to 0.00 min" in window.status_label.text()
    assert np.abs(window.image_item.image).max() < 1e-4
    limit = window.fold_change_limit()
    np.testing.assert_allclose(window.color_bar.levels(), (-np.log(limit), np.log(limit)))

    window.view_box.setRange(xRange=(0.5, 1.0), yRange=(200.0, 201.0), padding=0)
    wait_for_tasks(app)
    assert window.status_label.text().startswith("10 ppm bins")
    assert map_presenter._detail_layers[1] is not None
    assert np.abs(window.image_item.image).max() < 1e-4

    window.compare_combo.setCurrentIndex(0)
    wait_for_tasks(app)
    assert map_presenter.compare_model is None
    assert not window.diff_group.isVisibleTo(window)
    assert window.color_bar.axis.labelText == "Intensity"
    assert window.color_bar.colorMap().name == "inferno"
    assert window.image_item.image.max() > 0


def test_failing_alignment_is_reported_in_the_status(monkeypatch):
    from RAPID2_Agilent.views.mz_rt_map import mz_rt_map_presenter

    def missing_scipy(*_args):
        raise ImportError("No module named 'scipy'", name="scipy")

    monkeypatch.setattr(mz_rt_map_presenter, "align_maps", missing_scipy)
    app, view, model, presenter = build_app()
    window = view.mz_rt_map_window
    view.mz_rt_map_requested.emit()
    wait_for_tasks(app)

    window.compare_combo.setCurrentIndex(2)
    wait_for_tasks(app)

    assert window.status_label.text() == "RT alignment needs scipy: pip install scipy"


def test_diagnostics_panel_shows_each_kind_to_the_right_of_the_difference():
    pytest.importorskip("scipy")
    from RAPID2_Agilent.views.mz_rt_map.diagnostics import Diagnostics

    app, view, model, presenter = build_app()
    window = view.mz_rt_map_window
    view.mz_rt_map_requested.emit()
    wait_for_tasks(app)
    window.compare_combo.setCurrentIndex(2)
    wait_for_tasks(app)
    plot_item = window.diagnostics_plot.getPlotItem()
    assert not window.diagnostics_plot.isVisibleTo(window)
    assert window.plot_splitter.orientation() == Qt.Orientation.Horizontal
    assert window.plot_splitter.widget(0) is window.plot_widget
    assert window.plot_splitter.widget(1) is window.diagnostics_panel

    for kind in list(Diagnostics)[1:]:
        window.diagnostics_combo.setCurrentText(kind.value)
        wait_for_tasks(app)
        assert window.diagnostics_plot.isVisibleTo(window), kind
        assert plot_item.titleLabel.text == "", kind  # not waiting for anything
        assert plot_item.listDataItems(), kind
        assert window.diagnostics_color_bar.isVisible() == (kind is Diagnostics.DTW_PATH)
        assert window.diagnostics_color_controls.isVisibleTo(window) == (kind is Diagnostics.DTW_PATH)
    assert window.diagnostics_plot.getViewBox().viewRange()[0] == window.view_box.viewRange()[0]  # same RT range

    window.compare_combo.setCurrentIndex(0)  # A alone: no diagnostics
    assert not window.diagnostics_plot.isVisibleTo(window)
    window.compare_combo.setCurrentIndex(2)
    assert window.diagnostics_plot.isVisibleTo(window)
    wait_for_tasks(app)
    window.diagnostics_combo.setCurrentText(Diagnostics.NONE.value)
    assert not window.diagnostics_plot.isVisibleTo(window)


@pytest.mark.parametrize("x,y", [(0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1), (0.5, 1), (0, 1), (0, 0.5)])
def test_each_rectangle_handle_updates_main_ranges_without_editing_target(x, y):
    app, view, model, presenter = build_app()
    stack, window = presenter.data_stack_presenter, view.mz_rt_map_window
    target = presenter.target_panel_presenter.add_target_from_view()
    window.show_range((0.0, 2.0), (100.0, 400.0))
    stack.set_rt_window((0.5, 1.0))
    stack.set_mz_window((200.0, 201.0))
    handle = next(handle['item'] for handle in window.marker.handles if tuple(handle['pos']) == (x, y))

    window.marker.movePoint(handle, (0.5 + x * 0.5 + (x - 0.5) * 0.2, 200.0 + y + (y - 0.5) * 0.4))

    rt_range = (0.4 if x == 0 else 0.5, 1.1 if x == 1 else 1.0)
    mz_range = (199.8 if y == 0 else 200.0, 201.2 if y == 1 else 201.0)
    for child in stack.child_presenters:
        np.testing.assert_allclose(child.current_rt_window, rt_range)
        np.testing.assert_allclose(child.current_mz_window, mz_range)
        np.testing.assert_allclose(child.view.chromatogram_window.get_region(), rt_range)
        np.testing.assert_allclose(child.view.spectrum_window.get_region(), mz_range)
    np.testing.assert_allclose(view.target_panel.rt_values(), (sum(rt_range) / 2, (rt_range[1] - rt_range[0]) / 2))
    np.testing.assert_allclose(view.target_panel.mz_values()[:2], (sum(mz_range) / 2, (mz_range[1] - mz_range[0]) / 2))
    np.testing.assert_allclose(window.marker.pos(), (rt_range[0], mz_range[0]))
    np.testing.assert_allclose(window.marker.size(), (rt_range[1] - rt_range[0], mz_range[1] - mz_range[0]))
    assert model.target_list_model.get(target.id) == target
    assert presenter.target_panel_presenter.current_id is None


def test_rectangle_resize_keeps_mz_for_tic_and_snaps_ppm_for_xic():
    app, view, model, presenter = build_app()
    stack, window = presenter.data_stack_presenter, view.mz_rt_map_window
    window.show_range((0.0, 2.0), (100.0, 400.0))
    view.target_panel.trace_type_combo.setCurrentText(TraceType.TIC.value)
    mz_range = stack.current_mz_window
    window.marker_range_changed.emit(0.5, 0.8, 200.0, 201.0)
    np.testing.assert_allclose(stack.current_rt_window, (0.5, 0.8))
    np.testing.assert_allclose(stack.current_mz_window, mz_range)
    np.testing.assert_allclose(window.marker.pos(), (0.5, mz_range[0]))

    view.target_panel.trace_type_combo.setCurrentText(TraceType.XIC.value)
    view.target_panel.mz_range_control.set_tolerance_mode(ToleranceUnit.PPM.value)
    window.marker_range_changed.emit(0.5, 0.8, 299.9996, 300.0004)
    np.testing.assert_allclose(stack.current_mz_window, (299.9997, 300.0003))
    np.testing.assert_allclose(window.marker.size(), (0.3, 0.0006))


def test_diagnostics_color_bar_palette_and_manual_limits_survive_redraws():
    from RAPID2_Agilent.views.mz_rt_map.diagnostics import Diagnostics, DiagnosticsImage, DiagnosticsPlot

    app, view, model, presenter = build_app()
    window = view.mz_rt_map_window
    window.set_diff_mode(True)
    window.diagnostics_combo.setCurrentText(Diagnostics.DTW_PATH.value)
    plot = DiagnosticsPlot("B's RT (min)", [], DiagnosticsImage(np.array([[0.0, 0.5], [0.5, 1.0]]), (0.0, 0.0, 2.0, 2.0), (0.01, 0.99)))
    window.set_diagnostics(plot)
    np.testing.assert_allclose(window.diagnostics_color_bar.levels(), (0.01, 0.99))
    assert window.diagnostics_color_bar.axis.labelText == "Similarity (cosine)"
    main_lut = window.image_item.lut.copy()
    window.diagnostics_color_map_combo.setCurrentText("viridis")
    assert window.diagnostics_color_bar.colorMap().name == "viridis"
    assert not np.array_equal(window.diagnostics_image_item.lut, main_lut)
    np.testing.assert_array_equal(window.image_item.lut, main_lut)

    window.diagnostics_auto_range_check_box.setChecked(False)
    window.diagnostics_min_spin_box.setValue(0.2)
    window.diagnostics_max_spin_box.setValue(0.8)
    window.set_diagnostics(plot)
    np.testing.assert_allclose(window.diagnostics_color_bar.levels(), (0.2, 0.8))
    np.testing.assert_allclose(window.diagnostics_image_item.getLevels(), (0.2, 0.8))
    window.set_diagnostics(DiagnosticsPlot("Shift", []))
    assert not window.diagnostics_color_bar.isVisible()
    window.set_diagnostics(plot)
    np.testing.assert_allclose(window.diagnostics_color_bar.levels(), (0.2, 0.8))
    window.diagnostics_auto_range_check_box.setChecked(True)
    np.testing.assert_allclose(window.diagnostics_image_item.getLevels(), (0.01, 0.99))

    window.set_diagnostics(DiagnosticsPlot("B's RT", [], DiagnosticsImage(np.zeros((2, 2)), (0, 0, 2, 2), (0.0, 0.0))))
    lower, upper = window.diagnostics_color_bar.levels()
    assert lower < 0 < upper
    window.diagnostics_auto_range_check_box.setChecked(False)
    window.diagnostics_min_spin_box.setValue(0.5)
    lower, upper = window.diagnostics_color_bar.levels()
    assert lower < upper
