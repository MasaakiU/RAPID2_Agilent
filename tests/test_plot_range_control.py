import os
import sys

import numpy as np
from PyQt6.QtWidgets import QApplication

from RAPID2_Agilent.models.rpd_model import AgilentDModel
from RAPID2_Agilent.views.data_view import ChromatogramSpectrumPresenter, ChromatogramSpectrumView
from tests.data_paths import DATA_DIR, require


DATA_PATH = DATA_DIR / "dummy_RP_2min_neg.d"


def build_presenter():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)
    model = AgilentDModel()
    model.load(require(DATA_PATH))
    view = ChromatogramSpectrumView()
    presenter = ChromatogramSpectrumPresenter(view, model)
    presenter.initialize(scan_index=5, mz_window=(150.0, 151.0))
    return app, model, view, presenter


def test_spectrum_click_keeps_chromatogram_x_range():
    app, model, view, presenter = build_presenter()
    chromatogram_window = view.chromatogram_window
    before_x_range = tuple(chromatogram_window.plotItem.viewRange_x())

    presenter.on_spectrum_clicked(200.0, 0.0)

    np.testing.assert_allclose(chromatogram_window.plotItem.viewRange_x(), before_x_range)


def test_chromatogram_click_keeps_spectrum_x_range():
    app, model, view, presenter = build_presenter()
    spectrum_window = view.spectrum_window
    before_x_range = tuple(spectrum_window.plotItem.viewRange_x())

    presenter.on_chromatogram_clicked(float(model.RT_list[6]), 0.0)

    np.testing.assert_allclose(spectrum_window.plotItem.viewRange_x(), before_x_range)


def test_region_change_keeps_x_ranges():
    app, model, view, presenter = build_presenter()
    chromatogram_window = view.chromatogram_window
    spectrum_window = view.spectrum_window
    before_chrom_x_range = tuple(chromatogram_window.plotItem.viewRange_x())
    before_spec_x_range = tuple(spectrum_window.plotItem.viewRange_x())

    chromatogram_window.region0.setRegion((float(model.RT_list[4]), float(model.RT_list[7])))
    presenter.on_chromatogram_region_changed()
    spectrum_window.region0.setRegion((200.0, 201.0))
    presenter.on_spectrum_region_changed()

    np.testing.assert_allclose(chromatogram_window.plotItem.viewRange_x(), before_chrom_x_range)
    np.testing.assert_allclose(spectrum_window.plotItem.viewRange_x(), before_spec_x_range)


def test_auto_y_range_off_keeps_y_range_on_data_update():
    app, model, view, presenter = build_presenter()
    chromatogram_window = view.chromatogram_window
    chromatogram_window.set_auto_y_range_enabled(False)
    chromatogram_window.plotItem.setYRange(0.0, 123.0, padding=0)
    before_y_range = tuple(chromatogram_window.plotItem.viewRange_y())

    presenter.on_spectrum_clicked(200.0, 0.0)

    np.testing.assert_allclose(chromatogram_window.plotItem.viewRange_y(), before_y_range)


def test_auto_y_range_on_overrides_y_range_after_view_range_change():
    app, model, view, presenter = build_presenter()
    chromatogram_window = view.chromatogram_window
    chromatogram_window.set_auto_y_range_enabled(True)
    x_min = float(model.RT_list[10])
    x_max = float(model.RT_list[20])

    chromatogram_window.plotItem.setRange(xRange=(x_min, x_max), yRange=(0.0, 1.0), padding=0)

    expected_y_min, expected_y_max = chromatogram_window.get_displayed_y_bounds(x_min, x_max)
    expected_y_range = chromatogram_window.plotItem.calc_my_y_range(expected_y_min, expected_y_max)
    np.testing.assert_allclose(chromatogram_window.plotItem.viewRange_y(), expected_y_range)


def test_auto_y_range_off_keeps_dragged_y_range_after_view_range_change():
    app, model, view, presenter = build_presenter()
    chromatogram_window = view.chromatogram_window
    chromatogram_window.set_auto_y_range_enabled(False)
    x_min = float(model.RT_list[10])
    x_max = float(model.RT_list[20])
    dragged_y_range = (0.0, 1.0)

    chromatogram_window.plotItem.setRange(xRange=(x_min, x_max), yRange=dragged_y_range, padding=0)

    np.testing.assert_allclose(chromatogram_window.plotItem.viewRange_y(), dragged_y_range)


def test_rt_window_outside_scans_gives_no_spectrum():
    app, model, view, presenter = build_presenter()
    rt_first, rt_last = float(model.RT_list[0]), float(model.RT_list[-1])

    warning = view.spectrum_window.warning_label.textItem.toPlainText
    for rt_window in [(rt_first - 2.0, rt_first - 1.0), (rt_last + 1.0, rt_last + 2.0)]:
        mz_list, inten_list, _ = presenter.extract_spectrum_for_rt_window(rt_window)
        assert len(mz_list) == 0 and len(inten_list) == 0
        presenter.set_rt_window(rt_window)
        assert warning() == "RT window beyond the scanned RT range"

    presenter.set_rt_window((rt_first, rt_last))
    assert warning() == ""


def test_rt_window_between_scans_gives_nearest_scan_spectrum():
    app, model, view, presenter = build_presenter()
    rt_center = (float(model.RT_list[5]) + float(model.RT_list[6])) / 2
    half_width = (float(model.RT_list[6]) - float(model.RT_list[5])) / 10

    mz_list, _, scan_index = presenter.extract_spectrum_for_rt_window((rt_center - half_width, rt_center + half_width))

    assert len(mz_list) > 0
    assert scan_index in (5, 6)
