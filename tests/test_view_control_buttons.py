import os

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from RAPID2_Agilent.models.main_model import MainModel
from RAPID2_Agilent.presenters.main_presenter import MainPresenter
from RAPID2_Agilent.views.main_window import MainWindow
from RAPID2_Agilent.views.view_constants import ZOOM_MZ_PADDING
from tests.test_chromatogram_spectrum_stack import load_model


@pytest.fixture(params=["chromatogram_window", "spectrum_window"])
def viewer(request):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    view = MainWindow()
    presenter = MainPresenter(view, MainModel())
    controls = (
        view.rt_view_control_buttons
        if request.param == "chromatogram_window"
        else view.mz_view_control_buttons
    )
    yield presenter, request.param, controls
    presenter.data_stack_presenter.clear()
    view.close()
    view.deleteLater()
    app.processEvents()


def add_windows(presenter, window_name, count=2):
    windows = []
    for index in range(count):
        child = presenter.add_ms_data_model(load_model())
        window = getattr(child.view, window_name)
        # Off-screen peaks must not affect Auto Y in the central X interval.
        window.set_primary_plot_data(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([10000.0, 10.0, 20.0, 20000.0]) * (index + 1),
        )
        windows.append(window)
    windows[0].plotItem.setXRange(1.0, 2.0)
    return windows


def test_link_y_button_controls_y_sync_but_keeps_x_sync(viewer):
    presenter, name, controls = viewer
    first, second = add_windows(presenter, name)
    assert controls.link_y_button.isChecked()
    assert controls.auto_y_button.isChecked()
    controls.auto_y_button.click()
    controls.link_y_button.click()
    second.plotItem.setYRange(100.0, 200.0)

    first.plotItem.setRange(xRange=(0.5, 2.5), yRange=(-5.0, 50.0), padding=0)

    np.testing.assert_allclose(second.plotItem.viewRange_x(), (0.5, 2.5))
    np.testing.assert_allclose(second.plotItem.viewRange_y(), (100.0, 200.0))
    controls.link_y_button.click()
    np.testing.assert_allclose(second.plotItem.viewRange_y(), (-5.0, 50.0))
    second.plotItem.setYRange(-10.0, 80.0)
    np.testing.assert_allclose(first.plotItem.viewRange_y(), (-10.0, 80.0))


def test_auto_y_uses_pooled_or_individual_visible_bounds(viewer):
    presenter, name, controls = viewer
    first, second = add_windows(presenter, name)
    for window in (first, second):
        np.testing.assert_allclose(window.plotItem.viewRange_y(), (7.0, 43.0))

    controls.link_y_button.click()
    np.testing.assert_allclose(first.plotItem.viewRange_y(), (9.0, 21.0))
    np.testing.assert_allclose(second.plotItem.viewRange_y(), (18.0, 42.0))

    controls.auto_y_button.click()
    first.plotItem.setYRange(-1.0, 1.0)
    first.set_primary_plot_data(np.array([1.0, 2.0]), np.array([100.0, 200.0]))
    np.testing.assert_allclose(first.plotItem.viewRange_y(), (-1.0, 1.0))
    controls.auto_y_button.click()
    np.testing.assert_allclose(first.plotItem.viewRange_y(), (90.0, 210.0))
    np.testing.assert_allclose(second.plotItem.viewRange_y(), (18.0, 42.0))
    controls.link_y_button.click()
    for window in (first, second):
        np.testing.assert_allclose(window.plotItem.viewRange_y(), (2.0, 218.0))


def test_linked_auto_y_refreshes_after_data_and_x_range_changes(viewer):
    presenter, name, controls = viewer
    first, second = add_windows(presenter, name)
    second.set_primary_plot_data(np.array([1.0, 2.0]), np.array([-20.0, 80.0]))
    for window in (first, second):
        np.testing.assert_allclose(window.plotItem.viewRange_y(), (-30.0, 90.0))

    second.plotItem.setRange(xRange=(1.9, 2.1), yRange=(0.0, 1.0), padding=0)
    for window in (first, second):
        np.testing.assert_allclose(window.plotItem.viewRange_x(), (1.9, 2.1))
        np.testing.assert_allclose(window.plotItem.viewRange_y(), (14.0, 86.0))

    controls.reset_x_range_button.click()
    for window in (first, second):
        np.testing.assert_allclose(window.plotItem.viewRange_y(), (-2022.0, 22002.0))
    controls.auto_y_button.click()
    second.plotItem.setYRange(0.0, 1.0)
    first.set_primary_plot_data(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
    for window in (first, second):
        np.testing.assert_allclose(window.plotItem.viewRange_y(), (0.0, 1.0))
    controls.auto_y_button.click()
    for window in (first, second):
        np.testing.assert_allclose(window.plotItem.viewRange_y(), (-30.0, 90.0))


@pytest.mark.parametrize("count", [1, 2])
def test_reset_x_range_button_includes_all_files_and_keeps_manual_y(viewer, count):
    presenter, name, controls = viewer
    windows = add_windows(presenter, name, count)
    controls.auto_y_button.click()
    controls.link_y_button.click()
    windows[-1].set_primary_plot_data(np.array([-2.0, 5.0]), np.array([1.0, 2.0]))
    for index, window in enumerate(windows):
        window.plotItem.setYRange(index + 10.0, index + 20.0)

    controls.reset_x_range_button.click()

    for index, window in enumerate(windows):
        np.testing.assert_allclose(window.plotItem.viewRange_x(), (-2.7, 5.7))
        np.testing.assert_allclose(window.plotItem.viewRange_y(), (index + 10.0, index + 20.0))


def test_zoom_to_window_button_shows_window_with_room(viewer):
    presenter, name, controls = viewer
    windows = add_windows(presenter, name)
    stack = presenter.data_stack_presenter
    if name == "chromatogram_window":
        stack.set_rt_window((1.0, 1.4))
        expected = (0.6, 1.8)  # as wide again on each side
    else:
        stack.set_mz_window((1.0, 1.4))
        expected = (1.0 - ZOOM_MZ_PADDING, 1.4 + ZOOM_MZ_PADDING)

    controls.zoom_to_window_button.click()

    for window in windows:
        np.testing.assert_allclose(window.plotItem.viewRange_x(), expected)


def test_controls_before_loading_and_new_files_inherit_settings(viewer):
    presenter, name, controls = viewer
    controls.reset_x_range_button.click()
    controls.zoom_to_window_button.click()
    controls.link_y_button.click()
    controls.auto_y_button.click()
    first, = add_windows(presenter, name, count=1)
    first.plotItem.setYRange(5.0, 6.0)

    child = presenter.add_ms_data_model(load_model())
    second = getattr(child.view, name)

    assert not first.auto_y_range_enabled
    assert not second.auto_y_range_enabled
    np.testing.assert_allclose(second.plotItem.viewRange_x(), (1.0, 2.0))
    second.plotItem.setYRange(100.0, 200.0)
    np.testing.assert_allclose(first.plotItem.viewRange_y(), (5.0, 6.0))
    other_name = "spectrum_window" if name == "chromatogram_window" else "chromatogram_window"
    assert getattr(child.view, other_name).auto_y_range_enabled
    assert presenter.data_stack_presenter.link_y_enabled[other_name]


def test_linked_auto_y_handles_empty_and_flat_visible_data(viewer):
    presenter, name, controls = viewer
    first, second = add_windows(presenter, name)
    first.set_primary_plot_data(np.array([5.0, 6.0]), np.array([1.0, 2.0]))
    np.testing.assert_allclose(first.plotItem.viewRange_y(), (18.0, 42.0))
    second.set_primary_plot_data(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    for window in (first, second):
        np.testing.assert_allclose(window.plotItem.viewRange_y(), (-0.6, 0.6))

    second.plotItem.setXRange(10.0, 11.0)
    for window in (first, second):
        np.testing.assert_allclose(window.plotItem.viewRange_y(), (-0.6, 0.6))


def test_file_control_buttons_sit_above_info_panels():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    view = MainWindow()
    buttons = view.file_control_buttons
    for button in (
        buttons.move_to_top_button,
        buttons.move_up_button,
        buttons.move_down_button,
        buttons.move_to_bottom_button,
        buttons.close_button,
    ):
        assert not button.icon().isNull()
    assert buttons.width() == view.data_stack_view.info_w
    view.close()
    view.deleteLater()
    app.processEvents()


def test_file_control_buttons_reorder_and_close_selected_files():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    view = MainWindow()
    presenter = MainPresenter(view, MainModel())
    children = [presenter.add_ms_data_model(load_model()) for _ in range(3)]
    stack_view = view.data_stack_view
    buttons = view.file_control_buttons
    map_presenter = presenter.mz_rt_map_presenter
    map_presenter.current_model = children[0].model

    assert not buttons.isEnabled()
    stack_view.set_selected_items([children[2].view])
    assert buttons.isEnabled()

    def data_set_models():
        return [data_set.original for data_set in presenter.model.rpd_data_set_list._data_sets]

    buttons.move_to_top_button.click()
    assert presenter.data_stack_presenter.child_presenters == [children[2], children[0], children[1]]
    assert data_set_models() == presenter.data_stack_presenter.models

    stack_view.set_selected_items([children[0].view])
    buttons.close_button.click()
    assert presenter.data_stack_presenter.child_presenters == [children[2], children[1]]
    assert data_set_models() == [children[2].model, children[1].model]
    # the one below is selected, for the next close
    assert stack_view.selected_items == [children[1].view]
    assert buttons.isEnabled()
    # A was closed while the map's window is hidden: none until it is shown
    assert map_presenter.current_model is None
    assert view.mz_rt_map_window.file_combo.count() == 2

    presenter.data_stack_presenter.clear()
    view.close()
    view.deleteLater()
    app.processEvents()


def test_button_groups_line_up_with_their_columns_beside_the_scroll_bar():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    view = MainWindow()
    presenter = MainPresenter(view, MainModel())
    # at the size it opens at, its smallest
    view.show()
    stack_view = view.data_stack_view

    def edges(widget):
        return widget.mapTo(view, widget.rect().topLeft()).x(), widget.mapTo(view, widget.rect().topRight()).x()

    def check_lined_up():
        for _ in range(3):
            app.processEvents()
        item = stack_view.stack_items[0]
        # nothing of the rows under the scroll bar
        assert edges(item.file_info_panel)[1] <= edges(stack_view.viewport())[1]
        assert edges(view.rt_view_control_buttons)[0] == edges(item.chromatogram_window)[0]
        assert edges(view.mz_view_control_buttons)[0] == edges(item.spectrum_window)[0]
        # the file buttons in the room above the info panels, from its left
        assert edges(view.file_control_buttons) == edges(item.file_info_panel)
        assert edges(view.file_control_buttons.move_to_top_button)[0] == edges(item.file_info_panel)[0]

    presenter.add_ms_data_model(load_model())
    assert stack_view.scroll_bar_extent() == 0
    check_lined_up()
    while stack_view.scroll_bar_extent() == 0:
        presenter.add_ms_data_model(load_model())
        app.processEvents()
    check_lined_up()
    while stack_view.scroll_bar_extent() != 0:
        stack_view.select_item(stack_view.stack_items[-1])
        presenter.data_stack_presenter.close_selected()
        app.processEvents()
    check_lined_up()

    presenter.data_stack_presenter.clear()
    view.close()
    view.deleteLater()
    app.processEvents()
