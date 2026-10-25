import os

from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWidget

from RAPID2_Agilent.models.main_model import MainModel
from RAPID2_Agilent.presenters.main_presenter import MainPresenter
from RAPID2_Agilent.views.main_window import MainWindow
from RAPID2_Agilent.views.widgets import RowChangeOverlay
from RAPID2_Agilent.views.widgets.row_change_overlay import ROW_CHANGE_MS
from tests.test_chromatogram_spectrum_stack import load_model


def make_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def picture(height=20):
    pixmap = QPixmap(100, height)
    pixmap.fill(QColor("white"))
    return pixmap


def test_overlay_plays_only_a_change_and_covers_from_the_first_changed_row():
    app = make_app()
    viewport = QWidget()
    viewport.resize(100, 200)
    rows = {"a": (0, 20), "b": (20, 20), "c": (40, 20), "d": (60, 20)}
    asked = []

    def picture_of(key):
        asked.append(key)
        return picture()

    assert RowChangeOverlay.play(viewport, rows, rows, {}, picture_of, QColor("white")) is None

    # b removed: c and d move up; a stays above the overlay
    after = {"a": (0, 20), "c": (20, 20), "d": (40, 20)}
    overlay = RowChangeOverlay.play(viewport, rows, after, {"b": picture()}, picture_of, QColor("white"), on_top=["c"])
    assert overlay is not None
    assert overlay.geometry().top() == 20
    assert asked == ["d", "c"]  # a is not reached; c, on top, is drawn last
    assert [(old, new) for _picture, old, new in overlay._moved] == [(60, 40), (40, 20)]
    assert [top for _picture, top in overlay._removed] == [20]

    RowChangeOverlay.finish_all(viewport)
    assert overlay not in RowChangeOverlay._playing
    viewport.deleteLater()
    app.processEvents()


def test_cut_row_leaves_the_part_out_of_view_clear():
    app = make_app()
    viewport = QWidget()
    viewport.resize(50, 40)
    viewport.setAutoFillBackground(True)
    image = RowChangeOverlay.picture_of_rows(viewport)

    row = RowChangeOverlay.cut_row(image, (-10, 20)).toImage()

    assert row.height() == 20
    assert QColor.fromRgba(row.pixel(0, 0)).alpha() == 0  # above the viewport
    assert QColor.fromRgba(row.pixel(0, 15)).alpha() == 255
    viewport.deleteLater()
    app.processEvents()


def build_shown_window(file_count):
    app = make_app()
    view = MainWindow()
    presenter = MainPresenter(view, MainModel())
    children = [presenter.add_ms_data_model(load_model()) for _ in range(file_count)]
    view.show()
    QTest.qWait(20)
    return app, view, presenter, children


def close_window(app, view, presenter):
    presenter.data_stack_presenter.clear()
    view.close()
    view.deleteLater()
    app.processEvents()


def overlays_over(viewport):
    return [overlay for overlay in RowChangeOverlay._playing if overlay.parentWidget() is viewport]


def test_stack_plays_a_close_after_making_it_and_is_left_as_it_is():
    app, view, presenter, children = build_shown_window(3)
    stack_view = view.data_stack_view
    stack_view.select_item(children[0].view)

    view.file_control_buttons.close_button.click()

    # made at once
    assert presenter.data_stack_presenter.child_presenters == children[1:]
    assert stack_view.selected_items == [children[1].view]
    app.processEvents()
    assert len(overlays_over(stack_view.viewport())) == 1
    # a second close while the first plays ends it, and plays its own
    view.file_control_buttons.close_button.click()
    app.processEvents()
    assert len(overlays_over(stack_view.viewport())) == 1
    QTest.qWait(ROW_CHANGE_MS + 100)
    assert overlays_over(stack_view.viewport()) == []
    close_window(app, view, presenter)


def test_stack_plays_a_move():
    app, view, presenter, children = build_shown_window(3)
    stack_view = view.data_stack_view
    stack_view.select_item(children[2].view)

    view.file_control_buttons.move_to_top_button.click()

    assert presenter.data_stack_presenter.child_presenters == [children[2], children[0], children[1]]
    app.processEvents()
    assert len(overlays_over(stack_view.viewport())) == 1
    close_window(app, view, presenter)


def test_target_list_plays_a_removal_but_not_an_add():
    app, view, presenter, _children = build_shown_window(1)
    target_presenter = presenter.target_panel_presenter
    table_view = view.target_panel.table_view
    viewports = (table_view.viewport(), table_view.verticalHeader().viewport())
    ids = [target_presenter.add_target_from_view().id for _ in range(3)]
    app.processEvents()
    assert all(overlays_over(viewport) == [] for viewport in viewports)

    target_presenter.select_target(ids[0])
    target_presenter.remove_selected_targets()

    assert target_presenter.current_id == ids[1]
    app.processEvents()
    # the rows and their header alike
    assert all(len(overlays_over(viewport)) == 1 for viewport in viewports)
    QTest.qWait(ROW_CHANGE_MS + 100)
    assert all(overlays_over(viewport) == [] for viewport in viewports)
    close_window(app, view, presenter)
