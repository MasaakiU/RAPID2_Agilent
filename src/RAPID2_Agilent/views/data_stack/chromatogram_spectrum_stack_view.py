from contextlib import contextmanager

from PyQt6.QtCore import QEvent, QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from RAPID2_Agilent.views.data_view import ChromatogramWindow, SpectrumWindow
from RAPID2_Agilent.views.widgets import RowChangeOverlay

from .file_info_panel import FileInfoPanel


class ChromatogramSpectrumStackItem(QWidget):
    def __init__(
        self,
        *,
        data_min_w: int,
        info_w: int,
        data_info_h: int,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("ChromatogramSpectrumStackItem")

        self.chromatogram_window = ChromatogramWindow()
        self.spectrum_window = SpectrumWindow()
        self.file_info_panel = FileInfoPanel()

        self.chromatogram_window.setMinimumWidth(data_min_w)
        self.spectrum_window.setMinimumWidth(data_min_w)
        self.file_info_panel.setFixedWidth(info_w)
        self.chromatogram_window.setFixedHeight(data_info_h)
        self.spectrum_window.setFixedHeight(data_info_h)
        self.file_info_panel.setFixedHeight(data_info_h)
        self.setFixedHeight(data_info_h)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.chromatogram_window, 1)
        layout.addWidget(self.spectrum_window, 1)
        layout.addWidget(self.file_info_panel)


class ChromatogramSpectrumStackView(QScrollArea):
    selection_changed = pyqtSignal()
    # the width the vertical scroll bar takes beside the rows, 0 while it is hidden
    scroll_bar_extent_changed = pyqtSignal(int)
    data_min_w = 350
    info_w = 150
    data_info_h = 200
    min_h = 550
    margin = 1

    def __init__(self, parent=None, *, image_on: bool = False):
        super().__init__(parent)
        self.setObjectName("ChromatogramSpectrumStackView")
        self.image_on = image_on
        width_factor = 3 if self.image_on else 2
        self.min_w = self.data_min_w * width_factor + self.info_w
        self.stack_items: list[ChromatogramSpectrumStackItem] = []
        self._selection_anchor: ChromatogramSpectrumStackItem | None = None
        # (rows, pictures of the removed ones, on_top) of a change being made, and of one made but not played yet
        self._row_change = None
        self._row_change_before = None

        self.scroll_widget = QWidget()
        self.row_layout = self._make_row_layout()
        self.scroll_widget.setLayout(self.row_layout)

        # wide enough for the rows beside the scroll bar as well: the rows cannot scroll sideways, and narrower, the
        # info panels' right went under the bar once it showed
        self.setMinimumSize(
            self.min_w + self.margin * 2 + self.verticalScrollBar().sizeHint().width(),
            self.min_h + self.margin * 2,
        )
        self.setWidgetResizable(True)
        self.setWidget(self.scroll_widget)
        self.horizontalScrollBar().setEnabled(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.verticalScrollBar().installEventFilter(self)

    def scroll_bar_extent(self) -> int:
        bar = self.verticalScrollBar()
        return bar.sizeHint().width() if bar.isVisibleTo(self) else 0

    def eventFilter(self, watched, event):
        if watched is self.verticalScrollBar() and event.type() in (QEvent.Type.Show, QEvent.Type.Hide):
            self.scroll_bar_extent_changed.emit(self.scroll_bar_extent())
        return super().eventFilter(watched, event)

    @staticmethod
    def _make_row_layout() -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        return layout

    def add_data_view(self) -> ChromatogramSpectrumStackItem:
        item = ChromatogramSpectrumStackItem(
            data_min_w=self.data_min_w,
            info_w=self.info_w,
            data_info_h=self.data_info_h,
        )
        self.stack_items.append(item)
        self.row_layout.addWidget(item)
        item.file_info_panel.clicked.connect(
            lambda modifiers, source=item: self.on_item_clicked(source, modifiers)
        )
        return item

    ### selection ###
    @property
    def selected_items(self) -> list[ChromatogramSpectrumStackItem]:
        return [item for item in self.stack_items if item.file_info_panel.is_selected()]

    def set_selected_items(self, items):
        for item in self.stack_items:
            item.file_info_panel.set_selected(item in items)
        self.selection_changed.emit()

    def select_item(self, item: ChromatogramSpectrumStackItem):
        """Select ``item`` alone, as a click on it does: a Shift+click then selects from it."""
        self.set_selected_items([item])
        self._selection_anchor = item

    def on_item_clicked(self, item: ChromatogramSpectrumStackItem, modifiers):
        if modifiers & Qt.KeyboardModifier.ShiftModifier and self._selection_anchor in self.stack_items:
            start = self.stack_items.index(self._selection_anchor)
            end = self.stack_items.index(item)
            low, high = sorted((start, end))
            self.set_selected_items(self.stack_items[low:high + 1])
            return
        if modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            item.file_info_panel.set_selected(not item.file_info_panel.is_selected())
            self.selection_changed.emit()
        elif self.selected_items == [item]:
            # clicking the only selected item again clears the selection
            self.set_selected_items([])
        else:
            self.set_selected_items([item])
        self._selection_anchor = item

    def set_item_order(self, items: list[ChromatogramSpectrumStackItem]):
        """Lay the rows out in the order of ``items``, the same items as stack_items."""
        self.stack_items = list(items)
        for item in self.stack_items:
            self.row_layout.removeWidget(item)
        for item in self.stack_items:
            self.row_layout.addWidget(item)

    @contextmanager
    def animated_row_changes(self, on_top=()):
        """Rows moved or removed in this block are shown going there (see RowChangeOverlay), the ones in on_top
        over the others; while hidden, the change is only made."""
        if not self.isVisible():
            yield
            return
        RowChangeOverlay.finish_all(self.viewport())
        # remove_data_view keeps a picture of each row it removes, as the row was
        self._row_change = (self._row_spans(), {}, list(on_top))
        try:
            yield
        finally:
            self._row_change_before, self._row_change = self._row_change, None
        # once the layout has put the rows in their places, and before they are painted there
        QTimer.singleShot(0, self._play_row_change)

    def _row_spans(self) -> dict[ChromatogramSpectrumStackItem, tuple[int, int]]:
        viewport = self.viewport()
        return {item: (item.mapTo(viewport, QPoint(0, 0)).y(), item.height()) for item in self.stack_items}

    def _play_row_change(self):
        before, self._row_change_before = self._row_change_before, None
        if before is None:
            return
        rows, removed_pictures, on_top = before
        RowChangeOverlay.play(
            self.viewport(),
            rows,
            self._row_spans(),
            removed_pictures,
            lambda item: item.grab(),
            self.scroll_widget.palette().color(self.scroll_widget.backgroundRole()),
            on_top,
        )

    def remove_data_view(self, item: ChromatogramSpectrumStackItem):
        if item not in self.stack_items:
            return
        if item is self._selection_anchor:
            self._selection_anchor = None
        if self._row_change is not None:
            rows, removed_pictures, _on_top = self._row_change
            top, height = rows[item]
            if top + height > 0 and top < self.viewport().height():
                removed_pictures[item] = item.grab()
        was_selected = item.file_info_panel.is_selected()
        self.stack_items.remove(item)
        self.row_layout.removeWidget(item)
        item.setParent(None)
        item.deleteLater()
        if was_selected:
            self.selection_changed.emit()

    def clear_data_views(self):
        for item in list(self.stack_items):
            self.remove_data_view(item)
