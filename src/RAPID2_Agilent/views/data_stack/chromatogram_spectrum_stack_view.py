from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from RAPID2_Agilent.views.data_view import ChromatogramWindow, SpectrumWindow

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

        self.scroll_widget = QWidget()
        self.row_layout = self._make_row_layout()
        self.scroll_widget.setLayout(self.row_layout)

        self.setMinimumSize(self.min_w, self.min_h + self.margin * 2)
        self.setWidgetResizable(True)
        self.setWidget(self.scroll_widget)
        self.horizontalScrollBar().setEnabled(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

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
        return item

    def remove_data_view(self, item: ChromatogramSpectrumStackItem):
        if item not in self.stack_items:
            return
        self.stack_items.remove(item)
        self.row_layout.removeWidget(item)
        item.setParent(None)
        item.deleteLater()

    def clear_data_views(self):
        for item in list(self.stack_items):
            self.remove_data_view(item)
