from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QWidget

from RAPID2_Agilent.views.widgets.my_push_button import make_icon_button
from RAPID2_Agilent.views.layout_metrics import SMALL


class FileControlButtons(QWidget):
    """Reorder and close the files selected in the data stack, left-aligned above its info panels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FileControlButtons")

        self.move_to_top_button = make_icon_button("angles-up-solid", "Move the selected files to the top")
        self.move_up_button = make_icon_button("angle-up-solid", "Move the selected files up one row")
        self.move_down_button = make_icon_button("angle-down-solid", "Move the selected files down one row")
        self.move_to_bottom_button = make_icon_button("angles-down-solid", "Move the selected files to the bottom")
        self.close_button = make_icon_button("xmark-solid", "Close the selected files")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SMALL)
        layout.addWidget(self.move_to_top_button, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.move_up_button, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.move_down_button, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.move_to_bottom_button, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.close_button, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addStretch(1)
