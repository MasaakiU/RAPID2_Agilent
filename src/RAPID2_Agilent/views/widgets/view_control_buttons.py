from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QWidget

from RAPID2_Agilent.views.widgets.my_push_button import make_icon_button


class ViewControlButtons(QWidget):
    def __init__(self, target_name: str, parent=None):
        super().__init__(parent)
        self.setObjectName("ViewControlButtons")
        self.target_name = target_name

        self.link_y_button = make_icon_button("link-solid", f"Link {target_name} Y")
        self.auto_y_button = make_icon_button("arrows-up-down_withA_BOLD", f"Auto {target_name} Y")
        self.reset_x_range_button = make_icon_button("arrows-left-right_BOLD", f"Reset {target_name} X range")

        self.link_y_button.setCheckable(True)
        self.auto_y_button.setCheckable(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.link_y_button, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.auto_y_button, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.reset_x_range_button, alignment=Qt.AlignmentFlag.AlignTop)
