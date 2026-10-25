from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from RAPID2_Agilent.views.widgets.my_push_button import make_icon_button
from RAPID2_Agilent.views.layout_metrics import SMALL


class ViewControlButtons(QWidget):
    """Y range and X range of one column of plots: ``plot_name`` e.g. "chromatograms", ``x_name`` e.g. "RT"."""

    def __init__(self, plot_name: str, x_name: str, parent=None):
        super().__init__(parent)
        self.setObjectName("ViewControlButtons")

        self.link_y_button = make_icon_button("link-solid", f"Share one Y range among the {plot_name} of all files")
        self.auto_y_button = make_icon_button("arrows-up-down_withA_BOLD", f"Fit the Y range to the {plot_name} in the shown {x_name} range")
        self.reset_x_range_button = make_icon_button("arrows-left-right_BOLD", f"Show the whole {x_name} range of all files")
        self.zoom_to_window_button = make_icon_button("arrows-to-window_BOLD", f"Zoom to the selected {x_name} range, with room on each side")

        self.link_y_button.setCheckable(True)
        self.auto_y_button.setCheckable(True)

        self.title_label = QLabel(plot_name.capitalize())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SMALL)
        layout.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.link_y_button, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.auto_y_button, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.reset_x_range_button, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.zoom_to_window_button, alignment=Qt.AlignmentFlag.AlignTop)
