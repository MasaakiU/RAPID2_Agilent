from PyQt6.QtCore import Qt, pyqtBoundSignal, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QSplitter, QVBoxLayout, QWidget

from RAPID2_Agilent.views.data_stack import ChromatogramSpectrumStackView
from RAPID2_Agilent.views.mz_rt_map import MzRtMapWindow
from RAPID2_Agilent.views.target_panel import TargetPanel
from RAPID2_Agilent.views.widgets import ViewControlButtons


class MainWindow(QMainWindow):
    open_rpd_files_requested = pyqtSignal()
    open_d_folders_requested = pyqtSignal()
    convert_files_requested = pyqtSignal()
    export_targets_requested = pyqtSignal()
    load_targets_requested = pyqtSignal()
    export_images_requested = pyqtSignal()
    export_all_images_requested = pyqtSignal()
    add_compound_requested = pyqtSignal()
    remove_all_targets_requested = pyqtSignal()
    undo_target_edit_requested = pyqtSignal()
    redo_target_edit_requested = pyqtSignal()
    undo_view_range_requested = pyqtSignal()
    redo_view_range_requested = pyqtSignal()
    mz_rt_map_requested = pyqtSignal()
    set_view_range_requested = pyqtSignal()
    atomic_ratio_window_requested = pyqtSignal()
    execute_deisotoping_requested = pyqtSignal()
    generate_mz_rt_image2d_requested = pyqtSignal()
    show_adduct_ion_syntax_requested = pyqtSignal()
    about_requested = pyqtSignal()
    preferences_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MainWindow")
        self.setWindowTitle("RAPID2 Agilent")

        self.target_panel = TargetPanel()
        self.mz_rt_map_window = MzRtMapWindow(self)
        self.rt_view_control_buttons = ViewControlButtons("RT")
        self.mz_view_control_buttons = ViewControlButtons("mz")
        self.data_stack_view = ChromatogramSpectrumStackView()

        self._central_widget = QWidget()
        self._control_layout = QHBoxLayout()
        self._control_layout.setContentsMargins(0, 0, 0, 0)
        self._control_layout.setSpacing(0)
        self._control_layout.addWidget(self.rt_view_control_buttons)
        self._control_layout.addStretch(1)
        self._control_layout.addWidget(self.mz_view_control_buttons)
        self._control_layout.addStretch(1)

        self._main_layout = QVBoxLayout(self._central_widget)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)
        self._main_layout.addLayout(self._control_layout, 0)
        self._main_layout.addWidget(self.data_stack_view, 1)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.addWidget(self.target_panel)
        self._splitter.addWidget(self._central_widget)
        self._splitter.setStretchFactor(1, 1)
        self.setCentralWidget(self._splitter)

        self._setup_menu()
        self.resize(self.minimumSizeHint())

    def _setup_menu(self):
        file_menu = self.menuBar().addMenu("File")
        self._add_menu_action(file_menu, "Open RPD file", self.open_rpd_files_requested, "Ctrl+O")
        self._add_menu_action(file_menu, "Open .d folder", self.open_d_folders_requested, "Ctrl+Shift+O")
        self._add_menu_action(file_menu, "Convert Files", self.convert_files_requested, "Ctrl+Shift+C")
        self._add_menu_action(file_menu, "Export Targets", self.export_targets_requested, "Ctrl+E")
        self._add_menu_action(file_menu, "Load Targets", self.load_targets_requested, "Ctrl+L")
        self._add_menu_action(file_menu, "Export Images for Current Targets", self.export_images_requested, "Ctrl+Shift+E")
        self._add_menu_action(file_menu, "Export Images for All Targets", self.export_all_images_requested, "Ctrl+Alt+E")
        edit_menu = self.menuBar().addMenu("Edit")
        self.undo_target_edit_action = self._add_menu_action(edit_menu, "Undo Target Edit", self.undo_target_edit_requested, "Ctrl+Z")
        self.redo_target_edit_action = self._add_menu_action(edit_menu, "Redo Target Edit", self.redo_target_edit_requested, "Ctrl+Shift+Z")
        edit_menu.addSeparator()
        self._add_menu_action(edit_menu, "Add Target", self.add_compound_requested)
        self._add_menu_action(edit_menu, "Remove All Targets", self.remove_all_targets_requested)
        view_menu = self.menuBar().addMenu("View")
        self.undo_view_range_action = self._add_menu_action(view_menu, "Undo View Range", self.undo_view_range_requested, "Ctrl+Alt+Z")
        self.redo_view_range_action = self._add_menu_action(view_menu, "Redo View Range", self.redo_view_range_requested, "Ctrl+Alt+Shift+Z")
        view_menu.addSeparator()
        self._add_menu_action(view_menu, "m/z-RT Map", self.mz_rt_map_requested, "Ctrl+Shift+M")
        self._add_menu_action(view_menu, "Set View Range", self.set_view_range_requested, "Ctrl+Meta+V")
        self.set_target_undo_state(False, False)
        self.set_view_range_undo_state(False, False)
        tools_menu = self.menuBar().addMenu("Tools")
        self._add_menu_action(tools_menu, "Atomic Ratio Calculator", self.atomic_ratio_window_requested, "Ctrl+T")
        self._add_menu_action(tools_menu, "Execute Deisotoping", self.execute_deisotoping_requested, "Ctrl+D")
        self._add_menu_action(tools_menu, "Generate mz-RT Image2D", self.generate_mz_rt_image2d_requested, "Ctrl+G")
        self._add_menu_action(tools_menu, "Use adduct ion syntax", self.show_adduct_ion_syntax_requested, "Ctrl+H")
        help_menu = self.menuBar().addMenu("Help")
        self._add_menu_action(help_menu, "About", self.about_requested)
        self._add_menu_action(help_menu, "Settings", self.preferences_requested)

    def set_target_undo_state(self, can_undo: bool, can_redo: bool):
        self.undo_target_edit_action.setEnabled(can_undo)
        self.redo_target_edit_action.setEnabled(can_redo)

    def set_view_range_undo_state(self, can_undo: bool, can_redo: bool):
        self.undo_view_range_action.setEnabled(can_undo)
        self.redo_view_range_action.setEnabled(can_redo)

    @staticmethod
    def _add_menu_action(
        menu,
        text: str,
        signal: pyqtBoundSignal,
        shortcut: str | None = None,
    ) -> QAction:
        action = menu.addAction(text)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(lambda _checked=False: signal.emit())
        return action
