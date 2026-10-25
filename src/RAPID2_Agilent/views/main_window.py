from PyQt6.QtCore import pyqtBoundSignal, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QMainWindow, QSizePolicy, QSpacerItem, QVBoxLayout, QWidget

from RAPID2_Agilent.views.data_stack import ChromatogramSpectrumStackView
from RAPID2_Agilent.views.layout_metrics import MEDIUM, SMALL
from RAPID2_Agilent.views.mz_rt_map import MzRtMapWindow
from RAPID2_Agilent.views.target_panel import TargetPanel
from RAPID2_Agilent.views.widgets import FileControlButtons, ViewControlButtons


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
    ai_assistant_access_toggled = pyqtSignal(bool)
    copy_ai_assistant_command_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MainWindow")
        self.setWindowTitle("RAPID2 Agilent")

        self.target_panel = TargetPanel()
        self.mz_rt_map_window = MzRtMapWindow(self)
        self.rt_view_control_buttons = ViewControlButtons("chromatograms", "RT")
        self.mz_view_control_buttons = ViewControlButtons("spectra", "m/z")
        # while a target is edited: its RT window one for all files, or each file's own, the target's peak in it
        self.link_rt_windows_check_box = QCheckBox("Link RT windows")
        self.data_stack_view = ChromatogramSpectrumStackView()
        self.file_control_buttons = FileControlButtons()
        # as wide as the info panels below, so that each set of buttons starts above its column
        self.file_control_buttons.setFixedWidth(self.data_stack_view.info_w)

        self._central_widget = QWidget()
        self._control_layout = QHBoxLayout()
        # room above and below the buttons; at the top, as much as above the target panel's heading. On the left,
        # the stack's frame, so that the buttons start where the plots below do
        self._control_layout.setContentsMargins(self.data_stack_view.frameWidth(), SMALL, 0, SMALL)
        self._control_layout.setSpacing(0)
        # each column's controls in a half of the room, so that they start above its plots, however wide they are
        rt_controls = QHBoxLayout()
        rt_controls.setSpacing(MEDIUM)
        rt_controls.addWidget(self.rt_view_control_buttons)
        rt_controls.addWidget(self.link_rt_windows_check_box)
        rt_controls.addStretch(1)
        mz_controls = QHBoxLayout()
        mz_controls.addWidget(self.mz_view_control_buttons)
        mz_controls.addStretch(1)
        self._control_layout.addLayout(rt_controls, 1)
        self._control_layout.addLayout(mz_controls, 1)
        self._control_layout.addWidget(self.file_control_buttons)
        # what lies right of the info panels -- the stack's frame, and its scroll bar while shown -- lies right of the
        # buttons above them as well
        self._scroll_bar_spacer = QSpacerItem(0, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        self._control_layout.addItem(self._scroll_bar_spacer)
        self._on_scroll_bar_extent_changed(self.data_stack_view.scroll_bar_extent())
        self.data_stack_view.scroll_bar_extent_changed.connect(self._on_scroll_bar_extent_changed)

        self._main_layout = QVBoxLayout(self._central_widget)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)
        self._main_layout.addLayout(self._control_layout, 0)
        self._main_layout.addWidget(self.data_stack_view, 1)

        # the target panel at its own width, the plots in the rest
        window_widget = QWidget()
        window_layout = QHBoxLayout(window_widget)
        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.setSpacing(0)
        window_layout.addWidget(self.target_panel)
        window_layout.addWidget(self._central_widget, 1)
        self.setCentralWidget(window_widget)

        self._setup_menu()
        self.resize(self.minimumSizeHint())

    def _setup_menu(self):
        not_implemented = []  # greyed out until they do something
        file_menu = self.menuBar().addMenu("File")
        self._add_menu_action(file_menu, "Open RPD file", self.open_rpd_files_requested, "Ctrl+Shift+O")
        self._add_menu_action(file_menu, "Open .d folder", self.open_d_folders_requested, "Ctrl+O")
        not_implemented.append(self._add_menu_action(file_menu, "Convert Files", self.convert_files_requested, "Ctrl+Shift+C"))
        self._add_menu_action(file_menu, "Export Targets", self.export_targets_requested, "Ctrl+E")
        self._add_menu_action(file_menu, "Load Targets", self.load_targets_requested, "Ctrl+L")
        not_implemented.append(self._add_menu_action(file_menu, "Export Images for Current Targets", self.export_images_requested, "Ctrl+Shift+E"))
        not_implemented.append(self._add_menu_action(file_menu, "Export Images for All Targets", self.export_all_images_requested, "Ctrl+Alt+E"))
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
        not_implemented.append(self._add_menu_action(view_menu, "Set View Range", self.set_view_range_requested, "Ctrl+Meta+V"))
        self.set_target_undo_state(False, False)
        self.set_view_range_undo_state(False, False)
        tools_menu = self.menuBar().addMenu("Tools")
        not_implemented.append(self._add_menu_action(tools_menu, "Atomic Ratio Calculator", self.atomic_ratio_window_requested, "Ctrl+T"))
        not_implemented.append(self._add_menu_action(tools_menu, "Execute Deisotoping", self.execute_deisotoping_requested, "Ctrl+D"))
        not_implemented.append(self._add_menu_action(tools_menu, "Generate mz-RT Image2D", self.generate_mz_rt_image2d_requested, "Ctrl+G"))
        not_implemented.append(self._add_menu_action(tools_menu, "Use adduct ion syntax", self.show_adduct_ion_syntax_requested, "Ctrl+H"))
        tools_menu.addSeparator()
        self.ai_assistant_access_action = tools_menu.addAction("AI Assistant Access")
        self.ai_assistant_access_action.setCheckable(True)
        self.ai_assistant_access_action.toggled.connect(lambda checked: self.ai_assistant_access_toggled.emit(checked))
        self.copy_ai_assistant_command_action = self._add_menu_action(tools_menu, "Copy AI Assistant Setup Command", self.copy_ai_assistant_command_requested)
        self.set_ai_assistant_state(False, None)
        help_menu = self.menuBar().addMenu("Help")
        not_implemented.append(self._add_menu_action(help_menu, "About", self.about_requested))
        not_implemented.append(self._add_menu_action(help_menu, "Settings", self.preferences_requested))
        for action in not_implemented:
            action.setEnabled(False)

    def _on_scroll_bar_extent_changed(self, extent: int):
        width = self.data_stack_view.frameWidth() + extent
        self._scroll_bar_spacer.changeSize(width, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        self._control_layout.invalidate()

    def set_target_undo_state(self, can_undo: bool, can_redo: bool):
        self.undo_target_edit_action.setEnabled(can_undo)
        self.redo_target_edit_action.setEnabled(can_redo)

    def set_view_range_undo_state(self, can_undo: bool, can_redo: bool):
        self.undo_view_range_action.setEnabled(can_undo)
        self.redo_view_range_action.setEnabled(can_redo)

    def set_ai_assistant_state(self, running: bool, label: str | None):
        """Follow the MCP server: the menu check, and the letter an assistant addresses this window by."""
        blocked = self.ai_assistant_access_action.blockSignals(True)
        self.ai_assistant_access_action.setChecked(running)
        self.ai_assistant_access_action.blockSignals(blocked)
        self.copy_ai_assistant_command_action.setEnabled(running)
        self.setWindowTitle(f"RAPID2 Agilent — {label}" if running and label else "RAPID2 Agilent")

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
