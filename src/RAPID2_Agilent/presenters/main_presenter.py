from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import (
    QFileDialog, 
)

from RAPID2_Agilent.models.rpd_model import AgilentDModel, BaseMSDataModel, RPDFileModel
from RAPID2_Agilent.views.data_stack import ChromatogramSpectrumStackPresenter
from RAPID2_Agilent.views.mz_rt_map import MzRtMapPresenter
from RAPID2_Agilent.views.target_panel import TargetPanelPresenter

if TYPE_CHECKING:
    from RAPID2_Agilent.models.main_model import MainModel
    from RAPID2_Agilent.views.main_window import MainWindow


class MainPresenter:
    def __init__(self, view: MainWindow, model: MainModel) -> None:
        self.view = view
        self.model = model
        self.data_stack_presenter = ChromatogramSpectrumStackPresenter(self.view.data_stack_view)
        self.target_panel_presenter = TargetPanelPresenter(
            self.view.target_panel,
            self.model.target_list_model,
            self.data_stack_presenter,
        )
        self.mz_rt_map_presenter = MzRtMapPresenter(
            self.view.mz_rt_map_window,
            self.data_stack_presenter,
            self.target_panel_presenter,
        )
        self._connect_view_signals()
        self.model.target_list_model.undo_state_changed.connect(self.view.set_target_undo_state)
        self.data_stack_presenter.view_range_undo_state_changed.connect(self.view.set_view_range_undo_state)

    def _connect_view_signals(self) -> None:
        self.view.open_rpd_files_requested.connect(self.open_rpd_files_clicked)
        self.view.open_d_folders_requested.connect(self.open_d_folders_clicked)
        self.view.convert_files_requested.connect(self.convert_files_clicked)
        self.view.export_targets_requested.connect(self.export_targets_clicked)
        self.view.load_targets_requested.connect(self.load_targets_clicked)
        self.view.export_images_requested.connect(self.export_images_clicked)
        self.view.export_all_images_requested.connect(self.export_allimages_clicked)
        self.view.add_compound_requested.connect(self.add_compound_clicked)
        self.view.remove_all_targets_requested.connect(self.remove_all_targets_clicked)
        self.view.undo_target_edit_requested.connect(self.model.target_list_model.undo)
        self.view.redo_target_edit_requested.connect(self.model.target_list_model.redo)
        self.view.undo_view_range_requested.connect(self.data_stack_presenter.undo_view_range)
        self.view.redo_view_range_requested.connect(self.data_stack_presenter.redo_view_range)
        self.view.mz_rt_map_requested.connect(self.mz_rt_map_presenter.show)
        self.view.set_view_range_requested.connect(self.set_view_range_clicked)
        self.view.atomic_ratio_window_requested.connect(self.show_atomic_ratio_window)
        self.view.execute_deisotoping_requested.connect(self.execute_Deisotoping)
        self.view.generate_mz_rt_image2d_requested.connect(self.generate_mz_RT_Image2D)
        self.view.show_adduct_ion_syntax_requested.connect(self.show_adduct_ion_syntax)
        self.view.about_requested.connect(self.show_about)
        self.view.preferences_requested.connect(self.show_preferences)
        for window_name, controls in (
            ("chromatogram_window", self.view.rt_view_control_buttons),
            ("spectrum_window", self.view.mz_view_control_buttons),
        ):
            stack = self.data_stack_presenter
            controls.link_y_button.setChecked(stack.link_y_enabled[window_name])
            controls.auto_y_button.setChecked(stack.auto_y_enabled[window_name])
            controls.link_y_button.toggled.connect(
                lambda enabled, name=window_name: stack.set_link_y_enabled(name, enabled)
            )
            controls.auto_y_button.toggled.connect(
                lambda enabled, name=window_name: stack.set_auto_y_enabled(name, enabled)
            )
            controls.reset_x_range_button.clicked.connect(
                lambda _checked=False, name=window_name: stack.reset_x_range(name)
            )

    def add_ms_data_model(self, ms_data_model: BaseMSDataModel):
        self.model.add_ms_data_model(ms_data_model)
        return self.data_stack_presenter.add_model(ms_data_model)

    def open_rpd_files_clicked(self, file_path_list: Sequence[str] | None = None) -> None:
        if file_path_list is None:
            file_path_list, _file_type = QFileDialog.getOpenFileNames(self.view, "Select rpd file", self.model.settings_model.get_last_dir(), filter="RPD files (*.rpd)")
        if not file_path_list:
            return

        self.model.settings_model.save_last_dir(file_path_list[-1])
        for file_path in file_path_list:
            ms_data_model = RPDFileModel()
            ms_data_model.load(Path(file_path))
            self.add_ms_data_model(ms_data_model)

    def open_d_folders_clicked(self, folder_path_list: Sequence[str] | None = None) -> None:
        if folder_path_list is None:
            folder_path_list = self.open_native_file(title="Select Agilent .d folder", start_dir=self.model.settings_model.get_last_dir(), choose_directories=True, allow_multiple=True)
        folder_path_list = [
            folder_path
            for folder_path in folder_path_list
            if Path(folder_path).suffix.lower() == ".d"
        ]
        if not folder_path_list:
            return

        self.model.settings_model.save_last_dir(folder_path_list[-1])
        for folder_path in folder_path_list:
            ms_data_model = AgilentDModel()
            ms_data_model.load(Path(folder_path))
            self.add_ms_data_model(ms_data_model)

    def open_native_file(
        self,
        title: str,
        start_dir: str,
        *,
        choose_directories: bool = False,
        allow_multiple: bool = False,
    ) -> list[str]:
        if sys.platform == "darwin" and choose_directories:
            return self._open_native_macos_folders(
                title=title,
                start_dir=start_dir,
                allow_multiple=allow_multiple,
            )

        file_mode = (
            QFileDialog.FileMode.Directory
            if choose_directories
            else QFileDialog.FileMode.ExistingFiles
        )
        dialog = QFileDialog(self.view, title)
        dialog.setDirectory(start_dir)
        dialog.setFileMode(file_mode)
        if choose_directories:
            dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        if not dialog.exec():
            return []
        return dialog.selectedFiles()

    def _open_native_macos_folders(
        self,
        title: str,
        start_dir: str,
        *,
        allow_multiple: bool,
    ) -> list[str]:
        multiple_clause = " with multiple selections allowed" if allow_multiple else ""
        script = [
            "on run argv",
            "set promptText to item 1 of argv",
            "set defaultDir to POSIX file (item 2 of argv)",
            f"set chosenItems to choose folder with prompt promptText default location defaultDir{multiple_clause}",
            "set outputText to \"\"",
            "if class of chosenItems is list then",
            "repeat with chosenItem in chosenItems",
            "set outputText to outputText & POSIX path of chosenItem & linefeed",
            "end repeat",
            "else",
            "set outputText to POSIX path of chosenItems",
            "end if",
            "return outputText",
            "end run",
        ]
        command = ["osascript"]
        for line in script:
            command.extend(["-e", line])
        command.extend([title, start_dir])

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return []
        return [line for line in result.stdout.splitlines() if line]

    def convert_files_clicked(self):
        pass

    def export_targets_clicked(self):
        pass

    def load_targets_clicked(self):
        pass

    def export_images_clicked(self):
        pass

    def export_allimages_clicked(self):
        pass

    def add_compound_clicked(self):
        self.target_panel_presenter.add_target_from_view()

    def remove_all_targets_clicked(self):
        self.target_panel_presenter.remove_all_targets()

    def set_view_range_clicked(self):
        pass

    def show_atomic_ratio_window(self):
        pass

    def execute_Deisotoping(self):
        pass

    def generate_mz_RT_Image2D(self):
        pass

    def show_adduct_ion_syntax(self):
        pass

    def show_about(self):
        pass

    def show_preferences(self):
        pass
