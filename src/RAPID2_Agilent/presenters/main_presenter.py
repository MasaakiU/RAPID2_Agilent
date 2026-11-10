from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
)

from RAPID2_Agilent.io import mzdata
from RAPID2_Agilent.io.agilent_centroid_reader import stored_spectra
from RAPID2_Agilent.models.rpd_model import AgilentCentroidModel, AgilentDModel, BaseMSDataModel, RPDFileModel
from RAPID2_Agilent.models.run_peaks import RunPeaksStore
from RAPID2_Agilent.services import peak_detection, rpd_csv
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
            self.model.settings_model,
        )
        link_rt_windows_check_box = self.view.link_rt_windows_check_box
        link_rt_windows_check_box.setChecked(self.data_stack_presenter.link_rt_windows)
        link_rt_windows_check_box.toggled.connect(self.target_panel_presenter.set_link_rt_windows)
        # the files' RT windows are one but for a target's
        link_rt_windows_check_box.setEnabled(self.target_panel_presenter.editing_target)
        self.target_panel_presenter.editing_target_changed.connect(link_rt_windows_check_box.setEnabled)
        self.mz_rt_map_presenter = MzRtMapPresenter(
            self.view.mz_rt_map_window,
            self.data_stack_presenter,
            self.target_panel_presenter,
        )
        self._mcp_controller = None
        self._connect_view_signals()
        self.model.target_list_model.undo_state_changed.connect(self.view.set_target_undo_state)
        self.data_stack_presenter.view_range_undo_state_changed.connect(self.view.set_view_range_undo_state)
        self.data_stack_presenter.models_changed.connect(
            lambda: self.model.keep_ms_data_models(self.data_stack_presenter.models)
        )

    def _connect_view_signals(self) -> None:
        self.view.open_rpd_files_requested.connect(self.open_rpd_files_clicked)
        self.view.open_d_folders_requested.connect(self.open_d_folders_clicked)
        self.view.convert_files_requested.connect(self.convert_files_clicked)
        self.view.export_targets_requested.connect(self.export_targets_clicked)
        self.view.load_targets_requested.connect(self.load_targets_clicked)
        self.view.target_panel.export_requested.connect(self.export_targets_clicked)
        self.view.target_panel.import_requested.connect(self.load_targets_clicked)
        self.view.target_panel.auto_requested.connect(self.detect_peaks_clicked)
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
        self.view.ai_assistant_access_toggled.connect(self.set_ai_assistant_access)
        self.view.copy_ai_assistant_command_requested.connect(self.copy_ai_assistant_command)
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
            controls.zoom_to_window_button.clicked.connect(
                lambda _checked=False, name=window_name: stack.zoom_to_window(name)
            )
        file_controls = self.view.file_control_buttons
        for button, action in (
            (file_controls.move_to_top_button, self.data_stack_presenter.move_selected_to_top),
            (file_controls.move_up_button, self.data_stack_presenter.move_selected_up),
            (file_controls.move_down_button, self.data_stack_presenter.move_selected_down),
            (file_controls.move_to_bottom_button, self.data_stack_presenter.move_selected_to_bottom),
            (file_controls.close_button, self.data_stack_presenter.close_selected),
        ):
            button.clicked.connect(lambda _checked=False, action=action: action())
        # usable while a file is selected
        data_stack_view = self.view.data_stack_view
        data_stack_view.selection_changed.connect(
            lambda: file_controls.setEnabled(bool(data_stack_view.selected_items))
        )
        file_controls.setEnabled(bool(data_stack_view.selected_items))

    def add_ms_data_model(self, ms_data_model: BaseMSDataModel):
        self.model.add_ms_data_model(ms_data_model)
        return self.data_stack_presenter.add_model(ms_data_model)

    def open_rpd_files_clicked(self, file_path_list: Sequence[str] | None = None) -> None:
        if file_path_list is None:
            file_path_list, _file_type = QFileDialog.getOpenFileNames(self.view, "Select rpd file", self.model.settings_model.get_last_dir(), filter="RPD files (*.rpd)")
        if not file_path_list:
            return

        self.model.settings_model.save_last_dir(file_path_list[-1])
        to_open = self._not_open(file_path_list, "Open RPD file")
        if not to_open:
            return
        progress, opened = self._opening_progress("Open RPD file", len(to_open))
        try:
            for position, file_path in enumerate(to_open):
                name = Path(file_path).name
                opened(name, position, 0, 1)
                try:
                    self.open_rpd_file(file_path)
                except (OSError, ValueError) as error:
                    QMessageBox.warning(self.view, "Open RPD file", f"{name} could not be opened.\n\n{error}")
                    continue
                opened(name, position, 1, 1)
        except rpd_csv.Cancelled:
            pass  # those opened stay open
        finally:
            progress.close()

    def open_d_folders_clicked(self, folder_path_list: Sequence[str] | None = None, *, centroid: bool = False) -> None:
        """Of the folders chosen in the dialog, each with centroid spectra as well as the profile is read as the user
        answers, all asked before any is read; those given (the command line) are read of their centroid spectra if
        centroid, else of their profile. A run of centroid spectra only is read of them. One that cannot be read is
        told of. They are read with their progress shown; once it is cancelled, those read stay open."""
        ask = folder_path_list is None
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
        to_open = self._not_open(folder_path_list, "Open .d folder")
        chosen = []  # (folder, of its centroid spectra)
        applied = None  # (centroid,) once "Apply to all" is checked: the answer for the rest
        for position, folder_path in enumerate(to_open):
            sizes = stored_spectra(folder_path)
            use_centroid = centroid
            if ask and len(sizes) == 2:
                if applied is None:
                    answer = self._ask_centroid(Path(folder_path).name, sizes, more=position < len(to_open) - 1)
                    if answer is None:
                        return
                    use_centroid, apply_to_all = answer
                    if apply_to_all:
                        applied = (use_centroid,)
                else:
                    use_centroid = applied[0]
            chosen.append((folder_path, use_centroid))
        if not chosen:
            return
        progress, opened = self._opening_progress("Open .d folder", len(chosen))
        try:
            for position, (folder_path, use_centroid) in enumerate(chosen):
                name = Path(folder_path).name
                opened(name, position, 0, 1)
                try:
                    self.open_d_folder(folder_path, centroid=use_centroid, on_progress=partial(opened, name, position))
                except (OSError, ValueError) as error:
                    QMessageBox.warning(self.view, "Open .d folder", f"{name} could not be opened.\n\n{error}")
                opened(name, position, 1, 1)
        except rpd_csv.Cancelled:
            pass  # those opened stay open
        finally:
            progress.close()

    def open_rpd_file(self, file_path: str | Path) -> BaseMSDataModel:
        """Open a .rpd file, of the spectra it holds. OSError or ValueError if it cannot be read, as the AI assistant
        is told."""
        ms_data_model = RPDFileModel()
        ms_data_model.load(Path(file_path))
        self.add_ms_data_model(ms_data_model)
        return ms_data_model

    def open_d_folder(
        self, folder_path: str | Path, *, centroid: bool = False, on_progress: Callable[[int, int], None] | None = None
    ) -> BaseMSDataModel:
        """Open a .d folder of its centroid spectra if centroid, else of its profile; a run of centroid spectra only,
        of them. on_progress(done, total) told of the scans of a profile read. OSError or ValueError if it cannot be
        read, as the AI assistant is told."""
        sizes = stored_spectra(folder_path)
        if centroid or ("centroid" in sizes and "profile" not in sizes):
            ms_data_model = AgilentCentroidModel()
        else:
            ms_data_model = AgilentDModel()
        ms_data_model.load(Path(folder_path), on_progress)
        self.add_ms_data_model(ms_data_model)
        return ms_data_model

    def _ask_centroid(self, name: str, sizes: dict[str, int], *, more: bool) -> tuple[bool, bool] | None:
        """Whether to read a run's centroid spectra rather than its profile, and to the rest too ("Apply to all",
        offered if more are to be opened); None if cancelled."""
        box = QMessageBox(
            QMessageBox.Icon.Question, "Open .d folder",
            f"{name} holds centroid spectra as well as the profile. Which to read?\n\n"
            f"Profile: {sizes['profile'] / 1e6:,.0f} MB\n"
            f"Centroid: {sizes['centroid'] / 1e6:,.0f} MB, the peaks as stored at acquisition; quicker from a server",
            parent=self.view,
        )
        profile_button = box.addButton("Profile", QMessageBox.ButtonRole.AcceptRole)
        centroid_button = box.addButton("Centroid", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(profile_button)
        apply_to_all = QCheckBox("Apply to all")
        if more:
            box.setCheckBox(apply_to_all)
        box.exec()
        if box.clickedButton() not in (profile_button, centroid_button):
            return None
        return box.clickedButton() is centroid_button, apply_to_all.isChecked()

    def _not_open(self, path_list: Sequence[str], title: str) -> list[str]:
        """The paths not open yet, each once, the others told of: the files are told apart by their paths."""
        open_paths = {Path(model.file_path).resolve() for model in self.data_stack_presenter.models}
        to_open, already_open = [], []
        for path in path_list:
            resolved = Path(path).resolve()
            (already_open if resolved in open_paths else to_open).append(path)
            open_paths.add(resolved)
        if already_open:
            names = "\n".join(Path(path).name for path in already_open)
            QMessageBox.information(self.view, title, f"Open already:\n\n{names}")
        return to_open

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
        # the folder last used may be gone (a server unmounted), and choose folder fails on it without a dialog:
        # the nearest folder of it that is, else the home
        start = Path(start_dir).expanduser()
        while not start.is_dir() and start != start.parent:
            start = start.parent
        if not start.is_dir():
            start = Path.home()
        command = ["osascript"]
        for line in script:
            command.extend(["-e", line])
        command.extend([title, str(start)])

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            # -128: cancelled by the user; any other failure is told of, not left as nothing happening
            if "(-128)" not in result.stderr:
                QMessageBox.warning(self.view, title, f"The folder dialog could not be shown.\n\n{result.stderr.strip()}")
            return []
        return [line for line in result.stdout.splitlines() if line]

    def convert_files_clicked(self, file_path_list: Sequence[str] | None = None) -> None:
        """Convert mzData files to .rpd files next to them, as RAPID2 does, asked for unless given."""
        if file_path_list is None:
            file_path_list, _file_type = QFileDialog.getOpenFileNames(self.view, "Select mzData file", self.model.settings_model.get_last_dir(), filter="mzData files (*.xml)")
        if not file_path_list:
            return

        self.model.settings_model.save_last_dir(file_path_list[-1])
        existing = [mzdata.rpd_path(file_path) for file_path in file_path_list if mzdata.rpd_path(file_path).exists()]
        if existing and QMessageBox.question(
            self.view, "Convert Files", "Replace these?\n\n" + "\n".join(path.name for path in existing)
        ) != QMessageBox.StandardButton.Yes:
            return
        progress, on_progress = self._progress("Convert Files", "Converting")
        written = []
        try:
            for file_path in file_path_list:
                written.append(mzdata.convert(Path(file_path), on_progress))
        except rpd_csv.Cancelled:
            pass
        except (OSError, ValueError) as error:
            QMessageBox.warning(self.view, "Convert Files", f"{Path(file_path).name} could not be converted.\n\n{error}")
        finally:
            progress.close()
        if written:
            QMessageBox.information(self.view, "Convert Files", "Written:\n\n" + "\n".join(path.name for path in written))

    # notes on rows listed after an export, the rest counted
    EXPORT_NOTES_SHOWN = 20

    def export_targets_clicked(self, file_path: str | None = None):
        """Write the targets and their values in the open files as a .rpd.csv file, asked for unless given."""
        targets = self.model.target_list_model.targets
        if not targets:
            QMessageBox.information(self.view, "Export Targets", "There are no targets to export.")
            return
        if file_path is None:
            last_dir = Path(self.model.settings_model.get_last_dir())
            file_path, _filter = QFileDialog.getSaveFileName(
                self.view, "Export Targets", str(last_dir / (last_dir.name + rpd_csv.SUFFIX)),
                f"RAPID2 targets (*{rpd_csv.SUFFIX})",
            )
            if not file_path:
                return
            # the dialog asked about overwriting the name it gave, not the one completed here
            path = rpd_csv.rpd_csv_path(file_path)
            if path != Path(file_path) and path.exists() and QMessageBox.question(
                self.view, "Export Targets", f"{path.name} already exists. Replace it?"
            ) != QMessageBox.StandardButton.Yes:
                return
        path = rpd_csv.rpd_csv_path(file_path)
        runs = [rpd_csv.Run(model, Path(model.file_path)) for model in self.data_stack_presenter.models]
        progress, on_progress = self._progress("Export Targets", "Exporting targets")
        try:
            rows, notes = rpd_csv.export_rows(
                targets, runs, merge_ppm=self.model.settings_model.ms2_merge_ppm, run_peaks=self.run_peaks_to_export(),
                on_progress=on_progress,
            )
            rpd_csv.write_rpd_csv(path, rows)
        except rpd_csv.Cancelled:
            return
        except (OSError, ValueError) as error:
            QMessageBox.warning(self.view, "Export Targets", f"The targets could not be exported.\n\n{error}")
            return
        finally:
            progress.close()
            # with the peaks found for it, in the files they were not looked for in yet
            self.target_panel_presenter.refresh_run_peaks()
        self.model.settings_model.save_last_dir(path)
        if notes:
            listed = "\n".join(notes[:self.EXPORT_NOTES_SHOWN])
            more = len(notes) - self.EXPORT_NOTES_SHOWN
            QMessageBox.information(
                self.view, "Export Targets",
                f"Written to {path.name}. Of its rows, {len(notes)} have no values or leave scans out:\n\n{listed}"
                + (f"\n... and {more} more" if more > 0 else ""),
            )

    def detect_peaks_clicked(self):
        """Find the peaks of the selected targets, of all if none is, in the open files, and take each target's; the
        files' RT windows are their own then, as what is found is."""
        selected = set(self.view.target_panel.selected_target_ids())
        targets = [target for target in self.model.target_list_model.targets if not selected or target.id in selected]
        models = self.data_stack_presenter.models
        if not targets or not models:
            QMessageBox.information(self.view, "Auto", "There are no targets." if not targets else "No files are open.")
            return
        pairs = [(target, model) for target in targets for model in models]
        run_peaks = self.model.run_peaks
        hand_set = [
            (target, model) for target, model in pairs
            if (kept := run_peaks.get(target.id, model.file_path)) is not None and kept.manual
        ]
        keep_manual = False
        if hand_set:
            box = QMessageBox(
                QMessageBox.Icon.Question, "Auto",
                f"{len(hand_set)} of the RT windows were set by hand. Overwrite them with the peaks found as well, "
                "or keep them?",
                parent=self.view,
            )
            overwrite_button = box.addButton("Overwrite all", QMessageBox.ButtonRole.DestructiveRole)
            keep_button = box.addButton("Keep hand-set", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() not in (overwrite_button, keep_button):
                return
            keep_manual = box.clickedButton() is keep_button
        merge_ppm = self.model.settings_model.ms2_merge_ppm
        progress, on_progress = self._progress("Auto", "Finding the peaks")
        found = []
        try:
            for done, (target, model) in enumerate(pairs, start=1):
                found.append((target, model, peak_detection.find_run_peaks(model, target, merge_ppm=merge_ppm)))
                on_progress("Finding the peaks", done, len(pairs))
        except rpd_csv.Cancelled:
            return  # nothing changed
        finally:
            progress.close()
        self.model.target_list_model.set_found_run_peaks(
            [(target.id, model.file_path, run_peaks_found) for target, model, run_peaks_found in found],
            keep_manual=keep_manual,
        )
        self.view.link_rt_windows_check_box.setChecked(False)
        self.target_panel_presenter.refresh_run_peaks()

    def _progress(self, title: str, label: str) -> tuple[QProgressDialog, Callable[[str, int, int], None]]:
        """A dialog of the progress of a long task, and the on_progress(what, done, total) to tell it, which raises
        rpd_csv.Cancelled once Cancel is clicked."""
        progress = QProgressDialog(label, "Cancel", 0, 0, self.view)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        def on_progress(what: str, done: int, total: int):
            progress.setLabelText(what)
            progress.setMaximum(max(total, 1))
            progress.setValue(min(done, max(total, 1)))
            QApplication.processEvents()
            if progress.wasCanceled():
                raise rpd_csv.Cancelled

        return progress, on_progress

    # of the progress dialog of opening files, the steps of each file's part of the bar
    OPEN_STEPS = 100

    def _opening_progress(self, title: str, count: int) -> tuple[QProgressDialog, Callable[[str, int, int, int], None]]:
        """A dialog of the progress of opening count files, shown at once, as a run's profile takes seconds to read,
        and the opened(name, position, done, total) to tell it of the file at position (from 0), done of its total
        parts read; it raises rpd_csv.Cancelled once Cancel is clicked."""
        progress, on_progress = self._progress(title, "Opening")
        progress.setMinimumDuration(0)
        shown = [None]  # the step shown last: told again only of another

        def opened(name: str, position: int, done: int, total: int):
            step = position * self.OPEN_STEPS + self.OPEN_STEPS * done // max(total, 1)
            if step != shown[0] or done == 0:
                shown[0] = step
                on_progress(f"Opening {name} ({position + 1} of {count})", step, count * self.OPEN_STEPS)

        return progress, opened

    def run_peaks_to_export(self) -> RunPeaksStore | None:
        """The files' own RT windows of the targets, as the plots show them: None while they are linked."""
        return None if self.data_stack_presenter.link_rt_windows else self.model.run_peaks

    # problems of a file listed when it cannot be loaded, the rest counted
    LOAD_PROBLEMS_SHOWN = 20

    def load_targets_clicked(self, file_path: str | None = None):
        """Import the targets of a .rpd.csv file, or of a file of RAPID2's Export Targets, asked for unless given."""
        if file_path is None:
            file_path, _filter = QFileDialog.getOpenFileName(
                self.view, "Load Targets", self.model.settings_model.get_last_dir(),
                f"RAPID2 targets (*{rpd_csv.SUFFIX});;RAPID2 (*.csv)",
            )
            if not file_path:
                return
        try:
            groups = rpd_csv.read_targets_file(file_path)
        except rpd_csv.TargetFileError as error:
            listed = "\n".join(error.problems[:self.LOAD_PROBLEMS_SHOWN])
            more = len(error.problems) - self.LOAD_PROBLEMS_SHOWN
            QMessageBox.warning(
                self.view, "Load Targets",
                f"No targets were loaded. The file has these problems:\n\n{listed}" + (f"\n... and {more} more" if more > 0 else ""),
            )
            return
        except OSError as error:
            QMessageBox.warning(self.view, "Load Targets", f"The file could not be read.\n\n{error}")
            return
        self.model.settings_model.save_last_dir(file_path)
        if not groups:
            QMessageBox.information(self.view, "Load Targets", "The file has no targets.")
            return
        choices = self._choose_target_variants(groups)
        if choices is None:
            return
        target_list_model = self.model.target_list_model
        replace = False
        if target_list_model.targets:
            box = QMessageBox(
                QMessageBox.Icon.Question, "Load Targets",
                f"The list has targets already. Replace them with those of {Path(file_path).name}, or append those?",
                parent=self.view,
            )
            replace_button = box.addButton("Replace", QMessageBox.ButtonRole.DestructiveRole)
            append_button = box.addButton("Append", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() not in (replace_button, append_button):
                return
            replace = box.clickedButton() is replace_button
        targets = rpd_csv.resolve_groups(groups, choices, first_index=1 if replace else target_list_model.next_index())
        target_list_model.import_targets(targets, replace=replace)
        self.target_panel_presenter.select_target(None)

    def _choose_target_variants(self, groups: list[rpd_csv.ImportedGroup]) -> dict[int, int | None] | None:
        """Of each '#' whose rows hold different targets, the variant to use, or None to keep them apart, as asked;
        None if the import is cancelled. "Apply to all" takes the same choice for the rest: the variant at the same
        position, or the last one of those with fewer."""
        choices = {}
        applied = None  # once "Apply to all" is ticked, (the choice,)
        for group in groups:
            if len(group.variants) == 1:
                continue
            if applied is not None:
                choice = applied[0]
                choices[group.number] = None if choice is None else min(choice, len(group.variants) - 1)
                continue
            differing = [
                name for name in ("name", "formula", "adduct", "trace_type", "ms_level", "rt", "rt_tolerance", "mz",
                                  "mz_tolerance", "mz_tolerance_unit", "precursor_mz", "collision_energy",
                                  "isolation_width", "note")
                if len({getattr(variant, name) for variant in group.variants}) > 1
            ]
            box = QMessageBox(
                QMessageBox.Icon.Warning, "Load Targets",
                f"The rows of target #{group.number} hold {len(group.variants)} different targets, "
                f"differing in {', '.join(differing)}. Keep them as separate targets, or use the values of one row?",
                parent=self.view,
            )
            separate_button = box.addButton("Keep as separate targets", QMessageBox.ButtonRole.AcceptRole)
            use_buttons = [
                box.addButton(f"Use values of line {lines[0]}", QMessageBox.ButtonRole.AcceptRole) for lines in group.lines
            ]
            box.addButton(QMessageBox.StandardButton.Cancel)
            apply_to_all = QCheckBox("Apply to all")
            box.setCheckBox(apply_to_all)
            box.exec()
            if box.clickedButton() is separate_button:
                choice = None
            elif box.clickedButton() in use_buttons:
                choice = use_buttons.index(box.clickedButton())
            else:
                return None
            choices[group.number] = choice
            if apply_to_all.isChecked():
                applied = (choice,)
        return choices

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

    ### AI assistant (MCP) ###
    @property
    def mcp_controller(self):
        """Built on first use, so a window that never turns AI access on pays nothing."""
        if self._mcp_controller is None:
            from RAPID2_Agilent.mcp.controller import McpController

            self._mcp_controller = McpController(self, parent=self.view)
            self._mcp_controller.state_changed.connect(self._refresh_ai_assistant_state)
        return self._mcp_controller

    def apply_ai_assistant_setting(self) -> None:
        """Turn AI access on at startup if the user left it on."""
        if self.model.settings_model.mcp_server_enabled:
            self.mcp_controller.apply_enabled(True)

    def set_ai_assistant_access(self, enabled: bool) -> None:
        controller = self.mcp_controller
        controller.apply_enabled(enabled)
        self.model.settings_model.mcp_server_enabled = controller.running
        if enabled and not controller.running:
            QMessageBox.warning(self.view, "AI Assistant Access", f"AI assistant access could not start.\n\n{controller.last_error}")
        self._refresh_ai_assistant_state()

    def copy_ai_assistant_command(self, client: str = "claude") -> None:
        """Copy the one-time `claude mcp add ...` (or `codex mcp add ...`) line for this computer."""
        command = self.mcp_controller.client_command(client)
        if command:
            QApplication.clipboard().setText(command)

    def _refresh_ai_assistant_state(self) -> None:
        controller = self._mcp_controller
        running = controller is not None and controller.running
        self.view.set_ai_assistant_state(running, controller.label if running else None)

    def shutdown(self) -> None:
        """Stop serving and give up the window letter. Called as the application quits."""
        if self._mcp_controller is not None:
            self._mcp_controller.release()
