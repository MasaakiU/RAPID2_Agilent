

from pathlib import Path
from PyQt6.QtCore import QSettings, pyqtSignal, QObject

from .. import config

class SettingsModel(QObject):
    settings_status_changed = pyqtSignal(str, object)
    def __init__(self):
        super().__init__()
        self.settings = QSettings(config.ORG_NAME, config.APP_NAME)
        # """保存場所
        # win: HKEY_CURRENT_USER\Software\MU_APPS\MyChemApp
        # mac: ~/Library/Preferences/com.mu_apps.MyChemApp.plist
        # linux: ~/.config/MU_APPS/MyChemApp.conf
        # """
    def get_last_dir(self) -> str:
        return self.settings.value("last_dir", config.DEFAULT_HOME, type=str)

    def save_last_dir(self, file_path) -> None:
        if file_path:
            # ファイルのフルパスからディレクトリ部分だけを抜き出す
            directory = Path(file_path).parent.as_posix()
            self.settings.setValue("last_dir", directory)
            self.settings_status_changed.emit("last_dir", directory)
    @property
    def mcp_server_enabled(self) -> bool:
        """Whether AI assistant access (the MCP server) starts with the window."""
        return self.settings.value("mcp_server_enabled", False, type=bool)
    @mcp_server_enabled.setter
    def mcp_server_enabled(self, value: bool):
        self.settings.setValue("mcp_server_enabled", bool(value))
        self.settings_status_changed.emit("mcp_server_enabled", bool(value))
    @property
    def target_table_shown_fields(self) -> list[str] | None:
        """Target fields whose columns are shown; None until chosen."""
        if not self.settings.contains("target_table_shown_fields"):
            return None
        # a comma-separated string: an empty list does not survive every QSettings backend
        text = self.settings.value("target_table_shown_fields", "", type=str)
        return [field for field in text.split(",") if field]
    @target_table_shown_fields.setter
    def target_table_shown_fields(self, value: list[str]):
        self.settings.setValue("target_table_shown_fields", ",".join(value))
        self.settings_status_changed.emit("target_table_shown_fields", list(value))
    @property
    def target_table_partial_fields(self) -> list[str] | None:
        """The last choice of some target table columns, which "All" returns to from none; None until chosen."""
        if not self.settings.contains("target_table_partial_fields"):
            return None
        text = self.settings.value("target_table_partial_fields", "", type=str)
        return [field for field in text.split(",") if field]
    @target_table_partial_fields.setter
    def target_table_partial_fields(self, value: list[str]):
        self.settings.setValue("target_table_partial_fields", ",".join(value))
        self.settings_status_changed.emit("target_table_partial_fields", list(value))
    @property
    def link_rt_windows(self) -> bool:
        """Whether a target's RT window is one for all files, rather than each file's own, the target's peak in it."""
        return self.settings.value("link_rt_windows", True, type=bool)
    @link_rt_windows.setter
    def link_rt_windows(self, value: bool):
        self.settings.setValue("link_rt_windows", bool(value))
        self.settings_status_changed.emit("link_rt_windows", bool(value))
    @property
    def ms2_merge_ppm(self) -> int:
        """MS2 precursors within this many ppm of each other are listed as one; 0 lists each value."""
        return self.settings.value("ms2_merge_ppm", 0, type=int)
    @ms2_merge_ppm.setter
    def ms2_merge_ppm(self, value: int):
        self.settings.setValue("ms2_merge_ppm", int(value))
        self.settings_status_changed.emit("ms2_merge_ppm", int(value))
    # @property
    # def show_atom_indices(self) -> bool:
    #     return self.settings.value("show_atom_indices", True, type=bool)
    # @show_atom_indices.setter
    # def show_atom_indices(self, value: bool):
    #     self.settings.setValue("show_atom_indices", value)
    #     self.settings_status_changed.emit("show_atom_indices", value)
    # @property
    # def automatic_indices_update(self) -> bool:
    #     return self.settings.value("automatic_indices_update", True, type=bool)
