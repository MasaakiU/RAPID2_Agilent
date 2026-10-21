

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
