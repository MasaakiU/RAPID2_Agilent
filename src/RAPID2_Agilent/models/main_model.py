from __future__ import annotations

import json
import time

from .rpd_model import BaseMSDataModel, RPDDataSet, RPDDataSetList
from .. import config
from ..models.settings_model import SettingsModel
from .target_model import TargetListModel

class MainModel:
    def __init__(self):
        self.rpd_data_set_list = RPDDataSetList()
        self.settings_model = SettingsModel()
        self.target_list_model = TargetListModel()
        self.run_peaks = self.target_list_model.run_peaks  # its edits are undone with those of the targets

    def add_ms_data_model(self, ms_data_model: BaseMSDataModel) -> RPDDataSet:
        return self.rpd_data_set_list.add_original(ms_data_model)

    def keep_ms_data_models(self, ms_data_models: list[BaseMSDataModel]) -> None:
        self.rpd_data_set_list.keep_originals(ms_data_models)

    # def on_save_clicked(self, save_path: str):
    #     meta_info = {
    #         "app_info": {
    #             "app_name": config.APP_NAME,
    #             "app_version": config.APP_VERSION,
    #             "schema_version": config.SCHEMA_VERSION,
    #             "save_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    #             "note": config.DEFAULT_NOTE,
    #         }, 
    #         "data_list": {
    #             "smiles": self.chem_model.smiles,
    #             "command_data_list": self.command_model.serialize(),
    #         }
    #     }
    #     self.svg_model.update_meta_info(json.dumps(meta_info, ensure_ascii=False, indent=2))
    #     self.svg_model.save(save_path)
    def on_open_clicked(self, open_path: str) -> dict:
        self.svg_model = SVGModel(file_path=open_path)
        meta_info = self.svg_model.extract_meta_info()
        if meta_info is not None:
            meta_dict = json.loads(meta_info)
            smiles = meta_dict["data_list"]["smiles"]
            command_data_list = meta_dict["data_list"]["command_data_list"]
            self.load_chem_model(smiles=smiles)
            self.command_model.deserialize(command_data_list)
            self.update_svg_model()
            return meta_dict["app_info"]
        else:
            raise Exception("No meta info found in the loaded SVG file.")
