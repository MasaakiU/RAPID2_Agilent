"""Qt-facing MS data models; raw-data access lives in RAPID2_Agilent.io."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PyQt6.QtCore import pyqtSignal, QObject

from ..io.agilent_reader import AgilentReader, MSCalibInfo, MSProfile, MSScanInfo


class BaseMSDataModel(QObject):
    settings_status_changed = pyqtSignal(str, object)

    @property
    def file_path(self) -> Path | None:
        raise NotImplementedError

    @property
    def N_scans(self) -> int:
        raise NotImplementedError

    @property
    def RT_list(self) -> np.ndarray | None:
        raise NotImplementedError

    @property
    def MS_level_list(self) -> np.ndarray | None:
        raise NotImplementedError

    def load(self, path: Path):
        raise NotImplementedError

    def scan_indices_for_ms_level(self, ms_level: int | None) -> np.ndarray:
        raise NotImplementedError

    def extract_spectrum(self, scan_index: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def extract_averaged_spectrum(self, rt_min: float, rt_max: float, mode: str = "mean", *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, int]:
        raise NotImplementedError

    def extract_chromatogram(self, low_mz: float, high_mz: float, *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def extract_tic(self, *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def profile_mz_range(self, scan_indices: np.ndarray) -> Tuple[float, float]:
        raise NotImplementedError

    def max_into_bins(self, scan_index: int, edges: np.ndarray, out: np.ndarray) -> None:
        raise NotImplementedError

    def extract_bpc(self, *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_scan_type_summary(self) -> list[tuple[str, int]]:
        return []

class RPDDataSet(QObject):
    """Group one loaded MS data model with derived variants such as deisotoped data."""

    settings_status_changed = pyqtSignal(str, object)

    def __init__(self, original: BaseMSDataModel):
        super().__init__()
        self.original = original

class RPDDataSetList(QObject):
    """Container for multiple loaded data sets."""

    settings_status_changed = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        self._data_sets: List[RPDDataSet] = []

    def add(self, data_set: RPDDataSet) -> None:
        self._data_sets.append(data_set)

    def add_original(self, ms_data_model: BaseMSDataModel) -> RPDDataSet:
        """Create a new data set from an original MS data model."""
        data_set = RPDDataSet(original=ms_data_model)
        self.add(data_set)
        return data_set

class AgilentDModel(BaseMSDataModel):
    """Qt-facing model delegating raw-data access to an AgilentReader."""

    def __init__(self):
        super().__init__()
        self.reader = AgilentReader()

    @property
    def file_path(self) -> Path | None:
        return self.reader.file_path

    @property
    def N_scans(self) -> int:
        return self.reader.N_scans

    @property
    def RT_list(self) -> np.ndarray | None:
        return self.reader.RT_list

    @property
    def MS_level_list(self) -> np.ndarray | None:
        return self.reader.MS_level_list

    @property
    def AcqData_path(self) -> Path:
        return self.reader.AcqData_path

    @property
    def ms_scan_info(self) -> MSScanInfo | None:
        return self.reader.ms_scan_info

    @property
    def ms_calib_info(self) -> MSCalibInfo | None:
        return self.reader.ms_calib_info

    @property
    def ms_profile(self) -> MSProfile | None:
        return self.reader.ms_profile

    @property
    def spectrum_param_index(self) -> int:
        return self.reader.spectrum_param_index

    @spectrum_param_index.setter
    def spectrum_param_index(self, value: int) -> None:
        self.reader.spectrum_param_index = value

    def load(self, folder_path: Path):
        self.reader.load(folder_path)

    def close(self):
        self.reader.close()

    def scan_indices_for_ms_level(self, ms_level: int | None) -> np.ndarray:
        return self.reader.scan_indices_for_ms_level(ms_level)

    def get_scan_type_summary(self) -> list[tuple[str, int]]:
        return self.reader.get_scan_type_summary()

    def extract_spectrum(self, scan_index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.reader.extract_spectrum(scan_index)

    def extract_averaged_spectrum(self, rt_min: float, rt_max: float, mode: str = "mean", *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, int]:
        return self.reader.extract_averaged_spectrum(rt_min, rt_max, mode, scan_indices=scan_indices)

    def extract_averaged_spectrum_python(self, rt_min: float, rt_max: float, mode: str = "mean", *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, int]:
        return self.reader.extract_averaged_spectrum_python(rt_min, rt_max, mode, scan_indices=scan_indices)

    def extract_averaged_spectrum_native_bins(self, rt_min: float, rt_max: float, mode: str = "mean", *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, int]:
        return self.reader.extract_averaged_spectrum_native_bins(rt_min, rt_max, mode, scan_indices=scan_indices)

    def extract_chromatogram(self, low_mz: float, high_mz: float, *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        return self.reader.extract_chromatogram(low_mz, high_mz, scan_indices=scan_indices)

    def extract_tic(self, *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        return self.reader.extract_tic(scan_indices=scan_indices)

    def profile_mz_range(self, scan_indices: np.ndarray) -> Tuple[float, float]:
        return self.reader.profile_mz_range(scan_indices)

    def max_into_bins(self, scan_index: int, edges: np.ndarray, out: np.ndarray) -> None:
        self.reader.max_into_bins(scan_index, edges, out)

    def extract_bpc(self, *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        return self.reader.extract_bpc(scan_indices=scan_indices)

class RPDFileModel(BaseMSDataModel):
    def load(self, file_path: Path):
        raise NotImplementedError("Loading .rpd files is not implemented yet.")
