"""RAPID2's .rpd files read as AgilentReader reads .d folders (MSData), from the arrays of all their scans in memory."""

from pathlib import Path

import numpy as np

from .rpd_file import read_rpd
from .scan_points_reader import ScanPointsReader

# the codes AgilentReader reads of the scan records, for the names RAPID2 kept: IonMode, ScanType and IonPolarity
ION_MODES = {"Esi": 64}
SCAN_TYPES = {"Scan": 1}
POLARITIES = {"Positive": 1, "Negative": -1}


class RPDReader(ScanPointsReader):
    """Read a RAPID2 .rpd file: MS1 scans converted from mzData, of one polarity in RAPID2's own files (they keep one
    setting; those converted here keep the polarity of each scan as well). Profile data
    ("continuous") is read as AgilentReader reads it; of centroid data ("discrete") a scan with no peak in an m/z
    window has 0 there, not no data points, so mz_window_gaps leaves none out."""

    def __init__(self):
        super().__init__()
        self.data_hash = ""  # the SHA-256 the file holds, of its content

    def load(self, file_path: Path):
        self.close()
        data = read_rpd(file_path)
        scan_mode = data.spectrum_settings_dict.get("ScanMode")
        if scan_mode not in SCAN_TYPES:
            raise ValueError(f"{Path(file_path).name}: scans of ScanMode {scan_mode!r} are not read, only MS1 (Scan)")
        self.file_path = Path(file_path)
        self.N_scans = len(data.RT_list)
        self.spectrum_type = data.spectrum_type
        self.data_hash = data.data_hash
        self.RT_list = data.RT_list
        # one setting for all the scans, but for the polarity of each in the files converted here
        self.MS_level_list = np.ones(self.N_scans, dtype=np.int64)
        polarities = data.polarity_list or [data.spectrum_settings_dict.get("Polarity")] * self.N_scans
        self.polarity_list = np.array([POLARITIES.get(polarity, 0) for polarity in polarities], dtype=np.int64)
        self.ion_mode_list = np.full(self.N_scans, ION_MODES.get(data.ionization_type, 0), dtype=np.int64)
        self.scan_type_list = np.full(self.N_scans, SCAN_TYPES[scan_mode], dtype=np.int64)
        self.precursor_mz_list, self.isolation_width_list, self.collision_energy_list = (
            np.zeros(self.N_scans) for _ in range(3)
        )
        # a row a scan, NaN and 0 after its points: the rows one after another, without a copy
        columns = data.mz_set.shape[1]
        self._set_points(
            data.mz_set.astype(np.float64, copy=False).reshape(-1),
            data.inten_set.reshape(-1),
            np.arange(self.N_scans, dtype=np.int64) * columns,
            data.point_counts,
        )
