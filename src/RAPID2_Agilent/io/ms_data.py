"""What every MS data source answers, whoever reads it.

`AgilentReader` implements it, and the GUI's `models.rpd_model.BaseMSDataModel`
delegates to a reader with the same names. Code above this layer -- the m/z x
RT map, `services.operations`, the GUI presenters -- is written against this
protocol, so it runs unchanged on either.
"""

from typing import Protocol

import numpy as np


class MSData(Protocol):
    @property
    def RT_list(self) -> np.ndarray | None: ...

    @property
    def MS_level_list(self) -> np.ndarray | None: ...

    # of each scan: +1 for positive ions, -1 for negative, as the sign of an adduct's charge
    @property
    def polarity_list(self) -> np.ndarray | None: ...

    # of each scan: the codes of the ionization (IonMode) and of the kind of scan (ScanType)
    @property
    def ion_mode_list(self) -> np.ndarray | None: ...

    @property
    def scan_type_list(self) -> np.ndarray | None: ...

    # of each scan: the precursor m/z isolated, its isolation width and the collision energy; 0 for MS1 scans
    @property
    def precursor_mz_list(self) -> np.ndarray | None: ...

    @property
    def isolation_width_list(self) -> np.ndarray | None: ...

    @property
    def collision_energy_list(self) -> np.ndarray | None: ...

    def scan_indices_for_ms_level(self, ms_level: int | None) -> np.ndarray: ...

    def get_scan_type_summary(self) -> list[tuple[str, int]]: ...

    def extract_chromatogram(self, low_mz: float, high_mz: float, *, scan_indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]: ...

    def mz_window_gaps(self, low_mz: float, high_mz: float, *, scan_indices: np.ndarray | None = None) -> np.ndarray: ...

    def extract_tic(self, *, scan_indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]: ...

    def extract_bpc(self, *, scan_indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]: ...

    def extract_averaged_spectrum(self, rt_min: float, rt_max: float, mode: str = "mean", *, scan_indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]: ...

    def extract_averaged_sub_spectrum(self, low_mz: float, high_mz: float, *, scan_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def profile_mz_range(self, scan_indices: np.ndarray) -> tuple[float, float]: ...

    def max_into_bins(self, scan_index: int, edges: np.ndarray, out: np.ndarray) -> None: ...
