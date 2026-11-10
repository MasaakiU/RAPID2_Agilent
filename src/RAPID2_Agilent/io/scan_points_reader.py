"""MSData of spectra held in memory: the points of all the scans in one array, each scan's from its start."""

from collections import Counter
from pathlib import Path
from typing import Tuple

import numpy as np

POLARITY_NAMES = {1: "Positive", -1: "Negative", 0: "Polarity unknown"}
# MSData's spectra, of a spectrum_type
SPECTRA_OF_TYPE = {"continuous": "profile", "discrete": "centroid"}


class ScanPointsReader:
    """The MSData of a reader that holds the points of its scans in memory: RPDReader's .rpd files and
    AgilentCentroidReader's centroids. A subclass's load sets the attributes and calls _set_points. Profile data
    ("continuous") is read as AgilentReader reads it; of centroid data ("discrete") a scan with no peak in an m/z
    window has 0 there, not no data points, so mz_window_gaps leaves none out."""

    def __init__(self):
        self.file_path: Path | None = None
        self.N_scans = 0
        self.RT_list: np.ndarray | None = None
        self.MS_level_list: np.ndarray | None = None
        self.polarity_list: np.ndarray | None = None
        self.ion_mode_list: np.ndarray | None = None
        self.scan_type_list: np.ndarray | None = None
        self.precursor_mz_list: np.ndarray | None = None
        self.isolation_width_list: np.ndarray | None = None
        self.collision_energy_list: np.ndarray | None = None
        self.spectrum_type: str | None = None  # "continuous" (profile) or "discrete" (centroid)
        self._mz: np.ndarray | None = None  # the points of all the scans, float64; a scan's ascend
        self._inten: np.ndarray | None = None
        self._starts: np.ndarray | None = None  # of each scan, where its points start
        self._point_counts: np.ndarray | None = None
        self._tic: np.ndarray | None = None
        self._bpc: np.ndarray | None = None

    @property
    def spectra(self) -> str | None:
        """MSData's "profile" or "centroid", of spectrum_type."""
        return SPECTRA_OF_TYPE.get(self.spectrum_type)

    def _set_points(self, mz: np.ndarray, inten: np.ndarray, starts: np.ndarray, point_counts: np.ndarray,
                    tic: np.ndarray | None = None, bpc: np.ndarray | None = None) -> None:
        """The points of the scans; the TIC and BPC, if not given, are their sum and max."""
        self._mz, self._inten = mz, inten
        self._starts, self._point_counts = np.asarray(starts, dtype=np.int64), np.asarray(point_counts, dtype=np.int64)
        sum_dtype = self._sum_dtype()
        self._tic = tic if tic is not None else np.array(
            [self._points(i)[1].sum(dtype=sum_dtype) for i in range(len(self._starts))], dtype=sum_dtype
        )
        self._bpc = bpc if bpc is not None else np.array(
            [self._points(i)[1].max(initial=0) for i in range(len(self._starts))], dtype=inten.dtype
        )

    def _sum_dtype(self):
        """Intensities are summed as int64 if integers (as AgilentReader sums the profile's), else as float64."""
        return np.int64 if np.issubdtype(self._inten.dtype, np.integer) else np.float64

    def _points(self, scan_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """The m/z and intensities of a scan's points, views of the arrays."""
        start = self._starts[scan_index]
        stop = start + self._point_counts[scan_index]
        return self._mz[start:stop], self._inten[start:stop]

    def close(self):
        self.__init__()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def scan_indices_for_ms_level(self, ms_level: int | None) -> np.ndarray:
        """Indices of the scans of an MS level; None selects all scans."""
        if ms_level is None:
            return np.arange(self.N_scans)
        scan_indices = np.flatnonzero(self.MS_level_list == ms_level)
        if len(scan_indices) == 0:
            raise ValueError(f"No MS{ms_level} scans in {self.file_path}")
        return scan_indices

    def get_scan_type_summary(self) -> list[tuple[str, int]]:
        """The number of scans of each polarity and MS level, positive first."""
        if self.polarity_list is None:
            return []
        counts = Counter(zip(self.polarity_list.tolist(), self.MS_level_list.tolist()))
        order = list(POLARITY_NAMES)
        return [
            (f"{POLARITY_NAMES[polarity]} MS{ms_level}", count)
            for (polarity, ms_level), count in sorted(counts.items(), key=lambda item: (order.index(item[0][0]), item[0][1]))
        ]

    def _select_scans_in_rt_window(self, rt_min: float, rt_max: float, scan_indices: np.ndarray | None) -> Tuple[np.ndarray, int]:
        """As AgilentReader's: the scans in the RT window, or the nearest one if none; and the one closest to the
        window center."""
        if scan_indices is None:
            scan_indices = np.arange(self.N_scans)
        rt_center = (rt_min + rt_max) / 2
        rt_list = self.RT_list[scan_indices]
        window_scan_indices = scan_indices[(rt_min <= rt_list) & (rt_list < rt_max)]
        if len(window_scan_indices) == 0:
            window_scan_indices = scan_indices[[int(np.argmin(np.abs(rt_list - rt_center)))]]
        representative_scan_index = int(window_scan_indices[np.argmin(np.abs(self.RT_list[window_scan_indices] - rt_center))])
        return window_scan_indices, representative_scan_index

    def _window(self, scan_index: int, low_mz: float, high_mz: float) -> Tuple[int, int]:
        """The data points of a scan in [low_mz, high_mz): [start, stop), as AgilentReader finds them."""
        mz_list = self._points(scan_index)[0]
        return int(np.searchsorted(mz_list, low_mz, side="left")), int(np.searchsorted(mz_list, high_mz, side="left"))

    @staticmethod
    def _accumulate_nearest(common_mz: np.ndarray, mz_list: np.ndarray, inten_list: np.ndarray, accumulated: np.ndarray) -> None:
        """Each intensity added to the nearest m/z of common_mz, as AgilentReader's averaged spectra are made."""
        insert_indices = np.searchsorted(common_mz, mz_list)
        left_indices = np.clip(insert_indices - 1, 0, len(common_mz) - 1)
        right_indices = np.clip(insert_indices, 0, len(common_mz) - 1)
        use_right = np.abs(common_mz[right_indices] - mz_list) < np.abs(mz_list - common_mz[left_indices])
        accumulated += np.bincount(
            np.where(use_right, right_indices, left_indices),
            weights=inten_list.astype(np.float64, copy=False),
            minlength=len(common_mz),
        )

    ### extract data ###
    def extract_spectrum(self, scan_index: int) -> Tuple[np.ndarray, np.ndarray]:
        mz_list, inten_list = self._points(scan_index)
        return mz_list.copy(), inten_list.copy()

    def extract_averaged_spectrum(self, rt_min: float, rt_max: float, mode: str = "mean", *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """A sum or mean spectrum across scans in an RT window, on the m/z of the scan closest to the window center,
        each scan's intensities added to the nearest of them."""
        if mode not in {"mean", "sum"}:
            raise ValueError(f"mode must be 'mean' or 'sum', got {mode!r}")
        if self.RT_list is None or len(self.RT_list) == 0:
            raise ValueError("RT_list is empty; load a data file before extracting spectra")
        scan_indices, representative_scan_index = self._select_scans_in_rt_window(rt_min, rt_max, scan_indices)
        if self._point_counts[representative_scan_index] == 0:
            # a scan of no points (the first of an Agilent run's centroids): the nearest of the window's with any
            with_points = scan_indices[self._point_counts[scan_indices] > 0]
            if len(with_points):
                rt_center = (rt_min + rt_max) / 2
                representative_scan_index = int(with_points[np.argmin(np.abs(self.RT_list[with_points] - rt_center))])
        common_mz, _ = self.extract_spectrum(representative_scan_index)
        accumulated_intensity = np.zeros(len(common_mz), dtype=np.float64)
        for scan_index in scan_indices if len(common_mz) else ():
            self._accumulate_nearest(common_mz, *self._points(int(scan_index)), accumulated_intensity)
        if mode == "mean":
            accumulated_intensity /= len(scan_indices)
        return common_mz, accumulated_intensity, representative_scan_index

    def extract_chromatogram(self, low_mz: float, high_mz: float, *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        if scan_indices is None:
            scan_indices = np.arange(self.N_scans)
        sum_dtype = self._sum_dtype()
        inten_list = np.zeros(len(scan_indices), dtype=sum_dtype)
        for i, scan_index in enumerate(scan_indices):
            start_index, stop_index = self._window(int(scan_index), low_mz, high_mz)
            inten_list[i] = self._points(int(scan_index))[1][start_index:stop_index].sum(dtype=sum_dtype)
        return self.RT_list[scan_indices], inten_list

    def extract_averaged_sub_spectrum(self, low_mz: float, high_mz: float, *, scan_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """As AgilentReader's: the mean spectrum of the scans in [low_mz, high_mz) only, on the data points of the
        scan nearest to the middle of their RTs (or the next with any there), each scan's added to the nearest."""
        common_mz = None
        accumulated_intensity = None
        rt_list = self.RT_list[scan_indices]
        representative = int(scan_indices[np.argmin(np.abs(rt_list - (rt_list.min() + rt_list.max()) / 2))])
        for scan_index in [representative] + [int(i) for i in scan_indices if int(i) != representative]:
            start_index, stop_index = self._window(scan_index, low_mz, high_mz)
            if stop_index <= start_index:
                continue
            mz_list, inten_list = (points[start_index:stop_index] for points in self._points(scan_index))
            if common_mz is None:
                common_mz, accumulated_intensity = mz_list.copy(), np.zeros(len(mz_list))
            self._accumulate_nearest(common_mz, mz_list, inten_list, accumulated_intensity)
        if common_mz is None:
            return np.array([]), np.array([])
        return common_mz, accumulated_intensity / len(scan_indices)

    def mz_window_gaps(self, low_mz: float, high_mz: float, *, scan_indices: np.ndarray | None = None) -> np.ndarray:
        """As AgilentReader's, of each scan with no data point in [low_mz, high_mz): the m/z spacing of the data
        points on either side of the window, or NaN when the window is beyond the scan's data; 0 for a scan with
        one, and for every scan of centroid data."""
        if scan_indices is None:
            scan_indices = np.arange(self.N_scans)
        gaps = np.zeros(len(scan_indices))
        if self.spectrum_type == "discrete":
            return gaps
        for i, scan_index in enumerate(scan_indices):
            scan_index = int(scan_index)
            start_index, stop_index = self._window(scan_index, low_mz, high_mz)
            if start_index != stop_index:
                continue
            mz_list = self._points(scan_index)[0]
            if 0 < start_index < len(mz_list):
                gaps[i] = mz_list[start_index] - mz_list[start_index - 1]
            else:
                gaps[i] = np.nan
        return gaps

    def extract_tic(self, *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        if scan_indices is None:
            scan_indices = np.arange(self.N_scans)
        return self.RT_list[scan_indices], self._tic[scan_indices]

    def extract_bpc(self, *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        if scan_indices is None:
            scan_indices = np.arange(self.N_scans)
        return self.RT_list[scan_indices], self._bpc[scan_indices]

    def profile_mz_range(self, scan_indices: np.ndarray) -> Tuple[float, float]:
        """m/z range covered by the spectra of the scans."""
        scan_indices = np.asarray(scan_indices)
        counts = self._point_counts[scan_indices]
        starts = self._starts[scan_indices][counts > 0]
        if len(starts) == 0:
            return np.nan, np.nan
        return float(self._mz[starts].min()), float(self._mz[starts + counts[counts > 0] - 1].max())

    def max_into_bins(self, scan_index: int, edges: np.ndarray, out: np.ndarray) -> None:
        """Max intensity of a scan in each m/z bin [edges[i], edges[i+1]), written into out (float32) where higher."""
        start_index, stop_index = self._window(scan_index, float(edges[0]), float(edges[-1]))
        if stop_index <= start_index:
            return
        mz_list, inten_list = (points[start_index:stop_index] for points in self._points(scan_index))
        bins = np.searchsorted(edges, mz_list, side="right") - 1
        # the data points ascend, and so their bins: the max of each run of one bin
        starts = np.flatnonzero(np.diff(bins, prepend=-1))
        highest = np.maximum.reduceat(inten_list, starts).astype(np.float32)
        np.maximum(out[bins[starts]], highest, out=highest)
        out[bins[starts]] = highest
