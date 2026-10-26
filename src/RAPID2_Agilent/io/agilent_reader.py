"""Agilent raw-data access shared by the GUI and batch processing."""

import mmap
import pprint
import struct
from collections import Counter
import numpy as np
from lxml import etree
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Tuple

from ._native.decode_inten_list import decode_inten_list
from ._native.extract_sub_spectrum import extract_sub_spectrum, sum_sub_spectrum
from ._native.calibrate_mz import calculate_poly, calibrate_mz
from ._native.build_checkpoints import build_checkpoints
from ._native.average_spectrum import accumulate_decoded_calibrated_nearest_bins, accumulate_nearest_bins
from ._native.mz_rt_binning import max_into_bins
from ._native.lzf_profile import rewrite_lzf_records


class AgilentReader:
    """Read Agilent .d scans without importing Qt or GUI models."""

    def __init__(self):
        self.file_path: Path | None = None
        self.N_scans = 0
        self.RT_list: np.ndarray | None = None
        self.MS_level_list: np.ndarray | None = None
        self.polarity_list: np.ndarray | None = None  # +1 positive, -1 negative
        self.ion_mode_list: np.ndarray | None = None  # IonMode, e.g. 64 for ESI
        self.scan_type_list: np.ndarray | None = None  # ScanType, e.g. 1 for Scan, 512 for ProductIon
        # of each scan: the precursor m/z isolated, its isolation width and the collision energy; 0 for MS1 scans
        self.precursor_mz_list: np.ndarray | None = None
        self.isolation_width_list: np.ndarray | None = None
        self.collision_energy_list: np.ndarray | None = None
        self.ms_scan_info: MSScanInfo = None
        self.ms_calib_info: MSCalibInfo = None
        self.ms_profile: MSProfile = None
        self.spectrum_param_index: int = 0  # profile or centroid, depending on data

    def scan_indices_for_ms_level(self, ms_level: int | None) -> np.ndarray:
        """Indices of the scans of an MS level; None selects all scans."""
        if ms_level is None:
            return np.arange(self.N_scans)
        scan_indices = np.flatnonzero(self.MS_level_list == ms_level)
        if len(scan_indices) == 0:
            raise ValueError(f"No MS{ms_level} scans in {self.file_path}")
        return scan_indices

    def _select_scans_in_rt_window(self, rt_min: float, rt_max: float, scan_indices: np.ndarray | None) -> Tuple[np.ndarray, int]:
        """Scans (among scan_indices, or all scans if None) in the RT window, or the nearest one if none; and the one closest to the window center."""
        if scan_indices is None:
            scan_indices = np.arange(self.N_scans)
        rt_center = (rt_min + rt_max) / 2
        rt_list = self.RT_list[scan_indices]
        window_scan_indices = scan_indices[(rt_min <= rt_list) & (rt_list < rt_max)]
        if len(window_scan_indices) == 0:
            window_scan_indices = scan_indices[[int(np.argmin(np.abs(rt_list - rt_center)))]]
        representative_scan_index = int(window_scan_indices[np.argmin(np.abs(self.RT_list[window_scan_indices] - rt_center))])
        return window_scan_indices, representative_scan_index

    @property
    def AcqData_path(self) -> Path:
        return self.file_path / "AcqData"

    # load_file_related_methods
    def load(self, folder_path: Path):
        self.close()
        self.file_path = Path(folder_path)
        # parse MSTS
        for time_seg in etree.parse(self.AcqData_path / "MSTS.xml").getroot().findall("TimeSegment"):
            self.N_scans += int(time_seg.find("NumOfScans").text)
        # parse MSScan
        self.ms_scan_info = MSScanInfo(
            xsd_path=self.AcqData_path / "MSScan.xsd", 
            bin_path=self.AcqData_path / "MSScan.bin", 
            N_scans=self.N_scans
        )
        # parse_MSCalib
        self.ms_calib_info = MSCalibInfo(
            xml_path=self.AcqData_path / "DefaultMassCal.xml",
            bin_path=self.AcqData_path / "MSMassCal.bin", 
            mass_cal_offsets=[
                self.ms_scan_info.get_scan_info(i, "MassCalOffset")
                for i in range(self.N_scans)
            ]
        )
        # parse_MSProfile
        self.ms_profile = MSProfile(
            bin_path=self.AcqData_path / "MSProfile.bin",
        )
        # LZF records of older acquisitions (e.g. of 2022) are read from memory, written again as the newer ones are:
        # the offsets and byte counts are then of them there
        spectra = [scan["SpectrumParamValues"][self.spectrum_param_index] for scan in self.ms_scan_info.scan_info]
        for spectrum, (offset, byte_count) in zip(spectra, self.ms_profile.rewrite_lzf_records(spectra)):
            spectrum["SpectrumOffset"], spectrum["ByteCount"] = offset, byte_count
        self.ms_profile.set_checkpoints(
            [self.ms_scan_info.get_spectrum_info(i, self.spectrum_param_index, "SpectrumOffset") for i in range(self.N_scans)],
            [self.ms_scan_info.get_spectrum_info(i, self.spectrum_param_index, "ByteCount") for i in range(self.N_scans)]
        )
        self.ms_calib_info.configure_poly_cache(
            scan_grids=[
                self.ms_profile.decode_header(
                    self.ms_scan_info.get_spectrum_info(i, self.spectrum_param_index, "SpectrumOffset")
                )[:3]
                for i in range(self.N_scans)
            ],
            calibration_ids=[self.ms_scan_info.get_scan_info(i, "CalibrationID") for i in range(self.N_scans)],
        )
        # RT_list is extracted because it will be used frequently
        self.RT_list = np.array([
            self.ms_scan_info.get_scan_info(i, "ScanTime")
            for i in range(self.N_scans)
        ])
        self.MS_level_list = np.array([
            self.ms_scan_info.get_scan_info(i, "MSLevel")
            for i in range(self.N_scans)
        ])
        # the codes of the ionization and of the kind of scan, e.g. 64 (ESI); 1 (Scan) and 512 (ProductIon)
        self.ion_mode_list, self.scan_type_list = (
            np.array([self.ms_scan_info.get_scan_info(i, field) for i in range(self.N_scans)], dtype=np.int64)
            for field in ("IonMode", "ScanType")
        )
        # +1 for positive ions, -1 for negative: the sign of an adduct's charge (IonPolarity 0 and 1; 0 if neither)
        self.polarity_list = np.array([
            {0: 1, 1: -1}.get(self.ms_scan_info.get_scan_info(i, "IonPolarity"), 0)
            for i in range(self.N_scans)
        ])
        # which MS2 scans to use is chosen by these; the older MSScan.xsd (e.g. of 2022) has no MzOfInterest and
        # MzIsolationWidth, 0 then
        self.precursor_mz_list, self.isolation_width_list, self.collision_energy_list = (
            np.array([self.ms_scan_info.scan_info[i].get(field, 0.0) for i in range(self.N_scans)], dtype=np.float64)
            for field in ("MzOfInterest", "MzIsolationWidth", "CollisionEnergy")
        )

    def close(self):
        """Release the mapped profile and return to the unloaded state."""
        if self.ms_profile is not None:
            self.ms_profile.close()
        self.ms_profile = None
        self.ms_scan_info = None
        self.ms_calib_info = None
        self.file_path = None
        self.N_scans = 0
        self.RT_list = None
        self.MS_level_list = None
        self.polarity_list = None
        self.ion_mode_list = None
        self.scan_type_list = None
        self.precursor_mz_list = None
        self.isolation_width_list = None
        self.collision_energy_list = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def get_scan_type_summary(self) -> list[tuple[str, int]]:
        counter = Counter(
            (
                self.ms_scan_info.get_scan_info(scan_index, "IonPolarity"),
                self.ms_scan_info.get_scan_info(scan_index, "MSLevel"),
            )
            for scan_index in range(self.N_scans)
        )
        return [
            (f"{self._format_ion_polarity(polarity)} {self._format_ms_level(ms_level)}", count)
            for (polarity, ms_level), count in sorted(counter.items(), key=lambda item: (item[0][0], item[0][1]))
        ]

    @staticmethod
    def _format_ion_polarity(value: int) -> str:
        if value == 0:
            return "Positive"
        if value == 1:
            return "Negative"
        return f"Polarity {value}"

    @staticmethod
    def _format_ms_level(value: int) -> str:
        return f"MS{value}"

    ### extract data ###
    def extract_spectrum(self, scan_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract m/z and intensity arrays for a given scan index.
        """
        start_t, delta_t, inten_list = self.ms_profile.extract_full_spectrum(
            scan_index = scan_index,
            offset = self.ms_scan_info.get_spectrum_info(scan_index, self.spectrum_param_index, "SpectrumOffset"),
        )
        mz_list = self.ms_calib_info.calibrate_mz(
            scan_index, 
            N_datapoints = self.ms_scan_info.get_spectrum_info(scan_index, self.spectrum_param_index, "PointCount"), 
            calib_id = self.ms_scan_info.get_scan_info(scan_index, "CalibrationID"),
            start_t = start_t,
            delta_t = delta_t
        )
        return mz_list, inten_list
    def extract_averaged_spectrum_python(self, rt_min: float, rt_max: float, mode: str = "mean", *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Extract a sum or mean spectrum across scans in an RT window.

        The representative scan closest to the RT-window center provides the
        common m/z grid. Intensities from each selected scan are assigned to
        the nearest common m/z bin without interpolation.
        """
        if mode not in {"mean", "sum"}:
            raise ValueError(f"mode must be 'mean' or 'sum', got {mode!r}")
        if self.RT_list is None or len(self.RT_list) == 0:
            raise ValueError("RT_list is empty; load a data file before extracting spectra")

        scan_indices, representative_scan_index = self._select_scans_in_rt_window(rt_min, rt_max, scan_indices)
        common_mz, representative_intensity = self.extract_spectrum(representative_scan_index)
        accumulated_intensity = np.zeros(len(common_mz), dtype=np.float64)

        for scan_index in scan_indices:
            scan_index = int(scan_index)
            if scan_index == representative_scan_index:
                mz_list, inten_list = common_mz, representative_intensity
            else:
                mz_list, inten_list = self.extract_spectrum(scan_index)
            insert_indices = np.searchsorted(common_mz, mz_list)
            left_indices = np.clip(insert_indices - 1, 0, len(common_mz) - 1)
            right_indices = np.clip(insert_indices, 0, len(common_mz) - 1)
            use_right = (
                np.abs(common_mz[right_indices] - mz_list)
                < np.abs(mz_list - common_mz[left_indices])
            )
            nearest_indices = np.where(use_right, right_indices, left_indices)
            accumulated_intensity += np.bincount(
                nearest_indices,
                weights=inten_list.astype(np.float64, copy=False),
                minlength=len(common_mz),
            )
        if mode == "mean":
            accumulated_intensity /= len(scan_indices)
        return common_mz, accumulated_intensity, representative_scan_index
    def extract_averaged_spectrum_native_bins(self, rt_min: float, rt_max: float, mode: str = "mean", *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Native-assisted version of extract_averaged_spectrum_python.

        Scan selection and spectrum extraction stay in Python. The nearest-bin
        accumulation step is delegated to C++.
        """
        if mode not in {"mean", "sum"}:
            raise ValueError(f"mode must be 'mean' or 'sum', got {mode!r}")
        if self.RT_list is None or len(self.RT_list) == 0:
            raise ValueError("RT_list is empty; load a data file before extracting spectra")

        scan_indices, representative_scan_index = self._select_scans_in_rt_window(rt_min, rt_max, scan_indices)
        common_mz, representative_intensity = self.extract_spectrum(representative_scan_index)
        accumulated_intensity = np.zeros(len(common_mz), dtype=np.float64)

        for scan_index in scan_indices:
            scan_index = int(scan_index)
            if scan_index == representative_scan_index:
                mz_list, inten_list = common_mz, representative_intensity
            else:
                mz_list, inten_list = self.extract_spectrum(scan_index)
            accumulate_nearest_bins(common_mz, mz_list, inten_list, accumulated_intensity)

        if mode == "mean":
            accumulated_intensity /= len(scan_indices)
        return common_mz, accumulated_intensity, representative_scan_index
    def extract_averaged_spectrum(self, rt_min: float, rt_max: float, mode: str = "mean", *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Fused native extraction of a sum or mean spectrum across scans.

        Each scan's compressed intensity stream is decoded in C++ while m/z is
        calibrated on the fly and accumulated into the representative m/z grid.
        Zero runs are skipped without materializing intensity or m/z arrays.
        """
        if mode not in {"mean", "sum"}:
            raise ValueError(f"mode must be 'mean' or 'sum', got {mode!r}")
        if self.RT_list is None or len(self.RT_list) == 0:
            raise ValueError("RT_list is empty; load a data file before extracting spectra")

        scan_indices, representative_scan_index = self._select_scans_in_rt_window(rt_min, rt_max, scan_indices)
        representative_offset = self.ms_scan_info.get_spectrum_info(
            representative_scan_index,
            self.spectrum_param_index,
            "SpectrumOffset",
        )
        representative_start_t, representative_delta_t, representative_N_datapoints, _ = self.ms_profile.decode_header(representative_offset)
        common_mz = self.ms_calib_info.calibrate_mz(
            representative_scan_index,
            N_datapoints=representative_N_datapoints,
            calib_id=self.ms_scan_info.get_scan_info(representative_scan_index, "CalibrationID"),
            start_t=representative_start_t,
            delta_t=representative_delta_t,
        )
        accumulated_intensity = np.zeros(len(common_mz), dtype=np.float64)

        for scan_index in scan_indices:
            scan_index = int(scan_index)
            offset = self.ms_scan_info.get_spectrum_info(scan_index, self.spectrum_param_index, "SpectrumOffset")
            start_t, delta_t, N_datapoints, init_zero_repeat = self.ms_profile.decode_header(offset)
            check_points = self.ms_profile.check_points_list[scan_index]
            stop_len = min(N_datapoints, check_points.N_datapoints_no_end_zeros)
            coeff, base = self.ms_calib_info.calib_info[scan_index][:2]
            calib_id = self.ms_scan_info.get_scan_info(scan_index, "CalibrationID")
            poly = self.ms_calib_info.get_poly(scan_index, N_datapoints, calib_id, start_t, delta_t)

            accumulate_decoded_calibrated_nearest_bins(
                self.ms_profile._mm_view,
                int(check_points.offset_list[0]),
                int(check_points.out_index_list[0]),
                int(stop_len),
                int(check_points.cur_size_list[0]),
                common_mz,
                coeff,
                base,
                poly,
                start_t,
                delta_t,
                accumulated_intensity,
            )

        if mode == "mean":
            accumulated_intensity /= len(scan_indices)
        return common_mz, accumulated_intensity, representative_scan_index
    def extract_chromatogram(self, low_mz: float, high_mz: float, *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        if scan_indices is None:
            scan_indices = np.arange(self.N_scans)
        inten_list = []
        for scan_index in scan_indices:
            scan_index = int(scan_index)
            offset = self.ms_scan_info.get_spectrum_info(scan_index, self.spectrum_param_index, "SpectrumOffset")
            start_t, delta_t, N_datapoints, init_zero_repeat = self.ms_profile.decode_header(offset)
            calib_id = self.ms_scan_info.get_scan_info(scan_index, "CalibrationID")
            start_index = self.ms_calib_info.search_index_below_mz(scan_index, N_datapoints, calib_id, start_t, delta_t, low_mz) + 1
            stop_index = self.ms_calib_info.search_index_below_mz(scan_index, N_datapoints, calib_id, start_t, delta_t, high_mz) + 1
            inten_list.append(self.ms_profile.sum_sub_spectrum(
                scan_index, 
                offset,
                start_index,
                stop_index
            ))
        return self.RT_list[scan_indices], np.array(inten_list)
    def extract_averaged_sub_spectrum(self, low_mz: float, high_mz: float, *, scan_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The mean profile spectrum of the scans in [low_mz, high_mz) only, as extract_averaged_spectrum makes one over
        all m/z: on the data points of the scan nearest to the middle of their RTs (or the next with any there), each
        scan's added to the nearest of them."""
        common_mz = None
        accumulated_intensity = None
        rt_list = self.RT_list[scan_indices]
        representative = int(scan_indices[np.argmin(np.abs(rt_list - (rt_list.min() + rt_list.max()) / 2))])
        for scan_index in [representative] + [int(i) for i in scan_indices if int(i) != representative]:
            offset = self.ms_scan_info.get_spectrum_info(scan_index, self.spectrum_param_index, "SpectrumOffset")
            start_t, delta_t, N_datapoints, _ = self.ms_profile.decode_header(offset)
            calib_id = self.ms_scan_info.get_scan_info(scan_index, "CalibrationID")
            start_index = self.ms_calib_info.search_index_below_mz(scan_index, N_datapoints, calib_id, start_t, delta_t, low_mz) + 1
            stop_index = self.ms_calib_info.search_index_below_mz(scan_index, N_datapoints, calib_id, start_t, delta_t, high_mz) + 1
            if stop_index <= start_index:
                continue
            inten_list = self.ms_profile.extract_sub_spectrum(scan_index, offset, start_index, stop_index).astype(np.float64)
            mz_list = self.ms_calib_info.mz_values(scan_index, calib_id, start_t, delta_t, np.arange(start_index, stop_index))
            if common_mz is None:
                common_mz, accumulated_intensity = mz_list, np.zeros(len(mz_list))
            insert_indices = np.searchsorted(common_mz, mz_list)
            left_indices = np.clip(insert_indices - 1, 0, len(common_mz) - 1)
            right_indices = np.clip(insert_indices, 0, len(common_mz) - 1)
            use_right = np.abs(common_mz[right_indices] - mz_list) < np.abs(mz_list - common_mz[left_indices])
            accumulated_intensity += np.bincount(
                np.where(use_right, right_indices, left_indices), weights=inten_list, minlength=len(common_mz)
            )
        if common_mz is None:
            return np.array([]), np.array([])
        return common_mz, accumulated_intensity / len(scan_indices)
    def mz_window_gaps(self, low_mz: float, high_mz: float, *, scan_indices: np.ndarray | None = None) -> np.ndarray:
        """Of each scan with no profile data point in [low_mz, high_mz), so that extract_chromatogram sums nothing: the
        m/z spacing of the data points on either side of the window, or NaN when the window is beyond the scan's data;
        0 for a scan with one."""
        if scan_indices is None:
            scan_indices = np.arange(self.N_scans)
        gaps = np.zeros(len(scan_indices))
        for i, scan_index in enumerate(scan_indices):
            scan_index = int(scan_index)
            offset = self.ms_scan_info.get_spectrum_info(scan_index, self.spectrum_param_index, "SpectrumOffset")
            start_t, delta_t, N_datapoints, _ = self.ms_profile.decode_header(offset)
            calib_id = self.ms_scan_info.get_scan_info(scan_index, "CalibrationID")
            # as extract_chromatogram finds the data points in the window: [start_index, stop_index)
            start_index = self.ms_calib_info.search_index_below_mz(scan_index, N_datapoints, calib_id, start_t, delta_t, low_mz) + 1
            stop_index = self.ms_calib_info.search_index_below_mz(scan_index, N_datapoints, calib_id, start_t, delta_t, high_mz) + 1
            between_points = 0 < start_index < N_datapoints
            if start_index == stop_index and not between_points:
                gaps[i] = np.nan
                continue
            if not between_points:
                continue
            mz_at = self.ms_calib_info.mz_function(scan_index, calib_id, start_t, delta_t)
            spacing = mz_at(start_index) - mz_at(start_index - 1)
            if start_index == stop_index:
                gaps[i] = spacing
            # on one time grid the spacing differs between scans only by their calibration's drift: a window wider
            # than it in one scan holds a data point in every one
            elif i == 0 and self.ms_calib_info.shares_grid and high_mz - low_mz > 1.1 * spacing:
                break
        return gaps
    def profile_mz_range(self, scan_indices: np.ndarray) -> Tuple[float, float]:
        """m/z range covered by the profile spectra of the scans."""
        min_x = min(self.ms_scan_info.get_spectrum_info(int(i), self.spectrum_param_index, "MinX") for i in scan_indices)
        max_x = max(self.ms_scan_info.get_spectrum_info(int(i), self.spectrum_param_index, "MaxX") for i in scan_indices)
        return float(min_x), float(max_x)
    def max_into_bins(self, scan_index: int, edges: np.ndarray, out: np.ndarray) -> None:
        """Max profile intensity of a scan in each m/z bin [edges[i], edges[i+1]), written into out (float32)."""
        offset = self.ms_scan_info.get_spectrum_info(scan_index, self.spectrum_param_index, "SpectrumOffset")
        start_t, delta_t, N_datapoints, _ = self.ms_profile.decode_header(offset)
        calib_id = self.ms_scan_info.get_scan_info(scan_index, "CalibrationID")
        check_points = self.ms_profile.check_points_list[scan_index]
        start_index = self.ms_calib_info.search_index_below_mz(scan_index, N_datapoints, calib_id, start_t, delta_t, float(edges[0])) + 1
        stop_index = min(
            self.ms_calib_info.search_index_below_mz(scan_index, N_datapoints, calib_id, start_t, delta_t, float(edges[-1])) + 1,
            check_points.N_datapoints_no_end_zeros,
        )
        if stop_index <= start_index:
            return
        checkpoint_index = max(0, int(np.searchsorted(check_points.out_index_list[:check_points.N_filled], start_index, side="right")) - 1)
        coeff, base = self.ms_calib_info.calib_info[scan_index][:2]
        max_into_bins(
            self.ms_profile._mm_view,
            int(check_points.offset_list[checkpoint_index]),
            int(check_points.out_index_list[checkpoint_index]),
            int(start_index),
            int(stop_index),
            int(check_points.cur_size_list[checkpoint_index]),
            coeff,
            base,
            self.ms_calib_info.get_poly(scan_index, N_datapoints, calib_id, start_t, delta_t),
            start_t,
            delta_t,
            edges,
            out,
        )
    def extract_tic(self, *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        return self._extract_scan_info_trace("TIC", scan_indices)
    def extract_bpc(self, *, scan_indices: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        return self._extract_scan_info_trace("BasePeakValue", scan_indices)
    def _extract_scan_info_trace(self, field: str, scan_indices: np.ndarray | None) -> Tuple[np.ndarray, np.ndarray]:
        if scan_indices is None:
            scan_indices = np.arange(self.N_scans)
        return self.RT_list[scan_indices], np.array([self.ms_scan_info.get_scan_info(int(i), field) for i in scan_indices])


class MSScanInfo():
    def __init__(self, xsd_path: Path, bin_path: Path, N_scans: int):
        self.N_repeat_dict: Dict = {}
        self.scan_info: List = []
        # parse xsd file
        tree = etree.parse(xsd_path)
        namespace = tree.xpath('namespace-uri(.)')
        complextypes_dict = {}
        for complextype in tree.getroot().findall(f"{{{namespace}}}complexType"):
            complextypes_dict[complextype.get('name')] = [
                (element.get('name'), element.get('type')) 
                for element in complextype[0].findall(f"{{{namespace}}}element")
            ]
        # parse binary file
        with open(bin_path, "rb") as f:
            # fixed header
            fixed_header = struct.unpack("<22i", f.read(88))
            assert fixed_header[0] == 257, fixed_header[0]  # 257, unknown
            assert all(x == 0 for x in fixed_header[1:17])  # 16 fields of ScanRecordType from CalibrationID to NumOfActualsPerScan?
            # ScanRecordType/DataDependentScanParamType? and ScanRecordType/SpectrumParamValues?: 2 and 6, or 1 and 4 in
            # older files (e.g. of 2022, of the older MSScan.xsd)
            assert tuple(fixed_header[17:19]) in ((2, 6), (1, 4)), fixed_header[17:19]
            self.N_repeat_dict["SpectrumParamValues"] = fixed_header[19]
            assert fixed_header[20] == 2, fixed_header[20]  # ScanRecordType/DataDependentScanParamType?
            assert fixed_header[21] == 0, fixed_header[21]  # repeat count of DataDependentScanParamType?
            # skip variable header and jump to data section
            f.seek(struct.unpack("<i", f.read(4))[0])
            for _ in range(N_scans):
                self.scan_info.append(self.parse_complextype(f, complextypes_dict, "ScanRecordType"))
            # the records as MSScan.xsd has them fill the file to its end
            assert f.tell() == bin_path.stat().st_size, (f.tell(), bin_path.stat().st_size)
    def parse_complextype(self, f: BinaryIO, complextypes_dict: Dict[str, list], name: str) -> Dict[str, Any]:
        name = name.split(":", 1)[-1]  # remove namespace if exists
        out: Dict[str, Any] = {}
        for field_name, field_type in complextypes_dict[name]:
            if field_name in self.N_repeat_dict:
                N_repeat = self.N_repeat_dict[field_name]
                out[field_name] = [self.read_type(f, complextypes_dict, field_type) for _ in range(N_repeat)]
            else:
                out[field_name] = self.read_type(f, complextypes_dict, field_type)
        return out
    def read_type(self, f: BinaryIO, complextypes_dict: Dict[str, list], field_type: str) -> Any:
        if field_type == 'xs:byte':
            return struct.unpack('c', f.read(1))[0]
        elif field_type == 'xs:short':
            return struct.unpack('<H', f.read(2))[0]
        elif field_type == 'xs:int':
            return struct.unpack('<I', f.read(4))[0]
        elif field_type == 'xs:long':
            return struct.unpack('<Q', f.read(8))[0]
        elif field_type == 'xs:float':
            return struct.unpack('<f', f.read(4))[0]
        elif field_type == 'xs:double':
            return struct.unpack('<d', f.read(8))[0]
        else:
            return self.parse_complextype(f, complextypes_dict, field_type)
    def __str__(self):
        return pprint.pformat(self.scan_info)
    def get_scan_info(self, scan_index: int, field: str) -> Any:
        return self.scan_info[scan_index][field]
    def get_spectrum_info(self, scan_index: int, spectrum_param_index: int, field: str) -> Any:
        return self.scan_info[scan_index]["SpectrumParamValues"][spectrum_param_index][field]

class MSCalibInfo():
    def __init__(self, xml_path: Path, bin_path: Path, mass_cal_offsets: List[int]):
        self.poly: np.ndarray | None = None
        self._can_share_poly = False
        # load primary calibration info from binary file
        self.calib_info = [None] * len(mass_cal_offsets)
        with open(bin_path, "rb") as f:
            for i, mass_cal_offset in enumerate(mass_cal_offsets):
                f.seek(mass_cal_offset)
                length = struct.unpack("<I", f.read(4))[0]
                assert length == 10, length
                self.calib_info[i] = np.array(struct.unpack(f"<{length}d", f.read(length * 8)))
        # load polynomial calibration info from xml file
        self.default_calib_info = {}
        for calib in etree.parse(xml_path).getroot().findall("./DefaultCalibrations/DefaultCalibration"):
            calib_id = calib.get("DefaultCalibrationID")
            # Number="1" is for coeff and base, which we do not collect here because they varies per scan.
            # Get info for Step Number="2", which holds the polynomial calibration coefficients shared for all scans.
            step2 = calib.find("./Step[@Number='2']")
            if step2 is not None:
                assert (tech := step2.findtext("CalibrationTechnique")) == "ExternalReference", f"ID {calib_id}: Technique mismatch ({tech})"
                assert (formula := step2.findtext("CalibrationFormula")) == "Polynomial", f"ID {calib_id}: Formula mismatch ({formula})"
                self.default_calib_info[int(calib_id)] = {
                    "polynomial_flag": [char == '1' for char in reversed(format(int(step2.findtext("ValueUseFlags")), '08b'))],  # convert to binary and reverse to match the order of coefficients
                    "polynomial_coefficients":  [float(v.text) for v in step2.findall("./Values/Value")]
                }
    @property
    def shares_grid(self) -> bool:
        """Whether all scans have one time grid and polynomial correction, and differ only in coeff and base."""
        return self._can_share_poly

    def configure_poly_cache(self, scan_grids: list, calibration_ids: list[int]) -> None:
        """Check once per load whether all scans can share one correction array.

        Each grid is (start_t, delta_t, N_datapoints). Only coeff/base may
        vary when sharing poly. Nonuniform files use per-scan corrections.
        """
        self.poly = None
        self._can_share_poly = False
        if not self.calib_info:
            return
        if len(scan_grids) != len(self.calib_info) or len(calibration_ids) != len(self.calib_info):
            raise ValueError("Poly cache metadata must cover every scan")
        reference_flags = self.default_calib_info[calibration_ids[0]]["polynomial_flag"]
        self._can_share_poly = (
            all(grid == scan_grids[0] for grid in scan_grids)
            and all(np.array_equal(row[2:], self.calib_info[0][2:]) for row in self.calib_info)
            and all(self.default_calib_info[cid]["polynomial_flag"] == reference_flags for cid in calibration_ids)
        )

    def get_poly(self, scan_index: int, N_datapoints: int, calib_id: int, start_t: float, delta_t: float) -> np.ndarray:
        """Build lazily; retain one array only when all scan conditions match."""
        if self.poly is not None:
            return self.poly
        polynomial_flag = self.default_calib_info[calib_id]["polynomial_flag"]
        if len(polynomial_flag) > 8:
            # orders above 7 (older files, e.g. of 2022, have up to 11), which the native one has no room for
            _, _, left, right, poly_coeffs = self._calibration_terms(scan_index, calib_id)
            t_list = start_t + delta_t * np.arange(N_datapoints, dtype=np.float64)
            poly = np.polynomial.polynomial.polyval(np.clip(t_list, left, right), poly_coeffs)
        else:
            left, right, a2, b2, c2, d2, e2, f2 = self.calib_info[scan_index][2:]
            poly = calculate_poly(
                left, right, a2, b2, c2, d2, e2, f2,
                polynomial_flag,
                N_datapoints, start_t, delta_t,
            )
        poly.setflags(write=False)
        if self._can_share_poly:
            self.poly = poly
        return poly

    def calibrate_mz_python(self, scan_index: int, N_datapoints: int, calib_id: int, start_t: float, delta_t: float):
        # get variables
        coeff, base, left, right, a2, b2, c2, d2, e2, f2 = self.calib_info[scan_index]
        poly_flag = self.default_calib_info[calib_id]["polynomial_flag"]
        # t_list: linspaceより arange の方が軽いが、誤差が大きくなってしまう。
        # t_list = start_t + delta_t * np.arange(N_datapoints, dtype=np.float64)
        t_list = np.linspace(start_t, start_t + delta_t * (N_datapoints - 1), N_datapoints)
        # poly_coeffs[j] が t^j の係数（最大 7次まで、古いファイルでは 11次まで）
        coefficients = (a2, b2, c2, d2, e2, f2)
        poly_coeffs = np.zeros(max(8, len(poly_flag)), dtype=np.float64)
        coeff_idx = 0
        for j, F in enumerate(poly_flag):
            if F and coeff_idx < len(coefficients):
                poly_coeffs[j] = coefficients[coeff_idx]
                coeff_idx += 1
        t_list_clipped = np.clip(t_list, left, right)
        # Horner法で効率的に多項式計算する: 
        # poly = (((((c7*t) + c6)*t + c5)*t + c4)*t + ... + c0)
        poly = np.full_like(t_list_clipped, poly_coeffs[-1])
        for k in range(len(poly_coeffs) - 2, -1, -1):
            poly = poly * t_list_clipped + poly_coeffs[k]
        return (coeff * (t_list - base))**2 - poly
    def calibrate_mz(self, scan_index: int, N_datapoints: int, calib_id: int, start_t: float, delta_t: float):
        coeff, base = self.calib_info[scan_index][:2]
        poly = self.get_poly(scan_index, N_datapoints, calib_id, start_t, delta_t)
        return calibrate_mz(
            coeff=coeff,
            base=base,
            N_datapoints=N_datapoints,
            start_t=start_t,
            delta_t=delta_t,
            poly=poly,
        )
    def search_index_below_mz(self, scan_index: int, N_datapoints: int, calib_id: int, start_t: float, delta_t: float, low_mz: float) -> int:
        """
        Return the largest datapoint index whose calibrated m/z is below low_mz.
        This matches np.searchsorted(mz_list, low_mz, side="left") - 1 without
        materializing the full m/z array.
        """
        mz_at = self.mz_function(scan_index, calib_id, start_t, delta_t)
        if N_datapoints <= 0:
            return -1
        if low_mz <= mz_at(0):
            return -1
        if mz_at(N_datapoints - 1) < low_mz:
            return N_datapoints - 1

        low_idx = 0
        high_idx = N_datapoints
        while low_idx < high_idx:
            mid_idx = (low_idx + high_idx) // 2
            if mz_at(mid_idx) < low_mz:
                low_idx = mid_idx + 1
            else:
                high_idx = mid_idx
        return low_idx - 1

    def _calibration_terms(self, scan_index: int, calib_id: int):
        """coeff, base, the clipping limits and the polynomial coefficients (lowest order first; 8, or of the orders the
        flags have) of a scan."""
        coeff, base, left, right, a2, b2, c2, d2, e2, f2 = self.calib_info[scan_index]
        poly_flag = self.default_calib_info[calib_id]["polynomial_flag"]
        coefficients = (a2, b2, c2, d2, e2, f2)
        poly_coeffs = [0.0] * max(8, len(poly_flag))
        coeff_idx = 0
        for polynomial_order, flag in enumerate(poly_flag):
            if flag and coeff_idx < len(coefficients):
                poly_coeffs[polynomial_order] = coefficients[coeff_idx]
                coeff_idx += 1
        return coeff, base, left, right, poly_coeffs

    def mz_values(self, scan_index: int, calib_id: int, start_t: float, delta_t: float, datapoint_indices: np.ndarray) -> np.ndarray:
        """The calibrated m/z of datapoint indices of the scan, as mz_function gives them one at a time."""
        coeff, base, left, right, poly_coeffs = self._calibration_terms(scan_index, calib_id)
        t_value = start_t + delta_t * np.asarray(datapoint_indices, dtype=np.float64)
        polynomial_correction = np.polynomial.polynomial.polyval(np.clip(t_value, left, right), poly_coeffs)
        primary_term = coeff * (t_value - base)
        return primary_term * primary_term - polynomial_correction

    def mz_function(self, scan_index: int, calib_id: int, start_t: float, delta_t: float):
        """The calibrated m/z of a datapoint index of the scan, as a function: for a few points, without the full array."""
        coeff, base, left, right, poly_coeffs = self._calibration_terms(scan_index, calib_id)

        def mz_at(datapoint_index: int) -> float:
            t_value = start_t + delta_t * datapoint_index
            t_clipped = min(max(t_value, left), right)
            polynomial_correction = poly_coeffs[-1]
            for polynomial_order in range(len(poly_coeffs) - 2, -1, -1):
                polynomial_correction = polynomial_correction * t_clipped + poly_coeffs[polynomial_order]
            primary_term = coeff * (t_value - base)
            return primary_term * primary_term - polynomial_correction

        return mz_at



class CheckPoints:
    def __init__(self):
        self.offset_list: np.ndarray = None     # relative to payload start (payload: np.ndarray = comp_block[16:])
        self.out_index_list: np.ndarray = None  # output bin index at this token start
        self.cur_size_list: np.ndarray = None   # 1/2/4 (token width in bytes)
        self.N_filled: int = None               # number of filled checkpoints
        self.N_datapoints_no_end_zeros: int = None  # number of datapoints excluding the zeros added at the end to fill the last checkpoint
    def initialize_by_list(self, N_datapoints: int, stride_bytes: int):
        N = ((N_datapoints + stride_bytes - 1) // stride_bytes)
        self.offset_list = [None] * N
        self.out_index_list = [None] * N
        self.cur_size_list = [None] * N
    def to_ndarray(self):
        self.offset_list = np.array(self.offset_list, dtype=np.uint32)
        self.out_index_list = np.array(self.out_index_list, dtype=np.uint32)
        self.cur_size_list = np.array(self.cur_size_list, dtype=np.uint8)
    def add_checkpoint(self, checkpoint_index:int, offset: int, out_pos: int, cur_size: int):
        self.offset_list[checkpoint_index] = offset
        self.out_index_list[checkpoint_index] = out_pos
        self.cur_size_list[checkpoint_index] = cur_size
    def __str__(self):
        return pprint.pformat([self.offset_list, self.out_index_list, self.cur_size_list, f"N_filled: {self.N_filled}", f"N_datapoints_no_end_zeros: {self.N_datapoints_no_end_zeros}"])

class MSProfile():
    UNPACKERS = {
        1: struct.Struct('<b'),
        2: struct.Struct('<h'),
        3: struct.Struct('<i'),
        # 4: struct.Struct('<q'),
        "dd": struct.Struct('<dd'),
    }
    SIZES = {1:1, 2:2, 3:4}#, 4:8}
    checkpoint_bin = 2048
    def __init__(self, bin_path: Path):
        """
        # 1. keep file handle open
        # 2. フィアルを下層メモリ空間にマップさせる
        #       fileno はOSが持つファイル識別子
        #       0はファイル全体をマップすることを意味する
        #       access=mmap.ACCESS_READ は読み取り専用でマップすることを意味する
        # 3. memory_viewを作成
        #       スライスがコピー無しで取れる
        #       Cython/C++で扱うときに便利
        """
        self._fh = open(bin_path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._mm_view = memoryview(self._mm)
        self._buffer: np.ndarray | None = None  # the records written again in memory, of a file of LZF ones
        self.check_points_list: List[CheckPoints] = []
    def is_lzf(self, offset: int, compressed_len: int, N_datapoints: int) -> bool:
        """Whether a record is LZF compressed, not of the header decode_header reads (\\x90 and its datapoint count)."""
        return compressed_len < 24 or struct.unpack_from("<I", self._mm_view, offset + 16)[0] != (144 << 24 | N_datapoints)
    def rewrite_lzf_records(self, spectra: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """
        The offsets and byte counts of the records of the spectra (SpectrumParamValues) to read them by.
        Those of older acquisitions (e.g. of 2022) are LZF compressed, to UncompressedByteCount bytes of start_t,
        delta_t and an int32 intensity a datapoint. If any is, all are written again into memory, the LZF ones as
        the others are (about 0.6 of their bytes), and the profile is read from there instead of the file.
        """
        records = [(spectrum["SpectrumOffset"], spectrum["ByteCount"]) for spectrum in spectra]
        lzf = [self.is_lzf(offset, byte_count, spectrum["PointCount"]) for (offset, byte_count), spectrum in zip(records, spectra)]
        if not any(lzf):
            return records
        buffer, offset_list, byte_count_list = rewrite_lzf_records(
            self._mm_view,
            [offset for offset, _ in records],
            [byte_count for _, byte_count in records],
            [spectrum.get("UncompressedByteCount", 0) for spectrum in spectra],
            lzf,
        )
        self.close()
        self._buffer = buffer
        self._mm_view = memoryview(buffer)
        return list(zip(offset_list.tolist(), byte_count_list.tolist()))
    def set_checkpoints(self, offset_list: List[int], compressed_len_list: List[int]):
        for i, (offset, compressed_len) in enumerate(zip(offset_list, compressed_len_list)):
            self.check_points_list.append(self.build_checkpoints(offset, compressed_len, stride_bytes=2048))
    def build_checkpoints(self, offset: int, compressed_len: int, stride_bytes=2048) -> CheckPoints:
        start_t, delta_t, N_datapoints, cur_idx = self.decode_header(offset)
        offsets_np, out_index_np, cur_size_np, n_filled, cur_idx = build_checkpoints(self._mm_view, offset, compressed_len, N_datapoints, cur_idx, stride_bytes)
        check_points = CheckPoints()
        check_points.offset_list = offsets_np
        check_points.out_index_list = out_index_np
        check_points.cur_size_list = cur_size_np
        check_points.N_filled = n_filled
        check_points.N_datapoints_no_end_zeros = cur_idx
        return check_points
    def build_checkpoints_python(self, offset: int, compressed_len: int, stride_bytes=2048) -> CheckPoints:
        start_t, delta_t, N_datapoints, cur_idx = self.decode_header(offset)
        size_switch_flag = 3
        cur_size = self.SIZES[size_switch_flag]
        cur_unpacker: struct.Struct = self.UNPACKERS[size_switch_flag]
        cur_offset = offset + 24
        check_points = CheckPoints()
        check_points.initialize_by_list(N_datapoints, stride_bytes)
        check_points.add_checkpoint(checkpoint_index=0, offset=cur_offset, out_pos=cur_idx, cur_size=cur_size)  # 最初のチェックポイントを追加
        cur_checkpoint_index = 1
        accumulated_datapoints = 0
        while cur_offset < offset + compressed_len:
            # cur_bytes = cur_unpacker(self._mm_view[cur_offset:cur_offset+cur_size])[0]
            cur_bytes = cur_unpacker.unpack_from(self._mm_view, cur_offset)[0]
            cur_offset += cur_size
            # if positive, direct data
            if cur_bytes >= 0:
                # print(cur_bytes, cur_idx, cur_offset, len(comp_view), total_len)
                accumulated_datapoints += 1
                cur_idx += 1
            # if negative, zero run-length or size switch
            else:
                cur_bytes *= -1
                # zero-filling: 初期値が 0 なのでゼロ埋めは不要
                len_zero, size_switch_flag = divmod(cur_bytes, 4)
                # size switch
                if size_switch_flag != 0:
                    cur_size = self.SIZES[size_switch_flag]
                    cur_unpacker = self.UNPACKERS[size_switch_flag]
                else:
                    # size switch がない場合は、次のトークンも同じ: 最初しか起こらない
                    assert cur_checkpoint_index == 1, "Size switch without flag should only happen at the beginning"
                    assert accumulated_datapoints == 0, "Data should not appear before the first checkpoint if there is no size switch"
                accumulated_datapoints += len_zero
                cur_idx += len_zero
            if stride_bytes < accumulated_datapoints:  # チェックポイントを作るべき位置に来たら
                check_points.add_checkpoint(
                    checkpoint_index=cur_checkpoint_index, 
                    offset=cur_offset, 
                    out_pos=cur_idx, 
                    cur_size=cur_size
                )
                cur_checkpoint_index += 1
                accumulated_datapoints = 0
        check_points.to_ndarray()
        check_points.N_filled = cur_checkpoint_index
        check_points.N_datapoints_no_end_zeros = cur_idx
        return check_points
    def decode_header(self, offset: int) -> Tuple[float, float, int, int]:
        start_t, delta_t = self.UNPACKERS["dd"].unpack_from(self._mm_view, offset)
        inten_header_4 = struct.unpack(f'<I', self._mm_view[offset+16:offset+20])[0]
        N_datapoints = inten_header_4 & 0x00FFFFFF
        assert inten_header_4 >> 24 == 144, inten_header_4 >> 24  # byte seems to be fixed to \x90 ...?
        return start_t, delta_t, N_datapoints, -struct.Struct("<i").unpack_from(self._mm_view, offset+20)[0]    # cur_idx = init_zero_repeat
    def extract_full_spectrum(self, scan_index: int, offset: int) -> Tuple[float, float, np.ndarray]:
        start_t, delta_t, N_datapoints, cur_idx = self.decode_header(offset)
        check_points = self.check_points_list[scan_index]
        N_datapoints_no_end_zeros = check_points.N_datapoints_no_end_zeros
        inten_list = np.zeros(N_datapoints, dtype=np.uint32)
        decode_inten_list(
            self._mm_view, 
            offset + 24, 
            out_idx_start=cur_idx, 
            stop_len=N_datapoints_no_end_zeros, 
            cur_size_start=self.SIZES[3],  # initial size switch flag is always 3 (4 bytes)
            out=inten_list
        )
        return start_t, delta_t, inten_list
    def extract_full_spectrum_python(self, scan_index: int, offset: int) -> Tuple[float, float, np.ndarray]:
        start_t, delta_t, N_datapoints, cur_idx = self.decode_header(offset)
        check_points = self.check_points_list[scan_index]
        N_datapoints_no_end_zeros = check_points.N_datapoints_no_end_zeros
        # Initialize
        inten_list = [0] * N_datapoints
        size_switch_flag = 3
        cur_size = self.SIZES[size_switch_flag]
        cur_unpacker: struct.Struct = self.UNPACKERS[size_switch_flag]
        cur_offset = offset + 24
        while cur_idx < N_datapoints_no_end_zeros:
            # cur_bytes = cur_unpacker(self._mm_view[cur_offset:cur_offset+cur_size])[0]
            cur_bytes = cur_unpacker.unpack_from(self._mm_view, cur_offset)[0]
            cur_offset += cur_size
            # if positive, direct data
            if cur_bytes >= 0:
                # print(cur_bytes, cur_idx, cur_offset, len(comp_view), total_len)
                inten_list[cur_idx] = cur_bytes
                cur_idx += 1
            # if negative, zero run-length or size switch
            else:
                cur_bytes *= -1
                # zero-filling: 初期値が 0 なのでゼロ埋めは不要
                len_zero, size_switch_flag = divmod(cur_bytes, 4)
                cur_idx += len_zero
                # size switch
                cur_size = self.SIZES[size_switch_flag]
                cur_unpacker = self.UNPACKERS[size_switch_flag]
        return start_t, delta_t, np.array(inten_list, dtype=np.uint32)    # max size = 4 bytes = uint32
    def extract_sub_spectrum_python(self, scan_index: int, offset: int, start_index: int, stop_index: int) -> np.ndarray:
        start_t, delta_t, N_datapoints, cur_idx = self.decode_header(offset)
        check_points = self.check_points_list[scan_index]
        start_index = max(0, min(start_index, N_datapoints))
        stop_index = max(0, min(stop_index, N_datapoints))
        if stop_index <= start_index:
            return np.array([], dtype=np.uint32)

        decode_stop_index = min(stop_index, check_points.N_datapoints_no_end_zeros)
        if decode_stop_index <= start_index:
            return np.zeros(stop_index - start_index, dtype=np.uint32)

        valid_checkpoint_count = check_points.N_filled
        checkpoint_index = np.searchsorted(
            check_points.out_index_list[:valid_checkpoint_count],
            start_index,
            side="right"
        ) - 1
        checkpoint_index = max(0, checkpoint_index)

        out = np.zeros(stop_index, dtype=np.uint32)
        decode_inten_list(
            self._mm_view,
            token_offset=int(check_points.offset_list[checkpoint_index]),
            out_idx_start=int(check_points.out_index_list[checkpoint_index]),
            stop_len=decode_stop_index,
            cur_size_start=int(check_points.cur_size_list[checkpoint_index]),
            out=out
        )
        return out[start_index:stop_index]
    def extract_sub_spectrum(self, scan_index: int, offset: int, start_index: int, stop_index: int) -> np.ndarray:
        start_t, delta_t, N_datapoints, cur_idx = self.decode_header(offset)
        check_points = self.check_points_list[scan_index]
        start_index = max(0, min(start_index, N_datapoints))
        stop_index = max(0, min(stop_index, N_datapoints))
        if stop_index <= start_index:
            return np.array([], dtype=np.uint32)

        decode_stop_index = min(stop_index, check_points.N_datapoints_no_end_zeros)
        out = np.zeros(stop_index - start_index, dtype=np.uint32)
        if decode_stop_index <= start_index:
            return out

        valid_checkpoint_count = check_points.N_filled
        checkpoint_index = np.searchsorted(
            check_points.out_index_list[:valid_checkpoint_count],
            start_index,
            side="right"
        ) - 1
        checkpoint_index = max(0, checkpoint_index)

        decoded = extract_sub_spectrum(
            self._mm_view,
            token_offset=int(check_points.offset_list[checkpoint_index]),
            out_idx_start=int(check_points.out_index_list[checkpoint_index]),
            start_index=start_index,
            stop_index=decode_stop_index,
            cur_size_start=int(check_points.cur_size_list[checkpoint_index])
        )
        out[:decode_stop_index - start_index] = decoded
        return out
    def sum_sub_spectrum(self, scan_index: int, offset: int, start_index: int, stop_index: int) -> int:
        start_t, delta_t, N_datapoints, cur_idx = self.decode_header(offset)
        check_points = self.check_points_list[scan_index]
        start_index = max(0, min(start_index, N_datapoints))
        stop_index = max(0, min(stop_index, N_datapoints))
        if stop_index <= start_index:
            return 0

        decode_stop_index = min(stop_index, check_points.N_datapoints_no_end_zeros)
        if decode_stop_index <= start_index:
            return 0

        valid_checkpoint_count = check_points.N_filled
        checkpoint_index = np.searchsorted(
            check_points.out_index_list[:valid_checkpoint_count],
            start_index,
            side="right"
        ) - 1
        checkpoint_index = max(0, checkpoint_index)
        return sum_sub_spectrum(
            self._mm_view,
            token_offset=int(check_points.offset_list[checkpoint_index]),
            out_idx_start=int(check_points.out_index_list[checkpoint_index]),
            start_index=start_index,
            stop_index=decode_stop_index,
            cur_size_start=int(check_points.cur_size_list[checkpoint_index])
        )

    def close(self):
        try:
            self._mm_view.release()
        except Exception:
            pass
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        self._buffer = None
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.close()
