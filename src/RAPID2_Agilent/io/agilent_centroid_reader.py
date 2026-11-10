"""The centroid spectra of an Agilent .d run (MSPeak.bin), read into memory at load without the profile."""

from pathlib import Path

import numpy as np

from .agilent_reader import read_scan_records, scan_lists
from .scan_points_reader import ScanPointsReader

CENTROID_FORMAT_ID = 2  # SpectrumFormatID of a scan's centroid spectrum (MSPeak.bin); 1 is its profile's
POINT_BYTES = 12  # a centroid: its flight time (float64) and intensity (float32)


def stored_spectra(run: str | Path) -> dict[str, int]:
    """Of "profile" and "centroid", those a .d run holds, by the bytes of their files (MSProfile.bin and MSPeak.bin)
    that are not empty."""
    acq_data = Path(run) / "AcqData"
    sizes = {
        kind: (acq_data / name).stat().st_size if (acq_data / name).is_file() else 0
        for kind, name in (("profile", "MSProfile.bin"), ("centroid", "MSPeak.bin"))
    }
    return {kind: size for kind, size in sizes.items() if size > 0}


class AgilentCentroidReader(ScanPointsReader):
    """Read the centroid spectra of an Agilent .d run, as MassHunter stored them at acquisition, and not its profile:
    about 1/17 of the data, each file read once whole, for runs read from a server. All of them are held in memory
    (12 bytes a centroid, about 50 MB for a 30 min run) and read there, as RPDReader reads a .rpd file of centroid
    data. The scan records (RT, MS level, polarity, precursor, TIC and BPC) are those AgilentReader reads.

    A scan's record in MSPeak.bin: its centroids' flight times (float64), then their intensities (float32); their
    m/z by the scan's calibration, as the profile's. The first scan of a run has none."""

    def load(self, folder_path: Path):
        self.close()
        acq_data = Path(folder_path) / "AcqData"
        N_scans, ms_scan_info, ms_calib_info = read_scan_records(acq_data)
        records = ms_scan_info.scan_info
        spectra = [
            next((spectrum for spectrum in record["SpectrumParamValues"] if spectrum["SpectrumFormatID"] == CENTROID_FORMAT_ID), None)
            for record in records
        ]
        point_counts = np.array([spectrum["PointCount"] if spectrum else 0 for spectrum in spectra], dtype=np.int64)
        if not point_counts.any():
            raise ValueError(f"{Path(folder_path).name}: no centroid spectra in its scans")
        # one read of the file, its records copied into the arrays and calibrated; the bytes then let go
        peaks = (acq_data / "MSPeak.bin").read_bytes()
        starts = np.concatenate(([0], np.cumsum(point_counts)[:-1]))
        mz = np.empty(int(point_counts.sum()), dtype=np.float64)
        inten = np.empty(len(mz), dtype=np.float32)
        for scan_index, (spectrum, start, count) in enumerate(zip(spectra, starts, point_counts)):
            if count == 0:
                continue
            if spectrum["ByteCount"] != POINT_BYTES * count:
                raise ValueError(
                    f"{Path(folder_path).name}: the centroid record of scan {scan_index} has {spectrum['ByteCount']} "
                    f"bytes for {count} points, not {POINT_BYTES} a point (compressed?)"
                )
            offset = spectrum["SpectrumOffset"]
            stop = start + count
            mz[start:stop] = ms_calib_info.mz_at_times(
                scan_index, records[scan_index]["CalibrationID"], np.frombuffer(peaks, "<f8", count, offset)
            )
            inten[start:stop] = np.frombuffer(peaks, "<f4", count, offset + 8 * count)
            # the flight times ascend, and so their m/z; sorted if not, for the searches by m/z
            if np.any(np.diff(mz[start:stop]) < 0):
                order = np.argsort(mz[start:stop], kind="stable")
                mz[start:stop], inten[start:stop] = mz[start:stop][order], inten[start:stop][order]
        del peaks
        self.file_path = Path(folder_path)
        self.N_scans = N_scans
        self.spectrum_type = "discrete"
        for name, values in scan_lists(ms_scan_info).items():
            setattr(self, name, values)
        # the TIC and BPC of the records, of the profile, as AgilentReader has them
        self._set_points(
            mz, inten, starts, point_counts,
            tic=np.array([record["TIC"] for record in records]),
            bpc=np.array([record["BasePeakValue"] for record in records]),
        )
