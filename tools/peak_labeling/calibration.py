"""Scans whose mass calibration jumps away from that of the scans about them put back in line, for the peak detector's
XICs only: the app reads each scan's calibration as it is stored.

The instrument fits each scan's calibration (its coeff and base, in MSMassCal.bin) to its reference ions; in some
runs of 2026 single scans are off by up to 150 ppm, all their ions with them, so that an XIC drops to a few % in
them (data/peak_eval/README.md). Their coeff and base are made those interpolated, by RT, from the nearest scans not
off, in the reader loaded, in memory.
"""

import numpy as np

JUMP_PPM = 10.0  # off by more than this from the median of the scans about, at a REFERENCE_MZ (2022 runs: < 10)
ABOUT = 4  # scans either side, of the median
# where the calibrations are compared: not lower, where those of the 2022 runs scatter by up to 70 ppm
REFERENCE_MZ = (700.0, 1200.0)
PASSES = 3  # of repairs, for scans off one after another, which pull the median of those about them


def calibration_offsets(reader) -> np.ndarray:
    """Of each scan, how far (ppm) its calibration is, at each of REFERENCE_MZ, from the median of the ABOUT scans
    either side of it, in the order acquired: (scans, len(REFERENCE_MZ))."""
    calibration, scan_info = reader.ms_calib_info, reader.ms_scan_info
    # each scan's m/z at the flight times where the first scan has the reference m/z
    offset = scan_info.get_spectrum_info(0, reader.spectrum_param_index, "SpectrumOffset")
    start_t, delta_t, count, _ = reader.ms_profile.decode_header(offset)
    calib_id = scan_info.get_scan_info(0, "CalibrationID")
    points = [calibration.search_index_below_mz(0, count, calib_id, start_t, delta_t, mz) + 1 for mz in REFERENCE_MZ]
    mz = np.array([
        [calibration.mz_function(i, scan_info.get_scan_info(i, "CalibrationID"), start_t, delta_t)(point)
         for point in points]
        for i in range(reader.N_scans)
    ])
    padded = np.pad(mz, ((ABOUT, ABOUT), (0, 0)), mode="edge")
    about = np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * ABOUT + 1, axis=0), axis=-1)
    return (mz - about) / about * 1e6


def repair_calibration_jumps(reader) -> list[int]:
    """Put the scans off by more than JUMP_PPM at a REFERENCE_MZ back in line, in the reader, in up to PASSES; the
    scans repaired."""
    rows, rt = reader.ms_calib_info.calib_info, reader.RT_list
    repaired = set()
    for _ in range(PASSES):
        off = np.flatnonzero(np.abs(calibration_offsets(reader)).max(axis=1) > JUMP_PPM)
        if len(off) == 0:
            break
        good = np.setdiff1d(np.arange(reader.N_scans), off)
        coeff = np.interp(rt[off], rt[good], [rows[i][0] for i in good])
        base = np.interp(rt[off], rt[good], [rows[i][1] for i in good])
        for scan, scan_coeff, scan_base in zip(off, coeff, base):
            row = rows[scan].copy()
            row[0], row[1] = scan_coeff, scan_base
            rows[scan] = row
        repaired.update(off.tolist())
    return sorted(repaired)
