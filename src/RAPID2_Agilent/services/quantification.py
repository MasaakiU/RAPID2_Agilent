"""The values of a target's trace in one run -- its area and the like -- that the .rpd.csv export writes. No Qt.

They are RAPID2's (``calc_chromatogram_auc``), with what it got wrong put right: a scan whose profile data points
miss an XIC's m/z window is left out as not measured there, rather than made NaN or 0, and a trace of 0 has no
peak top. See docs/rpd_csv_format.md, sections 6 and 9.
"""

from dataclasses import dataclass, replace

import numpy as np

from ..io.ms_data import MSData
from ..models.chemistry import parse_adduct
from ..models.run_peaks import RunPeaks
from ..models.target import MSLevel, Target, TraceType
from . import operations

# the codes of the scan records named in scan_settings, those seen in data; others are written as numbers
ION_MODE_NAMES = {64: "ESI"}
SCAN_TYPE_NAMES = {1: "Scan", 512: "ProductIon"}
_POLARITY_SIGNS = {1: "+", -1: "-"}


@dataclass(frozen=True)
class Quantity:
    """A trace's values over the scans used: those of its choice in the RT window, less, for an XIC, those left out.

    None where there is no value: all but n_scans, n_missing and scan_settings with fewer than 2 scans used, and the
    peak top of a trace that is 0 throughout.
    """
    n_scans: int  # used
    rt_first: float | None  # the RTs of the first and last scans used
    rt_last: float | None
    area: float | None  # trapezoidal, in intensity x min, from the first scan used to the last
    baseline_adjusted_area: float | None  # less a flat baseline at the lowest intensity
    height: float | None
    baseline_height: float | None
    rt_peaktop: float | None
    mz_peaktop: float | None  # where the scans' mean spectrum is highest in the m/z window; anywhere, for TIC/BPC
    n_missing: int  # scans of an XIC left out, as its m/z window misses their data points
    scan_settings: str  # what the scans in the RT window are, see scan_settings_text
    # of the target's peak as the peak detector found it (peak_deconvolution), not of the RT window: the area of its
    # shape, in intensity x min, with a sigma within which 68 % of its errors are and an interval within which 95 %
    # are; None of another detector's peak, of a window set by hand, or of none taken
    deconvolved_area: float | None = None
    deconvolved_area_sigma: float | None = None
    deconvolved_area_interval: float | None = None


def quantify(
    data: MSData,
    trace_type: TraceType,
    rt_window: tuple[float, float],
    *,
    mz_window: tuple[float, float] | None = None,
    ms_level: int | None = None,
    ms2: operations.MS2Selection | None = None,
    polarity: int | None = None,
) -> Quantity:
    """The values of a trace, its scans chosen as the plots choose them (operations.trace_scans), in the RT window
    [rt_window[0], rt_window[1]], ends included."""
    trace_type = TraceType(trace_type)
    if trace_type is TraceType.XIC and mz_window is None:
        raise ValueError("An XIC needs an m/z window")
    scan_indices = operations.trace_scans(data, ms_level, ms2=ms2, polarity=polarity)
    rt = data.RT_list[scan_indices]
    scan_indices = scan_indices[(rt_window[0] <= rt) & (rt <= rt_window[1])]
    scan_settings = scan_settings_text(data, scan_indices)
    n_missing = 0
    if trace_type is TraceType.XIC and len(scan_indices):
        missing = operations.mz_window_misses(data, mz_window, scan_indices).missing
        n_missing = int(np.count_nonzero(missing))
        scan_indices = scan_indices[~missing]
    if len(scan_indices) < 2:
        return Quantity(len(scan_indices), None, None, None, None, None, None, None, None, n_missing, scan_settings)

    rt_list, intensity = operations.extract_trace(data, trace_type, mz_window=mz_window, scan_indices=scan_indices)
    rt_list, intensity = np.asarray(rt_list, dtype=np.float64), np.asarray(intensity, dtype=np.float64)
    area = float(((intensity[1:] + intensity[:-1]) * np.diff(rt_list)).sum() / 2)
    baseline_height = float(intensity.min())
    top = int(np.argmax(intensity))
    height = float(intensity[top])
    rt_peaktop = mz_peaktop = None
    if height > 0:
        rt_peaktop = float(rt_list[top])
        mz_peaktop = _mz_peaktop(data, trace_type, mz_window, scan_indices, rt_list)
    return Quantity(
        n_scans=len(scan_indices),
        rt_first=float(rt_list[0]),
        rt_last=float(rt_list[-1]),
        area=area,
        baseline_adjusted_area=area - baseline_height * float(rt_list[-1] - rt_list[0]),
        height=height,
        baseline_height=baseline_height,
        rt_peaktop=rt_peaktop,
        mz_peaktop=mz_peaktop,
        n_missing=n_missing,
        scan_settings=scan_settings,
    )


def quantify_target(
    data: MSData, target: Target, *, run_peaks: RunPeaks | None = None, merge_ppm: float = 0.0
) -> Quantity:
    """The values of the target's trace as the plots show it: its trace type and MS level -- for MS2 the scans of its
    precursor and CE, merged within merge_ppm as they are listed, all MS2 scans without one --, the polarity of its
    adduct, its RT window and, for an XIC, its m/z window; the windows of run_peaks, those of the target in this
    run, if given. With run_peaks of no RT window (no peak taken), no values."""
    rt_window, mz_window = (target.rt_window(), target.mz_window()) if run_peaks is None else (
        run_peaks.rt_window, run_peaks.mz_window
    )
    if rt_window is None:
        return Quantity(0, None, None, None, None, None, None, None, None, 0, "")
    quantity = quantify(
        data,
        target.trace_type,
        rt_window,
        mz_window=mz_window if target.trace_type is TraceType.XIC else None,
        **_trace_choice(target, merge_ppm),
    )
    peak = None if run_peaks is None or run_peaks.manual else next(
        (peak for peak in run_peaks.peaks if (peak.start, peak.end) == rt_window and peak.area is not None), None
    )
    if peak is None:
        return quantity
    return replace(quantity, deconvolved_area=peak.area, deconvolved_area_sigma=peak.area_sigma,
                   deconvolved_area_interval=peak.area_interval)


def target_trace(
    data: MSData, target: Target, *, mz_window: tuple[float, float] | None = None, merge_ppm: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """The RT and intensity of the target's trace over all the scans it uses, chosen as quantify_target chooses
    them; for an XIC in mz_window (the target's if None), the scans whose data points it misses left out."""
    choice = _trace_choice(target, merge_ppm)
    scan_indices = operations.trace_scans(data, choice["ms_level"], ms2=choice["ms2"], polarity=choice["polarity"])
    if target.trace_type is TraceType.XIC:
        mz_window = target.mz_window() if mz_window is None else mz_window
        if len(scan_indices):
            scan_indices = scan_indices[~operations.mz_window_misses(data, mz_window, scan_indices).missing]
    if len(scan_indices) == 0:
        return np.array([]), np.array([])
    rt, intensity = operations.extract_trace(data, target.trace_type, mz_window=mz_window, scan_indices=scan_indices)
    return np.asarray(rt, dtype=np.float64), np.asarray(intensity, dtype=np.float64)


def _trace_choice(target: Target, merge_ppm: float) -> dict:
    """quantify's ms_level, ms2 and polarity for the target's scans."""
    ms2 = None
    if target.ms_level is MSLevel.MS2 and target.precursor_mz is not None:
        ms2 = operations.MS2Selection(target.precursor_mz, target.collision_energy, merge_ppm)
    return {
        "ms_level": target.ms_level.number,
        "ms2": ms2,
        "polarity": 1 if parse_adduct(target.adduct).charge > 0 else -1,
    }


def _mz_peaktop(data: MSData, trace_type: TraceType, mz_window, scan_indices: np.ndarray, rt_list: np.ndarray) -> float | None:
    if trace_type is TraceType.XIC:
        # the mean spectrum in the window only: over all m/z it takes some 20 times as long
        mz, intensity = data.extract_averaged_sub_spectrum(mz_window[0], mz_window[1], scan_indices=scan_indices)
    else:
        # its RT window [rt_min, rt_max) holds the last scan too
        mz, intensity, _scan = data.extract_averaged_spectrum(
            float(rt_list[0]), float(np.nextafter(rt_list[-1], np.inf)), "mean", scan_indices=scan_indices
        )
    if len(intensity) == 0 or intensity.max() <= 0:
        return None
    return float(mz[np.argmax(intensity)])


def scan_settings_text(data: MSData, scan_indices: np.ndarray) -> str:
    """What the scans are, as ``ESI+ Scan``: the ionization, the polarity and the kind of scan, by the codes of the
    scan records, a code not named written as a number (``IonMode 32+ Scan``); each kind among them, joined by ", ".
    """
    kinds = sorted(set(zip(
        data.ion_mode_list[scan_indices].tolist(),
        data.polarity_list[scan_indices].tolist(),
        data.scan_type_list[scan_indices].tolist(),
    )))
    return ", ".join(
        f"{ION_MODE_NAMES.get(ion_mode, f'IonMode {ion_mode}')}{_POLARITY_SIGNS.get(polarity, '')} "
        f"{SCAN_TYPE_NAMES.get(scan_type, f'ScanType {scan_type}')}"
        for ion_mode, polarity, scan_type in kinds
    )
