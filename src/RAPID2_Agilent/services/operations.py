"""Operations on one run that every front end shares, as numpy arrays.

The GUI presenters plot these arrays; `reports` summarizes the same arrays for
the CLI and the MCP tools. So the choice of scans and of the extraction for a
trace type is made here once, and a change to it (a polarity filter, say)
reaches all three.

`data` is anything with the `io.MSData` methods: an `AgilentReader`, or a GUI
model that delegates to one. RTs are the reader's ScanTime values, in minutes.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..io.ms_data import MSData
from ..models.target import TraceType

# collision energies that agree to this many decimals are one; precursors are matched past floating-point noise
CE_DECIMALS = 2
PRECURSOR_EPSILON = 1e-6


@dataclass(frozen=True)
class Precursor:
    """A precursor and collision energy of MS2 scans, as listed for the choice of the scans to use."""
    precursor_mz: float
    collision_energy: float
    isolation_width: float  # the widest of its scans
    n_scans: int


@dataclass(frozen=True)
class MS2Selection:
    """The MS2 scans of one collision energy whose precursor lies within merge_ppm of precursor_mz."""
    precursor_mz: float
    collision_energy: float
    merge_ppm: float = 0.0


def precursors(datas: Iterable[MSData], merge_ppm: float = 0.0, polarity: int | None = None) -> list[Precursor]:
    """The precursors and collision energies of the MS2 scans of all the files, by precursor and then CE; with a
    polarity (+1, -1), of its scans only.

    With merge_ppm, the precursors of one CE that lie within it of each other are one: from the lowest
    up, each takes those up to 2 x merge_ppm above it, and lies midway between its lowest and highest,
    so that MS2Selection(it, CE, merge_ppm) selects those scans. 0 keeps each recorded value apart, as
    a targeted method has them; data-dependent scans record slightly different values of one ion.
    """
    rows = []
    for data in datas:
        ms2 = data.MS_level_list == 2
        if polarity is not None:
            ms2 &= data.polarity_list == polarity
        rows.append(np.column_stack((
            np.round(data.collision_energy_list[ms2], CE_DECIMALS),
            data.precursor_mz_list[ms2],
            data.isolation_width_list[ms2],
        )))
    if not rows or not sum(len(row) for row in rows):
        return []
    rows = np.concatenate(rows)
    values, inverse, counts = np.unique(rows[:, :2], axis=0, return_inverse=True, return_counts=True)
    widths = np.zeros(len(values))
    np.maximum.at(widths, inverse.ravel(), rows[:, 2])
    entries = []
    start = 0
    while start < len(values):
        collision_energy, lowest = values[start]
        stop = start + 1
        while (
            stop < len(values)
            and values[stop, 0] == collision_energy
            and values[stop, 1] <= lowest * (1 + 2 * merge_ppm * 1e-6)
        ):
            stop += 1
        entries.append(Precursor(
            precursor_mz=float((lowest + values[stop - 1, 1]) / 2),
            collision_energy=float(collision_energy),
            isolation_width=float(widths[start:stop].max()),
            n_scans=int(counts[start:stop].sum()),
        ))
        start = stop
    return sorted(entries, key=lambda entry: (entry.precursor_mz, entry.collision_energy))


def precursor_isolating(listed: list[Precursor], mz: float, collision_energy: float | None = None) -> Precursor | None:
    """The listed precursor nearest to mz whose isolation window holds it, of collision_energy if it has one;
    None when no window holds mz."""
    isolating = [precursor for precursor in listed if abs(precursor.precursor_mz - mz) <= precursor.isolation_width / 2]
    if not isolating:
        return None
    nearest_mz = min(isolating, key=lambda precursor: abs(precursor.precursor_mz - mz)).precursor_mz
    nearest = [precursor for precursor in isolating if precursor.precursor_mz == nearest_mz]
    return next(
        (
            precursor
            for precursor in nearest
            if collision_energy is not None
            and round(precursor.collision_energy, CE_DECIMALS) == round(collision_energy, CE_DECIMALS)
        ),
        nearest[0],
    )


def select_ms2_scans(data: MSData, selection: MS2Selection) -> np.ndarray:
    """Indices of the MS2 scans of the selection, in acquisition order; empty when the file has none."""
    precursor_mz = selection.precursor_mz
    return np.flatnonzero(
        (data.MS_level_list == 2)
        & (np.round(data.collision_energy_list, CE_DECIMALS) == round(selection.collision_energy, CE_DECIMALS))
        & (np.abs(data.precursor_mz_list - precursor_mz) <= precursor_mz * selection.merge_ppm * 1e-6 + PRECURSOR_EPSILON)
    )


def scans_of_polarity(data: MSData, scan_indices: np.ndarray, polarity: int | None) -> np.ndarray:
    """Those of the scans of a polarity: +1 positive, -1 negative, the sign of the adduct's charge; None, all."""
    if polarity is None:
        return scan_indices
    return scan_indices[data.polarity_list[scan_indices] == polarity]


def trace_scans(
    data: MSData,
    ms_level: int | None = None,
    *,
    ms2: MS2Selection | None = None,
    polarity: int | None = None,
) -> np.ndarray:
    """The scans a trace is made from, as the plots make it: of an MS level (None: all), with ms2 for MS2 only those
    of its precursor and CE, and with a polarity of it only; empty when there are none."""
    if ms_level == 2 and ms2 is not None:
        scan_indices = select_ms2_scans(data, ms2)
    else:
        try:
            scan_indices = data.scan_indices_for_ms_level(ms_level)
        except ValueError:  # no scans of the MS level
            return np.array([], dtype=int)
    return scans_of_polarity(data, scan_indices, polarity)


def polarity_name(polarity: int) -> str:
    return "positive" if polarity > 0 else "negative"


def select_scans(
    data: MSData,
    ms_level: int | None = None,
    rt_range: tuple[float, float] | None = None,
    *,
    ms2: MS2Selection | None = None,
    polarity: int | None = None,
) -> np.ndarray:
    """Indices of the scans of an MS level (None: all) within an RT range (None: all), in acquisition order;
    with ms2, of MS level 2, only those of its precursor and collision energy; with a polarity, of it only."""
    if ms2 is None:
        scan_indices = data.scan_indices_for_ms_level(ms_level)
    else:
        if ms_level != 2:
            raise ValueError("A precursor selects MS2 scans: the MS level must be 2")
        scan_indices = select_ms2_scans(data, ms2)
        if len(scan_indices) == 0:
            raise ValueError(f"No MS2 scans of precursor {ms2.precursor_mz:.4f} at CE {ms2.collision_energy:g}")
    if polarity is not None:
        scan_indices = scans_of_polarity(data, scan_indices, polarity)
        if len(scan_indices) == 0:
            raise ValueError(f"No {polarity_name(polarity)} scans of the choice")
    if rt_range is not None:
        rt = data.RT_list[scan_indices]
        scan_indices = scan_indices[(rt_range[0] <= rt) & (rt <= rt_range[1])]
        if len(scan_indices) == 0:
            raise ValueError(f"No scans in RT {rt_range[0]}-{rt_range[1]} min")
    return scan_indices


def extract_trace(
    data: MSData,
    trace_type: TraceType,
    *,
    mz_window: tuple[float, float] | None = None,
    scan_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(rt, intensity) of a trace: the summed intensity in mz_window for an XIC, or the
    scan records' TIC or base peak for TIC and BPC."""
    trace_type = TraceType(trace_type)
    if trace_type is TraceType.XIC:
        if mz_window is None:
            raise ValueError("An XIC needs an m/z window")
        return data.extract_chromatogram(mz_window[0], mz_window[1], scan_indices=scan_indices)
    if trace_type is TraceType.TIC:
        return data.extract_tic(scan_indices=scan_indices)
    if trace_type is TraceType.BPC:
        return data.extract_bpc(scan_indices=scan_indices)
    raise ValueError(f"Unknown trace type: {trace_type!r}")


@dataclass(frozen=True)
class MzWindowMisses:
    """Of each scan of an XIC, whether its m/z window holds none of the scan's profile data points: an XIC leaves
    such a scan out, as not measured there, rather than taking it for 0."""
    between_points: np.ndarray  # the window lies between two data points, narrower than their spacing
    beyond_data: np.ndarray  # the window lies beyond the m/z range of the scan's data
    spacing_ppm: float  # the widest spacing of those between_points, in ppm of the window's centre; 0 if none

    @property
    def missing(self) -> np.ndarray:
        return self.between_points | self.beyond_data

    def warning(self) -> str:
        """Why scans are left out, as the chromatogram says it; "" when none is."""
        if self.between_points.any():
            return (
                f"m/z window narrower than the data spacing ({self.spacing_ppm:.1f} ppm): "
                f"{np.count_nonzero(self.between_points)}/{len(self.between_points)} scans have no point in it"
            )
        if self.beyond_data.any():
            return "m/z window beyond the scanned m/z range"
        return ""


def mz_window_misses(data: MSData, mz_window: tuple[float, float], scan_indices: np.ndarray) -> MzWindowMisses:
    """The scans whose profile data points miss an XIC's m/z window, see MzWindowMisses.

    A window between two data points is narrower than their spacing there, which grows with m/z.
    """
    gaps = data.mz_window_gaps(mz_window[0], mz_window[1], scan_indices=scan_indices)
    beyond_data = np.isnan(gaps)
    between_points = np.nan_to_num(gaps) > 0
    spacing_ppm = 0.0
    if between_points.any():
        spacing_ppm = float(gaps[between_points].max() / ((mz_window[0] + mz_window[1]) / 2) * 1e6)
    return MzWindowMisses(between_points, beyond_data, spacing_ppm)


def averaged_spectrum(
    data: MSData,
    rt_window: tuple[float, float],
    *,
    mode: str = "mean",
    scan_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """(mz, intensity, representative scan) of the profile spectrum averaged ("mean") or summed
    ("sum") over the scans in [rt_window[0], rt_window[1]), or of the scan nearest to the
    window's center when none is in it."""
    return data.extract_averaged_spectrum(rt_window[0], rt_window[1], mode, scan_indices=scan_indices)


def scans_in_window(data: MSData, rt_window: tuple[float, float], scan_indices: np.ndarray) -> int:
    """How many scans `averaged_spectrum` combined for rt_window: at least the nearest one."""
    rt = data.RT_list[scan_indices]
    return max(int(np.count_nonzero((rt_window[0] <= rt) & (rt < rt_window[1]))), 1)


def profile_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    mz_range: tuple[float, float] | None = None,
    top_n: int | None = None,
) -> np.ndarray:
    """Indices of the nonzero local maxima of a profile spectrum, most intense first."""
    apexes = np.flatnonzero((intensity[1:-1] > intensity[:-2]) & (intensity[1:-1] >= intensity[2:]) & (intensity[1:-1] > 0)) + 1
    if mz_range is not None:
        apexes = apexes[(mz_range[0] <= mz[apexes]) & (mz[apexes] <= mz_range[1])]
    apexes = apexes[np.argsort(intensity[apexes], kind="stable")[::-1]]
    return apexes if top_n is None else apexes[:top_n]
