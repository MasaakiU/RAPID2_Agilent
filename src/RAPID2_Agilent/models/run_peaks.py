"""The peaks of a target's trace in each run, and the RT window integrated there; no Qt dependency.

A run is known by its .d folder, resolved, not by the object that opened it: a file of the targets and their values
(.rpd.csv) gives the folder, so that the windows of each run can be matched again once it is read back.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .chemistry import parse_adduct
from .target import Target, TraceType


def target_mz_window(target: Target) -> tuple[float, float] | None:
    """The m/z window of the target's XIC; None for TIC and BPC."""
    return target.mz_window() if target.trace_type is TraceType.XIC else None


def trace_of(target: Target) -> tuple:
    """What chooses the scans of the target's trace and what of them it is, but for the m/z window of an XIC."""
    positive = parse_adduct(target.adduct).charge > 0
    return target.trace_type, target.ms_level, target.precursor_mz, target.collision_energy, positive


@dataclass(frozen=True)
class Peak:
    """A peak found in a trace, by RT (min): where it starts and ends, and its top."""
    start: float
    end: float
    apex: float

    def __post_init__(self):
        if not self.start <= self.apex <= self.end:
            raise ValueError(f"A peak needs start <= apex <= end, got {self.start}, {self.apex}, {self.end}")


def assign_peak(peaks: Sequence[Peak], target: Target) -> Peak | None:
    """The target's peak among those found: of those whose top is in its RT window, the one nearest its RT."""
    low, high = target.rt_window()
    inside = [peak for peak in peaks if low <= peak.apex <= high]
    return min(inside, key=lambda peak: abs(peak.apex - target.rt), default=None)


def taken_window(peaks: Sequence[Peak], target: Target) -> tuple[float, float] | None:
    """The RT window of the target's peak among those found (assign_peak); None if none is."""
    peak = assign_peak(peaks, target)
    return None if peak is None else (peak.start, peak.end)


@dataclass(frozen=True)
class RunPeaks:
    """A target's trace in one run: the m/z window of its XIC, the peaks found in that trace, and the RT window
    integrated -- the target's peak, or one set by hand --, over which the spectrum is averaged too."""
    mz_window: tuple[float, float] | None  # None for TIC and BPC
    rt_window: tuple[float, float] | None  # None: no peak taken for the target's
    peaks: tuple[Peak, ...] = ()
    manual: bool = False  # rt_window set by hand (None too: no peak, by hand), not taken from the peaks found
    trace: tuple = ()  # trace_of the target, of which the peaks were found

    def __post_init__(self):
        for name in ("mz_window", "rt_window"):
            window = getattr(self, name)
            if window is not None and not window[0] <= window[1]:
                raise ValueError(f"{name} must be (low, high), got {window}")


def run_peaks_for(
    target: Target, rt_window: tuple[float, float] | None, peaks: Sequence[Peak] = (), *, manual: bool = False
) -> RunPeaks:
    """RunPeaks of the target's trace as it is now."""
    return RunPeaks(target_mz_window(target), rt_window, tuple(peaks), manual, trace_of(target))


class RunPeaksStore:
    """The RunPeaks of each target in each run. They stay when a target is removed or a run closed, so that undoing
    the removal, or opening the run again, finds them."""

    def __init__(self):
        self._run_peaks: dict[tuple[str, Path], RunPeaks] = {}

    def get(self, target_id: str, run: str | Path) -> RunPeaks | None:
        return self._run_peaks.get((target_id, Path(run).resolve()))

    def set(self, target_id: str, run: str | Path, run_peaks: RunPeaks) -> None:
        self._run_peaks[target_id, Path(run).resolve()] = run_peaks

    def set_found(self, target_id: str, run: str | Path, found: RunPeaks, *, keep_manual: bool) -> None:
        """found, the peaks found in the run and the one taken, in place of what it had; but for a window set by
        hand, which stays, with the peaks found, if keep_manual."""
        kept = self.get(target_id, run)
        if keep_manual and kept is not None and kept.manual:
            found = replace(found, rt_window=kept.rt_window, manual=True)
        self.set(target_id, run, found)

    def copy(self) -> RunPeaksStore:
        copied = RunPeaksStore()
        copied.restore(self.snapshot())
        return copied

    def snapshot(self) -> dict:
        """The RunPeaks as they are, for restore to take them back to."""
        return dict(self._run_peaks)

    def restore(self, snapshot: dict) -> None:
        self._run_peaks = dict(snapshot)

    def follow(self, target: Target) -> None:
        """Keep the target's RunPeaks to the target as it is now, as the files' m/z windows follow the target's for
        now. Of another trace, the peaks found in the former are dropped, and the windows taken from them; one set
        by hand stays. Of the same trace, the target's peak is taken again, as its RT may have moved."""
        mz_window, trace = target_mz_window(target), trace_of(target)
        for key, run_peaks in list(self._run_peaks.items()):
            if key[0] != target.id:
                continue
            if (run_peaks.mz_window, run_peaks.trace) != (mz_window, trace):
                if run_peaks.manual:
                    self._run_peaks[key] = RunPeaks(mz_window, run_peaks.rt_window, manual=True, trace=trace)
                else:
                    del self._run_peaks[key]
            elif not run_peaks.manual:
                self._run_peaks[key] = replace(run_peaks, rt_window=taken_window(run_peaks.peaks, target))
