from dataclasses import replace

import pytest

from RAPID2_Agilent.models.run_peaks import (
    Peak,
    RunPeaks,
    RunPeaksStore,
    assign_peak,
    run_peaks_for,
    taken_window,
)
from RAPID2_Agilent.models.target import MSLevel, Target, ToleranceUnit, TraceType


def test_a_peak_has_its_top_between_its_ends():
    assert Peak(1.0, 2.0, 1.0) == Peak(start=1.0, end=2.0, apex=1.0)
    with pytest.raises(ValueError, match="start <= apex <= end"):
        Peak(1.0, 2.0, 2.5)


def test_the_windows_of_a_run_go_from_low_to_high():
    assert RunPeaks(None, None).peaks == ()
    with pytest.raises(ValueError, match="rt_window"):
        RunPeaks((100.0, 100.2), (2.0, 1.0))
    with pytest.raises(ValueError, match="mz_window"):
        RunPeaks((100.2, 100.0), (1.0, 2.0))


def test_a_run_is_known_by_its_folder_however_its_path_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = RunPeaksStore()
    run_peaks = RunPeaks((100.0, 100.2), (1.0, 2.0), (Peak(1.0, 2.0, 1.5),))

    store.set("a", "x.d", run_peaks)

    assert store.get("a", tmp_path / "x.d") is run_peaks
    assert store.get("a", tmp_path / "y" / ".." / "x.d") is run_peaks
    assert store.get("b", "x.d") is None
    assert store.get("a", "y.d") is None


def _target(**fields):
    values = dict(
        index=1, name="x", formula="", adduct="[M+H]+", trace_type=TraceType.XIC, ms_level=MSLevel.MS1,
        rt=2.0, rt_tolerance=1.0, mz=100.1, mz_tolerance=0.1, mz_tolerance_unit=ToleranceUnit.MZ,
    )
    return Target(**(values | fields))


PEAKS = (Peak(1.0, 1.4, 1.2), Peak(1.6, 2.2, 1.9), Peak(3.5, 4.0, 3.8))


def test_the_target_peak_is_the_one_found_nearest_its_rt_with_its_top_in_its_window():
    assert assign_peak(PEAKS, _target(rt=2.0)) == PEAKS[1]
    assert assign_peak(PEAKS, _target(rt=1.0, rt_tolerance=0.3)) == PEAKS[0]
    # a top out of the window, whatever the peak's ends
    assert assign_peak(PEAKS, _target(rt=3.0, rt_tolerance=0.5)) is None
    assert taken_window(PEAKS, _target(rt=2.0)) == (1.6, 2.2)
    assert taken_window((), _target()) is None


def test_the_runs_follow_the_target_dropping_what_was_found_in_another_trace():
    target = _target()
    store = RunPeaksStore()
    store.set(target.id, "auto.d", run_peaks_for(target, (1.6, 2.2), PEAKS))
    store.set(target.id, "hand.d", run_peaks_for(target, (1.5, 2.5), PEAKS, manual=True))

    # the RT moved: the peak taken again among those found; none by hand
    moved = replace(target, rt=3.6)
    store.follow(moved)
    assert store.get(target.id, "auto.d").rt_window == (3.5, 4.0)
    assert store.get(target.id, "hand.d").rt_window == (1.5, 2.5)
    store.follow(replace(target, rt=3.0, rt_tolerance=0.2))
    assert store.get(target.id, "auto.d").rt_window is None

    # another trace: what was found goes, a window by hand stays without the peaks
    for changes in ({"mz": 150.0}, {"adduct": "[M-H]-"}, {"trace_type": TraceType.TIC}):
        store.set(target.id, "auto.d", run_peaks_for(target, (1.6, 2.2), PEAKS))
        store.set(target.id, "hand.d", run_peaks_for(target, (1.5, 2.5), PEAKS, manual=True))
        changed = replace(target, **changes)
        store.follow(changed)
        assert store.get(target.id, "auto.d") is None
        assert store.get(target.id, "hand.d") == run_peaks_for(changed, (1.5, 2.5), manual=True)


def test_peaks_found_replace_a_window_set_by_hand_unless_it_is_kept():
    target = _target()
    store = RunPeaksStore()
    found = run_peaks_for(target, (1.6, 2.2), PEAKS)
    for keep_manual, window in ((True, (1.5, 2.5)), (False, (1.6, 2.2))):
        store.set(target.id, "hand.d", run_peaks_for(target, (1.5, 2.5), manual=True))
        store.set_found(target.id, "hand.d", found, keep_manual=keep_manual)
        kept = store.get(target.id, "hand.d")
        assert (kept.rt_window, kept.peaks, kept.manual) == (window, PEAKS, keep_manual)
    store.set_found(target.id, "new.d", found, keep_manual=True)
    assert store.get(target.id, "new.d") == found
