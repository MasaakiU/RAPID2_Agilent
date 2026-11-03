import numpy as np
import pytest

from RAPID2_Agilent.models.chemistry import ion_mz
from RAPID2_Agilent.models.run_peaks import Peak, RunPeaks
from RAPID2_Agilent.models.target import MSLevel, Target, ToleranceUnit, TraceType
from RAPID2_Agilent.services import RunCache, operations
from RAPID2_Agilent.services.quantification import quantify, quantify_target, scan_settings_text
from tests.data_paths import DATA_DIR, require


CHOLINE_DATA_PATH = DATA_DIR / "9_P2-8-15_spl1_p0.d"  # choline, C5H14NO+, near RT 3.3 min
MSMS_DATA_PATH = DATA_DIR / "MSMS_data" / "9_PRJ_0-1-2_PA34-1_1uM_n1.d"


class _Run:
    """Positive MS1 ESI scans at ``rt`` whose XIC, in any m/z window, is ``intensity``; None where the window
    misses the scan's data points."""

    def __init__(self, rt, intensity, *, polarity=1):
        n = len(rt)
        self.RT_list = np.asarray(rt, dtype=float)
        self.MS_level_list = np.ones(n, dtype=int)
        self.polarity_list = np.full(n, polarity)
        self.ion_mode_list = np.full(n, 64)
        self.scan_type_list = np.ones(n, dtype=int)
        self._missing = np.array([value is None for value in intensity])
        self._intensity = np.array([0.0 if value is None else value for value in intensity])

    def scan_indices_for_ms_level(self, ms_level):
        if ms_level not in (None, 1):
            raise ValueError(f"No MS{ms_level} scans")
        return np.arange(len(self.RT_list))

    def mz_window_gaps(self, low_mz, high_mz, *, scan_indices):
        return np.where(self._missing[scan_indices], 0.001, 0.0)

    def extract_chromatogram(self, low_mz, high_mz, *, scan_indices):
        return self.RT_list[scan_indices], self._intensity[scan_indices]

    def extract_averaged_sub_spectrum(self, low_mz, high_mz, *, scan_indices):
        # its highest point in the middle of the window
        return np.array([low_mz, (low_mz + high_mz) / 2, high_mz]), np.array([0.0, self._intensity[scan_indices].mean(), 0.0])


def _xic(run, rt_window=(-1.0, 100.0), **kwargs):
    return quantify(run, TraceType.XIC, rt_window, mz_window=(100.0, 100.2), **kwargs)


def test_the_area_is_trapezoidal_above_a_flat_baseline_at_the_lowest_point():
    q = _xic(_Run([0, 1, 2, 3], [1, 3, 5, 1]))

    assert q.area == pytest.approx((1 + 3) / 2 + (3 + 5) / 2 + (5 + 1) / 2)
    assert q.baseline_height == 1
    assert q.baseline_adjusted_area == pytest.approx(9 - 1 * 3)
    assert (q.height, q.rt_peaktop, q.mz_peaktop) == (5, 2, pytest.approx(100.1))
    assert (q.n_scans, q.n_missing, q.rt_first, q.rt_last) == (4, 0, 0, 3)


def test_a_scan_missing_the_window_is_left_out_not_taken_for_0():
    # RAPID2 made it NaN (all values NaN) or 0 (no baseline taken off)
    q = _xic(_Run([0, 1, 2, 3, 4], [1, 3, None, 5, 1]))

    assert q.area == pytest.approx((1 + 3) / 2 + (3 + 5) / 2 * 2 + (5 + 1) / 2)
    assert q.baseline_height == 1
    assert (q.n_scans, q.n_missing) == (4, 1)


def test_the_rt_window_holds_its_ends():
    q = _xic(_Run([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]), rt_window=(1, 3))
    assert (q.n_scans, q.rt_first, q.rt_last) == (3, 1, 3)


def test_fewer_than_two_scans_give_no_values():
    q = _xic(_Run([0, 1, 2], [None, 3, None]))

    assert (q.n_scans, q.n_missing) == (1, 2)
    assert (q.area, q.baseline_adjusted_area, q.height, q.baseline_height) == (None, None, None, None)
    assert (q.rt_first, q.rt_last, q.rt_peaktop, q.mz_peaktop) == (None, None, None, None)
    assert q.scan_settings == "ESI+ Scan"


def test_a_trace_of_0_has_no_peak_top():
    # RAPID2 gave the first scan's RT
    q = _xic(_Run([0, 1, 2], [0, 0, 0]))
    assert (q.area, q.height, q.rt_peaktop, q.mz_peaktop) == (0, 0, None, None)


def test_scans_of_the_other_polarity_or_ms_level_are_not_used():
    run = _Run([0, 1, 2], [1, 2, 3], polarity=-1)
    assert _xic(run, polarity=1).n_scans == 0
    assert _xic(run, polarity=-1).n_scans == 3
    assert _xic(run, ms_level=2).n_scans == 0
    with pytest.raises(ValueError, match="m/z window"):
        quantify(run, TraceType.XIC, (0, 2))


def _xic_target(rt=1.0, rt_tolerance=1.0):
    return Target(
        index=1, name="x", formula="", adduct="[M+H]+", trace_type=TraceType.XIC, ms_level=MSLevel.MS1,
        rt=rt, rt_tolerance=rt_tolerance, mz=100.1, mz_tolerance=0.1, mz_tolerance_unit=ToleranceUnit.MZ,
    )


def test_a_target_is_quantified_in_the_windows_of_the_run_when_given():
    run, target = _Run([0, 1, 2, 3, 4, 5], [1, 3, 5, 1, 7, 9]), _xic_target()

    in_target = quantify_target(run, target)
    in_run = quantify_target(run, target, run_peaks=RunPeaks(mz_window=(200.0, 200.2), rt_window=(3.0, 5.0)))

    assert (in_target.rt_first, in_target.rt_last, in_target.area) == (0, 2, pytest.approx(6))
    assert (in_run.rt_first, in_run.rt_last, in_run.area) == (3, 5, pytest.approx((1 + 7) / 2 + (7 + 9) / 2))
    assert in_run.mz_peaktop == pytest.approx(200.1)  # the run's m/z window, not the target's


def test_the_deconvolved_area_is_of_the_peak_taken_and_of_none_of_a_window_set_by_hand():
    run, target = _Run([0, 1, 2, 3, 4, 5], [1, 3, 5, 1, 7, 9]), _xic_target()
    found = Peak(3.0, 5.0, 4.0, area=12.5, area_sigma=0.5, area_interval=1.5)
    taken = quantify_target(run, target, run_peaks=RunPeaks((100.0, 100.2), (3.0, 5.0), (Peak(0.0, 2.0, 1.0), found)))

    assert (taken.deconvolved_area, taken.deconvolved_area_sigma, taken.deconvolved_area_interval) == (12.5, 0.5, 1.5)
    assert taken.area == pytest.approx((1 + 7) / 2 + (7 + 9) / 2)  # of the window's points, as of any peak
    by_hand = quantify_target(run, target, run_peaks=RunPeaks((100.0, 100.2), (3.0, 5.0), (found,), manual=True))
    of_another = quantify_target(run, target, run_peaks=RunPeaks((100.0, 100.2), (3.0, 5.0), (Peak(3.0, 5.0, 4.0),)))
    assert by_hand.deconvolved_area is None and of_another.deconvolved_area is None
    assert quantify_target(run, target).deconvolved_area is None


def test_no_peak_taken_in_the_run_gives_no_values():
    q = quantify_target(_Run([0, 1, 2], [1, 2, 3]), _xic_target(), run_peaks=RunPeaks((100.0, 100.2), None))

    assert (q.n_scans, q.n_missing, q.scan_settings) == (0, 0, "")
    assert (q.area, q.height, q.rt_first, q.rt_peaktop) == (None, None, None, None)


def test_scan_settings_name_the_codes_seen_and_write_others_as_numbers():
    run = _Run([0, 1, 2, 3], [1, 1, 1, 1])
    run.scan_type_list = np.array([1, 512, 512, 256])
    run.ion_mode_list = np.array([64, 64, 64, 32])
    run.polarity_list = np.array([1, 1, 1, -1])
    assert scan_settings_text(run, np.arange(4)) == "IonMode 32- ScanType 256, ESI+ Scan, ESI+ ProductIon"
    assert scan_settings_text(run, np.array([], dtype=int)) == ""


@pytest.fixture
def runs():
    cache = RunCache()
    yield cache
    cache.close()


def test_choline_in_real_data_leaves_out_the_scans_a_narrow_window_misses(runs):
    reader = runs.get(require(CHOLINE_DATA_PATH))
    mz = ion_mz("C5H14NO", "[M]+")

    def choline(ppm):
        return quantify(
            reader, TraceType.XIC, (2.6345, 4.1701), mz_window=(mz * (1 - ppm * 1e-6), mz * (1 + ppm * 1e-6)),
            ms_level=1, polarity=1,
        )

    # each scan has one data point in a 6 ppm window, 7.65 ppm apart here
    wide = choline(3)
    assert (wide.n_scans, wide.n_missing) == (92, 0)
    assert wide.area == pytest.approx(18841.90, abs=0.01)
    assert (wide.baseline_height, wide.height) == (404, 50253)
    assert wide.rt_peaktop == pytest.approx(3.29198, abs=1e-5)
    assert wide.mz_peaktop == pytest.approx(mz, abs=1e-4)
    assert wide.scan_settings == "ESI+ Scan"
    # a 2 ppm window misses them in some scans: left out, the area stays near the same, the baseline too
    narrow = choline(1)
    assert (narrow.n_scans, narrow.n_missing) == (83, 9)
    assert narrow.area == pytest.approx(18918.62, abs=0.01)
    assert narrow.baseline_height == 404


def test_an_ms2_xic_uses_the_scans_of_its_precursor(runs):
    reader = runs.get(require(MSMS_DATA_PATH))  # negative ions only
    precursor = operations.precursors([reader])[0]
    ms2 = operations.MS2Selection(precursor.precursor_mz, precursor.collision_energy)

    q = quantify(reader, TraceType.TIC, (2.0, 4.0), ms_level=2, ms2=ms2, polarity=-1)

    in_window = operations.select_scans(reader, 2, (2.0, 4.0), ms2=ms2)
    assert q.n_scans == len(in_window)
    assert q.scan_settings == "ESI- ProductIon"
    assert q.area > 0 and q.mz_peaktop is not None
