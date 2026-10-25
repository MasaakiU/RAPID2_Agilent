"""The shared layer: operations (arrays, for every front end) and reports (JSON, for the CLI and MCP)."""

import json
import subprocess
import sys

import numpy as np
import pytest

from RAPID2_Agilent import services
from RAPID2_Agilent.models.chemistry import ion_mz
from RAPID2_Agilent.models.rpd_model import AgilentDModel
from RAPID2_Agilent.models.target import TraceType
from RAPID2_Agilent.services import operations, reports
from RAPID2_Agilent.services.reports import _trace
from tests.data_paths import DATA_DIR, require


DATA_PATH = DATA_DIR / "dummy_RP_2min_neg.d"
MSMS_DATA_PATH = DATA_DIR / "MSMS_data" / "9_PRJ_0-1-2_PA34-1_1uM_n1.d"
CHOLINE_DATA_PATH = DATA_DIR / "9_P2-8-15_spl1_p0.d"  # choline, C5H14NO+, near RT 3.3 min


@pytest.fixture
def runs():
    cache = services.RunCache()
    yield cache
    cache.close()


def loaded(script):
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout.strip()


def test_the_shared_layer_does_not_import_qt():
    assert loaded("import sys; import RAPID2_Agilent.services.reports; print(any(m.startswith('PyQt6') for m in sys.modules))") == "False"


def test_operations_do_not_bring_in_the_alignment_package():
    # The GUI imports operations at startup; scipy waits for the B - A map.
    assert loaded("import sys; import RAPID2_Agilent.services; print(any(m.startswith('scipy') for m in sys.modules))") == "False"


# -- operations ----------------------------------------------------------------
def test_a_gui_model_and_a_reader_give_the_same_arrays(runs):
    # The GUI presenters pass their model, the reports a bare reader: one operation serves both.
    reader = runs.get(require(DATA_PATH))
    model = AgilentDModel()
    model.load(DATA_PATH)
    for data in (reader, model):
        scans = operations.select_scans(data, 1, (0.5, 1.5))
        assert np.all((data.RT_list[scans] >= 0.5) & (data.RT_list[scans] <= 1.5))
    scans = operations.select_scans(reader, 1)
    for trace_type in TraceType:
        from_reader = operations.extract_trace(reader, trace_type, mz_window=(150.0, 151.0), scan_indices=scans)
        from_model = operations.extract_trace(model, trace_type, mz_window=(150.0, 151.0), scan_indices=scans)
        np.testing.assert_array_equal(from_reader[1], from_model[1])
    spectrum_reader = operations.averaged_spectrum(reader, (1.0, 1.1), scan_indices=scans)
    spectrum_model = operations.averaged_spectrum(model, (1.0, 1.1), scan_indices=scans)
    np.testing.assert_array_equal(spectrum_reader[1], spectrum_model[1])


def test_each_trace_type_is_the_reader_extraction(runs):
    reader = runs.get(require(DATA_PATH))
    scans = reader.scan_indices_for_ms_level(1)
    np.testing.assert_array_equal(
        operations.extract_trace(reader, TraceType.XIC, mz_window=(150.0, 151.0), scan_indices=scans)[1],
        reader.extract_chromatogram(150.0, 151.0, scan_indices=scans)[1],
    )
    np.testing.assert_array_equal(operations.extract_trace(reader, "TIC", scan_indices=scans)[1], reader.extract_tic(scan_indices=scans)[1])
    np.testing.assert_array_equal(operations.extract_trace(reader, "BPC", scan_indices=scans)[1], reader.extract_bpc(scan_indices=scans)[1])
    with pytest.raises(ValueError, match="m/z window"):
        operations.extract_trace(reader, TraceType.XIC, scan_indices=scans)


def test_an_rt_range_with_no_scans_is_an_error(runs):
    with pytest.raises(ValueError, match="No scans"):
        operations.select_scans(runs.get(require(DATA_PATH)), 1, (50.0, 60.0))


def test_profile_peaks_are_local_maxima_most_intense_first():
    mz = np.arange(10.0)
    intensity = np.array([0, 3, 0, 0, 7, 0, 5, 5, 0, 0], dtype=float)
    assert operations.profile_peaks(mz, intensity).tolist() == [4, 6, 1]
    assert operations.profile_peaks(mz, intensity, mz_range=(2.0, 9.0), top_n=1).tolist() == [4]


# -- reports -------------------------------------------------------------------
def test_long_traces_keep_the_maximum_of_each_group():
    rt = np.arange(1000) / 100
    intensity = np.zeros(1000)
    intensity[[3, 517]] = [5.0, 9.0]
    trace = _trace(rt, intensity, max_points=10)
    assert trace["downsampled"] and len(trace["rt"]) == 10
    assert trace["apex_rt"] == rt[517] and trace["apex_intensity"] == 9.0
    assert rt[3] in trace["rt"] and rt[517] in trace["rt"]
    assert _trace(rt[:5], intensity[:5], max_points=10)["rt"] == rt[:5].tolist()


def test_run_path_rejects_folders_without_acqdata(tmp_path):
    with pytest.raises(ValueError, match="AcqData"):
        services.run_path(tmp_path)


def test_find_runs_lists_d_folders_without_entering_them(tmp_path):
    for name in ("a.d", "nested/b.D", "nested/b.D/inner.d", "not_a_run.d/x"):
        (tmp_path / name / "AcqData").mkdir(parents=True)
    names = [run["name"] for run in services.find_runs(tmp_path)]
    assert names == ["a.d", "b.D"]


def test_calculate_ion_matches_chemistry():
    ion = reports.calculate_ion("C6H12O6", "[M-H]-")
    assert ion["mz"] == pytest.approx(179.056112, abs=1e-6)
    assert ion["neutral_mass"] == pytest.approx(180.063388, abs=1e-6)


def test_describe_run(runs):
    summary = reports.describe_run(runs.get(require(DATA_PATH)))
    assert summary["n_scans"] == 118
    assert summary["scan_types"] == [{"type": "Negative MS1", "scans": 118}]
    assert summary["ms1_profile_mz_range"][0] < summary["ms1_profile_mz_range"][1]
    json.dumps(summary)


def test_xic_matches_reader_and_is_serializable(runs):
    reader = runs.get(require(DATA_PATH))
    xic = reports.xic(reader, 150.5, 0.5, "m/z")
    rt, intensity = reader.extract_chromatogram(150.0, 151.0, scan_indices=reader.scan_indices_for_ms_level(1))
    assert xic["intensity"] == intensity.astype(float).tolist()
    assert xic["apex_intensity"] == float(intensity.max())
    json.dumps(xic)


def test_an_xic_window_narrower_than_the_data_spacing_misses_scans(runs):
    reader = runs.get(require(CHOLINE_DATA_PATH))
    mz = ion_mz("C5H14NO", "[M]+")
    scans = operations.select_scans(reader, 1, (2.635, 4.170))

    def misses(ppm):
        return operations.mz_window_misses(reader, (mz * (1 - ppm * 1e-6), mz * (1 + ppm * 1e-6)), scans)

    assert not misses(10).missing.any()
    # the data points lie 7.65 ppm apart here: a 2 ppm window falls between two of them in some scans
    narrow = misses(1)
    assert np.count_nonzero(narrow.between_points) == 9 and not narrow.beyond_data.any()
    assert narrow.spacing_ppm == pytest.approx(7.65, abs=0.01)
    _rt, intensity = reader.extract_chromatogram(mz * (1 - 1e-6), mz * (1 + 1e-6), scan_indices=scans)
    assert np.array_equal(intensity == 0, narrow.missing)
    # beyond the data, not between its points
    beyond = operations.mz_window_misses(reader, (5000.0, 5000.001), scans)
    assert beyond.beyond_data.all() and not beyond.between_points.any() and beyond.spacing_ppm == 0.0


def test_scans_and_precursors_of_a_polarity(runs):
    reader = runs.get(require(MSMS_DATA_PATH))  # negative ions only
    assert set(reader.polarity_list) == {-1}
    assert np.array_equal(operations.select_scans(reader, 2, polarity=-1), operations.select_scans(reader, 2))
    with pytest.raises(ValueError):
        operations.select_scans(reader, 2, polarity=1)
    assert operations.precursors([reader], polarity=1) == []
    assert operations.precursors([reader], polarity=-1) == operations.precursors([reader])


def test_rt_range_and_ms_level_select_scans(runs):
    reader = runs.get(require(MSMS_DATA_PATH))
    tic = reports.trace(reader, "tic", ms_level=2, rt_range=(2.0, 3.0), max_points=10_000)
    assert all(2.0 <= rt <= 3.0 for rt in tic["rt"])
    assert tic["n_scans"] == np.count_nonzero((reader.MS_level_list == 2) & (reader.RT_list >= 2.0) & (reader.RT_list <= 3.0))
    with pytest.raises(ValueError):
        reports.trace(reader, "XIC")


class _MS2Records:
    """The scan records of MS2 scans alone, as operations.precursors and select_ms2_scans read them."""

    def __init__(self, precursor_mz, collision_energy):
        self.precursor_mz_list = np.array(precursor_mz, dtype=float)
        self.collision_energy_list = np.array(collision_energy, dtype=float)
        self.MS_level_list = np.full(len(self.precursor_mz_list), 2)
        self.isolation_width_list = np.full(len(self.precursor_mz_list), 1.3)


def test_targeted_ms2_precursors_are_listed_by_collision_energy(runs):
    reader = runs.get(require(MSMS_DATA_PATH))

    listed = operations.precursors([reader])

    assert [(round(entry.precursor_mz, 4), entry.collision_energy, entry.n_scans) for entry in listed] == [
        (673.4814, 0.0, 205), (673.4814, 10.0, 205), (673.4814, 20.0, 205), (673.4814, 30.0, 205), (673.4814, 40.0, 204),
    ]
    scans = operations.select_ms2_scans(reader, operations.MS2Selection(listed[2].precursor_mz, 20.0))
    assert len(scans) == 205
    assert set(reader.MS_level_list[scans]) == {2} and set(reader.collision_energy_list[scans]) == {20.0}
    assert operations.precursors([runs.get(require(DATA_PATH))]) == []


def test_close_precursors_of_one_ce_merge_within_ppm():
    # data-dependent scans record slightly different values of one ion: 500.000 and 500.004 (8 ppm)
    data = _MS2Records([500.0, 500.004, 500.004, 500.05, 500.0], [20, 20, 20, 20, 30])
    assert [entry.n_scans for entry in operations.precursors([data])] == [1, 1, 2, 1]

    merged = operations.precursors([data], merge_ppm=10)

    assert [(entry.precursor_mz, entry.collision_energy, entry.n_scans) for entry in merged] == [
        (pytest.approx(500.0), 30.0, 1), (pytest.approx(500.002), 20.0, 3), (pytest.approx(500.05), 20.0, 1),
    ]
    selection = operations.MS2Selection(merged[1].precursor_mz, 20.0, merge_ppm=10)
    np.testing.assert_array_equal(operations.select_ms2_scans(data, selection), [0, 1, 2])


def test_spectrum_peaks_are_sorted_local_maxima_in_range(runs):
    reader = runs.get(require(DATA_PATH))
    spectrum = reports.spectrum(reader, rt=1.0, rt_tolerance=0.05, mz_range=(150.0, 400.0), top_n=5)
    intensities = [peak["intensity"] for peak in spectrum["peaks"]]
    assert len(intensities) == 5 and intensities == sorted(intensities, reverse=True)
    assert all(150.0 <= peak["mz"] <= 400.0 for peak in spectrum["peaks"])
    assert spectrum["n_scans"] >= 1


def test_run_cache_keeps_recent_runs_open():
    cache = services.RunCache(max_runs=1)
    first = cache.get(require(DATA_PATH))
    assert cache.get(DATA_PATH) is first
    cache.get(require(MSMS_DATA_PATH))
    assert first.file_path is None  # evicted and closed
    cache.close()


def test_alignment_reports_run_the_shared_jobs(runs, tmp_path):
    reference = require(DATA_PATH)
    query = require(DATA_DIR / "dummy_RP_2min_neg_copy.d")
    scales = reports.scale_candidates(runs, reference, max_scans=20)
    assert scales["n_scans"] == 20 and scales["scale_candidates"]["median"] > 0
    aligned = reports.align_runs(reference, query, tmp_path / "out", scale=scales["scale_candidates"]["median"], max_scans=20, device="cpu")
    assert aligned["rt_shift_seconds"] == {"min": 0.0, "median": 0.0, "max": 0.0}
    assert (tmp_path / "out" / "mapping.json").is_file()
    with pytest.raises(FileExistsError):
        reports.align_runs(reference, query, tmp_path / "out", scale=1.0, max_scans=20)


# -- the XIC as the plots make it ------------------------------------------------
def test_an_xic_leaves_out_the_scans_whose_data_points_miss_its_window(runs):
    reader = runs.get(require(CHOLINE_DATA_PATH))
    xic = reports.xic(reader, ion_mz("C5H14NO", "[M]+"), 1, "ppm", rt_range=(2.635, 4.170), max_points=None)
    # 9 of its 92 scans have no data point within 1 ppm (test above): left out, not taken for 0
    assert (xic["n_scans"], xic["n_missing"]) == (92 - 9, 9)
    assert 0 not in xic["intensity"]
    assert xic["warning"].startswith("m/z window narrower than the data spacing (7.7 ppm): 9/92")
    wide = reports.xic(reader, ion_mz("C5H14NO", "[M]+"), 10, "ppm", rt_range=(2.635, 4.170))
    assert wide["n_missing"] == 0 and "warning" not in wide


def test_an_xic_beyond_the_data_has_no_apex(runs):
    xic = reports.xic(runs.get(require(DATA_PATH)), 5000.0, 10)
    assert (xic["n_scans"], xic["n_missing"], xic["apex_rt"], xic["rt"]) == (0, 118, None, [])
    assert xic["warning"] == "m/z window beyond the scanned m/z range"
    # a trace of 0 has no peak top either
    assert _trace(np.arange(3.0), np.zeros(3), 300)["apex_rt"] is None


def test_an_xic_of_a_formula_and_adduct_uses_the_scans_of_its_polarity(runs):
    reader = runs.get(require(DATA_PATH))  # negative ions only
    xic = reports.xic(reader, formula="C6H12O6", adduct="[M-H]-", max_points=3)
    assert xic["mz"] == pytest.approx(ion_mz("C6H12O6", "[M-H]-")) and xic["polarity"] == "negative"
    with pytest.raises(ValueError, match="No positive scans"):
        reports.xic(reader, formula="C6H12O6", adduct="[M+H]+")
    with pytest.raises(ValueError, match="is negative, not positive"):
        reports.xic(reader, formula="C6H12O6", adduct="[M-H]-", polarity="positive")
    with pytest.raises(ValueError, match="not both"):
        reports.xic(reader, 179.05, formula="C6H12O6", adduct="[M-H]-")
    with pytest.raises(ValueError, match="needs an adduct"):
        reports.xic(reader, formula="C6H12O6")


class _TwoPolarities:
    """The scan records of a run that switches polarity every scan."""

    def __init__(self):
        self.RT_list = np.linspace(0.0, 1.0, 6)
        self.MS_level_list = np.ones(6, dtype=int)
        self.polarity_list = np.array([1, -1] * 3)

    def scan_indices_for_ms_level(self, ms_level):
        return np.arange(6)


def test_a_run_of_both_polarities_needs_one_chosen():
    data = _TwoPolarities()
    with pytest.raises(ValueError, match="both polarities"):
        reports._scans(data, 1)
    assert reports._scans(data, 1, polarity=-1).tolist() == [1, 3, 5]
    assert reports.polarity_sign("negative") == -1 and reports.polarity_sign(None, "[M+Na]+") == 1


def test_a_trace_of_every_scan(runs):
    tic = reports.trace(runs.get(require(DATA_PATH)), max_points=None)
    assert len(tic["rt"]) == tic["n_scans"] == 118 and not tic["downsampled"]


# -- targets and their values ------------------------------------------------------
def test_a_target_from_its_description(runs):
    reader = runs.get(require(MSMS_DATA_PATH))
    xic = reports.target_from_spec({"formula": "C6H12O6", "adduct": "[M-H]-", "rt": 1.0, "rt_tolerance": 0.2}, 3, [reader])
    assert (xic.index, xic.name, xic.trace_type, xic.mz_tolerance, xic.mz_tolerance_unit.value) == (3, "C6H12O6", TraceType.XIC, 10, "ppm")
    assert xic.mz == pytest.approx(ion_mz("C6H12O6", "[M-H]-"))
    # MS2: the listed precursor and CE nearest those given, with its isolation width
    ms2 = reports.target_from_spec(
        {"adduct": "[M-H]-", "ms_level": "MS2", "mz": 415.25, "precursor_mz": 673.48, "collision_energy": 20, "rt": 6.5, "rt_tolerance": 1},
        1, [reader],
    )
    assert (ms2.precursor_mz, ms2.collision_energy, ms2.isolation_width) == pytest.approx((673.4813842773438, 20.0, 1.3))
    tic = reports.target_from_spec({"adduct": "[M-H]-", "trace_type": "TIC", "rt": 1, "rt_tolerance": 1, "mz": None}, 1, [reader])
    assert tic.trace_type is TraceType.TIC and "mz" not in reports.target_dict(tic)
    with pytest.raises(ValueError, match="needs mz, or a formula"):
        reports.target_from_spec({"adduct": "[M-H]-", "rt": 1, "rt_tolerance": 1}, 1, [reader])
    with pytest.raises(ValueError, match="needs mz, a product ion"):
        reports.target_from_spec({"adduct": "[M-H]-", "ms_level": "MS2", "formula": "C6H12O6", "rt": 1, "rt_tolerance": 1}, 1, [reader])
    with pytest.raises(ValueError, match="are for MS2 targets"):
        reports.target_from_spec({"adduct": "[M-H]-", "mz": 100, "precursor_mz": 673.48, "rt": 1, "rt_tolerance": 1}, 1, [reader])


def test_quantified_targets_are_the_rows_of_the_export(tmp_path):
    from RAPID2_Agilent.services import rpd_csv
    from RAPID2_Agilent.services.quantification import quantify_target

    paths = [require(DATA_PATH), require(DATA_DIR / "dummy_RP_2min_neg_copy.d")]
    spec = {"formula": "C6H12O6", "adduct": "[M-H]-", "rt": 1.0, "rt_tolerance": 0.5}
    result = reports.quantify_targets(paths, specs=[spec], rpd_csv_file=tmp_path / "out", max_rows=1)
    assert (result["n_rows"], result["truncated"], len(result["rows"])) == (2, True, 1)
    assert result["rpd_csv"] == str(tmp_path / "out.rpd.csv")
    reader = services.RunCache().get(paths[0])
    target = reports.target_from_spec(spec, 1, [reader])
    assert result["rows"][0]["area"] == quantify_target(reader, target).area
    # the file holds every row, and reads back as the target
    groups = rpd_csv.read_targets_file(result["rpd_csv"])
    assert [group.variants[0].formula for group in groups] == ["C6H12O6"]
    assert len((tmp_path / "out.rpd.csv").read_text(encoding="utf-8-sig").splitlines()) == 1 + 2
    json.dumps(result)
    with pytest.raises(FileExistsError):
        reports.quantify_targets(paths, specs=[spec], rpd_csv_file=tmp_path / "out.rpd.csv")
    with pytest.raises(ValueError, match="No targets"):
        reports.quantify_targets(paths)


# -- the m/z x RT map --------------------------------------------------------------
def test_the_strongest_spots_of_a_run_lie_apart(runs):
    spots = reports.map_spots(runs.get(require(DATA_PATH)), top_n=5)["spots"]
    assert len(spots) == 5
    assert [spot["intensity"] for spot in spots] == sorted((spot["intensity"] for spot in spots), reverse=True)
    for spot in spots:
        assert spot["mz_bin"][0] <= spot["mz"] <= spot["mz_bin"][1]
    within = reports.map_spots(runs.get(require(DATA_PATH)), top_n=5, rt_range=(1.0, 1.5), mz_range=(200.0, 400.0))["spots"]
    assert all(1.0 <= spot["rt"] <= 1.5 and spot["mz_bin"][1] > 200 and spot["mz_bin"][0] < 400 for spot in within)


def test_two_runs_that_are_the_same_differ_nowhere(runs):
    compared = reports.compare_runs(runs, require(DATA_PATH), require(DATA_DIR / "dummy_RP_2min_neg_copy.d"), top_n=5)
    assert compared["higher_in_query"] == compared["higher_in_reference"] == []
    assert compared["normalization"] == pytest.approx(1.0)


def test_two_different_runs_differ_both_ways_by_at_least_the_fold_change(runs):
    compared = reports.compare_runs(
        runs, require(CHOLINE_DATA_PATH), require(DATA_DIR / "4_P2-10-12_BD_sv_MeOH100_ACN0_p0.d"), top_n=5, min_fold_change=4,
    )
    for name, sign in (("higher_in_query", 1), ("higher_in_reference", -1)):
        differences = [spot["difference"] * sign for spot in compared[name]]
        assert len(differences) == 5 and differences == sorted(differences, reverse=True)
        assert min(differences) >= np.log(4)
    json.dumps(compared)
