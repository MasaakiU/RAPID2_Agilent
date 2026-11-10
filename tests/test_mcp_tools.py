"""The tool surface, through the dispatcher a window answers with. No Qt."""

import json
import subprocess
import sys
from unittest import mock

import pytest

from RAPID2_Agilent.alignment import OPTIONS as ALIGNMENT_OPTIONS
from RAPID2_Agilent.mcp import guide, registry
from RAPID2_Agilent.mcp.protocol import INVALID_PARAMS, Dispatcher, unsupported_keywords
from RAPID2_Agilent.mcp.relay import launch
from RAPID2_Agilent.mcp.tools import AnalysisSession, build_registry
from RAPID2_Agilent.services import reports
from tests.data_paths import DATA_DIR, require


DATA_PATH = DATA_DIR / "dummy_RP_2min_neg.d"
MSMS_DATA_PATH = DATA_DIR / "MSMS_data" / "9_PRJ_0-1-2_PA34-1_1uM_n1.d"
IDENTITY = {"label": "A", "instance_id": "a1b2c3d4e5", "started_at": "2026-10-21T00:00:00+00:00"}
VIEWER_TOOLS = {
    "rapid2_get_status", "rapid2_open_files", "rapid2_set_view", "rapid2_capture",
    "rapid2_list_targets", "rapid2_add_target", "rapid2_select_target", "rapid2_import_targets", "rapid2_export_targets",
}


class FakeContext:
    """Stands in for the live window; records what the viewer tools asked of it."""

    def __init__(self):
        self.calls = []

    def status(self):
        self.calls.append(("status",))
        return {"files": []}

    def open_files(self, paths, spectra=None):
        self.calls.append(("open_files", paths, spectra))
        return {"opened": paths}

    def set_view(self, **changes):
        self.calls.append(("set_view", changes))
        return {"files": ["x"]}

    def capture(self, part):
        self.calls.append(("capture", part))
        return {"part": part}

    def add_target(self, spec):
        self.calls.append(("add_target", spec))
        return {"added": spec}

    def target_snapshot(self):
        self.calls.append(("target_snapshot",))
        return self.snapshot


@pytest.fixture
def context():
    return FakeContext()


@pytest.fixture
def analysis():
    session = AnalysisSession()
    yield session
    session.close()


@pytest.fixture
def dispatcher(context, analysis):
    on_main_thread = []

    def run_on_main_thread(call):
        on_main_thread.append(True)
        return call()

    tools = build_registry(context, analysis, run_on_main_thread=run_on_main_thread, identity=IDENTITY)
    dispatcher = Dispatcher(tools, "rapid2-agilent", "0.0.0", instructions=guide.server_instructions())
    dispatcher.on_main_thread = on_main_thread
    return dispatcher


@pytest.fixture
def one_window():
    """The registry lists only this window, whatever the suite's data dir holds."""

    with mock.patch.object(registry, "sweep_stale"), mock.patch.object(registry, "read_instances", return_value=[IDENTITY]):
        yield


def call(dispatcher, name, **arguments):
    response = dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
    if "error" in response:
        return response["error"], None
    result = response["result"]
    return result["structuredContent"], result["isError"]


def test_every_tool_is_prefixed_and_launch_is_listed(dispatcher):
    names = [tool["name"] for tool in dispatcher.registry.describe_all()]
    assert all(name.startswith("rapid2_") for name in names)
    assert launch.TOOL_NAME in names
    assert VIEWER_TOOLS <= set(names)


def test_every_schema_is_one_the_validator_enforces(dispatcher):
    for tool in dispatcher.registry.describe_all():
        assert unsupported_keywords(tool["inputSchema"]) == set(), tool["name"]


def test_only_viewer_tools_take_a_target(dispatcher):
    for tool in dispatcher.registry.describe_all():
        takes_target = "instance" in tool["inputSchema"]["properties"]
        assert takes_target == (tool["name"] in VIEWER_TOOLS), tool["name"]


def test_the_handshake_says_how_to_use_the_tools(dispatcher):
    response = dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
    assert "rapid2_launch" in response["result"]["instructions"]


def test_viewer_tools_run_on_the_main_thread_and_name_their_window(dispatcher, context, one_window):
    payload, is_error = call(dispatcher, "rapid2_get_status")
    assert not is_error
    assert payload["instance"] == "A" and payload["instance_id"] == IDENTITY["instance_id"]
    assert dispatcher.on_main_thread == [True]


def test_analysis_tools_stay_off_the_main_thread(dispatcher):
    payload, is_error = call(dispatcher, "rapid2_calculate_ion_mz", formula="C6H12O6", adduct="[M-H]-")
    assert not is_error
    assert payload["mz"] == pytest.approx(179.056112, abs=1e-6)
    assert dispatcher.on_main_thread == []


def test_a_call_for_another_window_is_refused(dispatcher, context, one_window):
    payload, is_error = call(dispatcher, "rapid2_get_status", instance="B")
    assert is_error and "asked for 'B'" in payload["message"]
    assert context.calls == []


def test_a_window_can_be_named_by_its_letter_or_id(dispatcher, context, one_window):
    for wanted in ("a", IDENTITY["instance_id"]):
        assert not call(dispatcher, "rapid2_get_status", instance=wanted)[1]


def test_an_untargeted_call_with_two_windows_open_gets_the_roster(dispatcher, context):
    other = {"label": "B", "instance_id": "ffff0000", "url": "http://127.0.0.1:1/mcp", "token": "t"}
    with mock.patch.object(registry, "sweep_stale"), mock.patch.object(registry, "read_instances", return_value=[IDENTITY, other]):
        payload, is_error = call(dispatcher, "rapid2_get_status")
    assert not is_error and payload["status"] == "ambiguous_instance"
    assert [row["instance"] for row in payload["windows"]] == ["A", "B"]
    assert context.calls == []


def test_set_view_passes_only_what_was_given(dispatcher, context, one_window):
    call(dispatcher, "rapid2_set_view", rt_min=1.0, rt_max=2.0, ms_level=None)
    assert context.calls == [("set_view", {
        "rt_window": (1.0, 2.0), "mz_window": None, "trace_type": None, "ms_level": None,
        "precursor_mz": None, "collision_energy": None,
    })]
    context.calls.clear()
    call(dispatcher, "rapid2_set_view", trace_type="TIC")
    assert context.calls[0][1]["ms_level"] is False  # left as it is


@pytest.mark.parametrize(
    ("arguments", "message"),
    [({"rt_min": 1.0}, "Give both rt_min and rt_max"), ({"mz_min": 151.0, "mz_max": 150.0}, "less than")],
)
def test_set_view_refuses_half_a_window(dispatcher, context, one_window, arguments, message):
    payload, is_error = call(dispatcher, "rapid2_set_view", **arguments)
    assert is_error and message in payload["message"]
    assert context.calls == []


def test_arguments_are_checked_before_anything_runs(dispatcher, context):
    error, _ = call(dispatcher, "rapid2_extract_xic", mz=150.5)
    assert error["code"] == INVALID_PARAMS and "'path' is required" in error["message"]
    error, _ = call(dispatcher, "rapid2_set_view", trace_type="SIM")
    assert error["code"] == INVALID_PARAMS
    error, _ = call(dispatcher, "rapid2_find_runs", directory="x", recursive=True)
    assert error["code"] == INVALID_PARAMS and "unknown argument" in error["message"]
    assert context.calls == []


def test_a_wrong_path_is_an_answer_not_a_crash(dispatcher, tmp_path):
    payload, is_error = call(dispatcher, "rapid2_describe_run", path=str(tmp_path))
    assert is_error
    assert payload["message"].startswith("Not an Agilent .d folder")  # not "ValueError: ..."


def test_analysis_tools_read_real_runs(dispatcher):
    path = str(require(DATA_PATH))
    described, is_error = call(dispatcher, "rapid2_describe_run", path=path)
    assert not is_error and described["n_scans"] == 118
    xic, is_error = call(dispatcher, "rapid2_extract_xic", path=path, mz=150.5, tolerance=0.5, unit="m/z", max_points=10)
    assert not is_error and len(xic["rt"]) == 10 and xic["n_scans"] == 118
    spectrum, is_error = call(dispatcher, "rapid2_extract_spectrum", path=path, rt=1.0, top_n=3)
    assert not is_error and len(spectrum["peaks"]) == 3
    json.dumps([described, xic, spectrum])


def test_analysis_tools_select_ms2_scans_by_precursor_and_ce(dispatcher):
    path = str(require(MSMS_DATA_PATH))
    described, _ = call(dispatcher, "rapid2_describe_run", path=path)
    assert described["n_ms2_precursors"] == 5
    listed, is_error = call(dispatcher, "rapid2_list_precursors", path=path)
    assert not is_error and listed["n_precursors"] == 5
    assert [entry["collision_energy"] for entry in listed["precursors"]] == [0.0, 10.0, 20.0, 30.0, 40.0]

    # the listed value as printed, matched within its isolation window; mz is a product ion
    xic, is_error = call(
        dispatcher, "rapid2_extract_xic", path=path, mz=415.25, tolerance=0.05, unit="m/z", precursor_mz=673.4814, collision_energy=20,
    )
    assert not is_error and (xic["ms_level"], xic["n_scans"], xic["ms2"]["collision_energy"]) == (2, 205, 20.0)
    tic, is_error = call(dispatcher, "rapid2_extract_trace", path=path, precursor_mz=673.4814, collision_energy=40)
    assert not is_error and tic["n_scans"] == 204
    spectrum, is_error = call(
        dispatcher, "rapid2_extract_spectrum", path=path, rt=6.5, rt_tolerance=0.1, precursor_mz=673.4814, collision_energy=20, top_n=3,
    )
    assert not is_error and len(spectrum["peaks"]) == 3 and spectrum["ms2"]["collision_energy"] == 20.0
    json.dumps([listed, xic, tic, spectrum])

    payload, is_error = call(dispatcher, "rapid2_extract_xic", path=path, mz=415.25, precursor_mz=673.4814)
    assert is_error and payload["message"].endswith("have CE 0, 10, 20, 30, 40")
    payload, is_error = call(dispatcher, "rapid2_extract_xic", path=path, mz=415.25, precursor_mz=500.0, collision_energy=20)
    assert is_error and payload["message"].startswith("No MS2 scans isolate m/z 500")


def test_the_tools_module_does_not_load_qt():
    script = "import sys; import RAPID2_Agilent.mcp.tools; print(any(m.startswith('PyQt6') for m in sys.modules))"
    assert subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout.strip() == "False"


def test_alignment_tools_name_their_options_as_the_alignment_table_does(dispatcher):
    # Passed through to `alignment.with_options` unchanged, so a renamed option fails here.
    tools = {tool["name"]: tool for tool in dispatcher.registry.describe_all()}
    run_arguments = {"path", "reference", "query", "output_dir"}
    for name in ("rapid2_suggest_alignment_scales", "rapid2_align_runs"):
        assert set(tools[name]["inputSchema"]["properties"]) - run_arguments <= set(ALIGNMENT_OPTIONS)


def test_a_null_for_an_argument_that_takes_none_is_its_default(dispatcher, context, one_window):
    path = str(require(DATA_PATH))
    xic, is_error = call(dispatcher, "rapid2_extract_xic", path=path, mz=150.5, tolerance=None, unit=None, max_points=None, precursor_mz=None)
    # 10 ppm, every scan (max_points takes null), and MS level 1 although precursor_mz was named
    assert not is_error and xic["mz_window"] == pytest.approx([150.5 * (1 - 1e-5), 150.5 * (1 + 1e-5)])
    assert (xic["ms_level"], len(xic["rt"])) == (1, 118)
    spectrum, is_error = call(dispatcher, "rapid2_extract_spectrum", path=path, rt=1.0, top_n=None, mode=None)
    assert not is_error and len(spectrum["peaks"]) == 50
    call(dispatcher, "rapid2_capture", part=None)
    assert context.calls == [("capture", "window")]


def test_a_range_upside_down_is_refused(dispatcher):
    payload, is_error = call(dispatcher, "rapid2_extract_spectrum", path=str(require(DATA_PATH)), rt=1.0, mz_min=500, mz_max=400)
    assert is_error and payload["message"] == "mz_min must not be greater than mz_max."


def test_an_xic_of_a_formula_and_adduct(dispatcher):
    xic, is_error = call(dispatcher, "rapid2_extract_xic", path=str(require(DATA_PATH)), formula="C6H12O6", adduct="[M-H]-", max_points=3)
    assert not is_error and (xic["adduct"], xic["polarity"]) == ("[M-H]-", "negative")
    payload, is_error = call(dispatcher, "rapid2_extract_xic", path=str(require(DATA_PATH)))
    assert is_error and payload["message"] == "Give mz, or formula and adduct"


def test_targets_are_quantified_from_a_file_and_described(dispatcher, tmp_path):
    from RAPID2_Agilent.services import rpd_csv

    path = str(require(DATA_PATH))
    spec = {"formula": "C6H12O6", "adduct": "[M-H]-", "rt": 1.0, "rt_tolerance": 0.5}
    targets_file = tmp_path / "targets.rpd.csv"
    rpd_csv.write_rpd_csv(targets_file, rpd_csv.export_rows([reports.target_from_spec(spec, 1, [])], [])[0])
    result, is_error = call(
        dispatcher, "rapid2_quantify_targets", runs=[path], targets_file=str(targets_file), targets=[{**spec, "name": "again"}],
    )
    assert not is_error
    assert [(row["target"], row["name"]) for row in result["rows"]] == [(1, "C6H12O6"), (2, "again")]
    assert result["rows"][0]["area"] == result["rows"][1]["area"] > 0
    error, _ = call(dispatcher, "rapid2_quantify_targets", runs=[path], targets=[{"formula": "C6H12O6"}])
    assert error["code"] == INVALID_PARAMS and "'adduct' is required" in error["message"]
    payload, is_error = call(dispatcher, "rapid2_quantify_targets", runs=[path])
    assert is_error and payload["message"] == "Give targets_file, or targets."


def test_the_window_targets_are_exported_off_the_main_thread(dispatcher, context, one_window, tmp_path):
    spec = {"formula": "C6H12O6", "adduct": "[M-H]-", "rt": 1.0, "rt_tolerance": 0.5}
    context.snapshot = {
        "targets": (reports.target_from_spec(spec, 1, []),),
        "paths": [str(require(DATA_PATH))],
        "spectra": ["centroid"],
        "left_out": ["old.rpd"],
        "merge_ppm": 0,
        "run_peaks": None,
    }
    exported, is_error = call(dispatcher, "rapid2_export_targets", path=str(tmp_path / "out"))
    assert not is_error and exported["rpd_csv"] == str(tmp_path / "out.rpd.csv") and exported["left_out"] == ["old.rpd"]
    # each file read of the spectra the window has it of
    assert exported["spectra"] == ["centroid"]
    # the snapshot on the main thread, the measuring here
    assert dispatcher.on_main_thread == [True] and context.calls == [("target_snapshot",)]
    payload, is_error = call(dispatcher, "rapid2_export_targets", path=str(tmp_path / "out.rpd.csv"))
    assert is_error and payload["message"].startswith("Choose a new file")


def test_a_target_is_added_as_described(dispatcher, context, one_window):
    call(dispatcher, "rapid2_add_target", adduct="[M-H]-", rt=1.0, rt_tolerance=0.1, mz=None)
    assert context.calls == [("add_target", {"adduct": "[M-H]-", "rt": 1.0, "rt_tolerance": 0.1})]


def test_runs_are_mapped_and_compared(dispatcher):
    path = str(require(DATA_PATH))
    spots, is_error = call(dispatcher, "rapid2_map_spots", path=path, top_n=3)
    assert not is_error and len(spots["spots"]) == 3
    compared, is_error = call(dispatcher, "rapid2_compare_runs", reference=path, query=str(require(DATA_DIR / "dummy_RP_2min_neg_copy.d")))
    assert not is_error and compared["higher_in_query"] == []


def test_analysis_tools_read_the_centroid_spectra_when_asked(dispatcher):
    path = str(require(DATA_PATH))
    described, is_error = call(dispatcher, "rapid2_describe_run", path=path, spectra="centroid")
    assert not is_error and described["spectra"] == "centroid" and "ms1_centroid_mz_range" in described
    assert described["stored_spectra"] == {"profile": 164178754, "centroid": 8424068}
    profile, _ = call(dispatcher, "rapid2_extract_xic", path=path, mz=965.9998, max_points=None)
    centroid, is_error = call(dispatcher, "rapid2_extract_xic", path=path, mz=965.9998, max_points=None, spectra="centroid")
    assert not is_error and (profile["spectra"], centroid["spectra"]) == ("profile", "centroid")
    # the heights of the centroids, below the sums of the profile's points
    assert 0.1 < centroid["apex_intensity"] / profile["apex_intensity"] < 0.5
    spectrum, is_error = call(dispatcher, "rapid2_extract_spectrum", path=path, rt=1.0, top_n=3, spectra="centroid")
    assert not is_error and spectrum["peaks"][0]["mz"] == pytest.approx(966.0, abs=0.002)
    spots, is_error = call(dispatcher, "rapid2_map_spots", path=path, top_n=3, spectra="centroid")
    assert not is_error and spots["spectra"] == "centroid" and len(spots["spots"]) == 3
    quantified, is_error = call(
        dispatcher, "rapid2_quantify_targets", runs=[path], spectra="centroid",
        targets=[{"formula": "C6H12O6", "adduct": "[M-H]-", "rt": 1.0, "rt_tolerance": 0.5}],
    )
    assert not is_error and quantified["spectra"] == ["centroid"]


def test_the_isotope_pattern_tool_leaves_out_the_rare_peaks(dispatcher):
    payload, is_error = call(dispatcher, "rapid2_calculate_isotope_pattern", formula="C6H12O6", adduct="[M-H]-")
    assert not is_error and payload["charge"] == -1
    # M+3, 0.087 % of M+0, is below the 0.1 % left in by default
    assert [peak["label"] for peak in payload["peaks"]] == ["M+0", "M+1", "M+2"]
    assert payload["peaks"][0]["relative"] == 1 and payload["peaks"][1]["relative"] == pytest.approx(0.0684, abs=1e-3)
    more, _ = call(dispatcher, "rapid2_calculate_isotope_pattern", formula="C6H12O6", adduct="[M-H]-", min_relative=0.0005)
    assert [peak["label"] for peak in more["peaks"]] == ["M+0", "M+1", "M+2", "M+3"]


def test_analysis_tools_read_a_rpd_file_of_the_spectra_it_holds(dispatcher, tmp_path):
    from RAPID2_Agilent.io.rpd_file import write_rpd
    from tests.test_rpd_file import _data

    write_rpd(tmp_path / "x.rpd", _data("discrete"))
    found, is_error = call(dispatcher, "rapid2_find_runs", directory=str(tmp_path))
    assert not is_error and [run["name"] for run in found["runs"]] == ["x.rpd"]
    described, is_error = call(dispatcher, "rapid2_describe_run", path=found["runs"][0]["path"])
    assert not is_error and described["spectra"] == "centroid" and described["n_scans"] == 4


def test_a_run_without_the_spectra_asked_for_is_an_answer(dispatcher, tmp_path):
    (tmp_path / "x.d" / "AcqData").mkdir(parents=True)
    (tmp_path / "x.d" / "AcqData" / "MSProfile.bin").write_bytes(b"x")
    payload, is_error = call(dispatcher, "rapid2_extract_trace", path=str(tmp_path / "x.d"), spectra="centroid")
    assert is_error and payload["message"] == "x.d has no centroid spectra, only profile spectra"


def test_files_are_opened_in_the_window_of_the_spectra_asked_for(dispatcher, context, one_window):
    call(dispatcher, "rapid2_open_files", paths=["a.d"])
    call(dispatcher, "rapid2_open_files", paths=["a.d"], spectra="centroid")
    # left out, the window chooses: a .d folder's profile, a .rpd file's own
    assert context.calls == [("open_files", ["a.d"], None), ("open_files", ["a.d"], "centroid")]
