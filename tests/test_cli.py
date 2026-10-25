"""rapid2-agilent-cli, which is generated from the MCP analysis tools."""

import json
import subprocess
import sys

import pytest

from RAPID2_Agilent import cli
from RAPID2_Agilent.mcp.tools import AnalysisSession, analysis_tools
from tests.data_paths import DATA_DIR, require


DATA_PATH = DATA_DIR / "dummy_RP_2min_neg.d"


@pytest.fixture
def tools():
    analysis = AnalysisSession()
    yield analysis_tools(analysis)
    analysis.close()


def run_cli(*arguments):
    return subprocess.run([sys.executable, "-m", "RAPID2_Agilent.cli", *map(str, arguments)], capture_output=True, text=True)


def test_every_analysis_tool_is_a_command_with_its_options(tools):
    parser = cli.build_parser(tools)
    commands = parser._subparsers._group_actions[0].choices
    assert set(commands) == {cli.command_name(tool.name) for tool in tools}
    for tool in tools:
        options = {action.dest for action in commands[cli.command_name(tool.name)]._actions} - {"help"}
        assert options == set(tool.input_schema["properties"]), tool.name


def test_a_command_prints_the_tools_answer(capsys):
    assert cli.main(["calculate-ion-mz", "--formula", "C6H12O6", "--adduct", "[M-H]-"]) == 0
    assert json.loads(capsys.readouterr().out)["mz"] == pytest.approx(179.056112, abs=1e-6)


def test_an_option_left_out_keeps_the_tools_default_and_all_means_null(capsys):
    path = require(DATA_PATH)
    cli.main(["extract-xic", "--path", str(path), "--mz", "150.5", "--tolerance", "0.5", "--unit", "m/z", "--max-points", "3"])
    assert json.loads(capsys.readouterr().out)["ms_level"] == 1
    cli.main(["extract-xic", "--path", str(path), "--mz", "150.5", "--tolerance", "0.5", "--unit", "m/z", "--ms-level", "all"])
    assert json.loads(capsys.readouterr().out)["ms_level"] is None


def test_a_failure_is_a_message_and_exit_status_one(tmp_path):
    finished = run_cli("describe-run", "--path", tmp_path)
    assert finished.returncode == 1
    assert "Not an Agilent .d folder" in finished.stderr and finished.stdout == ""


def test_arguments_are_checked_like_the_mcp_server_checks_them():
    finished = run_cli("extract-xic", "--path", "x.d", "--mz", "-1")
    assert finished.returncode == 2 and "must be greater than 0" in finished.stderr
    assert run_cli("extract-trace", "--path", "x.d", "--trace-type", "XIC").returncode == 2


def test_tsv_is_the_answers_table_and_output_writes_a_file(tmp_path, capsys):
    path = require(DATA_PATH)
    out = tmp_path / "xic.tsv"
    assert cli.main(["--format", "tsv", "-o", str(out), "extract-xic", "--path", str(path), "--mz", "150.5", "--max-points", "all"]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "rt\tintensity" and len(lines) == 1 + 118
    assert capsys.readouterr().out == ""
    assert cli.main(["--format", "tsv", "calculate-ion-mz", "--formula", "C6H12O6", "--adduct", "[M-H]-"]) == 1
    assert "no table" in capsys.readouterr().err


def test_several_lists_are_one_table_with_the_list_named():
    columns, rows = cli.table({"n": 2, "higher": [{"mz": 1.0, "bin": [1, 2]}], "lower": [{"mz": 2.0}]})
    assert columns == ["list", "mz", "bin"]
    assert rows == [["higher", 1.0, [1, 2]], ["lower", 2.0, None]]
    assert cli._tsv(columns, rows) == "list\tmz\tbin\nhigher\t1.0\t1,2\nlower\t2.0\t\n"


def test_an_object_argument_is_json(capsys):
    path = require(DATA_PATH)
    target = '{"formula": "C6H12O6", "adduct": "[M-H]-", "rt": 1.0, "rt_tolerance": 0.5}'
    assert cli.main(["quantify-targets", "--runs", str(path), "--targets", target]) == 0
    assert json.loads(capsys.readouterr().out)["rows"][0]["name"] == "C6H12O6"
    finished = run_cli("quantify-targets", "--runs", "x.d", "--targets", "{not json")
    assert finished.returncode == 2 and "invalid JSON object value" in finished.stderr
