"""The Claude Desktop extension (scripts/build_claude_desktop_mcpb.py), after MyWB's tests of its own.

Like the Codex plugins it carries no server code and names the launcher a RAPID2 window writes; and it is committed,
so it has to be the same file when nothing changed, and what the build produces now.
"""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_claude_desktop_mcpb as builder  # noqa: E402
import rapid2_distribution  # noqa: E402

from RAPID2_Agilent.mcp import paths as mcp_paths  # noqa: E402
from RAPID2_Agilent.mcp.relay import launcher  # noqa: E402


def test_the_extension_starts_the_launcher_rapid2_writes():
    config = builder.build_manifest("1.2.3")["server"]["mcp_config"]
    macos = f"{mcp_paths.home_relative_launcher_dir('darwin')}/{launcher.launcher_name('darwin')}"
    windows = f"{mcp_paths.home_relative_launcher_dir('win32')}/{launcher.launcher_name('win32')}".replace("/", "\\")
    assert config["command"] == "${HOME}/" + macos and config["args"] == []
    # a .cmd file, which Node does not start without a shell: through cmd, with backslashes cmd reads as a path
    assert config["platform_overrides"]["win32"] == {"command": "cmd", "args": ["/c", "${HOME}\\" + windows]}


def test_a_home_with_a_space_reaches_cmd_quoted_whole():
    # Node quotes an argument with a space as MSVCRT does; cmd /c keeps a quoted path of an executable whole
    _command, arguments = builder.build_manifest("1")["server"]["mcp_config"]["platform_overrides"]["win32"].values()
    path = arguments[-1].replace("${HOME}", r"C:\Users\Some Name")
    assert subprocess.list2cmdline(["cmd", *arguments[:-1], path]) == f'cmd /c "{path}"'


def test_the_manifest_has_what_claude_desktop_reads():
    manifest = builder.build_manifest("1.2.3")
    assert manifest["manifest_version"] == "0.3" and manifest["version"] == "1.2.3"
    assert manifest["name"] == launcher.SERVER_NAME and manifest["server"]["type"] == "binary"
    assert manifest["compatibility"]["platforms"] == ["darwin", "win32"] and manifest["icon"] == "icon.png"


def test_an_unchanged_build_is_the_same_file(tmp_path):
    builder.stage(tmp_path / "staged", "1.2.3")
    first = builder.package(tmp_path / "staged", tmp_path / "a.mcpb").read_bytes()
    second = builder.package(tmp_path / "staged", tmp_path / "b.mcpb").read_bytes()
    assert first == second
    with zipfile.ZipFile(tmp_path / "a.mcpb") as archive:
        assert archive.namelist() == ["icon.png", "manifest.json"]


def test_the_committed_extension_is_what_the_build_produces():
    """Claude Desktop reads the version inside to offer an update: run scripts/build_claude_desktop_mcpb.py."""
    committed = ROOT / builder.OUTPUT_RELATIVE
    assert committed.is_file(), f"{committed} is missing; run scripts/build_claude_desktop_mcpb.py"
    with zipfile.ZipFile(committed) as archive:
        assert "icon.png" in archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest == builder.build_manifest(rapid2_distribution.project_version()), \
        "stale .mcpb; run scripts/build_claude_desktop_mcpb.py"
