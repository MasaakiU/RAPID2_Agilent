"""The Codex plugins (scripts/build_codex_plugin.py), after MyWB's tests of its own.

A plugin carries no server code: it names the launcher a RAPID2 window writes. So what matters is what fails when
that stops being true -- a renamed launcher has to break here, not ship a plugin that points at nothing -- and that
the committed files, which Codex reads from the tree, are what the build produces now.
"""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_codex_plugin as builder  # noqa: E402
import rapid2_distribution  # noqa: E402

from RAPID2_Agilent.mcp import paths as mcp_paths  # noqa: E402
from RAPID2_Agilent.mcp.relay import launcher  # noqa: E402

HOME = Path("/Users/somebody")


def written_launcher(platform, environment):
    """Where a RAPID2 window on platform writes the launcher, relative to the home directory: the real place, not
    the redirect the suite installs."""
    with mock.patch.object(sys, "platform", platform), mock.patch.dict(os.environ, environment), \
            mock.patch.object(Path, "home", classmethod(lambda _cls: HOME)):
        return (mcp_paths.platform_launcher_dir() / launcher.launcher_name(platform)).relative_to(HOME).as_posix()


def test_the_macos_plugin_spawns_the_launcher_rapid2_writes():
    _command, arguments = builder.mcp_command(builder.DARWIN)
    assert arguments == ["-c", f'exec "$HOME/{written_launcher("darwin", {})}"']


@pytest.mark.parametrize("local_app_data", [r"C:\Users\somebody\AppData\Local", r"D:\Redirected\somebody\Local"])
def test_the_windows_plugin_spawns_the_launcher_rapid2_writes_wherever_local_app_data_is(local_app_data):
    _command, arguments = builder.mcp_command(builder.WIN32)
    expected = written_launcher("win32", {"LOCALAPPDATA": local_app_data}).replace("/", "\\")
    assert arguments == ["/c", f"%USERPROFILE%\\{expected}"]


def test_the_mcp_config_is_what_codex_reads_and_declares_no_server_code():
    for platform in rapid2_distribution.PLATFORMS:
        document = builder.build_mcp_config(platform)
        # `mcpServers`, not the `mcp_servers` of the published example, which Codex ignores (openai/codex#22105)
        assert list(document) == ["mcpServers"] and list(document["mcpServers"]) == [launcher.SERVER_NAME]
        command, _arguments = builder.mcp_command(platform)
        # Codex expands nothing in `command`
        assert "$" not in command and "%" not in command
        body = json.dumps(document)
        assert "${PLUGIN_ROOT}" not in body and "cwd" not in body
    with pytest.raises(ValueError):
        builder.mcp_command("linux")


def test_the_windows_path_reaches_cmd_as_written():
    # subprocess.list2cmdline is the MSVCRT quoting that Rust's std::process::Command implements: the line cmd is
    # handed. A quote in the argument would arrive escaped as \", which cmd reads literally.
    command, arguments = builder.mcp_command(builder.WIN32)
    assert subprocess.list2cmdline([command, *arguments]) == f"cmd /c {arguments[-1]}"
    # and the part RAPID2 names has nothing cmd would split or expand
    assert not set(rapid2_distribution.launcher_relative(builder.WIN32)) & set(' &|<>^%"()')


def test_the_manifest_has_what_a_listing_needs():
    for platform in rapid2_distribution.PLATFORMS:
        manifest = builder.build_plugin_manifest(platform, "1.2.3")
        assert {"name", "version", "description", "author", "mcpServers"} <= manifest.keys()
        assert {"displayName", "shortDescription", "longDescription", "developerName", "category", "capabilities"} \
            <= manifest["interface"].keys()
        assert manifest["mcpServers"] == "./.mcp.json" and manifest["interface"]["logo"].startswith("./assets/")
        assert manifest["interface"]["websiteURL"].startswith("https://")
    assert len(builder.DEFAULT_PROMPTS) <= 3 and all(len(prompt) <= 128 for prompt in builder.DEFAULT_PROMPTS)


def test_the_marketplace_names_each_plugin_by_its_directory_and_manifest():
    marketplace = builder.build_marketplace()
    assert builder.marketplace_path().relative_to(ROOT).as_posix() == ".agents/plugins/marketplace.json"
    for platform, entry in zip(rapid2_distribution.PLATFORMS, marketplace["plugins"]):
        name = builder.PLUGIN_NAMES[platform]
        assert entry["name"] == name == builder.plugin_dir(platform).name == builder.build_plugin_manifest(platform, "1")["name"]
        assert entry["source"] == {"source": "local", "path": f"./integrations/codex/{name}"}
        assert entry["policy"] == {"installation": "AVAILABLE", "authentication": "ON_USE"}


def test_a_build_elsewhere_writes_the_same_tree(tmp_path):
    assert builder.main(["--root", str(tmp_path)]) == 0
    assert json.loads(builder.marketplace_path(tmp_path).read_text(encoding="utf-8")) == builder.build_marketplace()
    for platform in rapid2_distribution.PLATFORMS:
        written = builder.plugin_dir(platform, tmp_path)
        assert sorted(path.name for path in (written / ".codex-plugin").iterdir()) == ["plugin.json"]
        logo = (written / "assets" / "logo.png").read_bytes()
        assert logo[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", logo[16:24]) == (rapid2_distribution.ICON_PIXELS,) * 2


def test_the_committed_plugins_are_what_the_build_produces(tmp_path):
    """Codex installs from the tree, so a stale commit is a broken install: run scripts/build_codex_plugin.py."""
    version = rapid2_distribution.project_version()
    committed = json.loads(builder.marketplace_path().read_text(encoding="utf-8"))
    assert committed == builder.build_marketplace(), "stale marketplace.json; run scripts/build_codex_plugin.py"
    for platform in rapid2_distribution.PLATFORMS:
        fresh = builder.write_plugin(platform, version, tmp_path / builder.PLUGIN_NAMES[platform])
        for name in (".codex-plugin/plugin.json", ".mcp.json", "assets/logo.png"):
            path = builder.plugin_dir(platform) / name
            assert path.is_file(), f"{path} is missing; run scripts/build_codex_plugin.py"
        for name in (".codex-plugin/plugin.json", ".mcp.json"):
            assert json.loads((builder.plugin_dir(platform) / name).read_text(encoding="utf-8")) == \
                json.loads((fresh / name).read_text(encoding="utf-8")), f"stale {name}; run scripts/build_codex_plugin.py"
