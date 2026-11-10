"""The Claude plugin (scripts/build_claude_plugin.py): a manifest and the .mcpb it names, in one committed archive."""

import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_claude_desktop_mcpb as mcpb  # noqa: E402
import build_claude_plugin as builder  # noqa: E402
import rapid2_distribution  # noqa: E402


def _contents(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(".claude-plugin/plugin.json"))
        with zipfile.ZipFile(io.BytesIO(archive.read(builder.BUNDLE_NAME))) as bundle:
            bundle_manifest = json.loads(bundle.read("manifest.json"))
        return archive.namelist(), manifest, bundle_manifest


def test_the_plugin_names_the_mcpb_inside_it(tmp_path):
    staged = builder.stage(tmp_path / "plugin", "1.2.3")
    manifest = json.loads((staged / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    # a path inside the plugin, with the ./ prefix, ending in .mcpb: the shape Claude reads a bundle of
    assert manifest["mcpServers"] == "./rapid2-agilent.mcpb" and (staged / "rapid2-agilent.mcpb").is_file()
    assert manifest["name"] == "rapid2-agilent" and manifest["version"] == "1.2.3" and manifest["author"]["name"]
    # nothing but the manifest in .claude-plugin/, and no bin/, which Cowork refuses
    assert sorted(path.name for path in (staged / ".claude-plugin").iterdir()) == ["plugin.json"]
    assert not (staged / "bin").exists()


def test_an_unchanged_build_is_the_same_file(tmp_path):
    builder.main(["--root", str(tmp_path / "a")])
    builder.main(["--root", str(tmp_path / "b")])
    first, second = (tmp_path / root / builder.OUTPUT_RELATIVE for root in ("a", "b"))
    assert first.read_bytes() == second.read_bytes()
    names, _manifest, _bundle = _contents(first)
    # the archive's root is the plugin's
    assert names == [".claude-plugin/plugin.json", "rapid2-agilent.mcpb"]


def test_the_committed_plugin_is_what_the_build_produces():
    """Run scripts/build_claude_plugin.py after a version bump."""
    committed = ROOT / builder.OUTPUT_RELATIVE
    assert committed.is_file(), f"{committed} is missing; run scripts/build_claude_plugin.py"
    version = rapid2_distribution.project_version()
    _names, manifest, bundle_manifest = _contents(committed)
    assert manifest == builder.build_manifest(version), "stale plugin; run scripts/build_claude_plugin.py"
    assert bundle_manifest == mcpb.build_manifest(version), "stale .mcpb inside the plugin; run scripts/build_claude_plugin.py"
