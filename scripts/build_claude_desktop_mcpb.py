# -*- coding: utf-8 -*-

"""Build the Claude Desktop extension, `integrations/claude-desktop/rapid2-agilent.mcpb`, as MyWB's does.

An .mcpb is a zip with a `manifest.json` at its root. This one carries no
server code: the relay it starts is the user's own RAPID2 installation,
through the launcher each RAPID2 window writes at a fixed path under the home
directory, so it always runs the RAPID2 the user has and never another
version. The paths are read from `mcp.paths` and `mcp.relay.launcher`, so a
rename of the launcher breaks the build. Install it in Claude Desktop from
Settings > Extensions (or by opening the file).

The file name has no version: what the bundle names is a path under the home
directory, the same whichever RAPID2 is installed. The version inside the
manifest still changes, which is what Claude Desktop reads to offer an update;
`tests/test_claude_desktop_mcpb.py` fails when the committed bundle is stale.

On Windows the launcher is a .cmd file, which Node -- Claude Desktop's -- does
not start without a shell since its fix of CVE-2024-27980; so it is started
through `cmd /c`, as the Codex plugin does. Unchecked on Windows yet.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from rapid2_distribution import (
    AUTHOR_NAME,
    DARWIN,
    DISPLAY_NAME,
    INTEGRATIONS_RELATIVE,
    KEYWORDS,
    LICENSE,
    LONG_DESCRIPTION,
    PLATFORMS,
    REPO_ROOT,
    REPOSITORY_URL,
    SERVER_NAME,
    SHORT_DESCRIPTION,
    WIN32,
    launcher_relative,
    project_version,
    render_icon,
)


NAME = SERVER_NAME
OUTPUT_RELATIVE = INTEGRATIONS_RELATIVE / "claude-desktop" / f"{NAME}.mcpb"

# A committed bundle has to be the same file when nothing changed: the zip epoch, the earliest a zip can express,
# for every entry's time.
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def launcher_command(platform):
    """The launcher, under `${HOME}`: the only substitution of `mcp_config` that names a per-user place."""

    relative = launcher_relative(platform)
    if platform == WIN32:
        # backslashes, as cmd reads a forward slash as the start of a switch
        return "${HOME}\\" + relative.replace("/", "\\")
    return "${HOME}/" + relative


def build_manifest(version):
    return {
        # `manifest_version`, not the `mcpb_version` that Anthropic's engineering post prints
        "manifest_version": "0.3",
        "name": NAME,
        "display_name": DISPLAY_NAME,
        "version": version,
        "description": SHORT_DESCRIPTION,
        "long_description": LONG_DESCRIPTION,
        "author": {"name": AUTHOR_NAME, "url": REPOSITORY_URL},
        "homepage": REPOSITORY_URL,
        "license": LICENSE,
        "keywords": list(KEYWORDS),
        "icon": "icon.png",
        "server": {
            "type": "binary",
            "entry_point": launcher_command(DARWIN),
            "mcp_config": {
                "command": launcher_command(DARWIN),
                "args": [],
                "env": {},
                "platform_overrides": {
                    WIN32: {"command": "cmd", "args": ["/c", launcher_command(WIN32)]},
                },
            },
        },
        "compatibility": {"platforms": list(PLATFORMS)},
    }


def stage(directory, version):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(version)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    render_icon(directory / "icon.png")
    return manifest


def package(staged, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(staged.rglob("*")):
            if path.is_file():
                entry = zipfile.ZipInfo(path.relative_to(staged).as_posix(), ZIP_EPOCH)
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o644 << 16
                archive.writestr(entry, path.read_bytes())
    return output


def validate(staged):
    """`mcpb validate`, when npx is here: the schema and its documentation disagree in places, and the CLI is the
    authority that is never out of date."""

    if shutil.which("npx") is None:
        print("Note: npx not found; skipped `mcpb validate`.")
        return True
    finished = subprocess.run(
        ["npx", "--yes", "@anthropic-ai/mcpb", "validate", "manifest.json"], cwd=staged, capture_output=True, text=True,
    )
    if finished.returncode != 0:
        print(finished.stdout)
        print(finished.stderr, file=sys.stderr)
        print("Error: `mcpb validate` rejected the manifest.", file=sys.stderr)
        return False
    print("OK: mcpb validate")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="The tree to write into; this repository by default.")
    parser.add_argument("--skip-validate", action="store_true", help="Do not run the mcpb CLI even when it is here.")
    args = parser.parse_args(argv)
    version = project_version()
    with tempfile.TemporaryDirectory() as directory:
        staged = Path(directory) / NAME
        manifest = stage(staged, version)
        if not args.skip_validate and not validate(staged):
            return 1
        output = package(staged, args.root.resolve() / OUTPUT_RELATIVE)
    print("Created:")
    print(f"  {output}  ({output.stat().st_size / 1024:.0f} KB)")
    print(f"  version: {manifest['version']}")
    print(f"  relay: {manifest['server']['mcp_config']['command']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
