# -*- coding: utf-8 -*-

"""Build the Claude plugin, `integrations/claude-desktop/rapid2-agilent-plugin.zip`.

Claude Desktop now installs plugins (Customize > Plugins > Upload, a .zip or .plugin) where it used to install
extensions. A plugin is `.claude-plugin/plugin.json` at the root of the archive, and its `mcpServers` may name a
`.mcpb` bundle inside it, which the client unpacks and reads as it reads an extension -- `${HOME}` and the Windows
override of `platform_overrides` included. So the plugin is the `.mcpb` of build_claude_desktop_mcpb.py, built afresh
inside it, and a manifest naming it: one archive for macOS and Windows alike, and no second copy of the launcher's
paths. The same layout is a plugin for Claude Code (`claude plugin validate`).

It has no skills: what an assistant is told of RAPID2 reaches it from the server itself (mcp/guide.py, the
`instructions` of its handshake), and a skill would be a second copy of that to keep in step.

Like the .mcpb it is committed, and the same file when nothing changed; `tests/test_claude_plugin.py` fails when it
is stale.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import build_claude_desktop_mcpb as mcpb
from rapid2_distribution import (
    AUTHOR_NAME,
    DISPLAY_NAME,
    INTEGRATIONS_RELATIVE,
    KEYWORDS,
    LICENSE,
    REPO_ROOT,
    REPOSITORY_URL,
    SERVER_NAME,
    SHORT_DESCRIPTION,
    project_version,
)


NAME = SERVER_NAME
BUNDLE_NAME = f"{NAME}.mcpb"
OUTPUT_RELATIVE = INTEGRATIONS_RELATIVE / "claude-desktop" / f"{NAME}-plugin.zip"


def build_manifest(version):
    return {
        "name": NAME,
        "displayName": DISPLAY_NAME,
        "version": version,
        "description": SHORT_DESCRIPTION,
        "author": {"name": AUTHOR_NAME, "url": REPOSITORY_URL},
        "homepage": REPOSITORY_URL,
        "repository": REPOSITORY_URL,
        "license": LICENSE,
        "keywords": list(KEYWORDS),
        "mcpServers": f"./{BUNDLE_NAME}",
    }


def stage(directory, version):
    """The plugin's files in directory: its manifest and the .mcpb it names."""
    manifest_path = directory / ".claude-plugin" / "plugin.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(build_manifest(version), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory() as bundle:
        mcpb.stage(Path(bundle), version)
        mcpb.package(Path(bundle), directory / BUNDLE_NAME)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="The tree to write into; this repository by default.")
    parser.add_argument("--staged", type=Path, help="Also leave the plugin's files unpacked here, e.g. for `claude plugin validate`.")
    args = parser.parse_args(argv)
    version = project_version()
    with tempfile.TemporaryDirectory() as directory:
        staged = stage(Path(directory) / NAME, version)
        if args.staged is not None:
            stage(args.staged.resolve(), version)
        # the archive's root is the plugin's, as the upload reads .claude-plugin/plugin.json there
        output = mcpb.package(staged, args.root.resolve() / OUTPUT_RELATIVE)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    print("Created:")
    print(f"  {output}  ({output.stat().st_size / 1024:.0f} KB): {', '.join(names)}")
    print(f"  version: {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
