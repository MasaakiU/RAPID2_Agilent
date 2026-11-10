# -*- coding: utf-8 -*-

"""Build the Codex plugins of RAPID2, as MyWB's scripts/build_codex_plugin.py does.

A Codex plugin is installed from a marketplace, a tree with its manifest at
`.agents/plugins/marketplace.json`; the plugins themselves can live anywhere,
because each entry names its own directory. Here the tree is this repository:

    codex plugin marketplace add <this repository, a clone or owner/repo>
    codex plugin add rapid2-agilent-macos@rapid2-agilent

The files are generated, and committed, because what a plugin encodes is the
launcher's path, read from the code that writes the launcher: a rename breaks
the build rather than shipping a plugin that points at nothing.
`tests/test_codex_plugin.py` fails when the committed files are stale -- after
a version bump, run this again.

Why there are two plugins, one per platform (from MyWB): a plugin's
`.mcp.json` gets no variable expansion in `command`, and the format has no
per-platform override. The launcher is at a per-user path, so the only way to
name it is to let a shell expand the home directory -- `/bin/sh` on macOS,
`cmd` on Windows -- and those are different commands.

Why the Windows path is not quoted (from MyWB): the host encodes the argument
list into one command line with the MSVCRT rules, which would escape an
embedded quote as `\\"`, and cmd would read a literal backslash-quote. Left
unquoted, Windows still finds a launcher below a home directory with a space
in it, by trying longer prefixes at each space. What is not checked here is
cmd.exe itself: run it on Windows once before announcing the plugin.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
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


PLUGINS_RELATIVE = INTEGRATIONS_RELATIVE / "codex"
MARKETPLACE_RELATIVE = Path(".agents") / "plugins" / "marketplace.json"
MARKETPLACE_NAME = "rapid2-agilent"

# A marketplace entry's name has to match both the plugin directory and the
# name in its manifest, so this one string names all three.
PLUGIN_NAMES = {DARWIN: "rapid2-agilent-macos", WIN32: "rapid2-agilent-windows"}
DISPLAY_NAMES = {DARWIN: f"{DISPLAY_NAME} (macOS)", WIN32: f"{DISPLAY_NAME} (Windows)"}

LOGO_RELATIVE = "./assets/logo.png"
CATEGORY = "Productivity"
# It drives a window the user is watching, and it writes files.
CAPABILITIES = ["Interactive", "Write"]

DEFAULT_PROMPTS = [
    "What does RAPID2 have open right now?",
    "List the runs in this folder and describe the first one.",
    "Quantify the targets of this .rpd.csv file in these runs.",
]


def mcp_command(platform):
    """How to start the relay on `platform`, as (command, args): through the shell, the one thing in the chain
    that expands the home directory."""

    relative = launcher_relative(platform)
    if platform == DARWIN:
        # `exec`, so the shell does not stay between the client and the relay
        return "/bin/sh", ["-c", f'exec "$HOME/{relative}"']
    if platform == WIN32:
        return "cmd", ["/c", "%USERPROFILE%\\" + relative.replace("/", "\\")]
    raise ValueError(f"No plugin is built for {platform}.")


def build_mcp_config(platform):
    """The plugin's `.mcp.json`: `mcpServers`, the key Codex reads, not the `mcp_servers` of its published example
    (openai/codex#22105)."""

    command, args = mcp_command(platform)
    return {"mcpServers": {SERVER_NAME: {"command": command, "args": args}}}


def build_plugin_manifest(platform, version):
    return {
        "name": PLUGIN_NAMES[platform],
        "version": version,
        "description": SHORT_DESCRIPTION,
        "author": {"name": AUTHOR_NAME, "url": REPOSITORY_URL},
        "homepage": REPOSITORY_URL,
        "repository": REPOSITORY_URL,
        "license": LICENSE,
        "keywords": list(KEYWORDS),
        "mcpServers": "./.mcp.json",
        "interface": {
            "displayName": DISPLAY_NAMES[platform],
            "shortDescription": SHORT_DESCRIPTION,
            "longDescription": LONG_DESCRIPTION,
            "developerName": AUTHOR_NAME,
            "category": CATEGORY,
            "capabilities": list(CAPABILITIES),
            "websiteURL": REPOSITORY_URL,
            "defaultPrompt": list(DEFAULT_PROMPTS),
            "logo": LOGO_RELATIVE,
        },
    }


def plugin_dir(platform, root=REPO_ROOT):
    return Path(root) / PLUGINS_RELATIVE / PLUGIN_NAMES[platform]


def marketplace_path(root=REPO_ROOT):
    return Path(root) / MARKETPLACE_RELATIVE


def build_marketplace(root=REPO_ROOT):
    return {
        "name": MARKETPLACE_NAME,
        "interface": {"displayName": DISPLAY_NAME},
        "plugins": [
            {
                "name": PLUGIN_NAMES[platform],
                # relative to the marketplace's root, so it reads the same wherever the tree is
                "source": {"source": "local", "path": "./" + plugin_dir(platform, root).relative_to(root).as_posix()},
                # nothing to authenticate at install: the relay proves itself to the window with a token it reads
                # when it is used
                "policy": {"installation": "AVAILABLE", "authentication": "ON_USE"},
                "category": CATEGORY,
            }
            for platform in PLATFORMS
        ],
    }


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_plugin(platform, version, directory):
    """Replace one plugin directory with a freshly generated one."""

    shutil.rmtree(directory, ignore_errors=True)
    write_json(directory / ".codex-plugin" / "plugin.json", build_plugin_manifest(platform, version))
    write_json(directory / ".mcp.json", build_mcp_config(platform))
    render_icon(directory / LOGO_RELATIVE)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="The tree to write into; this repository unless a copy is given.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    version = project_version()
    written = [write_plugin(platform, version, plugin_dir(platform, root)) for platform in PLATFORMS]
    marketplace = write_json(marketplace_path(root), build_marketplace(root))

    print("Created:")
    for directory in [*written, marketplace]:
        print(f"  {directory}")
    print(f"version: {version}")
    for platform in PLATFORMS:
        command, arguments = mcp_command(platform)
        print(f"  {PLUGIN_NAMES[platform]}: {command} {' '.join(arguments)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
