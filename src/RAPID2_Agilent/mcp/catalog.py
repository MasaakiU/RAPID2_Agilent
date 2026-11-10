# -*- coding: utf-8 -*-

"""What the relay answers with while RAPID2 is closed. No Qt.

An AI client asks for the tools once, as it starts, and a client whose
handshake failed gives up on the server for the rest of that run. When only a
running window could answer, opening the client before RAPID2 left the
assistant without RAPID2 until the client was restarted.

The catalog is that answer written down: the tools as the window's own
registry describes them, and the name and version it introduces itself by.
Each window writes it when it starts serving, and the relay answers the
handshake and the tool list from it when no window is listening. It also
names the analysis tools, which read runs by path and never touch a
window, so that the relay can run those itself while no window is listening.
Everything else is forwarded as before -- a call that needs RAPID2 still needs
RAPID2, and `rapid2_launch`, which the relay answers itself, can start it.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import paths
from ..utils.private_files import write_private_file


CATALOG_NAME = "catalog.json"
# Raised when the shape changes in a way an older relay could misread. A relay
# that finds a format it does not know answers nothing from it, which is the
# same as having no catalog: the client waits for RAPID2, as it always did.
FORMAT_VERSION = 1
# Optional, so the format stays 1: a relay that does not know the key ignores
# it, and a catalog without it is one whose tools the relay never runs.
ANALYSIS_TOOLS_KEY = "analysis_tools"


def runtime_catalog_path():
    return paths.app_local_data_dir() / "mcp" / CATALOG_NAME


def catalog_document(tools, *, server_name, server_version, toolset_version, analysis_tools=()):
    """The catalog for `tools`, the descriptions a registry gives out.

    `analysis_tools` names those of them the relay may run itself.
    """

    return {
        "format": FORMAT_VERSION,
        "server_name": server_name,
        "server_version": server_version,
        "toolset_version": toolset_version,
        "tools": list(tools),
        ANALYSIS_TOOLS_KEY: list(analysis_tools),
    }


def analysis_tool_names(document):
    """The analysis tools of a catalog: named as such, and described in it."""

    listed = {tool["name"] for tool in document.get("tools", ())}
    return frozenset(name for name in document.get(ANALYSIS_TOOLS_KEY, ()) if name in listed)


def encode(document):
    return (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def read_catalog(path=None):
    """The catalog at `path`, or None if there is none fit to answer from.

    Anything short of a whole, current catalog is None rather than a partial
    answer: a tool list missing half its tools would be believed.
    """

    path = runtime_catalog_path() if path is None else Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, paths.DataPathError):
        return None
    return document if _usable(document) else None


def _usable(document):
    if not isinstance(document, dict):
        return False
    if document.get("format") != FORMAT_VERSION:
        return False
    if not isinstance(document.get("server_name"), str):
        return False
    if not isinstance(document.get("server_version"), str):
        return False
    tools = document.get("tools")
    if not isinstance(tools, list) or not tools:
        return False
    analysis = document.get(ANALYSIS_TOOLS_KEY, [])
    if not isinstance(analysis, list) or not all(isinstance(name, str) for name in analysis):
        return False
    return all(
        isinstance(tool, dict)
        and isinstance(tool.get("name"), str)
        and isinstance(tool.get("inputSchema"), dict)
        for tool in tools
    )


def write_runtime_catalog(document):
    """Leave `document` where the relay looks for it."""

    return write_private_file(
        encode(document), runtime_catalog_path(), require_private_directory=True
    )
