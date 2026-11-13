# -*- coding: utf-8 -*-

"""The one path an AI client is ever told about.

A client is registered once and must keep working afterwards -- after RAPID2
moves to another Python environment, or the environment is rebuilt. The
interpreter's path cannot offer that: it is wherever the environment is this
week.

So each RAPID2 window writes a launcher of its own at a location that never
moves, and rewrites it on every start with the interpreter it is running on
now. The client points at the launcher; the launcher starts the relay with
that interpreter. Registration happens once and is never mentioned again.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from .. import paths
from ... import frozen
from ...utils.private_files import write_private_file


RELAY_MODULE = "RAPID2_Agilent.mcp.relay"

# The name a client knows RAPID2 by: in the registration line, the server's
# introduction and the plugins (scripts/build_codex_plugin.py). Here, without
# Qt, so the plugin builds can read it.
SERVER_NAME = "rapid2-agilent"

_HEADER = "Written by RAPID2 on every start. Edits here are overwritten."


def launcher_name(platform=None):
    """What the launcher is called on `platform` -- this one by default."""

    platform = sys.platform if platform is None else platform
    return "rapid2-mcp-relay.cmd" if platform == "win32" else "rapid2-mcp-relay"


def default_launcher_path():
    """Where RAPID2 writes the launcher: under the home directory.

    `paths.launcher_dir` says why it is not the application data directory --
    a client's configuration can name the home directory and nothing else.
    """

    return paths.launcher_dir() / launcher_name()


def relay_invocation():
    """How to start the relay from here: (command, environment).

    The interpreter this window runs on, which has RAPID2 installed; the
    package refuses to import from a bare source tree. In the standalone
    build, the relay's own executable beside this one.
    """

    if frozen.is_frozen():
        return [str(frozen.executable(frozen.RELAY))], {}
    return [sys.executable, "-m", RELAY_MODULE], {}


def launcher_source(command, environment):
    if sys.platform == "win32":
        lines = ["@echo off", f"rem {_HEADER}"]
        lines += [f'set "{name}={value}"' for name, value in environment.items()]
        arguments = " ".join(f'"{part}"' for part in command)
        lines.append(f"{arguments} %*")
        return "\r\n".join(lines) + "\r\n"

    lines = ["#!/bin/sh", f"# {_HEADER}"]
    lines += [
        f"export {name}={shlex.quote(value)}" for name, value in environment.items()
    ]
    arguments = " ".join(shlex.quote(part) for part in command)
    lines.append(f'exec {arguments} "$@"')
    return "\n".join(lines) + "\n"


def write_launcher(path=None):
    """Point the launcher at this interpreter. Returns where it is."""

    path = Path(path) if path is not None else default_launcher_path()
    encoded_source = launcher_source(*relay_invocation()).encode("utf-8")
    # Rewriting an identical file every start would churn its timestamp for
    # nothing, and some backup tools notice. Compare bytes: text-mode reads
    # translate the CRLF in a Windows launcher to LF, making an identical
    # `.cmd` file look different on every start.
    try:
        if path.read_bytes() == encoded_source:
            return path
    except OSError:
        pass
    # Vouched for as well as written: an AI client runs this file at the start
    # of every session, so another account being able to put their own script
    # here would be running their code as this user.
    return write_private_file(
        encoded_source,
        path,
        mode=0o700,
        require_private_directory=True,
    )


def _quote_for_powershell(text):
    """Quote one argument the way PowerShell reads it."""

    return "'{}'".format(text.replace("'", "''"))


# The clients a registration line is written for, by the command each is
# registered with. Both take the same `<name> -- <command>` form.
CLIENTS = {"claude": "Claude Code", "codex": "Codex"}


def registration_command(path, server_name, client="claude"):
    """The line a user runs once to register RAPID2 with Claude Code
    (`client` "claude") or Codex ("codex").

    Written for PowerShell on Windows and a POSIX shell elsewhere, because no
    single quoting is right in both.
    """

    if client not in CLIENTS:
        raise ValueError(f"No registration line is written for {client!r}.")
    quote = _quote_for_powershell if sys.platform == "win32" else shlex.quote
    return f"{client} mcp add {server_name} -- {quote(str(path))}"
