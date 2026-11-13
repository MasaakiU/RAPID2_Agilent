# -*- coding: utf-8 -*-

"""The standalone build: three programs sharing one bundled interpreter.

PyInstaller freezes RAPID2 (packaging/RAPID2_Agilent.spec) into one folder --
on macOS one .app -- holding the three commands `pip install` would give:

    rapid2-agilent       the window
    rapid2-agilent-cli   the analysis tools as commands
    rapid2-agilent-mcp   the stdio relay an AI client starts

There is no interpreter there to hand `-m` to, so where RAPID2 starts one of
its own programs -- the relay launcher, `rapid2_launch` -- it names the
executable beside the running one instead. All three are built from one entry
script, which runs the program its executable is named after.

Qt-free: the relay asks this module too.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


GUI = "rapid2-agilent"
CLI = "rapid2-agilent-cli"
RELAY = "rapid2-agilent-mcp"
PROGRAMS = (GUI, CLI, RELAY)

# Read by PyInstaller's bootloader: start as an application of its own, not as
# a child of the frozen program that started it.
RESET_ENVIRONMENT = "PYINSTALLER_RESET_ENVIRONMENT"


def is_frozen():
    return bool(getattr(sys, "frozen", False))


def executable(program, platform=None):
    """Where `program` is: beside the running executable."""

    platform = sys.platform if platform is None else platform
    suffix = ".exe" if platform == "win32" else ""
    return Path(sys.executable).with_name(program + suffix)


def program_name():
    """Which of the programs is running, by the executable's name; else the window."""

    name = Path(sys.executable).name.lower().removesuffix(".exe")
    return name if name in PROGRAMS else GUI


def child_environment():
    """The environment for a frozen program to start another with; None to inherit.

    A frozen program's environment tells the bootloader it is that program's
    child, which a program meant to outlive it is not.
    """

    if not is_frozen():
        return None
    return {**os.environ, RESET_ENVIRONMENT: "1"}


def run(argv=None):
    """The entry script's body: run the program this executable is."""

    program = program_name()
    if program == CLI:
        from .cli import main
    elif program == RELAY:
        from .mcp.relay.__main__ import main
    else:
        from .main import main
    return main(argv)
