# -*- coding: utf-8 -*-

"""Where RAPID2 keeps its MCP data, worked out without Qt.

The stdio relay is spawned by another application and must find exactly the
files the running window wrote, without importing PyQt6 to ask Qt where they
are. So both sides ask this module, and nothing else, for the directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


APPLICATION_NAME = "RAPID2_Agilent"

# Relocates everything RAPID2 keeps for itself, this module's callers included.
# It exists because a child process cannot be told any other way: the test
# suite redirects this directory so a test never writes a window letter or a
# launcher into the profile of whoever is running it, and a test that starts
# the relay in a subprocess would otherwise ignore that redirect and reach the
# real one. Same-user only, and the directory it names already holds the
# access token, so nothing is exposed that the reader could not already read.
DATA_DIR_ENV = "RAPID2_APP_LOCAL_DATA_DIR"


class DataPathError(Exception):
    """The operating system did not say where application data belongs."""


def _home():
    try:
        return Path.home()
    except (RuntimeError, OSError) as exc:
        raise DataPathError("This account has no home directory.") from exc


def app_local_data_dir(application_name=APPLICATION_NAME):
    """Where this run keeps its data: the override if set, else the real place."""

    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override)
    return platform_app_local_data_dir(application_name)


def platform_app_local_data_dir(application_name=APPLICATION_NAME):
    """The real per-user, machine-local data directory, ignoring the override.

    The same place as Qt's `QStandardPaths.AppLocalDataLocation` for an
    application of this name.
    """

    if sys.platform == "darwin":
        root = _home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA") or ""
        if not local_app_data:
            raise DataPathError("LOCALAPPDATA is not set.")
        root = Path(local_app_data)
    else:
        # XDG: a relative XDG_DATA_HOME is invalid and the default takes over.
        data_home = os.environ.get("XDG_DATA_HOME") or ""
        root = (
            Path(data_home)
            if os.path.isabs(data_home)
            else _home() / ".local" / "share"
        )
    return root / application_name


def launcher_dir(application_name=APPLICATION_NAME):
    """Where the relay launcher is written: the override if set, else home.

    The one file RAPID2 keeps that another program has to name, and the only
    thing that can name it is a client's configuration -- which can spell out
    the home directory and nothing else. On Windows the data directory is
    LOCALAPPDATA, which an administrator can redirect, so the launcher is
    anchored to the home directory instead.
    """

    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override) / "bin"
    return platform_launcher_dir(application_name)


def home_relative_launcher_dir(platform=None, application_name=APPLICATION_NAME):
    """Where the launcher's directory sits below the home directory, as POSIX.

    Answered for a platform rather than for this one, because the client
    integrations (scripts/build_*.py) are built on one machine and installed
    on another.
    """

    platform = sys.platform if platform is None else platform
    if platform == "darwin":
        parts = ("Library", "Application Support", application_name, "bin")
    elif platform == "win32":
        parts = ("AppData", "Local", application_name, "bin")
    else:
        parts = (".local", "share", application_name, "bin")
    return "/".join(parts)


def platform_launcher_dir(application_name=APPLICATION_NAME):
    """The real launcher directory, ignoring the override. See `launcher_dir`."""

    return _home().joinpath(*home_relative_launcher_dir(application_name=application_name).split("/"))
