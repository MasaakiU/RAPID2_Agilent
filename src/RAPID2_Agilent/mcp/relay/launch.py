# -*- coding: utf-8 -*-

"""`rapid2_launch`: starting RAPID2 for the assistant, answered by the relay.

The relay is running whenever an AI client is -- the client starts it -- and
RAPID2 need not be, so the relay is the one place a request to open RAPID2 can
be answered. It starts RAPID2 with the interpreter the relay runs on and waits
until a window takes requests, so that "open RAPID2 and show me this XIC" can
carry on in the same turn.

What it does not do is switch AI assistant access on. RAPID2 remembers that
switch and turns access back on by itself at the next start; a window that
comes up with it off is reported as such, for the user to decide.

The tool is registered in the window's registry too, so that it appears in the
tool list a running window gives out. There it only says that RAPID2 is
already open: the relay answers every call to it before it reaches a window.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

from .. import presence
from .. import registry
from ..protocol import ToolError


TOOL_NAME = "rapid2_launch"
TITLE = "Start RAPID2"
DESCRIPTION = (
    "Start RAPID2 when it is not running, and wait until its window takes "
    "requests. Call it when another RAPID2 tool answers that RAPID2 is not "
    "reachable, or when the user asks you to open RAPID2; then carry on with "
    "the task. If RAPID2 is already open it is left as it is. Starting RAPID2 "
    "does not switch AI assistant access on: if the user had it off, this says "
    "so, and only the user can turn it on (Tools > AI Assistant Access). "
    "Answered on this computer, so it works while RAPID2 is closed; takes no "
    "target."
)
INPUT_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}

# Long enough for a cold start of the interpreter and Qt, short of the minute a
# client allows one tool call.
LAUNCH_WAIT_SECONDS = 30.0
POLL_SECONDS = 0.25

ACCESS_OFF = (
    "Ask the user to turn on AI assistant access in RAPID2: Tools > AI "
    "Assistant Access. RAPID2 remembers it and turns access on by itself at "
    "the next start."
)


def application_command():
    """How to start RAPID2 from here: (argv, cwd)."""

    return [sys.executable, "-m", "RAPID2_Agilent.main"], None


def start_detached(argv, cwd):
    """Start `argv` so that it outlives this relay and its client."""

    options = {
        "cwd": None if cwd is None else str(cwd),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if sys.platform != "win32":
        subprocess.Popen(argv, start_new_session=True, **options)
        return
    detached = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        # A client may run its servers in a job that ends with it; RAPID2 must
        # not end when the assistant's app is closed.
        subprocess.Popen(
            argv, creationflags=detached | subprocess.CREATE_BREAKAWAY_FROM_JOB, **options
        )
    except OSError:
        subprocess.Popen(argv, creationflags=detached, **options)


def _window_rows():
    try:
        found = registry.read_instances()
    except Exception:  # noqa: BLE001 - a roster must not fail the launch
        found = []
    return [
        {
            "instance": document.get("label"),
            "instance_id": document.get("instance_id"),
            "started_at": document.get("started_at"),
            "app_version": document.get("app_version"),
        }
        for document in found
    ]


class Launcher:
    """The handler for `rapid2_launch`. Everything it touches is injectable."""

    def __init__(
        self,
        reachable,
        *,
        command=application_command,
        start=start_detached,
        window_running=presence.window_running,
        windows=_window_rows,
        wait_seconds=LAUNCH_WAIT_SECONDS,
        poll_seconds=POLL_SECONDS,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        self._reachable = reachable
        self._command = command
        self._start = start
        self._window_running = window_running
        self._windows = windows
        self._wait_seconds = wait_seconds
        self._poll_seconds = poll_seconds
        self._clock = clock
        self._sleep = sleep
        # The relay answers calls concurrently, and two calls that both found
        # RAPID2 closed would start it twice. The second waits, and then finds
        # the first one's window.
        self._lock = threading.Lock()

    def __call__(self, arguments):
        with self._lock:
            return self._launch(arguments)

    def _launch(self, _arguments):
        if self._reachable():
            return {
                "status": "ok",
                "started": False,
                "message": "RAPID2 is already open and taking requests.",
                "windows": self._windows(),
            }
        if self._window_running():
            raise ToolError(
                "RAPID2 is open, but AI assistant access is off in it, so "
                f"nothing was started. {ACCESS_OFF}"
            )

        argv, cwd = self._command()
        try:
            self._start(argv, cwd)
        except OSError as exc:
            raise ToolError(f"RAPID2 could not be started: {exc}") from exc

        deadline = self._clock() + self._wait_seconds
        while self._clock() < deadline:
            self._sleep(self._poll_seconds)
            if self._reachable():
                return {
                    "status": "ok",
                    "started": True,
                    "message": "RAPID2 started and is taking requests.",
                    "windows": self._windows(),
                }

        if self._window_running():
            raise ToolError(
                "RAPID2 started, but it is not taking requests: AI assistant "
                f"access is off in it. {ACCESS_OFF}"
            )
        raise ToolError(
            f"RAPID2 was started but no window opened within "
            f"{self._wait_seconds:g} seconds. It may still be opening: call "
            "rapid2_launch again in a moment, or ask the user to open RAPID2."
        )
