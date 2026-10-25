# -*- coding: utf-8 -*-

"""Run a callable on the Qt main thread on behalf of an HTTP request thread.

A viewer tool reads or changes what the window shows, and the window is a tree
of Qt widgets. Touching those from the request thread is undefined behaviour,
so nothing does: a request hands the work here, and this hands it to the main
thread through a queued connection.

That also supplies the concurrency guarantee those tools rely on. A queued
slot runs as one pass of the event loop, so a tool call cannot be interleaved
with the user's own mouse and keyboard events -- they queue behind it.

Analysis tools do not come here. They compute from the files, not the window,
and running them on the main thread would freeze the window for as long as
they take.
"""

from __future__ import annotations

import threading

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot


DEFAULT_TIMEOUT_SECONDS = 10.0
# If a call has already started when its deadline passes, it is not abandoned:
# stopping a half-applied change is worse than waiting a little longer. Opening
# a large run reads its scan index on this thread, which can take a while.
STARTED_CALL_GRACE_SECONDS = 60.0


class MainThreadTimeout(Exception):
    """The main thread did not run the call within its deadline."""


class _Call:
    """One unit of work, plus the handshake that reports it finished."""

    __slots__ = ("fn", "result", "error", "done", "_lock", "_claimed", "_abandoned")

    def __init__(self, fn):
        self.fn = fn
        self.result = None
        self.error = None
        self.done = threading.Event()
        self._lock = threading.Lock()
        self._claimed = False
        self._abandoned = False

    def claim(self):
        """True when this call may run. False once the caller gave up on it."""

        with self._lock:
            if self._abandoned or self._claimed:
                return False
            self._claimed = True
            return True

    def abandon(self):
        """True when the call had not started and will now never run."""

        with self._lock:
            if self._claimed:
                return False
            self._abandoned = True
            return True


class MainThreadBridge(QObject):
    """Create on the Qt main thread; call ``invoke`` from any thread."""

    _submitted = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._submitted.connect(self._run, Qt.ConnectionType.QueuedConnection)

    @pyqtSlot(object)
    def _run(self, call):
        if not call.claim():
            return
        try:
            call.result = call.fn()
        except Exception as exc:  # noqa: BLE001 - re-raised on the calling thread
            call.error = exc
        finally:
            call.done.set()

    def on_main_thread(self):
        return QThread.currentThread() is self.thread()

    def invoke(self, fn, *, timeout=DEFAULT_TIMEOUT_SECONDS):
        """Run ``fn`` on the main thread and return its value.

        Raises whatever ``fn`` raised, or ``MainThreadTimeout`` if the main
        thread never got to it.
        """

        if self.on_main_thread():
            # Re-entrant use (tests, or a tool invoked from the UI itself)
            # must not wait for an event loop it is already inside.
            return fn()

        call = _Call(fn)
        self._submitted.emit(call)

        if not call.done.wait(timeout):
            if call.abandon():
                raise MainThreadTimeout(
                    f"RAPID2 did not respond within {timeout:.0f}s. It is busy, "
                    "or a dialog is waiting for input."
                )
            # Already running: let it finish rather than leave a partial change.
            if not call.done.wait(STARTED_CALL_GRACE_SECONDS):
                raise MainThreadTimeout(
                    "The operation started but has not finished. RAPID2 may be "
                    "waiting for input."
                )

        if call.error is not None:
            raise call.error
        return call.result
