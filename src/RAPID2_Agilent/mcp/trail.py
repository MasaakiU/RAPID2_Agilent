# -*- coding: utf-8 -*-

"""A line per tool call, so that a crash leaves the sequence behind.

An assistant drives RAPID2 far harder than a person does -- dozens of files
opened and traces extracted in the time it takes someone to set one window --
and the failures that turns up are the ones that need the *order* to reproduce
rather than the last step. An operating system crash report names the frame the
process died in; it cannot say that the same run had been opened, closed and
re-read forty times first. This is the half it cannot see, and for anything that
corrupts memory it is the half that leads back to the cause.

What it records is what was asked and what came back -- deliberately not what
came back *in full*. A capture is several megabytes of base64, and a log that
holds one is a log nobody can read on a disk nobody expected to fill.

Two lines per call, not one. A single line written afterwards would be the one
line missing for the call that actually killed the process, which is the only
call anybody would be reading the file for.

Wrapped around every tool at the point they are registered, outside
targeting, so that a refusal there leaves a line too. A tool that had to remember to record itself would be the one that
forgot.
"""

from __future__ import annotations

import logging
import time

from .protocol import ToolError, ToolRefusal


log = logging.getLogger(__name__)

# Long, and already implied by the call that read the document before it.
# Everything else a caller sends is short enough to be worth keeping.
NOT_LOGGED = ("expected_stamp",)

# Enough to tell two calls apart, not enough for one argument to take the line
# over. A label's text matters; the whole of a pasted paragraph does not.
MAX_VALUE_CHARS = 80


def traced(handler, name):
    """One tool's handler, with a line before the call and one after it."""

    def traced_call(arguments):
        started = time.monotonic()
        log.info("%s <- %s", name, _arguments(arguments))
        try:
            payload = handler(arguments)
        except ToolRefusal as exc:
            log.info(
                "%s refused %s%s", name, _status(exc.payload), _took(started)
            )
            raise
        except ToolError as exc:
            log.info("%s failed: %s%s", name, _short(exc), _took(started))
            raise
        except Exception:  # noqa: BLE001 - logged, then handled as before
            # The interesting one: a tool that raised something nobody planned
            # for is a bug in RAPID2 rather than an answer for the caller, so it
            # is the one case worth a stack trace rather than a sentence.
            log.exception("%s raised%s", name, _took(started))
            raise
        log.info("%s -> %s%s", name, _status(payload), _took(started))
        return payload

    return traced_call


def _arguments(arguments):
    if not arguments:
        return "-"
    said = " ".join(
        f"{key}={_short(arguments[key])}"
        for key in sorted(arguments)
        if key not in NOT_LOGGED
    )
    return said or "-"


def _status(payload):
    """What the tool answered, in a word.

    A refusal names which kind it was -- `stale`, `work_cancelled` -- and that
    word is the difference between "the assistant was told no" and "the
    assistant was told no for a reason nobody expected".
    """

    if isinstance(payload, dict):
        status = payload.get("status")
        if status:
            return _short(status)
    return "ok"


def _took(started):
    return f" in {(time.monotonic() - started) * 1000:.0f}ms"


def _short(value):
    """One argument, on one line, at a length a person can scan.

    The newlines matter as much as the length: a label carrying one would
    otherwise split a call across two lines and quietly break the property
    this file is read for, which is that one line is one event.
    """

    text = " ".join(str(value).split())
    if len(text) > MAX_VALUE_CHARS:
        return f"{text[:MAX_VALUE_CHARS]}...[{len(text)} chars]"
    return text
