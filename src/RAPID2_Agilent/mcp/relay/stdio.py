# -*- coding: utf-8 -*-

"""The stdio adapter: newline-delimited JSON on a pair of pipes.

The disposable half of the relay. It knows how a client hands over bytes and
nothing else -- no HTTP, no endpoint file, no MCP methods. If stdio is ever
retired this file goes and `upstream` stays exactly as it is.

Two lines below reach for `announce`, and they are not an exception to that:
what passes through here is handed over whole, and comes back either
unchanged or with a correction this file never inspects. The knowledge of
which messages those are stays in `announce`, which is where the relay keeps
everything it says on its own behalf. Delegating it is what stops the rule
above from being spent the first time something has to be said. The same goes
for `local`, the answers the relay can give while RAPID2 is closed: which
messages it takes is its own business, asked of it here and never decided.

Two rules govern everything here.

**stdout carries the protocol and nothing else.** Diagnostics go to stderr,
which clients read as log output.

**Requests run concurrently; the session's opening does not.** A client emits
several tool calls in one turn and expects them to overlap -- and it sends
`notifications/cancelled` for a call that is still running, which is useless if
it queues behind that call. So each message is dispatched to a worker and only
the writes are serialised. The exception is the start of the session: the first
request is answered before anything after it is dispatched. That is the MCP
lifecycle rule, expressed as a position rather than as a method name -- the
protocol requires the handshake to be the first request, so this adapter never
has to learn what the handshake is called.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from ..protocol import INTERNAL_ERROR, PARSE_ERROR, error_response
from . import announce
from .upstream import UpstreamError, UpstreamUnavailable


log = logging.getLogger(__name__)

# Enough for the several tool calls a client issues in one turn, plus the
# notifications that arrive alongside them. Beyond this, messages queue, which
# is the right backpressure: the work itself happens in RAPID2, and one window
# runs one tool call at a time whatever we do here.
DEFAULT_MAX_WORKERS = 8


def serve(
    upstream, *, stdin=None, stdout=None, max_workers=DEFAULT_MAX_WORKERS, local=None
):
    """Pump messages between a client's pipes and `upstream` until stdin ends.

    `local` is a `local.LocalAnswers`, or None to forward everything.
    """

    reader = sys.stdin.buffer if stdin is None else stdin
    writer = sys.stdout.buffer if stdout is None else stdout
    write_lock = threading.Lock()

    # Leaving the pool means waiting for work already accepted, so a reply is
    # never abandoned half-way because stdin happened to close.
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="rapid2-relay"
    ) as pool:
        session_open = False
        for line in reader:
            message = line.strip()
            if not message:
                continue
            try:
                payload = json.loads(message)
            except ValueError:
                _write(writer, error_response(None, PARSE_ERROR, "Invalid JSON."), write_lock)
                continue

            if session_open:
                pool.submit(_relay, upstream, payload, writer, write_lock, local)
                continue

            # Still in the handshake. Run it here, in arrival order.
            _relay(upstream, payload, writer, write_lock, local)
            # A notification cannot be the handshake, so it does not open the
            # session; a request has been answered by the time this runs.
            session_open = _is_request(payload)


def _relay(upstream, payload, writer, write_lock, local=None):
    """Carry one message across and write whatever comes back.

    Nothing may escape: a worker that raises would lose the reply, and a client
    waiting on that id waits for as long as the session lasts.
    """

    try:
        response = announce.speaking_for_the_relay(_answer(upstream, payload, local))
    except UpstreamError as exc:
        log.warning("Could not reach RAPID2: %s", exc)
        response = _failure(payload, str(exc))
    except Exception as exc:  # noqa: BLE001 - see the docstring
        log.exception("Relaying a message failed")
        response = _failure(payload, f"{type(exc).__name__}: {exc}")

    if response is not None:
        _write(writer, response, write_lock)
    # After the answer, never instead of it. A client that reads strictly in
    # order gets what it asked for first, and is then told to ask again --
    # which is the order that makes sense to it, because the reply it is
    # waiting on was produced by the window it already knew about.
    #
    # Sent even when this message failed or was a notification: the window
    # changed underneath the session either way, and the whole point is that
    # the client is not relying on us to have noticed at a convenient moment.
    for message in announce.notifications(upstream.take_replacements()):
        _write(writer, message, write_lock)


def _answer(upstream, payload, local):
    """The reply to one message: from a window if one is listening.

    `local` answers in two cases only. A call it always takes, which never
    goes near a window. And one of the few questions a client asks before it
    uses RAPID2, when nothing is listening -- a window that is listening still
    answers those itself. A window that was reached and refused is a refusal
    the client has to see, so only "nothing there" falls back.
    """

    if local is not None and local.answers_always(payload):
        return local.answer(payload)
    if local is not None and local.answers_while_closed(payload):
        try:
            return upstream.send(payload)
        except UpstreamUnavailable:
            return local.answer(payload)
    return upstream.send(payload)


def _is_request(payload):
    return isinstance(payload, dict) and payload.get("id") is not None


def _failure(payload, message):
    """Report an undeliverable message in the caller's own terms.

    Copying the `id` across is the only JSON-RPC this adapter knows, and it has
    to: a caller waiting on an id would otherwise wait forever. A message with
    no id is a notification, which gets no answer even when it fails.
    """

    if isinstance(payload, list):
        answers = [_failure(entry, message) for entry in payload]
        return [answer for answer in answers if answer is not None] or None
    if not _is_request(payload):
        return None
    return error_response(payload["id"], INTERNAL_ERROR, message)


def _write(writer, response, write_lock):
    encoded = json.dumps(response, default=str).encode("utf-8") + b"\n"
    # One message reaches the pipe whole, or the client sees torn JSON.
    with write_lock:
        writer.write(encoded)
        writer.flush()
