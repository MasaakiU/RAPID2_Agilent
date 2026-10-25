# -*- coding: utf-8 -*-

"""The link to the running windows. No stdio, no Qt.

Everything peculiar to reaching RAPID2 lives here: finding the registry, choosing
which instance a message is for, presenting that instance's bearer token, and
surviving a window restarting underneath a client that has stayed connected the
whole time.

Choosing among several addresses is transport work -- it is what a proxy does --
and it is deliberately kept on this side of the relay rather than in `stdio`,
because it is the half that survives stdio being retired. The price is one
field of MCP vocabulary: `params.arguments.instance`, the tool argument by which
a caller names its target. Nothing else about the message is read, and the
argument is forwarded untouched for the instance itself to validate.
"""

from __future__ import annotations

import http.client
import json
import logging
import threading
import time
import urllib.error
import urllib.request

from .. import paths
from .. import registry
from ..protocol import (
    META_PROTOCOL_VERSION,
    METHOD_HEADER,
    NAME_HEADER,
    PROTOCOL_VERSION_HEADER,
    encode_header_value,
)


log = logging.getLogger(__name__)

# A viewer tool gives up on the Qt main thread by itself (see `bridge`) and
# reports that as a normal reply, but an analysis tool computes for as long as
# its data needs: aligning two full runs takes minutes. This ceiling only has
# to outlast those, so that a slow answer still arrives as an answer rather
# than as a dead connection. A client may stop waiting sooner on its own.
DEFAULT_TIMEOUT_SECONDS = 900.0

# A loopback TCP handshake normally completes immediately. Windows can leave a
# just-stopped listening port to time out instead of refusing it; using the
# response timeout for that dead address would stall routing for 30 seconds.
# Once connected, the ordinary response timeout is restored because a tool may
# legitimately spend several seconds on the Qt thread.
LOOPBACK_CONNECT_TIMEOUT_SECONDS = 1.0

# How long the first message may wait for RAPID2 to appear. A client starts its
# servers when the client starts, which can be moments before RAPID2 finishes
# opening; failing that first message outright makes the client give up on the
# session, and the user then has to reconnect by hand.
#
# The ceiling is not ours to choose: some clients allow a server about 10s to
# come up and kill it after that. So this covers a race with an application
# that is already starting --
# not an application nobody has opened yet -- and has to leave room for the
# handshake itself inside that budget.
#
# A relay with a catalog to answer from (see `local`) has no use for it: the
# handshake no longer depends on RAPID2, so it is answered at once.
FIRST_CONTACT_WAIT_SECONDS = 6.0
FIRST_CONTACT_POLL_SECONDS = 0.25

NOT_RUNNING = (
    "RAPID2 is not reachable. Start it with rapid2_launch, or ask the user to "
    "start RAPID2 and turn on AI assistant access (Tools > AI Assistant Access)."
)
# Asked of a window to learn whether it is there. Answered without touching
# the window, and by every revision this relay speaks.
_PROBE = {"jsonrpc": "2.0", "id": "rapid2-relay-probe", "method": "ping"}


def _mirrored_headers(payload):
    """The headers the modern era copies out of the message body.

    A relay is a client as far as the window is concerned, and a conforming
    modern client mirrors these so that anything sitting between the two can
    route on them without parsing JSON. Nothing sits between these two -- one
    loopback socket, no proxy -- but sending them is what makes the pair a
    worked example of the contract rather than an exception to it, and the
    window validates them against the body it receives.

    Nothing is mirrored for a legacy message or a batch: the headers belong to
    a single modern request, and a value invented for anything else would be a
    mismatch by construction.
    """

    if not isinstance(payload, dict):
        return {}
    params = payload.get("params")
    params = params if isinstance(params, dict) else {}
    meta = params.get("_meta")
    version = (meta or {}).get(META_PROTOCOL_VERSION) if isinstance(meta, dict) else None
    if not isinstance(version, str):
        return {}

    headers = {PROTOCOL_VERSION_HEADER: version}
    method = payload.get("method")
    if isinstance(method, str):
        headers[METHOD_HEADER] = method
    name = params.get("name") or params.get("uri")
    if isinstance(name, str):
        headers[NAME_HEADER] = encode_header_value(name)
    return headers


class UpstreamError(Exception):
    """The message could not be delivered, or its answer could not be used."""


class UpstreamUnavailable(UpstreamError):
    """Nothing is listening: RAPID2 is closed, or access is switched off."""


class UpstreamRejected(UpstreamError):
    """RAPID2 was reached but would not, or could not, answer."""


class _Unauthorised(Exception):
    """Internal: the token was refused. Worth one retry, never raised out."""


class _Undelivered(Exception):
    """Internal: the connection was refused, so nothing was received.

    Kept apart from every other failure on purpose. Moving to another window
    is only safe when this window provably never saw the message: a refused
    TCP handshake accepted no bytes, whereas a timeout may mean the call is
    being carried out right now. Retrying a change somewhere else on a
    timeout would change a window nobody asked about.
    """


class _ConnectTimeout(TimeoutError):
    """Internal: the TCP handshake timed out before a request was sent."""


def _requested_instance(payload):
    """The `instance` tool argument, if the caller named one.

    The single piece of MCP vocabulary in the relay. Read, never removed: the
    instance validates it too, and two readings of one field are how a
    misrouted call is caught rather than silently obeyed.
    """

    if not isinstance(payload, dict):
        return None
    params = payload.get("params")
    if not isinstance(params, dict):
        return None
    arguments = params.get("arguments")
    if not isinstance(arguments, dict):
        return None
    wanted = arguments.get("instance")
    if wanted is None:
        return None
    wanted = str(wanted).strip()
    return wanted or None


def _roster(instances):
    return ", ".join(
        f"{document.get('label') or '?'} ({document.get('instance_id', '?')})"
        for document in instances
    )


class _RejectRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse a redirect rather than follow it.

    The only address ever opened here is RAPID2's, and RAPID2 does not redirect.
    Anything that does is not RAPID2 -- another process holding the port after
    RAPID2 died -- and urllib keeps the Authorization header across a redirect,
    so following one would hand the bearer token to wherever it points.
    """

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


class _LoopbackHTTPConnection(http.client.HTTPConnection):
    """Distinguish a safe-to-retry connect timeout from a reply timeout."""

    def connect(self):
        response_timeout = self.timeout
        if isinstance(response_timeout, (int, float)):
            self.timeout = min(
                response_timeout,
                LOOPBACK_CONNECT_TIMEOUT_SECONDS,
            )
        else:
            self.timeout = LOOPBACK_CONNECT_TIMEOUT_SECONDS
        try:
            super().connect()
        except TimeoutError as exc:
            raise _ConnectTimeout(str(exc)) from exc
        finally:
            self.timeout = response_timeout

        # HTTPConnection uses the same timeout for connecting and reading.
        # Restore the caller's longer response timeout after the safe retry
        # boundary has passed.
        if self.sock is not None:
            socket_timeout = (
                response_timeout
                if response_timeout is None
                or isinstance(response_timeout, (int, float))
                else None
            )
            self.sock.settimeout(socket_timeout)


class _LoopbackHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, request):
        return self.do_open(_LoopbackHTTPConnection, request)


def loopback_urlopen():
    """An `urlopen` that reaches this machine and nowhere else.

    The default opener consults the environment and the operating system for a
    proxy, and neither treats 127.0.0.1 as special: with a proxy configured and
    no exception covering the loopback address -- `no_proxy=localhost` does not
    cover it, the two are matched as strings, and macOS excludes only `*.local`
    and `169.254/16` by default -- urllib would send this POST to the proxy
    instead, bearer token and document contents included, in the clear. The
    relay is started by the client and inherits whatever the user's environment
    says, so it carries its own client with proxies switched off.
    """

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _RejectRedirect,
        _LoopbackHTTPHandler,
    )
    return opener.open


class HttpUpstream:
    """Send one JSON-RPC message to the window and return its reply.

    `send` returns the decoded reply, or None when there is nothing to return
    -- a notification, which RAPID2 acknowledges with an empty 202.
    """

    def __init__(
        self,
        *,
        registry_root=None,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        first_contact_wait=FIRST_CONTACT_WAIT_SECONDS,
        poll_interval=FIRST_CONTACT_POLL_SECONDS,
        urlopen=None,
    ):
        self._registry_root = registry_root
        self._timeout = timeout
        self._first_contact_wait = first_contact_wait
        self._poll_interval = poll_interval
        self._urlopen = urlopen or loopback_urlopen()
        self._instances = {}
        self._replacements = []
        self._replacement_lock = threading.Lock()
        self._reached = False

    def send(self, payload):
        if self._reached:
            return self._deliver(payload)
        return self._deliver_first(payload)

    def reachable(self):
        """Whether a window takes requests right now. Waits for nothing."""

        try:
            self._deliver(_PROBE)
        except UpstreamError:
            return False
        return True

    def _deliver_first(self, payload):
        """Give RAPID2 a moment to appear, once.

        Only the first message waits. By then the client is established, and a
        later message is better answered with "RAPID2 is closed" than with a
        stall the model has to sit through.
        """

        deadline = time.monotonic() + self._first_contact_wait
        while True:
            try:
                response = self._deliver(payload)
            except UpstreamUnavailable:
                if time.monotonic() >= deadline:
                    raise
                log.info("Waiting for RAPID2 to become reachable.")
                time.sleep(self._poll_interval)
                continue
            except UpstreamError:
                # RAPID2 answered, even if the answer was a refusal. There is
                # nothing left to wait for.
                self._reached = True
                raise
            self._reached = True
            return response

    def _deliver(self, payload):
        candidates = self._resolve(payload)
        for document in candidates:
            try:
                return self._attempt(document, payload)
            except _Undelivered:
                # This window is gone but its address is still on file -- it
                # crashed, and the sweep that clears it runs when a window
                # starts. Nothing was received here, so the next one is safe
                # to try. Only ever more than one candidate for an untargeted
                # message: a message that named its window has exactly one.
                log.info(
                    "Window %s is on file but not listening.",
                    document.get("label"),
                )
                continue
        raise UpstreamUnavailable(NOT_RUNNING)

    def _attempt(self, document, payload):
        try:
            return self._post(document, payload)
        except _Unauthorised:
            # The window was restarted between reading its address and using
            # it: same relay process, same registration, new token. Re-reading
            # the registry is the whole recovery, so take it before giving up.
            log.info("Access token refused; re-reading the registry.")
            fresh = self._reread(document)
            if fresh is None:
                raise UpstreamRejected("RAPID2 refused the access token.")
            try:
                return self._post(fresh, payload)
            except _Unauthorised as exc:
                raise UpstreamRejected("RAPID2 refused the access token.") from exc

    def _reread(self, document):
        """The same window's current address, by letter.

        By letter rather than by instance_id: the point of re-reading is that
        the process behind the letter changed, so its id has changed with it.
        """

        label = document.get("label")
        for candidate in registry.read_instances(self._registry_root):
            if label is not None and candidate.get("label") == label:
                return candidate
        return None

    # -- addressing -------------------------------------------------------
    def _resolve(self, payload):
        """Read the registry and rank the windows this message may go to.

        Read every time, never cached: a token and a port are new on every
        launch of RAPID2, and this process outlives any number of them. A handful
        of small files is cheaper than being wrong.

        A message that named its window gets exactly one candidate. Falling
        back from a named window to a different one would be the worst
        available answer: the caller asked for that window, not any window.
        """

        try:
            instances = registry.read_instances(self._registry_root)
        except paths.DataPathError as exc:
            raise UpstreamUnavailable(str(exc)) from exc
        if not instances:
            raise UpstreamUnavailable(NOT_RUNNING)

        self._note_replacements(instances)

        wanted = _requested_instance(payload)
        if wanted is None:
            # Untargeted. Any listening window is a valid door: with several
            # open, the one that takes the message answers with the roster
            # rather than guessing, and that answer belongs to the side that
            # owns the tools.
            return instances
        document = registry.match_instance(instances, wanted)
        if document is None:
            raise UpstreamRejected(
                f"No RAPID2 window answers to {wanted!r}. "
                f"Open windows: {_roster(instances)}."
            )
        return [document]

    def _note_replacements(self, instances):
        """Record when a letter changed hands, for the log and for the client.

        A replacement is the one event in a session that invalidates something
        the client is holding: it took its tool list from a handshake with the
        process that has just gone. Logging it was never enough -- a client
        does not read the log -- so it is kept here as well, for whoever is
        driving this upstream to collect and pass on.

        The first sighting of a letter is not a replacement. That is the run
        the client handshook against.
        """

        for document in instances:
            label = document.get("label")
            instance_id = document.get("instance_id")
            # Comparing and recording under one lock, because several workers
            # resolve at once: read the previous id outside it and two of them
            # can both find it stale, and the client is asked to re-read its
            # tools twice for one restart.
            with self._replacement_lock:
                previous = self._instances.get(label)
                replaced = previous is not None and previous != instance_id
                self._instances[label] = instance_id
                if replaced:
                    self._replacements.append(label)
            if replaced:
                log.info(
                    "Window %s was replaced (%s -> %s, version %s).",
                    label,
                    previous,
                    instance_id,
                    document.get("app_version", "unknown"),
                )

    def take_replacements(self):
        """The windows replaced since this was last asked, and forget them.

        Drained rather than read, because every one of these is a thing to be
        said once. Several workers can be inside `send` at the same time, so
        the list and its emptying move together -- two of them reporting the
        same replacement would have the client re-read its tools twice for one
        event.
        """

        with self._replacement_lock:
            found, self._replacements = self._replacements, []
        return found

    # -- delivery ---------------------------------------------------------
    def _post(self, document, payload):
        url, token = document["url"], document["token"]
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, default=str).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                **_mirrored_headers(payload),
            },
        )
        try:
            with self._urlopen(request, timeout=self._timeout) as response:
                status = getattr(response, "status", 200)
                raw = response.read()
        except urllib.error.HTTPError as exc:
            # HTTPError is also the response object.  A refused redirect and
            # the status replies below otherwise keep their socket until GC.
            exc.close()
            if exc.code == 401:
                raise _Unauthorised() from exc
            if exc.code == 503:
                # AI access was turned off in that window while its address was
                # still on file. Refused before anything ran, so the next
                # window is safe to try -- and if there is none, the caller is
                # told to turn access back on.
                raise _Undelivered("AI assistant access is off there.") from exc
            raise UpstreamRejected(f"RAPID2 answered HTTP {exc.code}.") from exc
        except urllib.error.URLError as exc:
            if isinstance(
                exc.reason,
                (ConnectionRefusedError, _ConnectTimeout),
            ):
                raise _Undelivered(str(exc.reason)) from exc
            raise UpstreamUnavailable(NOT_RUNNING) from exc
        except (ConnectionRefusedError, _ConnectTimeout) as exc:
            raise _Undelivered(str(exc)) from exc
        except OSError as exc:
            # Once connected, a socket timeout is not provably undelivered,
            # so it must not fall through to another window.
            raise UpstreamUnavailable(f"RAPID2 did not answer: {exc}") from exc

        if status == 202 or not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise UpstreamRejected("RAPID2 sent a reply that was not JSON.") from exc
