# -*- coding: utf-8 -*-

"""Loopback HTTP transport for the MCP endpoint. No Qt.

Streamable HTTP, minus the parts a tools-only server does not need: every POST
is answered with a single JSON body, and no server-initiated SSE stream is
offered, so GET is refused outright.
"""

from __future__ import annotations

import hmac
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .protocol import (
    HEADER_MISMATCH,
    HTTP_STATUS_FOR_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    META_PROTOCOL_VERSION,
    METHOD_HEADER,
    METHOD_NOT_FOUND,
    MODERN_PROTOCOL_VERSIONS,
    NAME_HEADER,
    PARSE_ERROR,
    PROTOCOL_VERSION_HEADER,
    decode_header_value,
    error_response,
    missing_request_metadata,
    protocol_version_is_supported,
    request_meta,
    unsupported_version_error,
)


log = logging.getLogger(__name__)

LOOPBACK_HOST = "127.0.0.1"
ENDPOINT_PATH = "/mcp"
# Sent by an HTTP client on every request after the handshake, naming the
# revision the handshake settled on. Checked here rather than left to the
# dispatcher, because a revision this build does not speak is a fact about the
# connection: the specification has it answered with 400 before the body is
# treated as a message at all.
# The header names and the sentinel codec live in `protocol`, beside the rest
# of the wire vocabulary, and are re-exported here because this is the module
# that enforces them.
#
# The methods that carry a name worth mirroring. Only the first is answered
# here; the other two are listed because the rule belongs to the method rather
# than to what this build happens to implement, and a `resources/read` added
# later should not have to remember to come back and add itself.
NAMED_METHODS = frozenset({"tools/call", "resources/read", "prompts/get"})
# One JSON-RPC message. Tool arguments are small; anything larger is a mistake
# or an attempt to exhaust memory.
MAX_BODY_BYTES = 1 << 20
# An over-long body is drained up to this much so the 413 is actually
# deliverable. Past it the connection is closed instead of read.
DISCARD_LIMIT_BYTES = 8 << 20

_ALLOWED_ORIGIN_SCHEMES = ("http", "https")
# Compared against the parsed host, never as a prefix of the whole Origin:
# "http://localhost.evil.example" starts with "http://localhost" but is a
# foreign site.
_ALLOWED_ORIGIN_HOSTS = frozenset(("127.0.0.1", "localhost", "::1"))


class _Access:
    """The credentials and the handler that one run of the server answers with.

    Kept in one mutable object, read per request, rather than baked into the
    handler class at start: a connection that is already open outlives the
    listening socket, so turning access off has to reach the threads serving
    those connections. Once revoked, every request on every connection --
    including one already in flight -- is refused.
    """

    def __init__(self, dispatch_payload, token):
        self._lock = threading.Lock()
        self._dispatch_payload = dispatch_payload
        self._token = token
        self._revoked = False

    @property
    def revoked(self):
        with self._lock:
            return self._revoked

    def token(self):
        """The bearer token to compare against, or None once revoked."""

        with self._lock:
            return None if self._revoked else self._token

    def dispatch(self):
        """The dispatch function, or None once revoked."""

        with self._lock:
            return None if self._revoked else self._dispatch_payload

    def revoke(self):
        with self._lock:
            self._revoked = True
            # Dropped, not just flagged: nothing that arrives later can reach
            # the document through a reference the handler class still holds.
            self._dispatch_payload = None
            self._token = None


def origin_is_allowed(origin):
    """Reject cross-origin browser callers (the DNS-rebinding shape)."""

    if origin is None or origin == "null":
        # A non-browser client sends no Origin at all; the bearer token is what
        # authorises it.
        return origin is None
    try:
        parts = urlsplit(origin)
        host = parts.hostname
        parts.port  # Read only so that a malformed port raises here.
    except ValueError:
        return False
    if parts.scheme.lower() not in _ALLOWED_ORIGIN_SCHEMES:
        return False
    if parts.path or parts.query or parts.fragment:
        # A well-formed Origin is scheme + host + optional port and nothing
        # else; anything past that is not one.
        return False
    if parts.username is not None or parts.password is not None:
        return False
    if host is None:
        return False
    # urlsplit lowercases the host and strips the brackets around a literal
    # IPv6 address, so "::1" is what an "[::1]" Origin arrives as here.
    return host in _ALLOWED_ORIGIN_HOSTS


def _is_modern(payload):
    """Whether one decoded body belongs to the era that has no handshake."""

    if not isinstance(payload, dict):
        return False
    params = payload.get("params")
    version = request_meta(params if isinstance(params, dict) else {}).get(
        META_PROTOCOL_VERSION
    )
    return version in MODERN_PROTOCOL_VERSIONS


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # A listening socket has no reason to advertise the Python build it is
    # running on, which the default banner does.
    server_version = "RAPID2"
    sys_version = ""

    # Injected by McpHttpServer.
    access = None

    def log_message(self, fmt, *args):
        log.debug("mcp http: %s", fmt % args)

    # -- responses --------------------------------------------------------
    def _send_json(self, status, payload):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status, *, close=False):
        self.send_response(status)
        self.send_header("Content-Length", "0")
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()

    def _discard_body(self, length):
        """Drain an over-long upload so the client can read the refusal.

        Answering before the request body has been read leaves the client
        writing into a closed socket, which surfaces as a broken pipe instead
        of the status we are trying to report.
        """

        remaining = min(length, DISCARD_LIMIT_BYTES)
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 64 * 1024))
            if not chunk:
                break
            remaining -= len(chunk)
        return length <= DISCARD_LIMIT_BYTES

    # -- guards -----------------------------------------------------------
    def _refusal_status(self):
        """The status to refuse this request with, or None to let it through.

        A disallowed Origin and a bad token are separate refusals. The
        specification has an Origin that is present and not allowed answered
        with 403: the caller is turned away as a site, whatever credential it
        carries, and no token would change the answer. 401 stays for the
        credential itself being missing or wrong.
        """

        if not origin_is_allowed(self.headers.get("Origin")):
            log.warning("MCP request rejected: disallowed Origin")
            return 403
        return None if self._token_matches() else 401

    def _token_matches(self):
        token = None if self.access is None else self.access.token()
        if not token:
            return False
        header = self.headers.get("Authorization") or ""
        if not header.startswith("Bearer "):
            return False
        presented = header[len("Bearer ") :].strip()
        return hmac.compare_digest(presented, token)

    def _header_mismatch(self, payload):
        """Why the headers and the body disagree, or None if they do not.

        The modern era mirrors part of the message into headers so that a load
        balancer can route on them without parsing the body. Nothing routes in
        front of this server -- it is a loopback socket answering one relay --
        but the check is not about routing: two sources of truth that are
        allowed to differ are a way to have one component act on a value
        another component never saw.

        So for a modern request these headers are required, exactly as the
        specification has them, and a missing one is refused as firmly as a
        wrong one. A legacy request is held to nothing here: the headers were
        introduced with the modern era, and a client older than they are
        cannot be at fault for not sending them.
        """

        if not isinstance(payload, dict):
            # A batch mirrors no single method or name into headers. The
            # modern era sends one message per POST, so a batch is legacy by
            # construction.
            return None
        params = payload.get("params")
        params = params if isinstance(params, dict) else {}
        modern = _is_modern(payload)

        declared = request_meta(params).get(META_PROTOCOL_VERSION)
        header_version = self.headers.get(PROTOCOL_VERSION_HEADER)
        if header_version is None:
            if modern:
                return f"{PROTOCOL_VERSION_HEADER} header is required."
        elif isinstance(declared, str) and header_version.strip() != declared:
            return (
                f"{PROTOCOL_VERSION_HEADER} header {header_version.strip()!r} "
                f"does not match the protocol version {declared!r} in the body."
            )

        method = payload.get("method")
        expected = {METHOD_HEADER: method}
        if method in NAMED_METHODS:
            expected[NAME_HEADER] = params.get("name") or params.get("uri")
        for header, value in expected.items():
            sent = self.headers.get(header)
            if sent is None:
                # Only required where there is something for it to mirror. A
                # `tools/call` carrying no name at all is malformed in a way
                # the dispatcher explains far better than a missing-header
                # complaint would.
                if modern and isinstance(value, str):
                    return f"{header} header is required."
                continue
            if isinstance(value, str) and decode_header_value(sent) != value:
                return (
                    f"{header} header {sent!r} does not match the "
                    f"corresponding body value {value!r}."
                )
        return None

    @staticmethod
    def _metadata_problem(payload):
        """Why a modern body's request metadata is malformed, or None.

        The dispatcher checks this too, and has to: it answers over transports
        that have no status codes. It is checked here as well because the
        status is the transport's to set -- the specification requires a 400 --
        and both readings come from the same function, so they cannot come to
        different conclusions.
        """

        if not _is_modern(payload):
            return None
        params = payload.get("params")
        return missing_request_metadata(params if isinstance(params, dict) else {})

    @staticmethod
    def _status_for(response, payload):
        """The HTTP status one JSON-RPC answer is delivered with.

        Almost always 200: a JSON-RPC error is a successful HTTP exchange
        carrying a failure. The exceptions are the handful the specification
        pins to a status, because a dual-era client reads the status first and
        the body only to decide whether to fall back to `initialize`.
        """

        if not isinstance(response, dict):
            return 200
        code = (response.get("error") or {}).get("code")
        status = HTTP_STATUS_FOR_ERROR.get(code)
        if status is not None:
            return status
        if code == METHOD_NOT_FOUND and _is_modern(payload):
            # Only for a modern request: a legacy client reads a 404 as the
            # endpoint being absent rather than the method.
            return 404
        return 200

    # -- verbs ------------------------------------------------------------
    def do_GET(self):
        self._send_empty(405)

    def do_DELETE(self):
        self._send_empty(405)

    def do_POST(self):
        # The body is consumed before any guard answers. A refusal that leaves
        # it unread desynchronises a keep-alive connection: the socket still
        # holds the JSON, and the next request line the server parses is that
        # JSON, which it rejects as bad syntax. Clients that probe without
        # credentials before authenticating hit this on their very first try.
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_empty(400, close=True)
            return
        if length > MAX_BODY_BYTES:
            drained = self._discard_body(length)
            self._send_empty(413, close=not drained)
            return
        raw = self.rfile.read(length) if length > 0 else b""

        if self.path.split("?")[0] != ENDPOINT_PATH:
            self._send_empty(404)
            return
        if self.access is None or self.access.revoked:
            # Access was turned off while this connection stayed open. Say so
            # and hang up, so the client stops reusing it.
            self._send_empty(503, close=True)
            return
        refusal = self._refusal_status()
        if refusal is not None:
            self._send_empty(refusal)
            return

        version = self.headers.get(PROTOCOL_VERSION_HEADER)
        if not protocol_version_is_supported(version):
            # Checked after the credentials, so that an unauthenticated caller
            # learns nothing about this server it could not already guess.
            log.warning("MCP request rejected: protocol version %r", version)
            # The specification's own error, rather than a generic bad
            # request: `data.supported` is the list a client is expected to
            # pick its next attempt from, and a modern client reads the code
            # to tell "this server is modern and we disagree about versions"
            # from "this server is legacy, fall back to initialize".
            self._send_json(400, unsupported_version_error(None, version))
            return

        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            self._send_json(400, error_response(None, PARSE_ERROR, "Invalid JSON."))
            return

        request_id = payload.get("id") if isinstance(payload, dict) else None
        mismatch = self._header_mismatch(payload)
        if mismatch is not None:
            log.warning("MCP request rejected: %s", mismatch)
            self._send_json(
                400, error_response(request_id, HEADER_MISMATCH, mismatch)
            )
            return
        problem = self._metadata_problem(payload)
        if problem is not None:
            log.warning("MCP request rejected: %s", problem)
            self._send_json(
                400, error_response(request_id, INVALID_PARAMS, problem)
            )
            return

        dispatch = self.access.dispatch()
        if dispatch is None:
            self._send_empty(503, close=True)
            return

        try:
            response = dispatch(payload)
        except Exception as exc:  # noqa: BLE001 - a handler must never 500 silently
            log.exception("MCP dispatch failed")
            self._send_json(
                500, error_response(None, INVALID_REQUEST, f"{type(exc).__name__}: {exc}")
            )
            return

        if response is None:
            # Notifications get no body.
            self._send_empty(202)
            return
        self._send_json(self._status_for(response, payload), response)


class _ThreadingServer(ThreadingHTTPServer):
    """A ThreadingHTTPServer that knows which connections are still open.

    ``shutdown()`` only stops the accept loop; the threads already serving
    keep-alive connections carry on. Tracking the sockets is what lets a stop
    break them instead of leaving a door open behind the closed one.
    """

    daemon_threads = True

    def __init__(self, address, handler_class):
        super().__init__(address, handler_class)
        self._connections = set()
        self._connections_lock = threading.Lock()

    def process_request(self, request, client_address):
        with self._connections_lock:
            self._connections.add(request)
        super().process_request(request, client_address)

    def shutdown_request(self, request):
        with self._connections_lock:
            self._connections.discard(request)
        super().shutdown_request(request)

    def close_open_connections(self):
        """Break every connection still open, waking its blocked reader."""

        with self._connections_lock:
            open_connections = list(self._connections)
            self._connections.clear()
        for connection in open_connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                # Already gone, or closing on its own thread right now.
                pass


class McpHttpServer:
    """Owns the listening socket and the thread that serves it."""

    def __init__(self, dispatch_payload, token, *, host=LOOPBACK_HOST):
        self._dispatch_payload = dispatch_payload
        self._token = token
        self._access = _Access(dispatch_payload, token)
        self._host = host
        self._server = None
        self._thread = None

    @property
    def running(self):
        return self._server is not None

    @property
    def host(self):
        return None if self._server is None else self._server.server_address[0]

    @property
    def port(self):
        return None if self._server is None else int(self._server.server_address[1])

    @property
    def url(self):
        if self._server is None:
            return None
        return f"http://{self.host}:{self.port}{ENDPOINT_PATH}"

    def _make_handler_class(self):
        return type("_BoundHandler", (_Handler,), {"access": self._access})

    def start(self):
        if self._server is not None:
            return self.port

        # A run gets credentials of its own. The previous run's were revoked
        # when it stopped and must stay that way, so starting again cannot
        # reuse them.
        self._access = _Access(self._dispatch_payload, self._token)

        # Always an ephemeral port. The endpoint file publishes whichever one
        # the operating system handed out and the stdio bridge re-reads it per
        # request, so a fixed port would buy no stability -- only a collision
        # with whatever else on the machine claimed it, and a second code path
        # for the window that lost the race.
        try:
            server = _ThreadingServer((self._host, 0), self._make_handler_class())
        except OSError as exc:
            raise OSError(
                "No loopback port was available for the MCP server."
            ) from exc

        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="rapid2-mcp-http",
            daemon=True,
        )
        self._thread.start()
        log.info("MCP server listening on %s", self.url)
        return self.port

    def revoke(self):
        """Stop answering, on every connection, without touching the sockets.

        A stop does this first. It is separate because the sockets cannot be
        the whole answer: a request already on the wire has to be refused by
        the handler, and it can only do that by re-reading the credentials.
        """

        self._access.revoke()

    def stop(self, *, timeout=5.0):
        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        # First, before the accept loop even winds down: a request that is
        # mid-flight on an open connection must not be answered from here on.
        self.revoke()
        if server is None:
            return True
        try:
            server.shutdown()
        except Exception:  # noqa: BLE001 - shutting down must not raise
            log.exception("MCP server shutdown failed")
        finally:
            server.server_close()
            # The handler threads are never joined: stop() runs on the main
            # thread, and a handler waiting for that same thread to run its
            # tool call would deadlock the join. Breaking the sockets is what
            # ends them.
            server.close_open_connections()
        if thread is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True
