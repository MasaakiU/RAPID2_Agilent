# -*- coding: utf-8 -*-

import http.client
import json
import socket
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from RAPID2_Agilent.mcp.protocol import (
    HEADER_MISMATCH,
    LATEST_PROTOCOL_VERSION,
    META_CLIENT_CAPABILITIES,
    META_PROTOCOL_VERSION,
    METHOD_HEADER,
    METHOD_NOT_FOUND,
    MODERN_PROTOCOL_VERSIONS,
    NAME_HEADER,
    UNSUPPORTED_PROTOCOL_VERSION,
    Dispatcher,
    Tool,
    ToolRegistry,
)
from RAPID2_Agilent.mcp.transport import (
    ENDPOINT_PATH,
    MAX_BODY_BYTES,
    PROTOCOL_VERSION_HEADER,
    McpHttpServer,
    origin_is_allowed,
)


TOKEN = "test-token"


def build_dispatcher():
    registry = ToolRegistry()
    registry.add(
        Tool(
            name="probe",
            title="Probe",
            description="Test tool.",
            input_schema={"type": "object", "properties": {}},
            handler=lambda arguments: {"echo": arguments},
        )
    )
    return Dispatcher(registry, server_name="rapid2", server_version="0.0.0")


def post(
    url,
    body,
    *,
    token=TOKEN,
    origin=None,
    raw=None,
    protocol_version=None,
    headers=None,
):
    data = raw if raw is not None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json, text/event-stream")
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    if origin is not None:
        request.add_header("Origin", origin)
    if protocol_version is not None:
        request.add_header(PROTOCOL_VERSION_HEADER, protocol_version)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = response.read()
            return response.status, (json.loads(payload) if payload else None)
    except urllib.error.HTTPError as error:
        payload = error.read()
        return error.code, (json.loads(payload) if payload else None)


class OriginPolicyTests(unittest.TestCase):
    def test_absent_origin_is_allowed(self):
        # A non-browser client sends none; the bearer token authorises it.
        self.assertTrue(origin_is_allowed(None))

    def test_loopback_origins_are_allowed(self):
        for origin in ("http://127.0.0.1:9", "http://localhost:9", "http://[::1]:9"):
            self.assertTrue(origin_is_allowed(origin), origin)

    def test_loopback_origins_over_https_are_allowed(self):
        for origin in ("https://127.0.0.1:9", "https://localhost"):
            self.assertTrue(origin_is_allowed(origin), origin)

    def test_foreign_origins_are_refused(self):
        for origin in (
            "http://evil.example.com",
            "https://localhost.evil.example.com",
            "null",
        ):
            self.assertFalse(origin_is_allowed(origin), origin)

    def test_origins_that_merely_look_loopback_are_refused(self):
        # The host is compared, not the leading text of the Origin.
        for origin in (
            "http://localhost.evil.example",
            "http://127.0.0.1.evil.example",
            "http://[::1].evil.example",
            "http://127.0.0.10",
            "http://localhost@evil.example",
            "http://evil.example/http://localhost",
            "file://localhost",
            "http://localhost:notaport",
            "not a url",
        ):
            self.assertFalse(origin_is_allowed(origin), origin)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.server = McpHttpServer(build_dispatcher().dispatch_payload, TOKEN)
        self.server.start()
        self.url = self.server.url

    def tearDown(self):
        self.server.stop()

    # -- binding ---------------------------------------------------------
    def test_binds_to_loopback_only(self):
        self.assertEqual(self.server.host, "127.0.0.1")

    def test_not_reachable_from_a_non_loopback_address(self):
        probe = socket.socket()
        self.addCleanup(probe.close)
        probe.settimeout(2)
        try:
            hostname_ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            self.skipTest("No non-loopback address is available.")
        if hostname_ip.startswith("127."):
            self.skipTest("Host resolves to loopback.")
        with self.assertRaises(OSError):
            probe.connect((hostname_ip, self.server.port))

    # -- authentication --------------------------------------------------
    def test_request_without_a_token_is_refused(self):
        status, _ = post(self.url, {"jsonrpc": "2.0", "id": 1, "method": "ping"}, token=None)
        self.assertEqual(status, 401)

    def test_request_with_a_wrong_token_is_refused(self):
        status, _ = post(
            self.url, {"jsonrpc": "2.0", "id": 1, "method": "ping"}, token="wrong"
        )
        self.assertEqual(status, 401)

    def test_foreign_origin_is_refused_even_with_a_valid_token(self):
        # 403, not 401: the specification separates a caller refused as a site
        # from one refused over its credential.
        status, _ = post(
            self.url,
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            origin="http://evil.example.com",
        )
        self.assertEqual(status, 403)

    def test_foreign_origin_is_refused_before_the_token_is_looked_at(self):
        status, _ = post(
            self.url,
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            token=None,
            origin="http://evil.example.com",
        )
        self.assertEqual(status, 403)

    def test_a_loopback_origin_still_needs_a_token(self):
        status, _ = post(
            self.url,
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            token="wrong",
            origin="http://127.0.0.1:9",
        )
        self.assertEqual(status, 401)

    # -- routing ---------------------------------------------------------
    def test_valid_request_is_answered(self):
        status, body = post(self.url, {"jsonrpc": "2.0", "id": 1, "method": "ping"})
        self.assertEqual(status, 200)
        self.assertEqual(body["result"], {})

    def test_notification_is_accepted_without_a_body(self):
        status, body = post(
            self.url, {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        self.assertEqual(status, 202)
        self.assertIsNone(body)

    def test_get_is_refused_because_no_sse_stream_is_offered(self):
        request = urllib.request.Request(self.url, method="GET")
        request.add_header("Authorization", f"Bearer {TOKEN}")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 405)

    def test_unknown_path_is_not_found(self):
        base = self.url.rsplit("/", 1)[0]
        status, _ = post(f"{base}/elsewhere", {"jsonrpc": "2.0", "id": 1, "method": "ping"})
        self.assertEqual(status, 404)

    def test_invalid_json_is_reported_not_fatal(self):
        status, body = post(self.url, None, raw=b"{not json")
        self.assertEqual(status, 400)
        self.assertIsNotNone(body)
        # The server stays up for the next request.
        self.assertEqual(post(self.url, {"jsonrpc": "2.0", "id": 2, "method": "ping"})[0], 200)

    def test_oversized_body_is_refused(self):
        status, _ = post(self.url, None, raw=b"x" * (MAX_BODY_BYTES + 1))
        self.assertEqual(status, 413)

    # -- keep-alive integrity --------------------------------------------
    def keep_alive_request(self, connection, message, *, token=TOKEN):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        connection.request("POST", "/mcp", json.dumps(message), headers)
        response = connection.getresponse()
        body = response.read()
        return response.status, (json.loads(body) if body else None)

    def test_connection_survives_a_refused_request(self):
        # A guard that answers without consuming the body leaves the JSON in
        # the socket, and the server then parses it as the next request line.
        # Claude Code probes without credentials before authenticating, so it
        # meets this on its first connection.
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.port, timeout=5
        )
        self.addCleanup(connection.close)

        status, _ = self.keep_alive_request(
            connection, {"jsonrpc": "2.0", "id": 1, "method": "ping"}, token=None
        )
        self.assertEqual(status, 401)

        status, body = self.keep_alive_request(
            connection, {"jsonrpc": "2.0", "id": 2, "method": "ping"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["id"], 2)

    def test_connection_survives_a_wrong_path(self):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.port, timeout=5
        )
        self.addCleanup(connection.close)
        connection.request(
            "POST",
            "/elsewhere",
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TOKEN}",
            },
        )
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 404)

        status, body = self.keep_alive_request(
            connection, {"jsonrpc": "2.0", "id": 2, "method": "ping"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["id"], 2)

    def test_connection_survives_a_foreign_origin(self):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.port, timeout=5
        )
        self.addCleanup(connection.close)
        connection.request(
            "POST",
            "/mcp",
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TOKEN}",
                "Origin": "http://evil.example.com",
            },
        )
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 403)

        status, body = self.keep_alive_request(
            connection, {"jsonrpc": "2.0", "id": 2, "method": "ping"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["id"], 2)

    def test_full_handshake_sequence_on_one_connection(self):
        # The exact order a real client uses, unauthenticated probe included.
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.port, timeout=5
        )
        self.addCleanup(connection.close)

        probe = {
            "jsonrpc": "2.0",
            "id": "server-discover-probe-1",
            "method": "server/discover",
            "params": {"_meta": {}},
        }
        self.assertEqual(self.keep_alive_request(connection, probe, token=None)[0], 401)
        self.assertEqual(self.keep_alive_request(connection, probe)[0], 200)

        status, body = self.keep_alive_request(
            connection,
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {"protocolVersion": "2025-11-25"},
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["result"]["protocolVersion"], "2025-11-25")

        self.assertEqual(
            self.keep_alive_request(
                connection, {"jsonrpc": "2.0", "method": "notifications/initialized"}
            )[0],
            202,
        )

        status, body = self.keep_alive_request(
            connection, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["result"]["tools"][0]["name"], "probe")

    def test_a_supported_protocol_version_header_is_accepted(self):
        status, body = post(
            self.url,
            {"jsonrpc": "2.0", "id": 4, "method": "ping"},
            protocol_version=LATEST_PROTOCOL_VERSION,
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["id"], 4)

    def test_no_protocol_version_header_is_accepted(self):
        # stdio clients never send one, and neither does the relay that carries
        # them here; the revision predating the header is one we speak.
        status, _body = post(self.url, {"jsonrpc": "2.0", "id": 5, "method": "ping"})
        self.assertEqual(status, 200)

    def test_an_unsupported_protocol_version_header_is_refused(self):
        status, body = post(
            self.url,
            {"jsonrpc": "2.0", "id": 6, "method": "ping"},
            protocol_version="2099-01-01",
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], UNSUPPORTED_PROTOCOL_VERSION)
        self.assertEqual(body["error"]["data"]["requested"], "2099-01-01")
        self.assertIn(LATEST_PROTOCOL_VERSION, body["error"]["data"]["supported"])

    def test_an_unsupported_version_is_refused_after_the_token_not_before(self):
        # An unauthenticated caller must not learn which revisions we speak.
        status, body = post(
            self.url,
            {"jsonrpc": "2.0", "id": 7, "method": "ping"},
            token="wrong",
            protocol_version="2099-01-01",
        )
        self.assertEqual(status, 401)
        self.assertIsNone(body)

    def test_tool_call_round_trip(self):
        status, body = post(
            self.url,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "probe", "arguments": {"a": 1}},
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["result"]["structuredContent"], {"echo": {"a": 1}})


class RevocationTests(unittest.TestCase):
    """Turning access off has to reach connections that are already open."""

    def setUp(self):
        self.server = McpHttpServer(build_dispatcher().dispatch_payload, TOKEN)
        self.server.start()
        self.connection = http.client.HTTPConnection(
            self.server.host, self.server.port, timeout=5
        )
        self.addCleanup(self.connection.close)
        self.addCleanup(self.server.stop)

    def call(self):
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "probe", "arguments": {}},
            }
        )
        self.connection.request(
            "POST",
            ENDPOINT_PATH,
            body,
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TOKEN}",
            },
        )
        response = self.connection.getresponse()
        response.read()
        return response.status

    def test_the_connection_is_reusable_while_the_server_runs(self):
        self.assertEqual(self.call(), 200)
        self.assertEqual(self.call(), 200)

    def test_an_open_connection_is_broken_by_stop(self):
        # The bug this guards: shutdown() only closes the listening socket, so
        # a keep-alive connection went on being served with the old token and
        # the old dispatcher.
        self.assertEqual(self.call(), 200)
        self.server.stop()

        with self.assertRaises(
            (
                http.client.HTTPException,
                ConnectionError,
                OSError,
            )
        ):
            self.call()

    def test_a_request_already_on_the_wire_is_refused_after_revocation(self):
        # The socket teardown cannot catch a request that is mid-flight, so the
        # handler re-reads the credentials per request instead of holding them.
        self.assertEqual(self.call(), 200)
        self.server.revoke()

        self.assertEqual(self.call(), 503)

    def test_a_fresh_connection_is_refused_once_revoked(self):
        self.server.revoke()

        fresh = http.client.HTTPConnection(
            self.server.host, self.server.port, timeout=5
        )
        self.addCleanup(fresh.close)
        fresh.request(
            "POST",
            ENDPOINT_PATH,
            "{}",
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TOKEN}",
            },
        )
        response = fresh.getresponse()
        response.read()

        self.assertEqual(response.status, 503)


class LifecycleTests(unittest.TestCase):
    def test_stop_releases_the_port(self):
        server = McpHttpServer(build_dispatcher().dispatch_payload, TOKEN)
        port = server.start()
        self.assertTrue(server.running)
        self.assertTrue(server.stop())
        self.assertFalse(server.running)

        # The port is reusable, which proves the socket really closed.
        probe = socket.socket()
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
        probe.close()

    def test_a_restarted_server_answers_again(self):
        # The stop revokes the run's credentials; the restart must deal a new
        # set rather than inherit the revoked ones.
        server = McpHttpServer(build_dispatcher().dispatch_payload, TOKEN)
        server.start()
        server.stop()
        server.start()
        self.addCleanup(server.stop)

        status, body = post(
            server.url, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )

        self.assertEqual(status, 200)
        self.assertEqual(len(body["result"]["tools"]), 1)

    def test_stop_is_idempotent(self):
        server = McpHttpServer(build_dispatcher().dispatch_payload, TOKEN)
        server.start()
        self.assertTrue(server.stop())
        self.assertTrue(server.stop())

    def test_each_server_gets_its_own_port(self):
        # Two RAPID2 windows must both get a server, on ports of their own.
        first = McpHttpServer(build_dispatcher().dispatch_payload, TOKEN)
        second = McpHttpServer(build_dispatcher().dispatch_payload, TOKEN)
        try:
            first_port = first.start()
            second_port = second.start()
            self.assertTrue(first.running)
            self.assertTrue(second.running)
            self.assertNotEqual(first_port, second_port)
        finally:
            first.stop()
            second.stop()


MODERN = MODERN_PROTOCOL_VERSIONS[0]


def modern_body(method, params=None, *, request_id=1):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {
            **(params or {}),
            "_meta": {
                META_PROTOCOL_VERSION: MODERN,
                META_CLIENT_CAPABILITIES: {},
            },
        },
    }


class ModernHttpTests(unittest.TestCase):
    """The statuses a dual-era client reads before it reads the body."""

    def setUp(self):
        self.server = McpHttpServer(build_dispatcher().dispatch_payload, TOKEN)
        self.server.start()
        self.url = self.server.url
        self.addCleanup(self.server.stop)

    def conforming(self, body, **overrides):
        """A POST carrying the headers a modern client mirrors from the body."""

        params = body.get("params") or {}
        headers = {METHOD_HEADER: body["method"]}
        name = params.get("name")
        if isinstance(name, str):
            headers[NAME_HEADER] = name
        headers.update(overrides)
        return post(
            self.url, body, protocol_version=MODERN, headers=headers
        )

    def test_a_conforming_modern_call_goes_through(self):
        status, body = self.conforming(
            modern_body("tools/call", {"name": "probe", "arguments": {}})
        )

        self.assertEqual(status, 200)
        self.assertEqual(body["result"]["resultType"], "complete")

    def test_a_version_header_disagreeing_with_the_body_is_refused(self):
        # Two sources of truth allowed to differ are how one component acts on
        # a value another component never saw.
        status, body = post(
            self.url,
            modern_body("ping"),
            protocol_version=LATEST_PROTOCOL_VERSION,
            headers={METHOD_HEADER: "tools/list"},
        )

        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], HEADER_MISMATCH)

    def test_a_name_header_disagreeing_with_the_body_is_refused(self):
        status, body = self.conforming(
            modern_body("tools/call", {"name": "probe", "arguments": {}}),
            **{NAME_HEADER: "something_else"},
        )

        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], HEADER_MISMATCH)

    def test_a_base64_wrapped_name_header_is_decoded_before_comparing(self):
        status, _body = self.conforming(
            modern_body("tools/call", {"name": "probe", "arguments": {}}),
            **{NAME_HEADER: "=?base64?cHJvYmU=?="},
        )

        self.assertEqual(status, 200)

    def test_a_missing_method_header_is_refused(self):
        # Required of a modern request, and a missing one is refused as firmly
        # as a wrong one: a header nobody has to send is a header a router
        # cannot rely on.
        status, body = post(self.url, modern_body("ping"), protocol_version=MODERN)

        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], HEADER_MISMATCH)

    def test_a_missing_name_header_is_refused(self):
        status, body = post(
            self.url,
            modern_body("tools/call", {"name": "probe", "arguments": {}}),
            protocol_version=MODERN,
            headers={METHOD_HEADER: "tools/call"},
        )

        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], HEADER_MISMATCH)

    def test_a_missing_version_header_is_refused(self):
        status, body = post(
            self.url, modern_body("ping"), headers={METHOD_HEADER: "ping"}
        )

        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], HEADER_MISMATCH)

    def test_a_legacy_request_is_held_to_none_of_them(self):
        # The headers arrived with the modern era. A client older than they
        # are cannot be at fault for not sending them.
        status, _body = post(self.url, {"jsonrpc": "2.0", "id": 1, "method": "ping"})

        self.assertEqual(status, 200)

    def test_a_modern_body_missing_its_metadata_is_a_400(self):
        # The dispatcher rejects it too, but the status is the transport's to
        # set, and the specification requires this one.
        body = modern_body("ping")
        del body["params"]["_meta"][META_CLIENT_CAPABILITIES]

        status, answer = post(
            self.url,
            body,
            protocol_version=MODERN,
            headers={METHOD_HEADER: "ping"},
        )

        self.assertEqual(status, 400)
        self.assertEqual(answer["error"]["code"], -32602)
        self.assertEqual(answer["id"], 1)

    def test_caching_hints_ride_on_the_results_that_must_carry_them(self):
        for method, params in (("server/discover", None), ("tools/list", None)):
            body = modern_body(method, params)
            headers = {METHOD_HEADER: method}
            status, answer = post(
                self.url, body, protocol_version=MODERN, headers=headers
            )

            self.assertEqual(status, 200, method)
            self.assertGreater(answer["result"]["ttlMs"], 0, method)
            self.assertEqual(answer["result"]["cacheScope"], "public", method)

    def test_an_unsupported_version_in_the_body_is_a_400(self):
        body = modern_body("ping")
        body["params"]["_meta"][META_PROTOCOL_VERSION] = "1999-01-01"

        status, answer = post(
            self.url, body, protocol_version="1999-01-01", headers={METHOD_HEADER: "ping"}
        )

        # Refused by the header check, which is the earlier of the two places
        # that would have caught it, and answers with the same error.
        self.assertEqual(status, 400)
        self.assertEqual(answer["error"]["code"], UNSUPPORTED_PROTOCOL_VERSION)

    def test_an_unknown_modern_method_is_a_404(self):
        # A modern client reads the JSON-RPC body to tell this apart from a
        # 404 for an endpoint that is not there at all.
        status, body = self.conforming(modern_body("nonsense/method"))

        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], METHOD_NOT_FOUND)

    def test_an_unknown_legacy_method_stays_a_200(self):
        # A legacy client reads a 404 as the endpoint being gone.
        status, body = post(self.url, {"jsonrpc": "2.0", "id": 1, "method": "nope"})

        self.assertEqual(status, 200)
        self.assertEqual(body["error"]["code"], METHOD_NOT_FOUND)

    def test_discover_is_answered_over_http(self):
        status, body = post(
            self.url,
            {"jsonrpc": "2.0", "id": 1, "method": "server/discover"},
            protocol_version=MODERN,
            headers={METHOD_HEADER: "server/discover"},
        )

        self.assertEqual(status, 200)
        self.assertIn(MODERN, body["result"]["supportedVersions"])


if __name__ == "__main__":
    unittest.main()
