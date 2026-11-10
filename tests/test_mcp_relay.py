# -*- coding: utf-8 -*-

import http.server
import io
import json
import logging
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path, PureWindowsPath
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from RAPID2_Agilent.mcp import endpoint as endpoint_file
from RAPID2_Agilent.mcp import paths
from RAPID2_Agilent.mcp import registry
from RAPID2_Agilent.mcp.protocol import (
    INTERNAL_ERROR,
    META_CLIENT_CAPABILITIES,
    META_PROTOCOL_VERSION,
    LATEST_LEGACY_PROTOCOL_VERSION,
    META_CLIENT_INFO,
    METHOD_HEADER,
    MODERN_PROTOCOL_VERSIONS,
    NAME_HEADER,
    PARSE_ERROR,
    PROTOCOL_VERSION_HEADER,
)
from RAPID2_Agilent.mcp.relay import announce, launcher, stdio
from RAPID2_Agilent.mcp.relay.upstream import (
    HttpUpstream,
    _ConnectTimeout,
    _mirrored_headers,
    loopback_urlopen,
    UpstreamRejected,
    UpstreamUnavailable,
)
from RAPID2_Agilent.mcp.transport import McpHttpServer

from .test_mcp_transport import build_dispatcher


class UpstreamTests(unittest.TestCase):
    """The relay against a real RAPID2 HTTP server."""

    def setUp(self):
        self.token = "test-token"
        self.server = McpHttpServer(build_dispatcher().dispatch_payload, self.token)
        self.server.start()
        self.addCleanup(self.server.stop)

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "A.json"
        self.publish(self.token)
        # The wait is exercised on its own below; here it only slows failures.
        self.upstream = HttpUpstream(registry_root=self.root, first_contact_wait=0)

    def publish(self, token, *, instance_id="run-1", label="A", port=None):
        endpoint_file.write_endpoint(
            "127.0.0.1",
            self.server.port if port is None else port,
            token,
            path=self.root / f"{label}.json",
            instance_id=instance_id,
            app_version="0.0.0",
            label=label,
            started_at="2026-09-04T00:00:00+00:00",
        )

    def test_a_request_is_answered(self):
        response = self.upstream.send(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )

        self.assertEqual(response["id"], 1)
        self.assertIn("probe", [tool["name"] for tool in response["result"]["tools"]])

    def test_a_slow_reply_keeps_the_response_timeout_after_connecting(self):
        dispatch = build_dispatcher().dispatch_payload

        def slow_dispatch(payload):
            time.sleep(1.2)
            return dispatch(payload)

        slow_server = McpHttpServer(slow_dispatch, self.token)
        slow_server.start()
        self.addCleanup(slow_server.stop)
        self.publish(self.token, port=slow_server.port)
        upstream = HttpUpstream(
            registry_root=self.root,
            first_contact_wait=0,
            timeout=2.0,
        )

        response = upstream.send(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )

        self.assertEqual(response["id"], 1)

    def test_a_modern_request_survives_the_relay_intact(self):
        # The relay mirrors part of the body into headers, and the window
        # refuses a request whose headers and body disagree. So this passing
        # is the two halves of that contract agreeing with each other -- which
        # neither side can establish alone.
        response = self.upstream.send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "probe",
                    "arguments": {},
                    "_meta": {
                        META_PROTOCOL_VERSION: MODERN_PROTOCOL_VERSIONS[0],
                        META_CLIENT_CAPABILITIES: {},
                    },
                },
            }
        )

        self.assertEqual(response["result"]["resultType"], "complete")

    def test_the_relay_mirrors_the_modern_headers(self):
        headers = _mirrored_headers(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "probe",
                    "_meta": {
                        META_PROTOCOL_VERSION: MODERN_PROTOCOL_VERSIONS[0]
                    },
                },
            }
        )

        self.assertEqual(
            headers,
            {
                PROTOCOL_VERSION_HEADER: MODERN_PROTOCOL_VERSIONS[0],
                METHOD_HEADER: "tools/call",
                NAME_HEADER: "probe",
            },
        )

    def test_a_legacy_message_is_mirrored_into_no_headers(self):
        # There is nothing to mirror: the fields the headers copy arrived with
        # the modern era, and a value invented for a legacy message would be a
        # mismatch by construction.
        self.assertEqual(
            _mirrored_headers({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}), {}
        )
        self.assertEqual(_mirrored_headers([{"jsonrpc": "2.0", "id": 1}]), {})

    def test_a_discover_probe_is_answered_through_the_relay(self):
        # How a dual-era client tells the two eras apart over stdio.
        response = self.upstream.send(
            {"jsonrpc": "2.0", "id": 1, "method": "server/discover"}
        )

        self.assertIn(
            MODERN_PROTOCOL_VERSIONS[0], response["result"]["supportedVersions"]
        )

    def test_a_notification_has_no_answer(self):
        self.assertIsNone(
            self.upstream.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        )

    def test_a_rotated_token_is_picked_up_without_reconnecting(self):
        # RAPID2 restarted while the client stayed connected: new server, new
        # token, same relay process. It must recover on the next message.
        self.server.stop()
        self.server = McpHttpServer(build_dispatcher().dispatch_payload, "second-token")
        self.server.start()
        self.addCleanup(self.server.stop)
        self.publish("second-token", instance_id="run-2")

        response = self.upstream.send(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        )

        self.assertEqual(response["id"], 2)

    def test_a_stale_token_that_stays_wrong_is_reported(self):
        self.publish("wrong-token")

        with self.assertRaises(UpstreamRejected):
            self.upstream.send({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})

    def test_an_empty_registry_means_not_running(self):
        self.path.unlink()

        with self.assertRaises(UpstreamUnavailable):
            self.upstream.send({"jsonrpc": "2.0", "id": 4, "method": "tools/list"})

    def test_a_closed_server_means_not_running(self):
        self.server.stop()

        with self.assertRaises(UpstreamUnavailable):
            self.upstream.send({"jsonrpc": "2.0", "id": 5, "method": "tools/list"})


class RoutingTests(unittest.TestCase):
    """Which window a message reaches."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.servers = {}

    def publish(self, label, token):
        server = McpHttpServer(build_dispatcher().dispatch_payload, token)
        server.start()
        self.addCleanup(server.stop)
        self.servers[label] = server
        endpoint_file.write_endpoint(
            "127.0.0.1",
            server.port,
            token,
            path=self.root / f"{label}.json",
            instance_id=f"{label.lower() * 4}9999",
            app_version="0.0.0",
            label=label,
            started_at="2026-09-04T00:00:00+00:00",
        )
        return server

    def upstream(self):
        return HttpUpstream(registry_root=self.root, first_contact_wait=0)

    @staticmethod
    def call(instance=None):
        arguments = {} if instance is None else {"instance": instance}
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "probe", "arguments": arguments},
        }

    def test_a_named_window_is_reached_directly(self):
        # Each window only accepts its own token, so an answer at all proves
        # the message went to the named one and not through a neighbour.
        self.publish("A", "token-a")
        self.publish("B", "token-b")

        answer = self.upstream().send(self.call(instance="B"))

        self.assertEqual(answer["id"], 1)
        self.assertNotIn("error", answer)

    def test_a_window_can_be_named_by_instance_id_prefix(self):
        self.publish("A", "token-a")
        self.publish("B", "token-b")

        answer = self.upstream().send(self.call(instance="bbbb"))

        self.assertNotIn("error", answer)

    def test_an_unknown_target_is_refused_with_the_roster(self):
        self.publish("A", "token-a")

        with self.assertRaises(UpstreamRejected) as caught:
            self.upstream().send(self.call(instance="Z"))

        self.assertIn("A", str(caught.exception))

    def test_an_untargeted_message_goes_to_the_first_window(self):
        # Deciding between windows is the instance's business, not ours: it
        # answers with the roster. The relay only has to pick a door.
        self.publish("A", "token-a")
        self.publish("B", "token-b")

        answer = self.upstream().send(self.call())

        self.assertNotIn("error", answer)

    def test_a_window_that_crashed_is_stepped_over(self):
        # Its address is still on file -- the sweep runs when a window starts,
        # not when one dies -- and B must not be unreachable because of it.
        dead = McpHttpServer(build_dispatcher().dispatch_payload, "token-a")
        dead.start()
        port = dead.port
        dead.stop()
        endpoint_file.write_endpoint(
            "127.0.0.1",
            port,
            "token-a",
            path=self.root / "A.json",
            instance_id="aaaa9999",
            label="A",
            started_at="2026-09-04T00:00:00+00:00",
        )
        self.publish("B", "token-b")

        started = time.monotonic()
        answer = self.upstream().send(self.call())
        elapsed = time.monotonic() - started

        self.assertNotIn("error", answer)
        self.assertLess(elapsed, 5.0)

    def test_a_named_window_that_is_gone_does_not_fall_through(self):
        # The caller asked for that document, not for any document.
        dead = McpHttpServer(build_dispatcher().dispatch_payload, "token-a")
        dead.start()
        port = dead.port
        dead.stop()
        endpoint_file.write_endpoint(
            "127.0.0.1",
            port,
            "token-a",
            path=self.root / "A.json",
            instance_id="aaaa9999",
            label="A",
            started_at="2026-09-04T00:00:00+00:00",
        )
        self.publish("B", "token-b")

        with self.assertRaises(UpstreamUnavailable):
            self.upstream().send(self.call(instance="A"))

    def test_a_window_with_access_turned_off_is_stepped_over(self):
        # Its address can still be on file for a moment after the switch went
        # off; nothing ran there, so B answers instead.
        self.publish("A", "token-a").revoke()
        self.publish("B", "token-b")

        answer = self.upstream().send(self.call())

        self.assertNotIn("error", answer)

    def test_access_turned_off_everywhere_reads_as_not_reachable(self):
        # Not "HTTP 503": the message a user can act on says to turn the AI
        # assistant access switch back on.
        self.publish("A", "token-a").revoke()

        with self.assertRaises(UpstreamUnavailable) as caught:
            self.upstream().send(self.call())

        self.assertIn("AI assistant access", str(caught.exception))

    def test_a_timeout_never_falls_through_to_another_window(self):
        # A timeout may mean the edit is being applied right now, so trying
        # the next window could apply it twice.
        self.publish("A", "token-a")
        self.publish("B", "token-b")
        attempts = []

        def timing_out(request, timeout=None):
            attempts.append(request.full_url)
            raise TimeoutError("timed out")

        upstream = HttpUpstream(
            registry_root=self.root, first_contact_wait=0, urlopen=timing_out
        )

        with self.assertRaises(UpstreamUnavailable):
            upstream.send(self.call())

        self.assertEqual(len(attempts), 1)

    def test_a_connect_timeout_steps_over_to_the_next_window(self):
        self.publish("A", "token-a")
        self.publish("B", "token-b")
        actual_urlopen = loopback_urlopen()
        attempts = []

        def first_connect_times_out(request, timeout=None):
            attempts.append(request.full_url)
            if len(attempts) == 1:
                raise urllib.error.URLError(
                    _ConnectTimeout("connect timed out")
                )
            return actual_urlopen(request, timeout=timeout)

        upstream = HttpUpstream(
            registry_root=self.root,
            first_contact_wait=0,
            urlopen=first_connect_times_out,
        )

        answer = upstream.send(self.call())

        self.assertNotIn("error", answer)
        self.assertEqual(len(attempts), 2)

    def test_running_out_of_windows_reports_that_nothing_is_reachable(self):
        dead = McpHttpServer(build_dispatcher().dispatch_payload, "token-a")
        dead.start()
        port = dead.port
        dead.stop()
        endpoint_file.write_endpoint(
            "127.0.0.1",
            port,
            "token-a",
            path=self.root / "A.json",
            instance_id="aaaa9999",
            label="A",
            started_at="2026-09-04T00:00:00+00:00",
        )

        with self.assertRaises(UpstreamUnavailable):
            self.upstream().send(self.call())

    def test_a_message_with_no_arguments_still_routes(self):
        self.publish("A", "token-a")

        answer = self.upstream().send(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )

        self.assertNotIn("error", answer)


class LoopbackClientTests(unittest.TestCase):
    """The relay's own HTTP client: this machine, and nowhere else.

    Every message carries a bearer token and, for a tool call, the document
    being worked on. Both belong on the loopback interface, and urllib's
    default opener will take them off it.
    """

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def register(self, port, token="token-a"):
        endpoint_file.write_endpoint(
            "127.0.0.1",
            port,
            token,
            path=self.root / "A.json",
            instance_id="aaaa9999",
            app_version="0.0.0",
            label="A",
            started_at="2026-09-04T00:00:00+00:00",
        )

    def serve(self, handler_class):
        """Something on RAPID2's port that is not RAPID2."""

        server = http.server.HTTPServer(("127.0.0.1", 0), handler_class)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        # Cleanups run last-added first: stop the loop while its socket still
        # exists, wait for its thread, then close the socket.
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        return server

    def test_a_proxy_in_the_environment_is_not_used(self):
        # urllib treats no address as special, 127.0.0.1 included: with a proxy
        # set and no exception covering the loopback address, the default
        # opener would post the token and the document to the proxy instead.
        # Pointed here at a port nothing listens on, so going through it cannot
        # be mistaken for reaching the window.
        server = McpHttpServer(build_dispatcher().dispatch_payload, "token-a")
        server.start()
        self.addCleanup(server.stop)
        self.register(server.port)

        with mock.patch.dict(os.environ):
            os.environ["http_proxy"] = "http://127.0.0.1:1"
            os.environ["https_proxy"] = "http://127.0.0.1:1"
            os.environ.pop("no_proxy", None)
            os.environ.pop("NO_PROXY", None)
            upstream = HttpUpstream(registry_root=self.root, first_contact_wait=0)
            response = upstream.send({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

        self.assertEqual(response["id"], 1)

    def test_a_redirect_is_refused_rather_than_followed(self):
        # A process that took the port after RAPID2 died can answer with one, and
        # urllib keeps the Authorization header across a redirect.
        followed = []

        class Redirecting(http.server.BaseHTTPRequestHandler):
            target = None

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(302)
                self.send_header("Location", self.target)
                self.end_headers()

            def do_GET(self):
                followed.append(self.headers.get("Authorization"))
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = self.serve(Redirecting)
        Redirecting.target = f"http://127.0.0.1:{server.server_port}/taken"
        self.register(server.server_port)
        upstream = HttpUpstream(registry_root=self.root, first_contact_wait=0)

        with self.assertRaises(UpstreamRejected) as caught:
            upstream.send({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

        self.assertIn("302", str(caught.exception))
        self.assertEqual(followed, [])


class FirstContactTests(unittest.TestCase):
    """A client may start its servers before RAPID2 is open."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "A.json"
        self.message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}

    def start_server(self, token="tok"):
        server = McpHttpServer(build_dispatcher().dispatch_payload, token)
        server.start()
        self.addCleanup(server.stop)
        return server

    def publish(self, server, token="tok"):
        endpoint_file.write_endpoint(
            "127.0.0.1",
            server.port,
            token,
            path=self.path,
            instance_id="run-1",
            label="A",
            started_at="2026-09-04T00:00:00+00:00",
        )

    def test_the_first_message_waits_for_RAPID2_to_appear(self):
        server = self.start_server()
        upstream = HttpUpstream(
            registry_root=self.root, first_contact_wait=5.0, poll_interval=0.02
        )
        appears = threading.Timer(0.2, self.publish, args=(server,))
        appears.start()
        self.addCleanup(appears.cancel)

        response = upstream.send(self.message)

        self.assertEqual(response["id"], 1)

    def test_the_wait_is_bounded(self):
        upstream = HttpUpstream(
            registry_root=self.root, first_contact_wait=0.2, poll_interval=0.02
        )

        started = time.monotonic()
        with self.assertRaises(UpstreamUnavailable):
            upstream.send(self.message)

        self.assertGreaterEqual(time.monotonic() - started, 0.2)

    def test_later_messages_fail_at_once(self):
        # Once the client is established, an answer beats a stall.
        server = self.start_server()
        self.publish(server)
        upstream = HttpUpstream(
            registry_root=self.root, first_contact_wait=30.0, poll_interval=0.02
        )
        upstream.send(self.message)
        server.stop()

        started = time.monotonic()
        with self.assertRaises(UpstreamUnavailable):
            upstream.send(self.message)

        self.assertLess(time.monotonic() - started, 5.0)


class ConnectionTests(unittest.TestCase):
    """A whole conversation, from the pipe a client writes to.

    Everything else here tests one link. This drives the chain a client
    actually talks through -- stdin, the stdio adapter, the relay's routing,
    the loopback HTTP hop with its header validation, and the dispatcher -- so
    that a mistake at any of the joins shows up as a client that cannot
    connect, which is how a user would meet it.
    """

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.token = "tok"

    def start_window(self, *, publish=True):
        server = McpHttpServer(build_dispatcher().dispatch_payload, self.token)
        server.start()
        self.addCleanup(server.stop)
        if publish:
            self.publish(server)
        return server

    def publish(self, server):
        endpoint_file.write_endpoint(
            "127.0.0.1",
            server.port,
            self.token,
            path=self.root / "A.json",
            instance_id="run-1",
            label="A",
            started_at="2026-09-04T00:00:00+00:00",
        )

    def converse(self, messages, *, first_contact_wait=0.0):
        """Write a whole session down the pipe and read what comes back."""

        upstream = HttpUpstream(
            registry_root=self.root,
            first_contact_wait=first_contact_wait,
            poll_interval=0.02,
        )
        stdin = io.BytesIO(
            "".join(f"{json.dumps(message)}\n" for message in messages).encode("utf-8")
        )
        stdout = io.BytesIO()
        stdio.serve(upstream, stdin=stdin, stdout=stdout)
        written = stdout.getvalue().decode("utf-8").splitlines()
        answers = [json.loads(line) for line in written if line]
        return {answer["id"]: answer for answer in answers if "id" in answer}

    # -- the legacy era ---------------------------------------------------
    def test_a_legacy_client_connects_and_works(self):
        self.start_window()

        answers = self.converse(
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": LATEST_LEGACY_PROTOCOL_VERSION,
                        "clientInfo": {"name": "legacy-client", "version": "1.0"},
                        "capabilities": {},
                    },
                },
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "probe", "arguments": {}},
                },
            ]
        )

        handshake = answers[1]["result"]
        self.assertEqual(
            handshake["protocolVersion"], LATEST_LEGACY_PROTOCOL_VERSION
        )
        self.assertIn("tools", handshake["capabilities"])
        self.assertIn(
            "probe", [tool["name"] for tool in answers[2]["result"]["tools"]]
        )
        self.assertFalse(answers[3]["result"]["isError"])
        # None of the modern furniture leaks into an era that has no idea
        # what to do with it.
        self.assertNotIn("resultType", answers[2]["result"])
        self.assertNotIn("ttlMs", answers[2]["result"])

    # -- the modern era ---------------------------------------------------
    def modern(self, method, params=None, *, request_id):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": {
                **(params or {}),
                "_meta": {
                    META_PROTOCOL_VERSION: MODERN_PROTOCOL_VERSIONS[0],
                    META_CLIENT_CAPABILITIES: {},
                    META_CLIENT_INFO: {"name": "modern-client", "version": "1.0"},
                },
            },
        }

    def test_a_modern_client_connects_and_works(self):
        self.start_window()

        answers = self.converse(
            [
                self.modern("server/discover", request_id=1),
                self.modern("tools/list", request_id=2),
                self.modern(
                    "tools/call",
                    {"name": "probe", "arguments": {}},
                    request_id=3,
                ),
            ]
        )

        discovered = answers[1]["result"]
        self.assertEqual(discovered["resultType"], "complete")
        self.assertIn(MODERN_PROTOCOL_VERSIONS[0], discovered["supportedVersions"])
        self.assertGreater(discovered["ttlMs"], 0)

        listing = answers[2]["result"]
        self.assertEqual(listing["resultType"], "complete")
        self.assertEqual(listing["cacheScope"], "public")
        self.assertIn("probe", [tool["name"] for tool in listing["tools"]])

        called = answers[3]["result"]
        self.assertEqual(called["resultType"], "complete")
        self.assertFalse(called["isError"])
        # It never sent `initialize`, and never had to.
        self.assertNotIn("protocolVersion", called)

    def test_the_two_eras_can_share_one_relay(self):
        # There is no session to be in the wrong one of: the era is a property
        # of each request, so the same pipe answers both without either
        # colouring the other.
        self.start_window()

        answers = self.converse(
            [
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                self.modern("tools/list", request_id=2),
                {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
            ]
        )

        self.assertNotIn("resultType", answers[1]["result"])
        self.assertEqual(answers[2]["result"]["resultType"], "complete")
        self.assertNotIn("resultType", answers[3]["result"])

    # -- RAPID2 opening slowly ----------------------------------------------
    def test_a_modern_probe_waits_for_a_window_that_is_still_opening(self):
        # The client's servers start when the client does, which can be
        # moments before RAPID2 has finished opening. This is the case that
        # matters most: a `server/discover` that fails is read as "this server
        # is legacy", and a client MAY cache that judgement for the life of
        # the relay process -- so a slow launch would leave it speaking the
        # wrong era long after RAPID2 was ready.
        server = self.start_window(publish=False)
        appears = threading.Timer(0.3, self.publish, args=(server,))
        appears.start()
        self.addCleanup(appears.cancel)

        answers = self.converse(
            [self.modern("server/discover", request_id=1)], first_contact_wait=5.0
        )

        self.assertNotIn("error", answers[1])
        self.assertIn(
            MODERN_PROTOCOL_VERSIONS[0], answers[1]["result"]["supportedVersions"]
        )

    def test_a_legacy_handshake_waits_for_a_window_that_is_still_opening(self):
        server = self.start_window(publish=False)
        appears = threading.Timer(0.3, self.publish, args=(server,))
        appears.start()
        self.addCleanup(appears.cancel)

        answers = self.converse(
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": LATEST_LEGACY_PROTOCOL_VERSION},
                }
            ],
            first_contact_wait=5.0,
        )

        self.assertEqual(
            answers[1]["result"]["protocolVersion"], LATEST_LEGACY_PROTOCOL_VERSION
        )

    def test_a_window_that_never_opens_is_reported_rather_than_hung(self):
        answers = self.converse(
            [self.modern("server/discover", request_id=1)], first_contact_wait=0.2
        )

        self.assertIn("error", answers[1])


class _SwappingStdin:
    """A pipe that lets the window be replaced part-way through a session.

    `serve` reads its input by iterating, so a reader that does something
    between two lines is all it takes to put a RAPID2 restart in the middle of a
    client's conversation -- which is the only moment this behaviour exists
    for, and one a pre-filled buffer cannot express.
    """

    def __init__(self, lines, *, before_last=None):
        self._lines = list(lines)
        self._before_last = before_last

    def __iter__(self):
        for index, line in enumerate(self._lines):
            if self._before_last is not None and index == len(self._lines) - 1:
                self._before_last()
            yield line


class RelayAnnouncementTests(unittest.TestCase):
    """What the relay says on its own behalf, rather than RAPID2's.

    A client keeps the tool list it was given at the handshake. RAPID2 cannot
    tell it otherwise -- a window's tools are fixed for the life of the
    process, so `listChanged: false` is true of any one window and false of a
    session that outlives several. The relay is the only party that sees the
    seam.
    """

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.token = "tok"
        self.server = McpHttpServer(build_dispatcher().dispatch_payload, self.token)
        self.server.start()
        self.addCleanup(self.server.stop)
        self.publish("run-1")

    def publish(self, instance_id, *, label="A"):
        endpoint_file.write_endpoint(
            "127.0.0.1",
            self.server.port,
            self.token,
            path=self.root / f"{label}.json",
            instance_id=instance_id,
            label=label,
            started_at="2026-09-04T00:00:00+00:00",
        )

    def converse(self, messages, *, before_last=None):
        """Every line the relay wrote, notifications included and in order."""

        upstream = HttpUpstream(registry_root=self.root, first_contact_wait=0)
        lines = [
            f"{json.dumps(message)}\n".encode("utf-8") for message in messages
        ]
        stdout = io.BytesIO()
        stdio.serve(
            upstream,
            stdin=_SwappingStdin(lines, before_last=before_last),
            stdout=stdout,
        )
        written = stdout.getvalue().decode("utf-8").splitlines()
        return [json.loads(line) for line in written if line]

    @staticmethod
    def handshake(message_id=1):
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "method": "initialize",
            "params": {"protocolVersion": LATEST_LEGACY_PROTOCOL_VERSION},
        }

    @staticmethod
    def listing(message_id):
        return {"jsonrpc": "2.0", "id": message_id, "method": "tools/list"}

    def test_the_handshake_says_the_tool_list_can_change(self):
        # The window says it cannot, and for the window that is true.
        answers = self.converse([self.handshake()])

        self.assertFalse(
            build_dispatcher()
            .dispatch_payload(self.handshake())["result"]["capabilities"]["tools"]["listChanged"]
        )
        self.assertTrue(
            answers[0]["result"]["capabilities"]["tools"]["listChanged"]
        )

    def test_the_modern_probe_says_it_too(self):
        # Two replies carry this capability. A rule written around the name of
        # one of them would leave the other telling the client the opposite.
        answers = self.converse(
            [
                {"jsonrpc": "2.0", "id": 1, "method": "server/discover"}
            ]
        )

        self.assertTrue(
            answers[0]["result"]["capabilities"]["tools"]["listChanged"]
        )

    def test_a_window_replaced_mid_session_is_announced(self):
        written = self.converse(
            [self.handshake(), self.listing(2)],
            before_last=lambda: self.publish("run-2"),
        )

        methods = [one.get("method") for one in written]
        self.assertIn(announce.TOOLS_LIST_CHANGED, methods)

    def test_the_announcement_follows_the_answer_it_interrupts(self):
        # The client is waiting on that id, and the reply was produced by the
        # window it still believes in. Answer first, then say to ask again.
        written = self.converse(
            [self.handshake(), self.listing(2)],
            before_last=lambda: self.publish("run-2"),
        )

        answer = next(i for i, one in enumerate(written) if one.get("id") == 2)
        notice = next(
            i
            for i, one in enumerate(written)
            if one.get("method") == announce.TOOLS_LIST_CHANGED
        )
        self.assertLess(answer, notice)

    def test_a_session_against_one_window_is_never_interrupted(self):
        written = self.converse([self.handshake(), self.listing(2)])

        self.assertEqual(
            [one.get("method") for one in written if "id" not in one], []
        )

    def test_the_window_the_client_handshook_against_is_not_a_replacement(self):
        # The first sighting of a letter is the run the tool list came from.
        written = self.converse([self.handshake()])

        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["id"], 1)

    def test_concurrent_calls_announce_one_restart_once(self):
        # Every worker resolves the registry, so the compare-and-record has to
        # be atomic or one restart is announced several times.
        upstream = HttpUpstream(registry_root=self.root, first_contact_wait=0)
        upstream.send(self.handshake())
        self.publish("run-2")

        barrier = threading.Barrier(4)
        seen = []

        def call():
            barrier.wait()
            upstream.send(self.listing(2))
            seen.append(upstream.take_replacements())

        threads = [threading.Thread(target=call) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(sum(len(one) for one in seen), 1)

    def test_one_event_is_announced_once(self):
        written = self.converse(
            [self.handshake(), self.listing(2), self.listing(3)],
            before_last=None,
        )
        self.assertEqual(len(written), 3)

        written = self.converse(
            [self.handshake(), self.listing(2), self.listing(3)],
            before_last=lambda: self.publish("run-2"),
        )
        announced = [
            one
            for one in written
            if one.get("method") == announce.TOOLS_LIST_CHANGED
        ]

        self.assertEqual(len(announced), 1)


class AnnouncementShapeTests(unittest.TestCase):
    """`announce` on its own, including the replies it must not touch."""

    def test_a_capability_already_true_is_left_alone(self):
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"capabilities": {"tools": {"listChanged": True}}},
        }

        self.assertIs(announce.speaking_for_the_relay(response), response)

    def test_other_capabilities_survive_the_correction(self):
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "instructions": "read me",
                "capabilities": {
                    "resources": {"subscribe": True},
                    "tools": {"listChanged": False, "somethingElse": 1},
                },
            },
        }

        corrected = announce.speaking_for_the_relay(response)

        self.assertEqual(
            corrected["result"]["capabilities"]["resources"], {"subscribe": True}
        )
        self.assertEqual(corrected["result"]["capabilities"]["tools"]["somethingElse"], 1)
        self.assertEqual(corrected["result"]["instructions"], "read me")

    def test_the_reply_it_was_given_is_not_modified(self):
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"capabilities": {"tools": {"listChanged": False}}},
        }

        announce.speaking_for_the_relay(response)

        self.assertFalse(
            response["result"]["capabilities"]["tools"]["listChanged"]
        )

    def test_an_error_reply_is_carried_unchanged(self):
        response = {"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "no"}}

        self.assertIs(announce.speaking_for_the_relay(response), response)

    def test_a_reply_with_no_capabilities_is_carried_unchanged(self):
        response = {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}

        self.assertIs(announce.speaking_for_the_relay(response), response)

    def test_nothing_to_answer_is_nothing_to_correct(self):
        self.assertIsNone(announce.speaking_for_the_relay(None))

    def test_a_batch_is_corrected_entry_by_entry(self):
        corrected = announce.speaking_for_the_relay(
            [
                {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}},
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {"capabilities": {"tools": {"listChanged": False}}},
                },
            ]
        )

        self.assertEqual(corrected[0]["result"], {"tools": []})
        self.assertTrue(
            corrected[1]["result"]["capabilities"]["tools"]["listChanged"]
        )

    def test_several_windows_replaced_at_once_ask_for_one_re_read(self):
        # The client holds one tool list however many windows changed.
        self.assertEqual(len(announce.notifications(["A", "B"])), 1)

    def test_nothing_replaced_says_nothing(self):
        self.assertEqual(announce.notifications([]), [])


class LauncherTests(unittest.TestCase):
    """The path a client is registered with, which must never move."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = (
            Path(self.directory.name) / "bin" / launcher.launcher_name()
        )

    def test_it_has_the_platforms_private_runnable_form(self):
        launcher.write_launcher(self.path)

        if os.name == "nt":
            # Windows executes this by its extension and protects it with the
            # DACL installed by `write_private_file`; POSIX execute bits have
            # no equivalent meaning here.
            self.assertEqual(self.path.suffix.lower(), ".cmd")
            self.assertTrue(self.path.is_file())
            return

        mode = stat.S_IMODE(self.path.stat().st_mode)
        self.assertEqual(mode, 0o700)
        self.assertTrue(os.access(self.path, os.X_OK))

    def test_an_unchanged_launcher_is_left_alone(self):
        launcher.write_launcher(self.path)
        before = self.path.stat().st_mtime_ns

        launcher.write_launcher(self.path)

        self.assertEqual(before, self.path.stat().st_mtime_ns)

    def test_an_exact_windows_launcher_is_not_changed_by_newline_translation(self):
        command = [r"C:\Program Files\RAPID2\rapid2-mcp.exe"]
        environment = {"PYTHONPATH": r"C:\RAPID2\src"}
        with mock.patch.object(sys, "platform", "win32"):
            source = launcher.launcher_source(command, environment)
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(source.encode("utf-8"))

        with mock.patch.object(sys, "platform", "win32"), mock.patch.object(
            launcher,
            "relay_invocation",
            return_value=(command, environment),
        ), mock.patch.object(launcher, "write_private_file") as writer:
            returned = launcher.write_launcher(self.path)

        self.assertEqual(returned, self.path)
        writer.assert_not_called()
        self.assertIn(b"\r\n", self.path.read_bytes())

    def test_it_is_rewritten_when_the_application_moves(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("#!/bin/sh\nexec /somewhere/old/RAPID2 --mcp-relay\n")

        launcher.write_launcher(self.path)

        self.assertNotIn("/somewhere/old/RAPID2", self.path.read_text(encoding="utf-8"))

    @unittest.skipIf(sys.platform == "win32", "POSIX launcher")
    def test_a_path_with_spaces_survives_quoting(self):
        command = ["/Applications/My App.app/Contents/MacOS/RAPID2", "--mcp-relay"]

        source = launcher.launcher_source(command, {"PYTHONPATH": "/a b/src"})

        self.assertTrue(source.startswith("#!/bin/sh"))
        self.assertIn("'/Applications/My App.app/Contents/MacOS/RAPID2'", source)
        self.assertIn("export PYTHONPATH='/a b/src'", source)
        self.assertTrue(source.rstrip().endswith('"$@"'))

    def test_the_windows_launcher_quotes_a_path_with_spaces(self):
        # The last link in the Windows chain, and the only one that can be
        # checked here: Codex hands cmd an unquoted path, cmd finds this file,
        # and this file has to name the relay in a way cmd reads as one word.
        # `C:\Program Files` is the case that decides it.
        with mock.patch.object(sys, "platform", "win32"):
            source = launcher.launcher_source(
                [r"C:\Program Files\My App\rapid2-mcp.exe"],
                {"PYTHONPATH": r"C:\a b\src"},
            )

        self.assertIn(r'"C:\Program Files\My App\rapid2-mcp.exe"', source)
        # `set "NAME=value"` rather than `set NAME=value`: the quotes are what
        # keep a trailing space, and anything after an `&`, out of the value.
        self.assertIn(r'set "PYTHONPATH=C:\a b\src"', source)
        self.assertTrue(source.rstrip().endswith("%*"), source)
        self.assertTrue(source.startswith("@echo off\r\n"), source)

    @unittest.skipIf(sys.platform == "win32", "POSIX registration line")
    def test_the_registration_command_quotes_the_path(self):
        command = launcher.registration_command(
            Path("/Users/x/Library/Application Support/RAPID2/bin/rapid2-mcp-relay"),
            "rapid2",
        )

        self.assertIn("claude mcp add rapid2 -- ", command)
        self.assertIn("'/Users/x/Library/Application Support/RAPID2/bin/", command)

    @unittest.skipIf(sys.platform == "win32", "POSIX registration line")
    def test_codex_is_registered_by_the_same_line_of_its_own_command(self):
        command = launcher.registration_command(
            Path("/Users/x/Library/Application Support/RAPID2/bin/rapid2-mcp-relay"),
            "rapid2",
            "codex",
        )

        self.assertTrue(command.startswith("codex mcp add rapid2 -- '/Users/x/"), command)
        with self.assertRaises(ValueError):
            launcher.registration_command(Path("/x"), "rapid2", "cursor")

    def test_the_windows_registration_command_is_quoted_for_powershell(self):
        # An apostrophe in the user name is the case POSIX quoting gets wrong
        # here: it would backslash-escape one, which PowerShell reads as part
        # of the path.
        with mock.patch.object(sys, "platform", "win32"):
            command = launcher.registration_command(
                PureWindowsPath(r"C:\Users\O'Brien\AppData\Local\RAPID2\bin\x.cmd"),
                "rapid2",
            )

        self.assertEqual(
            command,
            "claude mcp add rapid2 -- "
            "'C:\\Users\\O''Brien\\AppData\\Local\\RAPID2\\bin\\x.cmd'",
        )

    def test_the_relay_is_started_with_this_interpreter(self):
        # RAPID2 is installed in the interpreter's environment, so the module
        # needs no search path of its own.
        command, environment = launcher.relay_invocation()

        self.assertEqual(command, [sys.executable, "-m", "RAPID2_Agilent.mcp.relay"])
        self.assertEqual(environment, {})

    def test_the_default_path_is_the_one_the_launcher_directory_names(self):
        self.assertEqual(
            launcher.default_launcher_path().parent,
            paths.launcher_dir(),
        )

    def test_a_redirected_data_directory_does_not_move_the_launcher(self):
        # On Windows LOCALAPPDATA can be redirected, and a client's own
        # configuration -- the Claude Desktop extension especially -- can name
        # the home directory and nothing else. So the launcher is written
        # below home, and RAPID2's own files follow the redirect without it.
        home = Path("/Users/somebody")
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": r"D:\Redirected\Local"}), \
                mock.patch.object(sys, "platform", "win32"), \
                mock.patch.object(Path, "home", classmethod(lambda _cls: home)):
            # The suite's own redirect would hide the computation under test.
            os.environ.pop(paths.DATA_DIR_ENV, None)
            self.assertEqual(
                paths.launcher_dir(),
                home / "AppData" / "Local" / paths.APPLICATION_NAME / "bin",
            )
            self.assertEqual(
                paths.platform_app_local_data_dir(),
                Path(r"D:\Redirected\Local") / paths.APPLICATION_NAME,
            )


class SpawnedLauncherTests(unittest.TestCase):
    """The registered command, run the way a client runs it."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.home = Path(self.directory.name)
        self.server = McpHttpServer(build_dispatcher().dispatch_payload, "tok")
        self.server.start()
        self.addCleanup(self.server.stop)
        # The registry the child will actually read, addressed the way the
        # registry addresses it: a `<label>.json` under the instances
        # directory. Written by hand rather than through `claim_label` because
        # the point of these tests is the path the child computes from HOME,
        # and a claim would need the lock this process is not holding.
        endpoint_file.write_endpoint(
            "127.0.0.1",
            self.server.port,
            "tok",
            path=registry.document_path(
                "A",
                root=self.home
                / "Library/Application Support"
                / paths.APPLICATION_NAME
                / "mcp"
                / registry.INSTANCES_DIRNAME,
            ),
            instance_id="run-spawned",
            app_version="0.0.0",
            label="A",
            started_at="2026-09-04T00:00:00+00:00",
        )

    def run_relay(self, command, message):
        return subprocess.run(
            command,
            input=(json.dumps(message) + "\n").encode("utf-8"),
            capture_output=True,
            env={"HOME": str(self.home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            timeout=60,
            cwd=str(ROOT),
        )

    def assert_listed_the_tools(self, finished, request_id):
        """The window answered -- not the relay apologising for not finding it.

        An unreachable window is also a well-formed reply carrying the right
        id, on a relay that exits 0, so the id alone proves nothing. Only the
        tool list comes from the far side of the connection.
        """

        self.assertEqual(finished.returncode, 0, finished.stderr.decode())
        answer = json.loads(finished.stdout.decode().splitlines()[0])
        self.assertNotIn("error", answer, finished.stderr.decode())
        self.assertEqual(answer["id"], request_id)
        self.assertIn("probe", [tool["name"] for tool in answer["result"]["tools"]])

    @unittest.skipIf(sys.platform != "darwin", "path layout is macOS-specific")
    def test_the_written_launcher_reaches_the_window(self):
        path = self.home / "bin" / "rapid2-mcp-relay"
        launcher.write_launcher(path)

        finished = self.run_relay(
            [str(path)], {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )

        self.assert_listed_the_tools(finished, 1)

    @unittest.skipIf(sys.platform != "darwin", "path layout is macOS-specific")
    def test_the_installed_entry_point_answers(self):
        # `rapid2-agilent-mcp`, which a client can be pointed at instead of
        # the launcher.
        entry_point = Path(sys.executable).parent / "rapid2-agilent-mcp"
        if not entry_point.is_file():
            self.skipTest("RAPID2 is not installed with its scripts here")
        finished = self.run_relay(
            [str(entry_point)],
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )

        self.assert_listed_the_tools(finished, 2)


class _NoReplacements:
    """The half of the upstream contract these fakes do not exercise.

    The stdio adapter drains replacements after every message, so an upstream
    is not optional here even when the test is about something else.
    """

    def take_replacements(self):
        return []


class _CountingUpstream(_NoReplacements):
    """Records how many sends are in flight at once."""

    def __init__(self, first_delay=0.0, delays=None):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.peak_during_first = 0
        self.first_delay = first_delay
        self.delays = delays or {}

    def send(self, payload):
        request_id = payload.get("id")
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if request_id == 1 and self.first_delay:
                time.sleep(self.first_delay)
                with self.lock:
                    self.peak_during_first = self.active
            elif request_id in self.delays:
                time.sleep(self.delays[request_id])
            return {"jsonrpc": "2.0", "id": request_id, "result": {"id": request_id}}
        finally:
            with self.lock:
                self.active -= 1


class _BarrierUpstream(_NoReplacements):
    """Every send after the first must meet the others or time out."""

    def __init__(self, parties, timeout=5.0):
        self.barrier = threading.Barrier(parties, timeout=timeout)
        self.lock = threading.Lock()
        self.calls = 0

    def send(self, payload):
        with self.lock:
            self.calls += 1
            first = self.calls == 1
        if not first:
            self.barrier.wait()
        return {"jsonrpc": "2.0", "id": payload.get("id"), "result": {}}


class ConcurrentStdioTests(unittest.TestCase):
    """Requests overlap; the session's opening does not."""

    @classmethod
    def setUpClass(cls):
        cls.silenced = logging.getLogger(stdio.__name__)
        cls.previous_level = cls.silenced.level
        cls.silenced.setLevel(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        cls.silenced.setLevel(cls.previous_level)

    def run_messages(self, messages, upstream, **kwargs):
        stdin = io.BytesIO(
            b"".join(json.dumps(m).encode("utf-8") + b"\n" for m in messages)
        )
        stdout = io.BytesIO()
        stdio.serve(upstream, stdin=stdin, stdout=stdout, **kwargs)
        return [
            json.loads(line)
            for line in stdout.getvalue().decode("utf-8").splitlines()
            if line
        ]

    @staticmethod
    def request(request_id, method="tools/call"):
        return {"jsonrpc": "2.0", "id": request_id, "method": method}

    def test_requests_after_the_handshake_run_at_the_same_time(self):
        # The barrier can only be met if both are in flight together; if they
        # serialise it times out and each answer comes back as an error.
        upstream = _BarrierUpstream(parties=2)

        written = self.run_messages(
            [self.request(1, "initialize"), self.request(2), self.request(3)],
            upstream,
        )

        self.assertEqual(len(written), 3)
        for answer in written:
            self.assertNotIn("error", answer, answer)

    def test_the_handshake_is_answered_before_anything_else_is_dispatched(self):
        # MCP's lifecycle rule, kept without this adapter learning a method name.
        # The later two are slow enough to overlap each other, so the same run
        # shows both halves: nothing beside the handshake, everything after it.
        upstream = _CountingUpstream(first_delay=0.2, delays={2: 0.15, 3: 0.15})

        self.run_messages(
            [self.request(1, "initialize"), self.request(2), self.request(3)],
            upstream,
        )

        self.assertEqual(upstream.peak_during_first, 1)
        self.assertGreater(upstream.peak, 1)

    def test_a_notification_does_not_open_the_session(self):
        # Only an answered request can be the handshake.
        upstream = _CountingUpstream(first_delay=0.2)

        self.run_messages(
            [
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                self.request(1, "initialize"),
                self.request(2),
            ],
            upstream,
        )

        self.assertEqual(upstream.peak_during_first, 1)

    def test_answers_may_come_back_out_of_order(self):
        # Ids correlate, so a slow call must not hold up a fast one behind it.
        upstream = _CountingUpstream(delays={2: 0.3})

        written = self.run_messages(
            [self.request(1, "initialize"), self.request(2), self.request(3)],
            upstream,
        )

        self.assertEqual([answer["id"] for answer in written], [1, 3, 2])

    def test_every_answer_reaches_the_pipe_whole(self):
        upstream = _CountingUpstream()
        messages = [self.request(1, "initialize")]
        messages += [self.request(n) for n in range(2, 22)]

        written = self.run_messages(messages, upstream)

        # json.loads of each line already proves no two writes interleaved.
        self.assertEqual(len(written), 21)
        self.assertEqual(
            sorted(answer["id"] for answer in written), sorted(range(1, 22))
        )

    def test_work_in_flight_is_finished_when_stdin_ends(self):
        upstream = _CountingUpstream(delays={2: 0.2, 3: 0.2, 4: 0.2})

        written = self.run_messages(
            [self.request(1, "initialize")] + [self.request(n) for n in (2, 3, 4)],
            upstream,
        )

        self.assertEqual(len(written), 4)

    def test_an_unexpected_failure_still_answers_the_waiting_id(self):
        class Exploding(_NoReplacements):
            def __init__(self):
                self.calls = 0

            def send(self, payload):
                self.calls += 1
                if self.calls == 1:
                    return {"jsonrpc": "2.0", "id": payload["id"], "result": {}}
                raise ValueError("boom")

        written = self.run_messages(
            [self.request(1, "initialize"), self.request(2)], Exploding()
        )

        self.assertEqual(written[1]["id"], 2)
        self.assertIn("ValueError", written[1]["error"]["message"])

    def test_one_failure_does_not_stop_the_pump(self):
        class ExplodingOnce(_NoReplacements):
            def __init__(self):
                self.calls = 0

            def send(self, payload):
                self.calls += 1
                if self.calls == 2:
                    raise ValueError("boom")
                return {"jsonrpc": "2.0", "id": payload["id"], "result": {}}

        written = self.run_messages(
            [self.request(1, "initialize"), self.request(2), self.request(3)],
            ExplodingOnce(),
            max_workers=1,
        )

        self.assertEqual(len(written), 3)
        self.assertIn("result", written[2])


class FakeUpstream:
    """Both halves of what the stdio adapter asks an upstream for.

    `take_replacements` is not optional padding: the adapter drains it after
    every message, including a failed one, so an upstream without it is one
    the adapter cannot drive.
    """

    def __init__(self, replies=None, raises=None, replacements=()):
        self.replies = replies or {}
        self.raises = raises
        self.sent = []
        self.replacements = list(replacements)

    def send(self, payload):
        self.sent.append(payload)
        if self.raises is not None:
            raise self.raises
        return self.replies.get(payload.get("id") if isinstance(payload, dict) else None)

    def take_replacements(self):
        found, self.replacements = self.replacements, []
        return found


class StdioAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These tests drive failures on purpose; their warnings are not news.
        cls.silenced = logging.getLogger(stdio.__name__)
        cls.previous_level = cls.silenced.level
        cls.silenced.setLevel(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        cls.silenced.setLevel(cls.previous_level)

    def run_lines(self, lines, upstream):
        stdin = io.BytesIO("".join(f"{line}\n" for line in lines).encode("utf-8"))
        stdout = io.BytesIO()
        stdio.serve(upstream, stdin=stdin, stdout=stdout)
        text = stdout.getvalue().decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line]

    def test_a_reply_is_one_line_of_json(self):
        upstream = FakeUpstream({1: {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}})

        written = self.run_lines(['{"jsonrpc": "2.0", "id": 1, "method": "x"}'], upstream)

        self.assertEqual(written, [{"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}])

    def test_the_message_is_forwarded_untouched(self):
        # The adapter is a pipe, not a participant.
        upstream = FakeUpstream({7: {"jsonrpc": "2.0", "id": 7, "result": {}}})
        message = {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"a": 1}}

        self.run_lines([json.dumps(message)], upstream)

        self.assertEqual(upstream.sent, [message])

    def test_a_notification_writes_nothing(self):
        upstream = FakeUpstream()

        written = self.run_lines(['{"jsonrpc": "2.0", "method": "notify"}'], upstream)

        self.assertEqual(written, [])

    def test_a_replaced_window_is_announced_even_when_the_message_failed(self):
        # The window changed underneath the session whatever became of this
        # message, and a client told only on success would be told last on the
        # run where it matters most -- the call that broke on the old schema.
        upstream = FakeUpstream(raises=ValueError("boom"), replacements=["A"])

        written = self.run_lines(['{"jsonrpc": "2.0", "id": 1, "method": "x"}'], upstream)

        self.assertEqual(written[0]["id"], 1)
        self.assertIn("error", written[0])
        self.assertEqual(written[1], announce.list_changed())

    def test_a_replaced_window_is_announced_after_a_message_with_no_reply(self):
        # A notification is answered with nothing, so this is the one case
        # where the announcement is the only thing written.
        upstream = FakeUpstream(replacements=["A"])

        written = self.run_lines(['{"jsonrpc": "2.0", "method": "notify"}'], upstream)

        self.assertEqual(written, [announce.list_changed()])

    def test_blank_lines_are_ignored(self):
        upstream = FakeUpstream()

        self.run_lines(["", "   "], upstream)

        self.assertEqual(upstream.sent, [])

    def test_malformed_json_is_answered_not_forwarded(self):
        upstream = FakeUpstream()

        written = self.run_lines(["{not json"], upstream)

        self.assertEqual(written[0]["error"]["code"], PARSE_ERROR)
        self.assertEqual(upstream.sent, [])

    def test_an_unreachable_window_answers_the_waiting_id(self):
        upstream = FakeUpstream(raises=UpstreamUnavailable("RAPID2 is not reachable."))

        written = self.run_lines(['{"jsonrpc": "2.0", "id": 9, "method": "x"}'], upstream)

        self.assertEqual(written[0]["id"], 9)
        self.assertEqual(written[0]["error"]["code"], INTERNAL_ERROR)
        self.assertIn("not reachable", written[0]["error"]["message"])

    def test_an_unreachable_window_stays_silent_for_a_notification(self):
        upstream = FakeUpstream(raises=UpstreamUnavailable("gone"))

        written = self.run_lines(['{"jsonrpc": "2.0", "method": "notify"}'], upstream)

        self.assertEqual(written, [])

    def test_a_failed_batch_answers_every_id_in_it(self):
        upstream = FakeUpstream(raises=UpstreamUnavailable("gone"))
        batch = '[{"jsonrpc":"2.0","id":1,"method":"a"},{"jsonrpc":"2.0","method":"b"}]'

        written = self.run_lines([batch], upstream)

        self.assertEqual([entry["id"] for entry in written[0]], [1])

    def test_the_pump_ends_with_stdin(self):
        upstream = FakeUpstream({1: {"jsonrpc": "2.0", "id": 1, "result": {}}})

        written = self.run_lines(['{"jsonrpc": "2.0", "id": 1, "method": "x"}'], upstream)

        self.assertEqual(len(written), 1)



if __name__ == "__main__":
    unittest.main()
