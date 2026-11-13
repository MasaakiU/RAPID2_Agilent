# -*- coding: utf-8 -*-

"""What the relay answers itself: the catalog and the analysis tools while
RAPID2 is closed, and rapid2_launch. Ported from MyWB's tests of the same
modules, the analysis tools added."""

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from RAPID2_Agilent.mcp import catalog as catalog_file
from RAPID2_Agilent.mcp import endpoint as endpoint_file
from RAPID2_Agilent.mcp import guide
from RAPID2_Agilent.mcp import presence
from RAPID2_Agilent.mcp.protocol import (
    LATEST_LEGACY_PROTOCOL_VERSION,
    META_CLIENT_CAPABILITIES,
    META_PROTOCOL_VERSION,
    MODERN_PROTOCOL_VERSIONS,
    ToolError,
)
from RAPID2_Agilent.mcp.relay import __main__ as relay_main
from RAPID2_Agilent.mcp.relay import launch, local, stdio
from RAPID2_Agilent.mcp.relay.upstream import HttpUpstream
from RAPID2_Agilent.mcp.transport import McpHttpServer

from .test_mcp_transport import build_dispatcher


def probe_catalog(server_version="9.9.9", analysis_tools=()):
    """A catalog of the transport tests' window: one tool, `probe`."""

    return catalog_file.catalog_document(
        build_dispatcher().registry.describe_all(),
        server_name="rapid2-agilent",
        server_version=server_version,
        toolset_version=1,
        analysis_tools=analysis_tools,
    )


def analysis_catalog():
    """The same, with `probe` named an analysis tool."""

    return probe_catalog(analysis_tools=["probe"])


class FakeSession:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def legacy_initialize(request_id=1):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": LATEST_LEGACY_PROTOCOL_VERSION,
            "clientInfo": {"name": "legacy-client", "version": "1.0"},
            "capabilities": {},
        },
    }


def modern(method, params=None, *, request_id):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {
            **(params or {}),
            "_meta": {
                META_PROTOCOL_VERSION: MODERN_PROTOCOL_VERSIONS[0],
                META_CLIENT_CAPABILITIES: {},
            },
        },
    }


def call(name, request_id):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": {}},
    }


class LocalAnswerConversationTests(unittest.TestCase):
    """A whole conversation from the pipe, with and without a window."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.token = "tok"
        self.launches = []
        self.builds = []

    def start_window(self, *, token=None):
        server = McpHttpServer(build_dispatcher().dispatch_payload, self.token)
        server.start()
        self.addCleanup(server.stop)
        endpoint_file.write_endpoint(
            "127.0.0.1",
            server.port,
            token or self.token,
            path=self.root / "A.json",
            instance_id="run-1",
            label="A",
            started_at="2026-10-12T00:00:00+00:00",
        )
        return server

    def fake_launch(self, arguments):
        self.launches.append(arguments)
        return {"status": "ok", "started": True}

    def relay_analysis(self, handlers):
        """A builder of the relay's analysis tools that records each build."""

        def build():
            self.builds.append(True)
            return FakeSession(), handlers

        return build

    def converse(self, messages, *, catalog=None, analysis_tools=None):
        upstream = HttpUpstream(registry_root=self.root, first_contact_wait=0.0)
        answers = local.LocalAnswers(
            catalog,
            launch_handler=self.fake_launch,
            analysis_tools=analysis_tools or self.relay_analysis({}),
        )
        stdin = io.BytesIO(
            "".join(f"{json.dumps(message)}\n" for message in messages).encode("utf-8")
        )
        stdout = io.BytesIO()
        stdio.serve(upstream, stdin=stdin, stdout=stdout, local=answers)
        written = stdout.getvalue().decode("utf-8").splitlines()
        replies = [json.loads(line) for line in written if line]
        return {reply["id"]: reply for reply in replies if "id" in reply}

    # -- RAPID2 closed -----------------------------------------------------
    def test_a_legacy_client_is_introduced_to_rapid2_while_it_is_closed(self):
        answers = self.converse(
            [
                legacy_initialize(),
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ],
            catalog=probe_catalog(),
        )

        handshake = answers[1]["result"]
        self.assertEqual(handshake["protocolVersion"], LATEST_LEGACY_PROTOCOL_VERSION)
        self.assertEqual(handshake["serverInfo"], {"name": "rapid2-agilent", "version": "9.9.9"})
        # Said by the relay about itself, on whatever answered.
        self.assertTrue(handshake["capabilities"]["tools"]["listChanged"])
        self.assertEqual(handshake["instructions"], guide.server_instructions())
        names = [tool["name"] for tool in answers[2]["result"]["tools"]]
        self.assertIn("probe", names)
        self.assertIn(launch.TOOL_NAME, names)

    def test_a_modern_client_is_introduced_to_rapid2_while_it_is_closed(self):
        answers = self.converse(
            [
                modern("server/discover", request_id=1),
                modern("tools/list", request_id=2),
            ],
            catalog=probe_catalog(),
        )

        self.assertIn(MODERN_PROTOCOL_VERSIONS[0], answers[1]["result"]["supportedVersions"])
        self.assertEqual(answers[2]["result"]["resultType"], "complete")
        self.assertIn("probe", [tool["name"] for tool in answers[2]["result"]["tools"]])

    def test_a_tool_call_is_not_answered_from_the_catalog(self):
        answers = self.converse([legacy_initialize(), call("probe", 2)], catalog=probe_catalog())

        self.assertIn("error", answers[2])
        self.assertIn(launch.TOOL_NAME, answers[2]["error"]["message"])
        self.assertEqual(self.builds, [])

    def test_an_analysis_tool_is_answered_here_while_rapid2_is_closed(self):
        answers = self.converse(
            [legacy_initialize(), call("probe", 2)],
            catalog=analysis_catalog(),
            analysis_tools=self.relay_analysis({"probe": lambda arguments: {"answered_by": "relay"}}),
        )

        self.assertFalse(answers[2]["result"]["isError"])
        self.assertEqual(answers[2]["result"]["structuredContent"], {"answered_by": "relay"})

    def test_an_analysis_tool_this_relay_does_not_have_needs_rapid2(self):
        # Named by a window whose code differs from the relay's.
        answers = self.converse([legacy_initialize(), call("probe", 2)], catalog=analysis_catalog())

        self.assertTrue(answers[2]["result"]["isError"])
        self.assertIn("needs RAPID2 open", answers[2]["result"]["structuredContent"]["message"])

    def test_with_no_catalog_the_handshake_still_needs_rapid2(self):
        answers = self.converse([legacy_initialize()], catalog=None)

        self.assertIn("error", answers[1])

    # -- RAPID2 open -------------------------------------------------------
    def test_a_window_that_is_listening_answers_for_itself(self):
        self.start_window()

        answers = self.converse(
            [legacy_initialize(), {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}],
            catalog=probe_catalog(server_version="9.9.9"),
        )

        # The window introduces itself as 0.0.0; the catalog would have said 9.9.9.
        self.assertEqual(answers[1]["result"]["serverInfo"]["version"], "0.0.0")
        names = [tool["name"] for tool in answers[2]["result"]["tools"]]
        self.assertEqual(names, ["probe"])

    def test_a_window_that_is_listening_answers_its_analysis_tools(self):
        self.start_window()

        answers = self.converse(
            [legacy_initialize(), call("probe", 2)],
            catalog=analysis_catalog(),
            analysis_tools=self.relay_analysis({"probe": lambda arguments: {"answered_by": "relay"}}),
        )

        self.assertEqual(answers[2]["result"]["structuredContent"], {"echo": {}})
        self.assertEqual(self.builds, [])

    def test_a_window_that_refuses_is_not_covered_up(self):
        # Reached, and refused: the token on file is wrong. That is RAPID2's
        # answer, and the client has to see it rather than a cheerful
        # introduction from the catalog.
        self.start_window(token="wrong")

        answers = self.converse([legacy_initialize()], catalog=probe_catalog())

        self.assertIn("error", answers[1])
        self.assertIn("access token", answers[1]["error"]["message"])

    # -- rapid2_launch -----------------------------------------------------
    def test_launch_is_answered_here_even_with_a_window_listening(self):
        self.start_window()

        answers = self.converse([legacy_initialize(), call(launch.TOOL_NAME, 2)], catalog=probe_catalog())

        self.assertFalse(answers[2]["result"]["isError"])
        self.assertEqual(self.launches, [{}])

    def test_launch_is_answered_here_with_no_catalog_too(self):
        # A client can hold rapid2_launch from a window's own list. Forwarded,
        # a call to start RAPID2 would meet the closed RAPID2 it is for.
        answers = self.converse([call(launch.TOOL_NAME, 1)], catalog=None)

        self.assertFalse(answers[1]["result"]["isError"])
        self.assertEqual(self.launches, [{}])


class EntryPointTests(unittest.TestCase):
    def test_a_relay_with_a_catalog_does_not_wait_for_rapid2(self):
        upstream, answers = relay_main.build(probe_catalog())

        self.assertEqual(upstream._first_contact_wait, 0.0)
        self.assertTrue(answers.answers_while_closed(legacy_initialize()))

    def test_a_relay_runs_only_the_tools_its_catalog_names_analysis_tools(self):
        _upstream, answers = relay_main.build(analysis_catalog())
        self.assertTrue(answers.answers_while_closed(call("probe", 1)))

        _upstream, answers = relay_main.build(probe_catalog())
        self.assertFalse(answers.answers_while_closed(call("probe", 1)))

    def test_a_relay_without_one_waits_as_it_always_did(self):
        upstream, answers = relay_main.build(None)

        self.assertGreater(upstream._first_contact_wait, 0.0)
        self.assertFalse(answers.answers_while_closed(legacy_initialize()))
        self.assertTrue(answers.answers_always(call(launch.TOOL_NAME, 1)))


class FindCatalogTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "data" / catalog_file.CATALOG_NAME

    def test_the_relay_reads_the_one_the_window_left(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(catalog_file.encode(probe_catalog()))
        with mock.patch.object(catalog_file, "runtime_catalog_path", return_value=self.path):
            found = local.find_catalog()

        self.assertEqual(found["server_version"], "9.9.9")

    def test_a_partial_catalog_is_no_catalog(self):
        document = probe_catalog()
        document["tools"] = []
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(catalog_file.encode(document))
        with mock.patch.object(catalog_file, "runtime_catalog_path", return_value=self.path):
            self.assertIsNone(local.find_catalog())

    def test_a_catalog_with_a_malformed_analysis_list_is_no_catalog(self):
        document = analysis_catalog()
        document[catalog_file.ANALYSIS_TOOLS_KEY] = "probe"
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(catalog_file.encode(document))
        with mock.patch.object(catalog_file, "runtime_catalog_path", return_value=self.path):
            self.assertIsNone(local.find_catalog())

    def test_no_file_is_no_catalog(self):
        with mock.patch.object(catalog_file, "runtime_catalog_path", return_value=self.path):
            self.assertIsNone(local.find_catalog())


class LocalAnalysisTests(unittest.TestCase):
    """The analysis tools the relay runs: built once, on the first call, and closed."""

    def setUp(self):
        self.sessions = []

    def build(self):
        time.sleep(0.05)  # long enough for calls arriving together to overlap
        session = FakeSession()
        self.sessions.append(session)
        return session, {"probe": lambda arguments: {"ok": True}}

    def test_calls_arriving_together_build_the_tools_once(self):
        answers = local.LocalAnswers(analysis_catalog(), analysis_tools=self.build)
        replies = []
        threads = [
            threading.Thread(target=lambda index=index: replies.append(answers.answer(call("probe", index))))
            for index in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)

        self.assertEqual(len(self.sessions), 1)
        self.assertEqual([reply["result"]["structuredContent"] for reply in replies], [{"ok": True}] * 4)

    def test_closing_closes_the_runs_and_builds_nothing_first(self):
        answers = local.LocalAnswers(analysis_catalog(), analysis_tools=self.build)
        answers.close()
        self.assertEqual(self.sessions, [])

        answers.answer(call("probe", 1))
        answers.close()
        self.assertTrue(self.sessions[0].closed)

    def test_a_call_with_a_name_that_is_not_text_is_not_taken(self):
        answers = local.LocalAnswers(analysis_catalog(), analysis_tools=self.build)
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": ["probe"]}}

        self.assertFalse(answers.answers_while_closed(payload))


class RelayAnalysisToolsTests(unittest.TestCase):
    """The relay's own analysis tools, from a catalog of the window's real registry."""

    def setUp(self):
        from RAPID2_Agilent.mcp.tools import AnalysisSession, analysis_tools, build_registry

        session = AnalysisSession()
        self.addCleanup(session.close)
        tools = build_registry(None, session, run_on_main_thread=lambda run: run())
        self.answers = local.LocalAnswers(
            catalog_file.catalog_document(
                tools.describe_all(),
                server_name="rapid2-agilent",
                server_version="0.0.0",
                toolset_version=1,
                analysis_tools=[tool.name for tool in analysis_tools(session)],
            )
        )
        self.addCleanup(self.answers.close)

    def tool_call(self, name, **arguments):
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
        return payload, self.answers.answer(payload)

    def test_an_analysis_tool_is_answered_with_the_windows_code(self):
        payload, reply = self.tool_call("rapid2_calculate_ion_mz", formula="C6H12O6", adduct="[M-H]-")

        self.assertTrue(self.answers.answers_while_closed(payload))
        self.assertFalse(reply["result"]["isError"])
        self.assertAlmostEqual(reply["result"]["structuredContent"]["mz"], 179.056112, places=6)

    def test_a_wrong_path_is_an_answer_too(self):
        _payload, reply = self.tool_call("rapid2_describe_run", path=tempfile.gettempdir())

        self.assertTrue(reply["result"]["isError"])
        self.assertIn("Not an Agilent .d folder", reply["result"]["structuredContent"]["message"])

    def test_arguments_are_checked_against_the_catalogs_schema(self):
        _payload, reply = self.tool_call("rapid2_calculate_ion_mz", formula="C6H12O6")

        self.assertIn("'adduct' is required", reply["error"]["message"])

    def test_a_viewer_tool_still_needs_a_window(self):
        payload = call("rapid2_get_status", 1)

        self.assertFalse(self.answers.answers_while_closed(payload))

    def test_the_analysis_code_is_loaded_on_the_first_call_only(self):
        # The handshake must not pay for numpy, the reader and scipy.
        script = (
            "import sys\n"
            "from RAPID2_Agilent.mcp import catalog\n"
            "from RAPID2_Agilent.mcp.relay import local\n"
            "schema = {'type': 'object', 'properties': {'formula': {'type': 'string'}, 'adduct': {'type': 'string'}}}\n"
            "document = catalog.catalog_document([{'name': 'rapid2_calculate_ion_mz', 'inputSchema': schema}],\n"
            "    server_name='rapid2-agilent', server_version='0', toolset_version=1,\n"
            "    analysis_tools=['rapid2_calculate_ion_mz'])\n"
            "answers = local.LocalAnswers(document)\n"
            "answers.answer({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-06-18'}})\n"
            "answers.answer({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'})\n"
            "print('RAPID2_Agilent.mcp.tools' in sys.modules)\n"
            "reply = answers.answer({'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call',\n"
            "    'params': {'name': 'rapid2_calculate_ion_mz', 'arguments': {'formula': 'C6H12O6', 'adduct': '[M-H]-'}}})\n"
            "print('RAPID2_Agilent.mcp.tools' in sys.modules, reply['result']['isError'])\n"
            "answers.close()\n"
        )
        finished = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

        self.assertEqual(finished.stdout.split(), ["False", "True", "False"])


class LauncherTests(unittest.TestCase):
    """The handler, with the world replaced: nothing here starts anything."""

    def build(self, *, reachable, running=False, wait=1.0):
        self.started = []
        self.now = 0.0
        replies = iter(reachable)

        def clock():
            return self.now

        def sleep(seconds):
            self.now += seconds

        return launch.Launcher(
            lambda: next(replies),
            command=lambda: (["app"], None),
            start=lambda argv, cwd: self.started.append((argv, cwd)),
            window_running=lambda: running,
            windows=lambda: [{"instance": "A"}],
            wait_seconds=wait,
            poll_seconds=0.25,
            clock=clock,
            sleep=sleep,
        )

    def test_an_open_rapid2_is_left_as_it_is(self):
        answer = self.build(reachable=[True])({})

        self.assertFalse(answer["started"])
        self.assertEqual(self.started, [])

    def test_a_closed_rapid2_is_started_and_waited_for(self):
        answer = self.build(reachable=[False, False, True])({})

        self.assertTrue(answer["started"])
        self.assertEqual(answer["windows"], [{"instance": "A"}])
        self.assertEqual(self.started, [(["app"], None)])

    def test_an_open_window_with_access_off_is_not_started_twice(self):
        with self.assertRaises(ToolError) as caught:
            self.build(reachable=[False], running=True)({})

        self.assertIn("access is off", str(caught.exception))
        self.assertEqual(self.started, [])

    def test_a_window_that_never_takes_requests_is_reported_as_access_off(self):
        launcher_ = self.build(reachable=[False] * 10)
        launcher_._window_running = iter([False, True]).__next__

        with self.assertRaises(ToolError) as caught:
            launcher_({})

        self.assertIn("not taking requests", str(caught.exception))
        self.assertEqual(len(self.started), 1)

    def test_nothing_opening_at_all_is_reported(self):
        with self.assertRaises(ToolError) as caught:
            self.build(reachable=[False] * 10)({})

        self.assertIn("no window opened", str(caught.exception))

    def test_a_start_that_fails_is_reported(self):
        launcher_ = self.build(reachable=[False])
        launcher_._start = mock.Mock(side_effect=OSError("denied"))

        with self.assertRaises(ToolError) as caught:
            launcher_({})

        self.assertIn("denied", str(caught.exception))

    def test_two_calls_at_once_start_rapid2_once(self):
        started = []

        def start(argv, cwd):
            time.sleep(0.05)
            started.append(argv)

        launcher_ = launch.Launcher(
            lambda: False,
            command=lambda: (["app"], None),
            start=start,
            window_running=lambda: bool(started),
            windows=lambda: [],
            wait_seconds=0.1,
            poll_seconds=0.01,
        )
        answers = []

        def attempt():
            try:
                launcher_({})
            except ToolError as exc:
                answers.append(str(exc))

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)

        self.assertEqual(started, [["app"]])
        self.assertTrue(any("nothing was started" in answer for answer in answers), answers)

    def test_rapid2_is_started_with_this_interpreter(self):
        argv, cwd = launch.application_command()

        self.assertEqual(argv, [sys.executable, "-m", "RAPID2_Agilent.main"])
        self.assertIsNone(cwd)

    def test_the_standalone_relay_starts_the_window_beside_it(self):
        app = Path("/Applications/RAPID2_Agilent.app/Contents/MacOS")
        with mock.patch.object(sys, "frozen", True, create=True), \
                mock.patch.object(sys, "executable", str(app / "rapid2-agilent-mcp")), \
                mock.patch.object(sys, "platform", "darwin"):
            argv, cwd = launch.application_command()

        self.assertEqual(argv, [str(app / "rapid2-agilent")])
        self.assertIsNone(cwd)

    @unittest.skipIf(sys.platform == "win32", "the POSIX branch of start_detached")
    def test_the_standalone_window_is_started_as_an_application_of_its_own(self):
        # Not as the frozen relay's child, which it outlives.
        with mock.patch.object(sys, "frozen", True, create=True), \
                mock.patch.object(launch.subprocess, "Popen") as popen:
            launch.start_detached(["app"], None)

        self.assertEqual(popen.call_args.kwargs["env"]["PYINSTALLER_RESET_ENVIRONMENT"], "1")

    @unittest.skipIf(sys.platform == "win32", "the POSIX branch of start_detached")
    def test_from_source_the_window_inherits_the_environment(self):
        with mock.patch.object(launch.subprocess, "Popen") as popen:
            launch.start_detached(["app"], None)

        self.assertIsNone(popen.call_args.kwargs["env"])


class PresenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_no_window_is_no_window(self):
        self.assertFalse(presence.window_running(self.root))

    def test_a_held_slot_is_an_open_window_until_released(self):
        claim = presence.hold(self.root)
        self.addCleanup(claim.release)

        self.assertTrue(presence.window_running(self.root))
        claim.release()
        self.assertFalse(presence.window_running(self.root))

    def test_two_windows_take_two_slots(self):
        first, second = presence.hold(self.root), presence.hold(self.root)
        self.addCleanup(first.release)
        self.addCleanup(second.release)

        self.assertNotEqual(first.number, second.number)

    def test_asking_leaves_no_lock_behind(self):
        claim = presence.hold(self.root)
        claim.release()
        presence.window_running(self.root)

        again = presence.hold(self.root)
        self.addCleanup(again.release)
        self.assertEqual(again.number, claim.number)

    def test_a_window_in_another_process_is_seen(self):
        script = (
            "import sys, time; from RAPID2_Agilent.mcp import presence; "
            "claim = presence.hold(sys.argv[1]); print('held', flush=True); time.sleep(30)"
        )
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(self.root)], stdout=subprocess.PIPE, text=True
        )
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        self.assertEqual(child.stdout.readline().strip(), "held")

        self.assertTrue(presence.window_running(self.root))
        child.kill()
        child.wait()
        self.assertFalse(presence.window_running(self.root))


if __name__ == "__main__":
    unittest.main()
