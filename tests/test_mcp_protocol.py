# -*- coding: utf-8 -*-

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from RAPID2_Agilent.mcp.protocol import (
    ASSUMED_PROTOCOL_VERSION,
    HEADER_MISMATCH,
    IMAGES_KEY,
    INVALID_PARAMS,
    LATEST_LEGACY_PROTOCOL_VERSION,
    LATEST_PROTOCOL_VERSION,
    LEGACY_PROTOCOL_VERSIONS,
    MODERN_PROTOCOL_VERSIONS,
    METHOD_NOT_FOUND,
    META_CLIENT_CAPABILITIES,
    META_CLIENT_INFO,
    META_PROTOCOL_VERSION,
    META_SERVER_INFO,
    RESULT_TYPE_COMPLETE,
    SUPPORTED_PROTOCOL_VERSIONS,
    UNSUPPORTED_PROTOCOL_VERSION,
    Dispatcher,
    Tool,
    ToolError,
    ToolRegistry,
    decode_header_value,
    encode_header_value,
    negotiate_protocol_version,
    protocol_version_is_supported,
    unsupported_keywords,
    validate_arguments,
)


def build_dispatcher(handler=None, name="probe"):
    registry = ToolRegistry()
    registry.add(
        Tool(
            name=name,
            title="Probe",
            description="Test tool.",
            input_schema={"type": "object", "properties": {}},
            handler=handler or (lambda arguments: {"echo": arguments}),
        )
    )
    return Dispatcher(registry, server_name="rapid2", server_version="0.0.0")


class VersionNegotiationTests(unittest.TestCase):
    def test_the_assumed_header_version_is_one_we_speak(self):
        # A request with no MCP-Protocol-Version header is read as this
        # revision, so the transport must not then refuse it.
        self.assertIn(ASSUMED_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS)
        self.assertEqual(LATEST_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS[0])

    def test_a_supported_legacy_version_is_agreed_to(self):
        for version in LEGACY_PROTOCOL_VERSIONS:
            self.assertEqual(negotiate_protocol_version(version), version)

    def test_an_unknown_version_gets_our_latest_as_a_counter_offer(self):
        for requested in ("2099-01-01", "2020-01-01", "", 7, None, {"a": 1}):
            self.assertEqual(
                negotiate_protocol_version(requested),
                LATEST_LEGACY_PROTOCOL_VERSION,
            )

    def test_a_handshake_naming_a_modern_version_is_counter_offered(self):
        # `initialize` is the legacy era by definition, so a handshake asking
        # for a revision that has no handshake is asking for two things that
        # do not go together. Echoing it back would promise per-request
        # semantics to a client that has just proved it does not use them.
        for version in MODERN_PROTOCOL_VERSIONS:
            self.assertEqual(
                negotiate_protocol_version(version),
                LATEST_LEGACY_PROTOCOL_VERSION,
            )

    def test_the_newest_version_spoken_is_a_modern_one(self):
        self.assertEqual(LATEST_PROTOCOL_VERSION, MODERN_PROTOCOL_VERSIONS[0])
        self.assertNotEqual(LATEST_PROTOCOL_VERSION, LATEST_LEGACY_PROTOCOL_VERSION)

    def test_a_missing_header_is_not_a_refusal(self):
        self.assertTrue(protocol_version_is_supported(None))

    def test_a_header_naming_an_unknown_revision_is_refused(self):
        self.assertFalse(protocol_version_is_supported("2099-01-01"))
        self.assertFalse(protocol_version_is_supported("not-a-version"))

    def test_a_header_is_read_without_its_surrounding_space(self):
        self.assertTrue(protocol_version_is_supported(f" {LATEST_PROTOCOL_VERSION} "))


class HandshakeTests(unittest.TestCase):
    def test_initialize_agrees_to_a_version_the_client_and_server_share(self):
        # An older client that is still on a revision we speak keeps speaking
        # it; agreeing is not the same as answering with our newest.
        older = SUPPORTED_PROTOCOL_VERSIONS[-1]
        dispatcher = build_dispatcher()

        response = dispatcher.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": older,
                    "clientInfo": {"name": "older-client"},
                },
            }
        )

        self.assertEqual(response["result"]["protocolVersion"], older)
        self.assertEqual(response["result"]["serverInfo"]["name"], "rapid2")
        self.assertEqual(dispatcher.client_info["name"], "older-client")

    def test_initialize_counter_offers_rather_than_echoing_an_unknown_version(self):
        # Echoing "2099-01-01" back would claim we speak a revision nobody
        # here has read. The client is told what we do speak and decides.
        dispatcher = build_dispatcher()

        response = dispatcher.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2099-01-01",
                    "clientInfo": {"name": "future-client"},
                },
            }
        )

        self.assertEqual(
            response["result"]["protocolVersion"], LATEST_LEGACY_PROTOCOL_VERSION
        )
        self.assertEqual(
            dispatcher.protocol_version, LATEST_LEGACY_PROTOCOL_VERSION
        )

    def test_initialize_without_a_version_falls_back(self):
        dispatcher = build_dispatcher()

        response = dispatcher.dispatch(
            {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}}
        )

        self.assertEqual(
            response["result"]["protocolVersion"], LATEST_LEGACY_PROTOCOL_VERSION
        )
        self.assertTrue(dispatcher.initialized)

    def test_notifications_are_never_answered(self):
        dispatcher = build_dispatcher()

        self.assertIsNone(
            dispatcher.dispatch(
                {"jsonrpc": "2.0", "method": "notifications/initialized"}
            )
        )

    def test_unknown_method_is_reported_not_raised(self):
        dispatcher = build_dispatcher()

        response = dispatcher.dispatch(
            {"jsonrpc": "2.0", "id": 7, "method": "nonsense/method"}
        )

        self.assertEqual(response["error"]["code"], METHOD_NOT_FOUND)

    def test_ping_is_answered(self):
        dispatcher = build_dispatcher()

        response = dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "ping"})

        self.assertEqual(response["result"], {})


class ToolCallTests(unittest.TestCase):
    def test_tools_list_reports_registered_tools(self):
        dispatcher = build_dispatcher()

        response = dispatcher.dispatch(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        )

        tools = response["result"]["tools"]
        self.assertEqual([tool["name"] for tool in tools], ["probe"])
        self.assertIn("inputSchema", tools[0])

    def test_tool_result_carries_text_and_structured_content(self):
        dispatcher = build_dispatcher(handler=lambda arguments: {"ok": True})

        response = dispatcher.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "probe", "arguments": {}},
            }
        )

        result = response["result"]
        self.assertFalse(result["isError"])
        self.assertEqual(result["structuredContent"], {"ok": True})
        self.assertEqual(json.loads(result["content"][0]["text"]), {"ok": True})

    def test_a_rendered_image_travels_as_an_image_block(self):
        dispatcher = build_dispatcher(
            handler=lambda arguments: {
                "width": 8,
                IMAGES_KEY: [{"data": "QUJD", "mimeType": "image/png"}],
            }
        )

        result = self.call(dispatcher)["result"]

        self.assertEqual(result["content"][0]["type"], "text")
        self.assertEqual(
            result["content"][1],
            {"type": "image", "data": "QUJD", "mimeType": "image/png"},
        )

    def test_the_image_is_kept_out_of_the_structured_content(self):
        # A client reads structuredContent as data and may hold on to it.
        # A megabyte of base64 copied in there buries the fields the caller
        # asked for, in both halves of the answer.
        dispatcher = build_dispatcher(
            handler=lambda arguments: {
                "width": 8,
                IMAGES_KEY: [{"data": "QUJD", "mimeType": "image/png"}],
            }
        )

        result = self.call(dispatcher)["result"]

        self.assertEqual(result["structuredContent"], {"width": 8})
        self.assertEqual(json.loads(result["content"][0]["text"]), {"width": 8})

    def test_an_image_with_no_data_is_dropped_rather_than_sent_empty(self):
        dispatcher = build_dispatcher(
            handler=lambda arguments: {"ok": True, IMAGES_KEY: [{}, None, ""]}
        )

        result = self.call(dispatcher)["result"]

        self.assertEqual(len(result["content"]), 1)

    def test_a_result_with_no_images_is_unchanged(self):
        dispatcher = build_dispatcher(handler=lambda arguments: {"ok": True})

        result = self.call(dispatcher)["result"]

        self.assertEqual(len(result["content"]), 1)
        self.assertEqual(result["structuredContent"], {"ok": True})

    def call(self, dispatcher):
        return dispatcher.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "probe", "arguments": {}},
            }
        )

    def test_arguments_default_to_an_empty_object(self):
        dispatcher = build_dispatcher()

        response = dispatcher.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "probe"},
            }
        )

        self.assertEqual(response["result"]["structuredContent"], {"echo": {}})

    def test_unknown_tool_is_an_invalid_params_error(self):
        dispatcher = build_dispatcher()

        response = dispatcher.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "nope"},
            }
        )

        self.assertEqual(response["error"]["code"], INVALID_PARAMS)

    def test_tool_error_becomes_a_readable_failed_result(self):
        # A refusal is something the caller has to read and act on, so it must
        # not be buried in a protocol-level error.
        def refuse(_arguments):
            raise ToolError("RAPID2 is waiting for an answer in a dialog.")

        dispatcher = build_dispatcher(handler=refuse)

        response = dispatcher.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "probe", "arguments": {}},
            }
        )

        result = response["result"]
        self.assertTrue(result["isError"])
        self.assertIn("dialog", result["structuredContent"]["message"])

    def test_unexpected_tool_exception_does_not_escape(self):
        def explode(_arguments):
            raise ZeroDivisionError("boom")

        dispatcher = build_dispatcher(handler=explode)

        response = dispatcher.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {"name": "probe", "arguments": {}},
            }
        )

        self.assertTrue(response["result"]["isError"])
        self.assertIn("ZeroDivisionError", response["result"]["structuredContent"]["message"])

    def test_non_object_arguments_are_rejected(self):
        dispatcher = build_dispatcher()

        response = dispatcher.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {"name": "probe", "arguments": ["not", "an", "object"]},
            }
        )

        self.assertEqual(response["error"]["code"], INVALID_PARAMS)


CHECKED_SCHEMA = {
    "type": "object",
    "properties": {
        "width": {"type": "number", "exclusiveMinimum": 0},
        "kind": {"type": "string", "enum": ["sample", "marker"]},
        "expected_stamp": {"type": "string"},
    },
    "required": ["width"],
    "additionalProperties": False,
}


def build_checked_dispatcher(seen):
    registry = ToolRegistry()
    registry.add(
        Tool(
            name="probe",
            title="Probe",
            description="Test tool.",
            input_schema=CHECKED_SCHEMA,
            handler=lambda arguments: seen.append(arguments) or {"ok": True},
        )
    )
    return Dispatcher(registry, server_name="rapid2", server_version="0.0.0")


class ArgumentValidationTests(unittest.TestCase):
    """The schema `tools/list` hands out is the one calls are held to."""

    def setUp(self):
        self.seen = []
        self.dispatcher = build_checked_dispatcher(self.seen)

    def call(self, arguments, name="probe"):
        return self.dispatcher.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )

    def assertRefused(self, response, *fragments):
        """An invalid call is a protocol error, and the handler never ran."""

        self.assertEqual(response["error"]["code"], INVALID_PARAMS)
        for fragment in fragments:
            self.assertIn(fragment, response["error"]["message"])
        self.assertEqual(self.seen, [])

    def test_a_value_outside_its_declared_range_is_refused(self):
        # The point of the whole pass: a width the schema says cannot be
        # negative must not reach a handler that would change the document.
        self.assertRefused(
            self.call({"width": -10}), "width", "greater than 0", "-10"
        )

    def test_a_value_of_the_wrong_type_is_refused(self):
        self.assertRefused(self.call({"width": "10"}), "width", "must be number")

    def test_a_boolean_is_not_a_number(self):
        self.assertRefused(self.call({"width": True}), "width", "not boolean")

    def test_a_non_finite_number_is_refused(self):
        # json.loads accepts the NaN literal, and NaN compares false to every
        # bound, so a range check alone would let it through.
        self.assertRefused(
            self.call({"width": float("nan")}), "width", "not NaN"
        )
        self.assertRefused(
            self.call({"width": float("inf")}), "width", "not Infinity"
        )

    def test_a_missing_required_argument_is_refused(self):
        self.assertRefused(self.call({}), "'width' is required")

    def test_a_value_outside_an_enum_is_refused(self):
        self.assertRefused(self.call({"width": 5, "kind": "sample "}), "kind")

    def test_an_unknown_argument_is_refused_by_name(self):
        # A misspelled `expected_stamp` would otherwise be dropped in silence,
        # taking the staleness guard with it.
        response = self.call({"width": 5, "token": "abc"})

        self.assertRefused(response, "'token'", "expected_stamp")

    def test_an_explicit_null_optional_argument_is_taken_as_omitted(self):
        # Several clients spell "not sending this" as an explicit null.
        response = self.call({"width": 5, "expected_stamp": None})

        self.assertEqual(response["result"]["structuredContent"], {"ok": True})
        self.assertEqual(self.seen, [{"width": 5, "expected_stamp": None}])

    def test_a_valid_call_is_untouched(self):
        response = self.call({"width": 5, "kind": "marker"})

        self.assertEqual(response["result"]["structuredContent"], {"ok": True})

    def test_a_non_string_tool_name_is_an_error_not_a_crash(self):
        # The registry is a dict, so an unhashable name raises rather than
        # missing. Escaping here would cost a batch every other reply in it.
        self.assertRefused(self.call({}, name=["probe"]), "must be a string")

    def test_a_bad_name_in_a_batch_leaves_the_rest_answered(self):
        responses = self.dispatcher.dispatch_payload(
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": {"not": "a name"}},
                },
                {"jsonrpc": "2.0", "id": 2, "method": "ping"},
            ]
        )

        self.assertEqual([response["id"] for response in responses], [1, 2])
        self.assertEqual(responses[0]["error"]["code"], INVALID_PARAMS)
        self.assertEqual(responses[1]["result"], {})

    def test_an_argument_that_has_to_be_fetched_says_how(self):
        # Rejection happens before any handler runs, so advice left in a
        # handler is advice the caller never reads. An argument it cannot
        # invent -- one that comes from an earlier call -- has to carry the
        # way out in the refusal itself.
        schema = {
            "type": "object",
            "properties": {
                "ticket": {
                    "type": "string",
                    "guidance": "Call issue_a_ticket first.",
                }
            },
            "required": ["ticket"],
        }

        missing = validate_arguments(schema, {})
        wrong_type = validate_arguments(schema, {"ticket": 7})

        self.assertEqual(
            missing, "'ticket' is required. Call issue_a_ticket first."
        )
        self.assertIn("Call issue_a_ticket first.", wrong_type)
        self.assertIn("must be string", wrong_type)

    def test_a_property_without_guidance_gains_no_stray_space(self):
        self.assertEqual(
            validate_arguments(
                {"type": "object", "properties": {}, "required": ["ticket"]}, {}
            ),
            "'ticket' is required.",
        )

    def test_guidance_is_a_keyword_the_validator_knows(self):
        # It enforces nothing, like `description`, but it is read rather than
        # ignored -- so it must not be reported as a declaration nobody holds
        # callers to.
        self.assertEqual(
            unsupported_keywords(
                {
                    "type": "object",
                    "properties": {"a": {"type": "string", "guidance": "do x"}},
                }
            ),
            set(),
        )

    def test_an_undeclared_property_is_not_second_guessed(self):
        # A schema that allows extras still allows them.
        self.assertIsNone(
            validate_arguments(
                {"type": "object", "properties": {}}, {"anything": ["at", "all"]}
            )
        )

    def test_unsupported_keywords_are_reported(self):
        self.assertEqual(
            unsupported_keywords(
                {
                    "type": "object",
                    "properties": {"a": {"type": "string", "pattern": "^x"}},
                }
            ),
            {"pattern"},
        )


class BatchTests(unittest.TestCase):
    def test_batch_returns_only_the_answerable_messages(self):
        dispatcher = build_dispatcher()

        response = dispatcher.dispatch_payload(
            [
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            ]
        )

        self.assertEqual(len(response), 1)
        self.assertEqual(response[0]["id"], 1)

    def test_batch_of_only_notifications_has_no_response(self):
        dispatcher = build_dispatcher()

        self.assertIsNone(
            dispatcher.dispatch_payload(
                [{"jsonrpc": "2.0", "method": "notifications/initialized"}]
            )
        )


class RegistryTests(unittest.TestCase):
    def test_duplicate_tool_names_are_refused(self):
        registry = ToolRegistry()
        tool = Tool("a", "A", "d", {}, handler=lambda _a: None)
        registry.add(tool)

        with self.assertRaises(ValueError):
            registry.add(tool)


def modern_meta(version=None, **extra):
    """The `_meta` a conforming modern client puts on every request."""

    meta = {
        META_PROTOCOL_VERSION: version or MODERN_PROTOCOL_VERSIONS[0],
        META_CLIENT_CAPABILITIES: {},
        META_CLIENT_INFO: {"name": "test-client", "version": "1.0.0"},
    }
    meta.update(extra)
    return meta


def modern(method, params=None, *, request_id=1, version=None):
    request = {"jsonrpc": "2.0", "id": request_id, "method": method}
    request["params"] = {**(params or {}), "_meta": modern_meta(version)}
    return request


class DiscoveryTests(unittest.TestCase):
    """`server/discover`: what the handshake used to say, asked for instead."""

    def discover(self, dispatcher=None, params=None):
        dispatcher = dispatcher or build_dispatcher()
        message = {"jsonrpc": "2.0", "id": 3, "method": "server/discover"}
        if params is not None:
            message["params"] = params
        return dispatcher.dispatch(message)["result"]

    def test_discover_reports_the_versions_and_capabilities(self):
        result = self.discover()

        self.assertEqual(result["resultType"], RESULT_TYPE_COMPLETE)
        self.assertEqual(
            result["supportedVersions"], list(SUPPORTED_PROTOCOL_VERSIONS)
        )
        self.assertIn("tools", result["capabilities"])
        self.assertEqual(
            result["_meta"][META_SERVER_INFO],
            {"name": "rapid2", "version": "0.0.0"},
        )

    def test_discover_offers_the_legacy_revisions_too(self):
        # A dual-era server does support them -- through `initialize` -- and
        # the specification's own example of this field lists a legacy
        # revision beside a modern one.
        result = self.discover()

        for version in LEGACY_PROTOCOL_VERSIONS:
            self.assertIn(version, result["supportedVersions"])

    def test_discover_carries_the_instructions(self):
        registry = ToolRegistry()
        dispatcher = Dispatcher(
            registry, server_name="rapid2", server_version="0.0.0",
            instructions="read me first",
        )

        self.assertEqual(
            self.discover(dispatcher)["instructions"], "read me first"
        )

    def test_discover_is_answered_without_any_metadata(self):
        # It is the request a client sends before it knows what it is talking
        # to. Refusing it for want of the metadata it exists to make possible
        # would leave its own question unanswerable.
        self.assertIn("supportedVersions", self.discover(params={}))
        self.assertIn("supportedVersions", self.discover())

    def test_discover_naming_an_unknown_version_says_what_is_spoken(self):
        dispatcher = build_dispatcher()

        response = dispatcher.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "server/discover",
                "params": {"_meta": modern_meta("1999-01-01")},
            }
        )

        self.assertEqual(response["error"]["code"], UNSUPPORTED_PROTOCOL_VERSION)
        self.assertEqual(
            response["error"]["data"]["supported"],
            list(SUPPORTED_PROTOCOL_VERSIONS),
        )
        self.assertEqual(response["error"]["data"]["requested"], "1999-01-01")


class ModernRequestTests(unittest.TestCase):
    """The era with no handshake: every request says what it is on its own."""

    def test_a_modern_result_declares_its_type_and_the_server(self):
        dispatcher = build_dispatcher()

        result = dispatcher.dispatch(modern("tools/list"))["result"]

        self.assertEqual(result["resultType"], RESULT_TYPE_COMPLETE)
        self.assertEqual(
            result["_meta"][META_SERVER_INFO],
            {"name": "rapid2", "version": "0.0.0"},
        )
        self.assertTrue(result["tools"])

    def test_a_tool_call_answers_the_same_content_in_either_era(self):
        # The whole point of a tools-only server surviving the split: the two
        # eras wrap the answer differently and the answer itself is identical.
        dispatcher = build_dispatcher()
        call = {"name": "probe", "arguments": {}}

        legacy = dispatcher.dispatch(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": call}
        )["result"]
        current = dispatcher.dispatch(modern("tools/call", call))["result"]

        self.assertEqual(current["structuredContent"], legacy["structuredContent"])
        self.assertEqual(current["content"], legacy["content"])

    def test_a_legacy_result_says_nothing_about_result_types(self):
        # Absent means "complete" to a modern reader, but a 2024-11-05 client
        # is not expecting the field at all.
        dispatcher = build_dispatcher()

        result = dispatcher.dispatch(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )["result"]

        self.assertNotIn("resultType", result)
        self.assertNotIn("_meta", result)

    def test_an_unsupported_version_is_refused_with_the_list(self):
        dispatcher = build_dispatcher()

        response = dispatcher.dispatch(modern("tools/list", version="1999-01-01"))

        self.assertEqual(response["error"]["code"], UNSUPPORTED_PROTOCOL_VERSION)
        self.assertEqual(
            response["error"]["data"]["supported"],
            list(SUPPORTED_PROTOCOL_VERSIONS),
        )

    def test_a_modern_request_without_client_capabilities_is_malformed(self):
        dispatcher = build_dispatcher()

        response = dispatcher.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {
                    "_meta": {
                        META_PROTOCOL_VERSION: MODERN_PROTOCOL_VERSIONS[0]
                    }
                },
            }
        )

        self.assertEqual(response["error"]["code"], INVALID_PARAMS)

    def test_client_info_is_wanted_but_not_required(self):
        # The specification marks it SHOULD, and nothing here reads it.
        dispatcher = build_dispatcher()

        response = dispatcher.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {
                    "_meta": {
                        META_PROTOCOL_VERSION: MODERN_PROTOCOL_VERSIONS[0],
                        META_CLIENT_CAPABILITIES: {},
                    }
                },
            }
        )

        self.assertIn("result", response)

    def test_a_legacy_version_in_meta_is_served_as_legacy(self):
        # The era comes from the version, not from the presence of `_meta`.
        dispatcher = build_dispatcher()

        result = dispatcher.dispatch(
            modern("tools/list", version=LEGACY_PROTOCOL_VERSIONS[0])
        )["result"]

        self.assertNotIn("resultType", result)

    def test_a_modern_client_never_has_to_initialize(self):
        # There is no handshake to be missing: the first thing a modern client
        # sends may be the work itself.
        dispatcher = build_dispatcher()

        response = dispatcher.dispatch(modern("tools/call", {"name": "probe"}))

        self.assertFalse(response["result"]["isError"])
        self.assertFalse(dispatcher.initialized)


class MirroredHeaderValueTests(unittest.TestCase):
    def test_a_plain_value_travels_as_it_is(self):
        self.assertEqual(encode_header_value("rapid2_guide"), "rapid2_guide")

    def test_a_value_that_cannot_be_a_header_is_wrapped(self):
        for value in ("Hello, \u4e16\u754c", " padded ", "line1\nline2"):
            encoded = encode_header_value(value)
            self.assertTrue(encoded.startswith("=?base64?"), value)
            self.assertEqual(decode_header_value(encoded), value)

    def test_a_value_that_merely_looks_wrapped_is_wrapped(self):
        # Otherwise the reader cannot tell an encoding from a literal.
        literal = "=?base64?literal?="
        encoded = encode_header_value(literal)

        self.assertNotEqual(encoded, literal)
        self.assertEqual(decode_header_value(encoded), literal)

    def test_an_undecodable_wrapper_is_left_alone(self):
        # It then fails the comparison it was about to be used in, which is
        # the safe outcome: no value is invented for it.
        mangled = "=?base64?not base64!?="

        self.assertEqual(decode_header_value(mangled), mangled)


if __name__ == "__main__":
    unittest.main()
