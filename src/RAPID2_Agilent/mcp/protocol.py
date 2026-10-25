# -*- coding: utf-8 -*-

"""JSON-RPC 2.0 dispatch for the MCP tool surface. No Qt, no sockets."""

from __future__ import annotations

import base64
import json
import logging
import math
from dataclasses import dataclass, field


log = logging.getLogger(__name__)


# The methods a tools-only server has to answer: `server/discover`,
# `initialize`, `ping`, `tools/list` and `tools/call`.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# Allocated by the specification itself, in the range it reserves for its own
# use (-32020 to -32099). Only the meanings given there are permitted, which
# is why they are spelled out here rather than reused for anything local.
HEADER_MISMATCH = -32020
MISSING_CLIENT_CAPABILITY = -32021
UNSUPPORTED_PROTOCOL_VERSION = -32022

# MCP split in two at 2026-07-28. Everything before it opens with an
# `initialize` handshake and remembers what that settled; from it onwards there
# is no handshake at all, and every request carries its own protocol version,
# client identity and client capabilities in `_meta`. The specification calls
# the two eras modern and legacy, and a server that answers both -- which this
# one does -- a dual-era server.
#
# What decides the era is the version on the request, never the connection: the
# same process answers a modern request and a legacy one, and must not let
# either colour the other.
#
# A tools-only server is the part of MCP that has moved least across all of
# these: `tools/list` and `tools/call` are the same messages carrying the same
# shapes throughout, with the modern revision adding `resultType` to the
# results and `_meta` to the requests. So each revision here is genuinely
# spoken rather than merely tolerated.
#
# When a new revision appears, the entry is added here after the tool surface
# has been read against it -- not before.
MODERN_PROTOCOL_VERSIONS = ("2026-07-28",)
LEGACY_PROTOCOL_VERSIONS = (
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
SUPPORTED_PROTOCOL_VERSIONS = MODERN_PROTOCOL_VERSIONS + LEGACY_PROTOCOL_VERSIONS

# The newest revision spoken at all, and the newest one a handshake can settle
# on. Not the same value, and the difference matters: `initialize` exists only
# in the legacy era, so offering a modern revision as the counter-proposal to a
# handshake would promise a client semantics its handshake has already ruled
# out.
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]
LATEST_LEGACY_PROTOCOL_VERSION = LEGACY_PROTOCOL_VERSIONS[0]

# The `_meta` keys the modern era reserves. `protocolVersion` and
# `clientCapabilities` are required on every request, `clientInfo` is not;
# `serverInfo` is what a result sends back in place of the handshake that used
# to carry it.
META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
META_CLIENT_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"

# What a modern result says about itself. Only "complete" is ever sent from
# here: the other value in the core protocol, "input_required", belongs to the
# multi round-trip pattern this server does not implement -- a question for the
# user comes back as a refusal payload instead.
RESULT_TYPE_COMPLETE = "complete"

# What a client is told it may hold on to, and for how long. Required on every
# complete result of `server/discover` and `tools/list`, not optional.
#
# Both are public: neither answer varies by who is asking or by which window
# answered. The tool list is built the same way in every window of a build, so
# a cache shared between callers cannot leak anything or mislead anyone.
#
# The two lifetimes differ because what would go stale differs. Discovery
# describes the build and changes only when RAPID2 is replaced; the tool list is
# just as stable, but a client holding a stale one calls a tool that is not
# there, so it is given the shorter hint.
CACHE_SCOPE_PUBLIC = "public"
DISCOVER_CACHE_TTL_MS = 3_600_000
TOOLS_LIST_CACHE_TTL_MS = 300_000

# Errors the specification pins to an HTTP status. Kept beside the codes rather
# than in the transport, because it is the code that decides the status and not
# the other way round: a caller that learns to recognise one has learned the
# other.
HTTP_STATUS_FOR_ERROR = {
    HEADER_MISMATCH: 400,
    MISSING_CLIENT_CAPABILITY: 400,
    UNSUPPORTED_PROTOCOL_VERSION: 400,
}

# What an HTTP request carrying no `MCP-Protocol-Version` header is taken to
# mean. Fixed by the specification rather than chosen: the header was introduced
# in 2025-06-18, so a client that omits it is speaking the revision before that
# one. A stdio client never sends it at all, and the relay that carries stdio
# here sends it only for a request that declares a version of its own, which is
# why an absent header is not an error.
ASSUMED_PROTOCOL_VERSION = "2025-03-26"


# The header names the modern era mirrors selected body fields into, so that
# whatever sits between a client and a server can route on them without
# parsing JSON. Named here rather than in the transport that enforces them:
# the relay has to send exactly what the endpoint checks, and one of them
# reaching for the other's private spelling is how the two come to disagree.
PROTOCOL_VERSION_HEADER = "MCP-Protocol-Version"
METHOD_HEADER = "Mcp-Method"
NAME_HEADER = "Mcp-Name"
# How a mirrored value that will not survive an HTTP header travels instead.
# Case-sensitive, and fixed by the specification.
BASE64_PREFIX = "=?base64?"
BASE64_SUFFIX = "?="


def decode_header_value(value):
    """A mirrored header value, with the Base64 sentinel undone if present.

    A tool name or a resource URI need not be plain ASCII, and an HTTP header
    is. The specification's answer is to wrap such a value in `=?base64?...?=`,
    and to have a client wrap even a plain value that happens to look like the
    sentinel -- so an undecodable one is left exactly as it arrived and fails
    the comparison it was about to be used in, rather than being guessed at.
    """

    value = value.strip()
    if not (value.startswith(BASE64_PREFIX) and value.endswith(BASE64_SUFFIX)):
        return value
    encoded = value[len(BASE64_PREFIX) : -len(BASE64_SUFFIX)]
    try:
        return base64.b64decode(encoded, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return value


def encode_header_value(value):
    """One mirrored value, Base64-wrapped when it cannot travel as it is.

    The wrapping is also applied to a plain value that happens to look like a
    wrapped one, which the specification requires so that a reader never has to
    guess whether an unwrapping was intended.
    """

    printable = value.isascii() and all(0x20 <= ord(ch) <= 0x7E for ch in value)
    looks_wrapped = value.startswith(BASE64_PREFIX) and value.endswith(BASE64_SUFFIX)
    if printable and value == value.strip() and not looks_wrapped:
        return value
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return f"{BASE64_PREFIX}{encoded}{BASE64_SUFFIX}"


class ToolError(Exception):
    """A tool could not run. Reported to the caller as a failed tool result."""


class ToolRefusal(Exception):
    """The operation did not happen, and the payload explains why.

    Distinct from ToolError: this is an expected outcome with structure the
    caller is meant to act on -- a question for the user, or a document that
    moved on since it was read -- rather than a plain failure message.
    """

    def __init__(self, payload):
        super().__init__(payload.get("message") or payload.get("status", "refused"))
        self.payload = payload


@dataclass(frozen=True)
class Tool:
    name: str
    title: str
    description: str
    input_schema: dict
    handler: object = field(compare=False, default=None)
    # Declared only where the answer really is a fixed shape. MCP holds a
    # server to an output schema it publishes -- "servers MUST provide
    # structured results that conform" -- so a schema that describes the
    # common case and not the awkward one is worse than none at all: the
    # client validates against it and rejects a result that was correct.
    output_schema: dict = None

    def describe(self):
        described = {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.output_schema is not None:
            described["outputSchema"] = self.output_schema
        return described


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def add(self, tool):
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def describe_all(self):
        return [self._tools[name].describe() for name in sorted(self._tools)]

    def get(self, name):
        return self._tools.get(name)

    def __len__(self):
        return len(self._tools)

    def __contains__(self, name):
        return name in self._tools


def _result(request_id, payload):
    return {"jsonrpc": "2.0", "id": request_id, "result": payload}


def is_answered_request(response):
    """Whether this answer means a client spoke MCP and RAPID2 replied to it.

    True only for a result. A malformed body, an unknown method, an
    unsupported protocol revision and a bad tool name are all answered with a
    JSON-RPC error, and none of them is evidence that anything is configured
    correctly -- which is the one question the caller of this asks.

    A tool that ran and failed does count: `isError` travels inside a result,
    the exchange itself worked, and a client that can call a tool has a
    working plugin. Batches count if any message in them was answered, and a
    notification -- answered with nothing at all -- does not.
    """

    if isinstance(response, list):
        return any(is_answered_request(item) for item in response)
    if not isinstance(response, dict):
        return False
    return "error" not in response and isinstance(response.get("result"), dict)


def error_response(request_id, code, message, data=None):
    error = {"code": code, "message": message}
    if data is not None:
        # Carried by the errors the modern era defines: which versions are
        # supported, which capability was missing. A client is expected to act
        # on it rather than only show it, so it is part of the answer and not
        # a detail folded into the message.
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def unsupported_version_error(request_id, requested):
    """What every modern request naming a revision we do not speak is answered
    with. The client picks one from `supported` and sends the request again."""

    return error_response(
        request_id,
        UNSUPPORTED_PROTOCOL_VERSION,
        f"Unsupported protocol version: {requested!r}.",
        data={
            "supported": list(SUPPORTED_PROTOCOL_VERSIONS),
            "requested": requested,
        },
    )


# Where a tool leaves images for `tool_result` to carry. Chosen with a leading
# underscore because it is not part of any tool's answer: it is lifted out into
# content blocks and never reaches the caller under this name.
IMAGES_KEY = "_images"


def _image_blocks(images):
    """Content blocks for what a tool rendered, skipping anything malformed."""

    blocks = []
    for image in images or ():
        if not isinstance(image, dict):
            continue
        data = image.get("data")
        if not isinstance(data, str) or not data:
            continue
        blocks.append(
            {
                "type": "image",
                "data": data,
                "mimeType": str(image.get("mimeType") or "image/png"),
            }
        )
    return blocks


def tool_result(payload, *, is_error=False):
    """The dual-shape result MCP expects: readable text plus structured data.

    A tool that rendered something leaves it under `IMAGES_KEY`, and it is
    lifted out here into image content blocks. It is removed from the payload
    on the way: base64 belongs in a content block the client can display, and
    a megabyte of it copied into `structuredContent` -- which a client reads as
    data and may keep -- would bury the fields the caller actually asked for.
    """

    images = ()
    if isinstance(payload, dict) and IMAGES_KEY in payload:
        payload = dict(payload)
        images = payload.pop(IMAGES_KEY)

    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            },
            *_image_blocks(images),
        ],
        "structuredContent": payload,
        "isError": bool(is_error),
    }


# -- argument validation --------------------------------------------------
# A tool that declares a constraint and then does not hold callers to it is
# worse than one that never declared it: the caller reads the declaration and
# builds on it. So every call is checked against the schema `tools/list` handed
# out, before the handler can touch a document.
#
# Hand-written rather than a schema library, because the tool schemas use these
# keywords and no others. `unsupported_keywords` is what keeps that honest: a
# schema that grows a keyword this does not enforce is caught by a test rather
# than quietly accepting anything.
SUPPORTED_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "description",
        "default",
        "enum",
        # The schema for each element of an array. Enforced element by
        # element, with the same keywords, so a tool taking a list of objects
        # holds callers to the shape of them.
        "items",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        # Carries no constraint, like "description" beside it. It is the one
        # sentence a caller that got this argument wrong needs in the answer:
        # rejection happens here, before any handler runs, so advice left in
        # the handler is advice the caller never sees. See _guidance.
        "guidance",
    }
)


def _is_number(value):
    # bool is an int in Python, and `true` is not a number in JSON. Non-finite
    # floats are barred here rather than by a bound, so that a property with no
    # declared range cannot be handed a NaN that compares false to everything.
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


_TYPE_CHECKS = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "number": _is_number,
    "integer": lambda value: _is_number(value) and isinstance(value, int),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}

_BOUNDS = (
    ("minimum", "at least", lambda value, bound: value >= bound),
    ("maximum", "at most", lambda value, bound: value <= bound),
    ("exclusiveMinimum", "greater than", lambda value, bound: value > bound),
    ("exclusiveMaximum", "less than", lambda value, bound: value < bound),
)


def _type_name(value):
    if isinstance(value, float) and not math.isfinite(value):
        # Named rather than called a float: "must be number, not float" would
        # read as a bug in the checker rather than a problem with the value.
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    for name in ("null", "boolean", "string", "array", "object", "integer"):
        if _TYPE_CHECKS[name](value):
            return name
    return "number" if _is_number(value) else type(value).__name__


def _check_value(property_schema, value):
    """What is wrong with one argument, as a sentence fragment, or None."""

    if not isinstance(property_schema, dict):
        # Nothing was declared about it, so there is nothing to hold it to.
        return None

    declared = property_schema.get("type")
    if declared is not None:
        wanted = [declared] if isinstance(declared, str) else list(declared)
        if not any(_TYPE_CHECKS.get(one, lambda _value: True)(value) for one in wanted):
            return f"must be {' or '.join(wanted)}, not {_type_name(value)}."

    allowed = property_schema.get("enum")
    if allowed is not None and value not in allowed:
        readable = ", ".join(repr(option) for option in allowed)
        return f"must be one of {readable} (got {value!r})."

    for keyword, phrase, holds in _BOUNDS:
        bound = property_schema.get(keyword)
        if bound is not None and _is_number(value) and not holds(value, bound):
            return f"must be {phrase} {bound} (got {value!r})."
    return None


def _guidance(property_schema):
    """The property's own advice for a caller that got it wrong, if any.

    Appended to the message rather than replacing it: what is wrong and what
    to do about it are two different things, and an argument the caller
    cannot produce out of thin air -- one that has to be fetched by another
    call first -- needs both.
    """

    hint = (property_schema or {}).get("guidance")
    return f" {hint}" if hint else ""


def _problem(property_schema, value):
    """What is wrong with one value, at any depth, or None.

    Nesting is checked with the same keywords as the top level rather than a
    second, weaker set: a tool that declares the shape of the objects in a
    list has to hold callers to it, or the declaration is decoration.
    """

    basic = _check_value(property_schema, value)
    if basic is not None:
        return basic
    if not isinstance(property_schema, dict):
        return None
    if isinstance(value, dict) and property_schema.get("properties") is not None:
        return _object_problem(property_schema, value, subject="This takes")
    items = property_schema.get("items")
    if items is not None and isinstance(value, list):
        for index, element in enumerate(value):
            found = _problem(items, element)
            if found is not None:
                return f"item {index} {found}"
    return None


def _object_problem(schema, values, subject="This tool takes"):
    """What is wrong with one object against one schema, or None."""

    properties = schema.get("properties") or {}
    required = set(schema.get("required") or ())

    for name in sorted(required):
        if name not in values:
            return f"{name!r} is required.{_guidance(properties.get(name))}"

    if schema.get("additionalProperties") is False:
        unknown = sorted(set(values) - set(properties))
        if unknown:
            # Named individually, with the real list beside them. A tool
            # that quietly drops what it does not recognise is one where a
            # misspelled `expected_stamp` changes a document the caller never
            # read, and says nothing about it.
            spelled = ", ".join(repr(name) for name in unknown)
            takes = ", ".join(sorted(properties)) or "no arguments"
            return f"unknown argument {spelled}. {subject}: {takes}."

    for name in sorted(values):
        value = values[name]
        if value is None and name not in required:
            continue
        found = _problem(properties.get(name), value)
        if found is not None:
            return f"{name!r} {found}{_guidance(properties.get(name))}"
    return None


def validate_arguments(schema, arguments):
    """Say what is wrong with one call's arguments, or None if nothing is."""

    if not isinstance(schema, dict):
        return None
    return _object_problem(schema, arguments)


def unsupported_keywords(schema):
    """Keywords in a schema that `validate_arguments` would ignore."""

    if not isinstance(schema, dict):
        return set()
    found = set(schema) - SUPPORTED_KEYWORDS
    for property_schema in (schema.get("properties") or {}).values():
        found |= unsupported_keywords(property_schema)
    found |= unsupported_keywords(schema.get("items"))
    return found


# -- version negotiation ---------------------------------------------------
def negotiate_protocol_version(requested):
    """The revision to answer one handshake with.

    A client opens with the newest revision it speaks. Where that is one of
    ours, it is the answer and both sides are done. Where it is not -- a client
    newer than this build, or older than anything here, or a value that is not a
    revision at all -- the answer is our own newest, which is the counter-offer
    the specification asks for. The client then decides whether it can live with
    it, and disconnects if it cannot.

    Echoing an unknown version back is the one answer there is no recovering
    from: it tells the client we speak a revision nobody here has read, so
    whatever that revision added is discovered to be missing at the point some
    later call relies on it, rather than at the handshake where it belongs.

    Only the legacy revisions are on offer, because only they have a handshake.
    A client that opens with `initialize` and names a modern revision is asking
    for two things that do not go together, and is counter-offered the newest
    revision that actually works the way it just asked.
    """

    if isinstance(requested, str) and requested in LEGACY_PROTOCOL_VERSIONS:
        return requested
    return LATEST_LEGACY_PROTOCOL_VERSION


def request_meta(params):
    """The `_meta` of one request, or an empty mapping if it has none."""

    meta = params.get("_meta") if isinstance(params, dict) else None
    return meta if isinstance(meta, dict) else {}


def _request_protocol_version(params):
    """The revision a request declares for itself, or None if it declares none.

    None is how a legacy request looks: the field arrived with the modern era,
    so its absence says the request belongs to the era before it rather than
    that something is missing.
    """

    version = request_meta(params).get(META_PROTOCOL_VERSION)
    return version if isinstance(version, str) else None


def caching_hints(ttl_ms):
    """The freshness hint that rides on a cacheable result."""

    return {"ttlMs": ttl_ms, "cacheScope": CACHE_SCOPE_PUBLIC}


def missing_request_metadata(params):
    """Why a modern request is malformed, or None if it is not.

    Only the fields the specification marks required are insisted on.
    `clientInfo` is not among them -- clients SHOULD send it, and a server that
    refused without it would turn a recommendation into a barrier -- and
    nothing here reads it in any case.
    """

    meta = request_meta(params)
    capabilities = meta.get(META_CLIENT_CAPABILITIES)
    if not isinstance(capabilities, dict):
        return (
            f"A request declaring protocol version "
            f"{meta.get(META_PROTOCOL_VERSION)!r} must also carry "
            f"`_meta.{META_CLIENT_CAPABILITIES}` as an object."
        )
    return None


def protocol_version_is_supported(version):
    """Whether an `MCP-Protocol-Version` header names a revision we speak.

    An absent header is not a refusal: it means the revision that predates the
    header, which is on the list.
    """

    if version is None:
        return True
    return isinstance(version, str) and version.strip() in SUPPORTED_PROTOCOL_VERSIONS


class Dispatcher:
    """Turn one decoded JSON-RPC message into one response (or None)."""

    def __init__(self, registry, server_name, server_version, instructions=None):
        self.registry = registry
        self.server_name = server_name
        self.server_version = server_version
        self.instructions = instructions
        self.client_info = None
        self.protocol_version = None

    # -- lifecycle -------------------------------------------------------
    @property
    def initialized(self):
        return self.protocol_version is not None

    def _initialize(self, request_id, params):
        requested = params.get("protocolVersion")
        self.protocol_version = negotiate_protocol_version(requested)
        self.client_info = params.get("clientInfo") or {}
        if isinstance(requested, str) and requested != self.protocol_version:
            log.info(
                "MCP client asked for protocol %s; offering %s instead.",
                requested,
                self.protocol_version,
            )
        payload = {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": self.server_name,
                "version": self.server_version,
            },
        }
        # The one thing a server says before being asked anything. The same
        # field, under the same name, is what `server/discover` answers with
        # in the modern era: written once in `guide`, and reached by whichever
        # of the two a client happens to speak.
        if self.instructions:
            payload["instructions"] = self.instructions
        return _result(request_id, payload)

    # -- discovery -------------------------------------------------------
    def _server_info(self):
        return {"name": self.server_name, "version": self.server_version}

    def _discover(self, request_id, params):
        """What the handshake used to say, answered on demand instead.

        Required of every modern server, and answered here even when the
        request carries no `_meta` at all: this is the method a client sends
        before it knows what it is talking to, and refusing it for want of
        metadata would leave the one question it exists to answer unanswerable.

        A modern client also uses it to tell the two eras apart over stdio,
        where there is no HTTP status to read: a `DiscoverResult` means modern,
        any other error means legacy and a fallback to `initialize`. Both
        answers are correct here, because this server is both.
        """

        requested = _request_protocol_version(params)
        if requested is not None and requested not in SUPPORTED_PROTOCOL_VERSIONS:
            return unsupported_version_error(request_id, requested)
        payload = {
            "resultType": RESULT_TYPE_COMPLETE,
            # Every revision spoken, legacy ones included. They are reachable
            # -- through `initialize` -- and the specification's own example of
            # this field lists a legacy revision beside a modern one.
            "supportedVersions": list(SUPPORTED_PROTOCOL_VERSIONS),
            "capabilities": {"tools": {"listChanged": False}},
            "_meta": {META_SERVER_INFO: self._server_info()},
            **caching_hints(DISCOVER_CACHE_TTL_MS),
        }
        if self.instructions:
            payload["instructions"] = self.instructions
        return _result(request_id, payload)

    # -- tools -----------------------------------------------------------
    def _call_tool(self, request_id, params):
        name = params.get("name")
        if not isinstance(name, str):
            # A name that is not a string is not a lookup that misses, it is
            # one that raises: the registry is a dict. Answered here so it
            # comes back as this request's error, rather than escaping to the
            # transport -- where a batch would lose every other reply with it.
            return error_response(
                request_id, INVALID_PARAMS, "Tool name must be a string."
            )
        tool = self.registry.get(name)
        if tool is None:
            return error_response(request_id, INVALID_PARAMS, f"Unknown tool: {name}")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return error_response(
                request_id, INVALID_PARAMS, "Tool arguments must be an object."
            )
        problem = validate_arguments(tool.input_schema, arguments)
        if problem is not None:
            # A protocol error, not a tool result: nothing was attempted, and
            # MCP puts invalid arguments on the same footing as an unknown
            # tool. The message names the argument so the caller can fix the
            # call rather than re-read the schema.
            return error_response(request_id, INVALID_PARAMS, f"{name}: {problem}")
        try:
            payload = tool.handler(arguments)
        except ToolRefusal as exc:
            # The operation was not carried out. Reported as a failed result so
            # the caller cannot mistake it for success, but with the full
            # payload so it knows what to do about it.
            return _result(request_id, tool_result(exc.payload, is_error=True))
        except ToolError as exc:
            # An expected failure -- a precondition, a refusal, a timeout. The
            # caller needs to read it, so it travels as a tool result rather
            # than a protocol error.
            return _result(
                request_id,
                tool_result(
                    {"status": "error", "message": str(exc)}, is_error=True
                ),
            )
        except Exception as exc:  # noqa: BLE001 - never let one tool kill the server
            return _result(
                request_id,
                tool_result(
                    {
                        "status": "error",
                        "message": f"{type(exc).__name__}: {exc}",
                    },
                    is_error=True,
                ),
            )
        return _result(request_id, tool_result(payload))

    # -- entry point -----------------------------------------------------
    def dispatch(self, message):
        if not isinstance(message, dict):
            return error_response(None, INVALID_REQUEST, "Expected a JSON-RPC object.")

        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params")
        if not isinstance(params, dict):
            params = {}

        if "id" not in message:
            # Notification: never answered, whatever it is.
            return None

        if not isinstance(method, str):
            return error_response(request_id, INVALID_REQUEST, "Missing method.")

        # Answered before the era is worked out, because it is how the era is
        # worked out. `initialize` is the other way round: it *is* the legacy
        # era, whatever a `_meta` on it might claim.
        if method == "server/discover":
            return self._discover(request_id, params)
        if method == "initialize":
            return self._initialize(request_id, params)

        requested = _request_protocol_version(params)
        if requested is not None and requested not in SUPPORTED_PROTOCOL_VERSIONS:
            return unsupported_version_error(request_id, requested)
        modern = requested in MODERN_PROTOCOL_VERSIONS
        if modern:
            problem = missing_request_metadata(params)
            if problem is not None:
                return error_response(request_id, INVALID_PARAMS, problem)

        response = self._dispatch_method(request_id, method, params, modern=modern)
        return self._as_modern(response) if modern else response

    def _dispatch_method(self, request_id, method, params, *, modern):
        """The part that is the same in both eras. Every revision this server
        speaks carries these three messages unchanged.

        `modern` reaches only as far as the caching hints, which exist in one
        era and not the other. Everything else is answered identically and
        wrapped afterwards.
        """

        if method == "ping":
            return _result(request_id, {})
        if method == "tools/list":
            listing = {"tools": self.registry.describe_all()}
            if modern:
                listing.update(caching_hints(TOOLS_LIST_CACHE_TTL_MS))
            return _result(request_id, listing)
        if method == "tools/call":
            return self._call_tool(request_id, params)
        return error_response(request_id, METHOD_NOT_FOUND, f"Unknown method: {method}")

    def _as_modern(self, response):
        """The same answer, in the shape the modern era reads it in.

        Two additions, both to the result and neither to an error: the
        `resultType` every modern result must declare, and the server's own
        name and version, which a client used to learn from the handshake and
        now has to be told on each reply.

        Applied here, on the way out, rather than by each method building its
        own result. A method that had to remember would answer the same
        question twice, once per era, and the two would drift.
        """

        result = response.get("result")
        if not isinstance(result, dict):
            return response
        meta = dict(result.get("_meta") or {})
        meta.setdefault(META_SERVER_INFO, self._server_info())
        return {
            **response,
            "result": {
                "resultType": RESULT_TYPE_COMPLETE,
                **result,
                "_meta": meta,
            },
        }

    def dispatch_payload(self, payload):
        """Dispatch a decoded body that may be a single message or a batch."""

        if isinstance(payload, list):
            if not payload:
                return error_response(None, INVALID_REQUEST, "Empty batch.")
            responses = [
                response
                for response in (self.dispatch(message) for message in payload)
                if response is not None
            ]
            return responses or None
        return self.dispatch(payload)
