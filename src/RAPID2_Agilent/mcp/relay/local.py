# -*- coding: utf-8 -*-

"""What the relay answers itself: RAPID2's introduction while RAPID2 is closed,
the analysis tools while RAPID2 is closed, and `rapid2_launch`.

The rest of the relay is a courier and says so. This file is the second
exception after `announce`, and like it is kept to one file so that it stays
the only one. An AI client asks for the handshake and the tool list once, as
it starts, and a failure there loses RAPID2 for the rest of that run -- so when
no window is listening, those few questions are answered from the catalog
(see `catalog`) instead of failing. A window that is listening still answers
them itself; this is the fallback, not a replacement.

The answers are not written here. They come from the same `Dispatcher` the
window answers with, holding the catalog's tools, so the version a handshake
settles on, the shape of each era's reply and the caching hints are decided by
one piece of code wherever the answer is given. The handshake text is the
window's own too: `guide` is part of the same package.

Tool calls are answered from here in two cases. `rapid2_launch`, which exists
for the moment no window can answer anything. And, while no window is
listening, the tools the catalog names as analysis tools: they read .d folders
by path and never touch a window, so the relay runs the window's own code for
them (`mcp.tools.analysis_tools`) rather than send the assistant away. That
code brings numpy, the reader and scipy, so it is loaded on the first such
call, never for the handshake. Every other call needs a window.
"""

from __future__ import annotations

import threading

from .. import catalog as catalog_file
from .. import guide
from ..protocol import Dispatcher, Tool, ToolError, ToolRegistry
from . import launch


# What a client asks before it uses RAPID2: the legacy handshake, the modern
# probe, the tool list, and the liveness check. Nothing here touches a window,
# and the answer to each is fixed for a build.
ANSWERED_WHILE_CLOSED = frozenset({"initialize", "server/discover", "tools/list", "ping"})


def find_catalog():
    """The catalog the last serving window left, or None to forward everything."""

    return catalog_file.read_catalog()


def own_analysis_tools():
    """This relay's analysis tools, as (session, {name: handler}). Loads the analysis code."""

    from ..tools import AnalysisSession, analysis_tools

    session = AnalysisSession()
    return session, {tool.name: tool.handler for tool in analysis_tools(session)}


class _LocalAnalysis:
    """The analysis tools the relay runs, built on the first call and kept for
    the relay's lifetime: the runs they open stay open for the next call, as in
    a window.

    Built once even when the first calls arrive together: the relay answers
    concurrently.
    """

    def __init__(self, build):
        self._build = build
        self._lock = threading.Lock()
        self._session = None
        self._handlers = None

    def handler(self, name):
        def call(arguments):
            handler = self._loaded().get(name)
            if handler is None:
                # Listed by a window whose tools this relay's code does not have.
                raise ToolError(
                    f"{name} needs RAPID2 open: it was listed by a RAPID2 other than the one this "
                    "relay runs. Start it with rapid2_launch, or ask the user to start RAPID2 and "
                    "turn on AI assistant access (Tools > AI Assistant Access)."
                )
            return handler(arguments)

        return call

    def _loaded(self):
        with self._lock:
            if self._handlers is None:
                self._session, self._handlers = self._build()
            return self._handlers

    def close(self):
        with self._lock:
            session, self._session, self._handlers = self._session, None, None
        if session is not None:
            session.close()


class LocalAnswers:
    """`catalog` may be None: then only `rapid2_launch` is answered here.

    It is taken even then, because a client can hold it from a window's own
    tool list. Forwarded, a call meant to start RAPID2 would meet the closed
    RAPID2 it is for, and be told to call itself.

    `analysis_tools` builds the analysis tools, as `own_analysis_tools` does.
    """

    def __init__(self, catalog, *, launch_handler=None, analysis_tools=own_analysis_tools):
        self._analysis = _LocalAnalysis(analysis_tools)
        self._analysis_names = catalog_file.analysis_tool_names(catalog) if catalog else frozenset()
        tools = ToolRegistry()
        for described in (catalog or {}).get("tools", ()):
            name = described["name"]
            if name == launch.TOOL_NAME:
                handler = launch_handler
            elif name in self._analysis_names:
                handler = self._analysis.handler(name)
            else:
                # Forwarded, and never reaching this dispatcher.
                handler = None
            tools.add(
                Tool(
                    name=name,
                    title=described.get("title") or name,
                    description=described.get("description") or "",
                    input_schema=described["inputSchema"],
                    output_schema=described.get("outputSchema"),
                    handler=handler,
                )
            )
        if launch.TOOL_NAME not in tools:
            tools.add(
                Tool(
                    name=launch.TOOL_NAME,
                    title=launch.TITLE,
                    description=launch.DESCRIPTION,
                    input_schema=launch.INPUT_SCHEMA,
                    handler=launch_handler,
                )
            )
        self._has_catalog = catalog is not None
        self._launches = launch_handler is not None
        self._dispatcher = Dispatcher(
            tools,
            server_name=(catalog or {}).get("server_name", "rapid2-agilent"),
            server_version=(catalog or {}).get("server_version", "unknown"),
            instructions=guide.server_instructions(),
        )

    def answers_while_closed(self, payload):
        """Whether `payload` is one the catalog answers when no window can:
        a question asked before RAPID2 is used, or a call to an analysis tool."""

        return (
            self._has_catalog
            and isinstance(payload, dict)
            and payload.get("id") is not None
            and (payload.get("method") in ANSWERED_WHILE_CLOSED or self._calls_analysis(payload))
        )

    def _calls_analysis(self, payload):
        if payload.get("method") != "tools/call":
            return False
        params = payload.get("params")
        name = params.get("name") if isinstance(params, dict) else None
        return isinstance(name, str) and name in self._analysis_names

    def answers_always(self, payload):
        """Whether `payload` is a call to `rapid2_launch`, answered here always."""

        if not self._launches or not isinstance(payload, dict):
            return False
        if payload.get("id") is None or payload.get("method") != "tools/call":
            return False
        params = payload.get("params")
        return isinstance(params, dict) and params.get("name") == launch.TOOL_NAME

    def answer(self, payload):
        return self._dispatcher.dispatch(payload)

    def close(self):
        """Close the runs the analysis tools opened, if any were."""

        self._analysis.close()
