# -*- coding: utf-8 -*-

"""Carries MCP messages from an AI client into the running RAPID2 window.

This is a transport adapter, not a second MCP server. Nothing here knows what
`initialize` is, which tools exist, or what a tool result looks like: a message
arrives, it is handed to the window that already answers such things, and the
answer is handed back verbatim. The one exception is the `id` copied onto a
failure, which is unavoidable for anything that has to report that the far side
is unreachable in the caller's own terms. What the relay says on its own behalf
is kept in two files: ``announce``, and ``local``, which answers while no window
listens -- the handshake from the catalog, `rapid2_launch`, and the analysis
tools, which never needed a window.

Two halves, split along the line most likely to move:

``upstream``  The link to the window: endpoint discovery, bearer token, HTTP.
              Independent of how the message reached this process.
``stdio``     One way in -- newline-delimited JSON on a pair of pipes.

Today every client that can drive a local application speaks stdio, so that is
the only adapter shipped. If stdio is retired, or a client arrives wanting a
socket or a pipe instead, ``stdio`` is deleted or joined by a sibling and
``upstream`` is untouched.
"""

from .stdio import serve
from .upstream import (
    HttpUpstream,
    UpstreamError,
    UpstreamRejected,
    UpstreamUnavailable,
)


__all__ = [
    "HttpUpstream",
    "UpstreamError",
    "UpstreamRejected",
    "UpstreamUnavailable",
    "serve",
]
