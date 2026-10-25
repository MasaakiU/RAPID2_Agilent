# -*- coding: utf-8 -*-

"""The two things the relay says on its own behalf rather than RAPID2's.

Everything else here is a courier: `stdio` knows how bytes arrive, `upstream`
knows how to reach a window, and neither reads the message. Both say so in
their own first paragraph, and both are right to -- a proxy that understands
what it carries is a proxy that can be wrong about it.

This is the exception, kept in its own file so that it is the *only* one. Two
statements are the relay's to make, because they are about the relay:

**The tool list can change during a session.** RAPID2 answers `listChanged:
false`, and from inside one run of RAPID2 that is the truth: a window's tool set
is fixed when it starts. But the client is not talking to one run of RAPID2. It
is talking to this relay, which outlives any number of them -- the user
updates RAPID2, or simply restarts it, and the same stdio session carries on
against a process with a different tool set. Across that seam the answer is
`true`, and only the relay is in a position to say so.

**A window was replaced.** `upstream` notices when a letter changes hands and
has, until now, written it to the log. A client cannot read the log. It
carries a tool list, and tool schemas, taken from a handshake with the process
that has just been replaced -- so a tool added since is invisible, and an
argument that changed is a call the new window will refuse. Telling it to ask
again is one line of JSON.

WHAT THIS DOES NOT FIX. The notification goes out when the relay next talks to
RAPID2, which is when a call arrives -- so the call that discovers the
replacement is itself still built on the old schema, and may fail. The one
after it is fine. Closing that window entirely would mean polling the registry
with nothing to send, which costs every session something to save one call in
the rare one.

A client that ignores `notifications/tools/list_changed` is left where it was:
reconnecting picks the new tools up. Nothing here can compel a client to ask.
"""

from __future__ import annotations


TOOLS_LIST_CHANGED = "notifications/tools/list_changed"


def list_changed():
    """The notification, as it goes on the wire."""

    return {"jsonrpc": "2.0", "method": TOOLS_LIST_CHANGED}


def notifications(replacements):
    """What to send the client after `replacements` windows changed hands.

    One notification however many windows were replaced: the client has one
    tool list, and asking it to re-read that list twice tells it nothing the
    first did not.
    """

    return [list_changed()] if replacements else []


def speaking_for_the_relay(response):
    """One reply from RAPID2, with the capability it cannot answer for corrected.

    Applied to every reply rather than to the handshake, because there are two
    replies that carry this capability -- `initialize` answers it and, in the
    modern era, so does `server/discover` -- and a rule written in terms of
    method names would have to know both, and would silently stop covering a
    third. "Wherever RAPID2 states a tools capability" needs to know none of
    them, which is also what keeps the method-name ignorance `stdio` and
    `upstream` are built on from being quietly spent here.

    Copied rather than edited in place: the reply belongs to whoever produced
    it, and a relay that mutates what it carries is hard to reason about the
    moment anything else looks at the same object.
    """

    if isinstance(response, list):
        return [speaking_for_the_relay(entry) for entry in response]
    if not isinstance(response, dict):
        return response
    result = response.get("result")
    if not isinstance(result, dict):
        return response
    capabilities = result.get("capabilities")
    if not isinstance(capabilities, dict):
        return response
    tools = capabilities.get("tools")
    if not isinstance(tools, dict):
        return response
    if tools.get("listChanged") is True:
        return response
    return {
        **response,
        "result": {
            **result,
            "capabilities": {
                **capabilities,
                "tools": {**tools, "listChanged": True},
            },
        },
    }
