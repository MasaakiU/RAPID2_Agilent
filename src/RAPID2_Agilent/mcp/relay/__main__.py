# -*- coding: utf-8 -*-

"""Entry point for the relay: `python -m RAPID2_Agilent.mcp.relay`, or
`rapid2-agilent-mcp`.

Registered once with an AI client, which spawns it and keeps it for as long as
the client itself runs. It survives RAPID2 starting, stopping and starting
again underneath it, because it never holds on to where RAPID2 was -- and,
with a catalog to answer from, RAPID2 not having started yet at all.
"""

from __future__ import annotations

import logging
import os
import sys

from .launch import Launcher
from .local import LocalAnswers, find_catalog
from .stdio import serve
from .upstream import HttpUpstream


log = logging.getLogger(__name__)


def build(catalog):
    """The upstream and the local answers for a relay holding `catalog`."""

    if catalog is None:
        # Nothing to introduce RAPID2 with, so the handshake still has to
        # reach a window, and the wait for one to appear still earns its keep.
        upstream = HttpUpstream()
    else:
        upstream = HttpUpstream(first_contact_wait=0.0)
    local = LocalAnswers(catalog, launch_handler=Launcher(upstream.reachable))
    return upstream, local


def main(argv=None):
    logging.basicConfig(
        level=os.environ.get("RAPID2_RELAY_LOG", "INFO").upper(),
        stream=sys.stderr,
        format="%(levelname)s rapid2-relay: %(message)s",
    )

    # stdout belongs to the protocol from here on. A stray print() anywhere in
    # this process would corrupt the stream and take the session down with it,
    # so keep the real pipe and give the name to stderr.
    protocol_stream = sys.stdout.buffer
    sys.stdout = sys.stderr

    catalog = find_catalog()
    if catalog is None:
        log.info("No tool catalog: the handshake waits for RAPID2.")
    upstream, local = build(catalog)
    try:
        serve(upstream, stdout=protocol_stream, local=local)
    except KeyboardInterrupt:
        pass
    finally:
        local.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
