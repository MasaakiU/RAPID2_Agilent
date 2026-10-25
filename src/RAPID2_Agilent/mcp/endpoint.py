# -*- coding: utf-8 -*-

"""The token, and the 0600 file that tells a client where to connect.

One such file per running instance. Where it goes, and which letter names it,
is `registry`'s business; this module only writes and reads the document.

A loopback port alone is not a permission: every process on the machine can
reach 127.0.0.1. The bearer token is what separates "the user pointed a client
at RAPID2" from "something else on this machine found an open port", so it is
generated per launch and never persisted across runs.

The stdio bridge re-reads this file on every request, because the token -- and,
if the preferred port was taken, the port -- changes each launch. `instance_id`
is what lets it notice that the window it was talking to has been replaced
rather than merely rewritten: a client-side handshake or cached tool list from
the previous run no longer applies. `app_version` is for the same reader, whose
executable can be older or newer than the running application.

`pid` is neither of those. It is kept for diagnostics only, and deliberately
not for deciding whether RAPID2 is up: an instance's own lock in `registry`
answers that, and posting to the server answers it beyond doubt.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from ..utils.private_files import write_private_file


TOKEN_BYTES = 32
INSTANCE_ID_BYTES = 8


class EndpointError(Exception):
    """The endpoint file could not be written or removed."""


def new_token():
    return secrets.token_urlsafe(TOKEN_BYTES)


def new_instance_id():
    """Name one run of the server.

    Not a secret -- the token is. A pid cannot do this job: it stays the same
    when a run rewrites its own file, and the operating system hands it out
    again once the process is gone.
    """

    return secrets.token_hex(INSTANCE_ID_BYTES)


def _write_atomic_bytes(data, path):
    try:
        # The directory is vouched for as well as the file: this document
        # holds the bearer token, and one that another account could replace
        # would send a client to a server of their choosing.
        return write_private_file(data, path, require_private_directory=True)
    except OSError as exc:
        raise EndpointError(
            f"The MCP endpoint file could not be written: {exc}"
        ) from exc


def write_endpoint(
    host,
    port,
    token,
    *,
    path,
    pid=None,
    instance_id=None,
    app_version=None,
    label=None,
    started_at=None,
    toolset_version=None,
):
    # schema_version stays 1 while the additions are optional fields: a reader
    # written against the first shape keeps working by ignoring them. Bump it
    # only when an existing field changes meaning or disappears.
    document = {
        "schema_version": 1,
        "url": f"http://{host}:{port}/mcp",
        "host": host,
        "port": int(port),
        "token": token,
        "pid": os.getpid() if pid is None else int(pid),
        "instance_id": new_instance_id() if instance_id is None else str(instance_id),
        "app_version": "unknown" if app_version is None else str(app_version),
        # The letter a user says out loud, and the moment this instance began.
        # Nothing about the documents it holds: those change, and a file that
        # has to keep up with them is a cache, not a registry.
        "label": label,
        "started_at": started_at,
        # Static, bumped by hand when the tool set changes. Enough for a reader
        # to notice two simultaneously running builds disagree, without the
        # registry having to enumerate tool names it would then have to keep
        # in step with the code.
        "toolset_version": toolset_version,
    }
    encoded = json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
    return _write_atomic_bytes(encoded, path)


def remove_endpoint(path):
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        return False
    return True


def read_endpoint(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    return document
