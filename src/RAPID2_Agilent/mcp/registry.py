# -*- coding: utf-8 -*-

"""Which RAPID2 instances exist, and which letter each answers to.

A registry of addresses, not a mirror of anything. It says where an instance is
and how to reach it. What that instance currently has open is a question for
its tools, which answer from the live document -- putting it here would mean
keeping a file in step with every undo, and choosing what to do when it fell
behind. The only history recorded is `started_at`: a property of the instance's
existence rather than of its work, immutable once written, and the one thing
that tells a reused letter apart from the window that held it before.

Each instance owns two files, and no instance ever writes another's:

``<label>.lock``  Held open, and flocked, from the moment the letter is claimed
                  until the process exits. This IS the claim: the operating
                  system grants it to exactly one process and releases it on
                  exit, crash included.
``<label>.json``  The address: url, token, instance_id, label, started_at.
                  Written when the server starts, removed when it stops.

The lock doubles as the liveness oracle. A `.json` whose lock can be taken
belongs to no living process, so it is stale -- which is how staleness is
detected here, with no pid check and no network probe.
"""

from __future__ import annotations

import logging
import os
import string
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import paths
from ..utils.private_files import ensure_private_directory


log = logging.getLogger(__name__)

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

# Sayable, and short enough for a window title. A machine with more than 26
# RAPID2 windows open has a different problem.
LABELS = string.ascii_uppercase
INSTANCES_DIRNAME = "instances"


class RegistryError(Exception):
    """The registry could not be read or written."""


class NoLabelAvailable(RegistryError):
    """Every letter is held by a living instance."""


def instances_dir(root=None):
    if root is not None:
        return Path(root)
    return paths.app_local_data_dir() / "mcp" / INSTANCES_DIRNAME


def lock_path(label, root=None):
    return instances_dir(root) / f"{label}.lock"


def document_path(label, root=None):
    return instances_dir(root) / f"{label}.json"


def _try_lock(handle):
    """Take an exclusive lock without waiting. True if we got it."""

    try:
        if sys.platform == "win32":
            msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


class LabelClaim:
    """One letter, held for as long as this process lives.

    Deliberately not released when the MCP server stops. Turning AI access off
    and on again must not rename the window: the letter is printed in the title
    bar and is how a user tells an assistant which window to work in.
    """

    def __init__(self, label, handle, root=None):
        self.label = label
        self.started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self._handle = handle
        self._root = root

    @property
    def document_path(self):
        return document_path(self.label, self._root)

    def release(self):
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            os.close(handle)
        except OSError:
            pass


def _open_lock(path):
    # This is what usually creates the directory the endpoint file goes into,
    # so it is where the 0700 chain has to be established -- `mkdir` with
    # `parents=True` would leave every directory above the leaf at whatever
    # the umask allows.
    ensure_private_directory(path.parent)
    return os.open(path, os.O_CREAT | os.O_RDWR, 0o600)


def claim_label(root=None, *, labels=LABELS):
    """Take the first free letter. Atomic against any number of racing starts.

    Two RAPID2 processes starting at the same instant cannot come away with the
    same letter: the claim is an exclusive lock, not a check followed by a
    write, so there is no window between deciding and owning.
    """

    for label in labels:
        path = lock_path(label, root)
        try:
            handle = _open_lock(path)
        except OSError as exc:
            raise RegistryError(f"The registry could not be opened: {exc}") from exc
        if not _try_lock(handle):
            os.close(handle)
            continue
        # The letter is ours. Anything left behind under it was left by a
        # process that no longer exists, because we hold what it would have held.
        _discard(document_path(label, root))
        return LabelClaim(label, handle, root)
    raise NoLabelAvailable("Every window letter is in use.")


def sweep_stale(root=None, *, labels=LABELS):
    """Remove address files belonging to processes that are gone.

    A crash leaves a `.json` with nobody behind it. Taking its lock proves the
    owner is gone; the lock is dropped again immediately, so a letter is never
    reserved by the sweep.
    """

    removed = []
    for label in labels:
        document = document_path(label, root)
        if not document.is_file():
            continue
        try:
            handle = _open_lock(lock_path(label, root))
        except OSError:
            continue
        try:
            if _try_lock(handle) and _discard(document):
                removed.append(label)
        finally:
            os.close(handle)
    return removed


def _discard(path):
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        return False
    return True


def read_instances(root=None, *, labels=LABELS):
    """Every published address, ordered by letter.

    Reads only. A stale entry is possible -- a process can die between the
    sweep and now -- so a caller that must be certain should try the address.
    """

    from . import endpoint as endpoint_file

    found = []
    for label in labels:
        document = endpoint_file.read_endpoint(document_path(label, root))
        if not document:
            continue
        if not document.get("url") or not document.get("token"):
            continue
        found.append(document)
    return found


def names_instance(document, wanted):
    """Does `wanted` name this instance? A letter, a full id, or an id prefix.

    The one place the rule lives. Both the relay (choosing where to send) and
    the instance (checking it was the intended target) read the same field, and
    they must agree about what it means or a misrouted call looks correct to
    one of them.
    """

    wanted = str(wanted).strip()
    if not wanted:
        return False
    label = str(document.get("label") or "")
    instance_id = str(document.get("instance_id") or "")
    if label and label.casefold() == wanted.casefold():
        return True
    if instance_id and instance_id == wanted:
        return True
    # A prefix has to be long enough to be a deliberate abbreviation.
    return bool(instance_id) and len(wanted) >= 4 and instance_id.startswith(wanted)


def match_instance(instances, wanted):
    """The one instance `wanted` names, or None if none or several do."""

    found = [
        document for document in instances if names_instance(document, wanted)
    ]
    return found[0] if len(found) == 1 else None
