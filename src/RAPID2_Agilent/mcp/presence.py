# -*- coding: utf-8 -*-

"""Whether a RAPID2 window is open, whether or not AI access is on in it.

The instance registry lists only the windows that serve MCP. `rapid2_launch`
also has to know about a window with access off -- that is exactly the one it
must not start a second copy of -- so every window holds one of a few presence
locks from start to exit. The operating system releases a lock when its
process exits, crash included, so a held lock is a living window. No Qt.
"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

from . import paths
from ..utils.private_files import ensure_private_directory

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


SLOTS = 8
PRESENCE_DIRNAME = "windows"

# How a lock another process holds is refused: EWOULDBLOCK by flock, EACCES
# by msvcrt.locking.
_HELD_ERRNOS = frozenset({errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES})


def presence_dir(root=None):
    if root is not None:
        return Path(root)
    return paths.app_local_data_dir() / "mcp" / PRESENCE_DIRNAME


def slot_path(number, root=None):
    return presence_dir(root) / f"window-{number}.lock"


def _lock(handle):
    if sys.platform == "win32":
        msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle):
    if sys.platform == "win32":
        msvcrt.locking(handle, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(handle, fcntl.LOCK_UN)


class PresenceClaim:
    """One slot, held until `release` or process exit."""

    def __init__(self, number, handle):
        self.number = number
        self._handle = handle

    def release(self):
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            os.close(handle)
        except OSError:
            pass


def hold(root=None):
    """Take the first free slot, or None if there is none or none can be made.

    None rather than an error: a window that cannot announce itself still
    works, and only `rapid2_launch` is less sure what it will find.
    """

    try:
        ensure_private_directory(presence_dir(root))
    except OSError:
        return None
    for number in range(1, SLOTS + 1):
        try:
            handle = os.open(slot_path(number, root), os.O_CREAT | os.O_RDWR, 0o600)
        except OSError:
            return None
        try:
            _lock(handle)
        except OSError:
            os.close(handle)
            continue
        return PresenceClaim(number, handle)
    return None


def window_running(root=None):
    """Whether any slot is held, by a window of any process."""

    for number in range(1, SLOTS + 1):
        try:
            handle = os.open(slot_path(number, root), os.O_RDWR)
        except OSError:
            continue
        try:
            _lock(handle)
        except OSError as exc:
            if exc.errno in _HELD_ERRNOS:
                return True
        else:
            _unlock(handle)
        finally:
            os.close(handle)
    return False
