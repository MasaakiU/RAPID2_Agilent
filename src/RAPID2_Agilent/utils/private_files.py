# -*- coding: utf-8 -*-

"""Writing a file that only this account can read, or replace.

RAPID2 keeps things on disk that decide what other programs may do: the bearer
token an AI client authenticates with, and the launcher script such a client
executes at the start of every session. Both are written asking for `0o600`, which is a
real restriction on POSIX and **nothing at all** on Windows -- `os.chmod`
there only flips the read-only attribute. Who could read those files was
therefore decided entirely by the directory they landed in, and no code ever
looked at that directory: `mkdir(parents=True, exist_ok=True, mode=0o700)`
applies its mode only to the last component it creates, and not at all to one
that already exists.

So a private write happens here, once, for both platforms, and it separates
two things that a single `chmod` was being asked to do at the same time.

*The file itself is locked, rather than its neighbourhood.* On Windows the
file is created with a DACL of its own granting this account and nobody else,
so it is private whatever the directory would have allowed, and there is no
window between creating it and restricting it. Specifying a DACL at creation
does not by itself keep the parent's inheritable entries out: Windows merges
them into the DACL it was given unless that DACL is marked protected, so it
is marked. On POSIX the `0o600` that was always asked for does this job
already, and the directories are now created `0o700` all the way down rather
than only at the leaf.

*The directory is still examined, for the other reason.* A directory that
others may write to lets them delete our file and leave their own in its
place. No permission on the file can prevent that, and for the launcher it
means running their code as this user. A caller whose file must not be
substituted therefore asks for the directory to be vouched for, and is
refused when it cannot be.

What counts as unsafe is deliberately narrow, and on Windows is deliberately
not answered yet -- see `directory_is_shared`.
"""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path


# Whether private writes have to go through the Windows security APIs. A
# module-level name rather than a test at each call site, so that a test can
# drive the Windows path from a machine that is not Windows.
_IS_WINDOWS = os.name == "nt"

# The temporary name is unpredictable so that nothing in a directory we do not
# control can be sitting on the name we are about to use. Creation is
# exclusive either way; this only decides whether someone else can make us
# retry.
_TEMPORARY_NAME_BYTES = 8
_TEMPORARY_NAME_ATTEMPTS = 8

# Windows security plumbing. Values from the platform SDK headers.
_TOKEN_QUERY = 0x0008
_TOKEN_USER_CLASS = 1
_ERROR_INSUFFICIENT_BUFFER = 122
_SECURITY_DESCRIPTOR_REVISION = 1
_SE_DACL_PROTECTED = 0x1000
_ACL_REVISION = 2
_FILE_ALL_ACCESS = 0x001F01FF
_GENERIC_WRITE = 0x40000000
_CREATE_NEW = 1
_FILE_ATTRIBUTE_NORMAL = 0x00000080
# Room for the header plus the single entry we add. One entry for one user SID
# is an order of magnitude short of this.
_ACL_BYTES = 1024


class UnsafeLocationError(OSError):
    """The directory cannot be trusted to hold a file only we may replace.

    An OSError, because every caller here already reports the failure to write
    a file and says which file it was; this only narrows the reason.
    """


def directory_is_shared(path):
    """Whether others may put their own file where ours belongs.

    Three answers, not two. `True` means demonstrably unsafe, `False` means
    demonstrably safe, and `None` means this platform cannot say -- which is
    the honest answer on Windows until the shape of a real installation's ACLs
    has been measured rather than guessed. Only `True` refuses a write.

    That reticence is the point. A predicate that is too strict here does not
    fail where somebody would notice it; it turns AI assistant access off for
    every user on the platform, and no test on another operating system can
    catch that. So the POSIX answer covers the two states that are unambiguous
    -- a directory somebody else owns, and one that group or world may write
    to without the sticky bit that would stop them removing what is not theirs
    -- and Windows waits.
    """

    if _IS_WINDOWS:
        return None
    try:
        info = Path(path).stat()
    except OSError:
        return None
    if info.st_uid not in (os.getuid(), 0):
        # Its owner can widen it whenever they like, so its mode right now
        # says nothing. Root is excepted because root is not held out by
        # anything we could do here anyway.
        return True
    if info.st_mode & stat.S_ISVTX:
        # Sticky: others may create files here but may not remove ours.
        return False
    return bool(info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))


def ensure_private_directory(path, *, require_private=False):
    """Create `path` and every missing parent, private to this account.

    Each component we create is made `0o700` outright rather than left to
    `mkdir`'s mode, which the umask can only narrow and which `parents=True`
    does not apply to intermediate directories at all. A directory that was
    already there is never altered: it may be one the user pointed us at
    deliberately, and quietly rewriting the permissions of somebody's shared
    folder is not ours to do. It is examined instead.
    """

    path = Path(path)
    missing = []
    probe = path
    while not probe.exists():
        missing.append(probe)
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            # Another process got there first, which is not a failure.
            continue
        if not _IS_WINDOWS:
            # mkdir's mode is masked by the umask; this is not.
            os.chmod(directory, 0o700)
    if not path.is_dir():
        raise NotADirectoryError(f"{path} is not a directory.")
    if require_private and directory_is_shared(path) is True:
        raise UnsafeLocationError(
            f"{path} can be written to by other accounts on this computer, "
            "so a file placed there could be replaced with another."
        )
    return path


def write_private_file(
    data,
    path,
    *,
    mode=0o600,
    require_private_directory=False,
):
    """Replace `path` with `data`, readable only by this account.

    Whole-file replacement, so a reader never sees half a document: the bytes
    land in a temporary file beside the target -- created just as privately as
    the target, which is what the old `chmod`-afterwards did not manage on
    Windows -- are flushed to disk, and are then moved into place in one step.

    `mode` is the POSIX mode; on Windows the file's own DACL decides, and an
    executable file is one by extension rather than by permission, so the
    launcher's `0o700` needs no counterpart there.

    Raises OSError, `UnsafeLocationError` among them; callers say what the
    file was.
    """

    path = Path(path)
    ensure_private_directory(
        path.parent,
        require_private=require_private_directory,
    )
    temporary_path = None
    try:
        temporary_path, descriptor = _create_private_temporary(path, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return path


def _create_private_temporary(path, mode):
    """A new, private, exclusively created file beside `path`."""

    for _ in range(_TEMPORARY_NAME_ATTEMPTS):
        candidate = path.with_name(
            f".{path.name}.{secrets.token_hex(_TEMPORARY_NAME_BYTES)}.tmp"
        )
        try:
            return candidate, _open_new_private(candidate, mode)
        except FileExistsError:
            continue
    raise FileExistsError(
        f"No unused temporary name was available beside {path}."
    )


def _open_new_private(path, mode):
    """Create `path`, private to this account, and return its descriptor."""

    if _IS_WINDOWS:
        return _windows_open_new_private(path)
    return os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        mode,
    )


def _windows_open_new_private(path):
    """Create `path` with a DACL naming this account and nobody else.

    The security descriptor is attached at creation, so the file is never
    briefly governed by what the directory would have handed down to it. Its
    DACL is also marked protected: without that, Windows merges the
    directory's inheritable entries into it at creation, and adds any the
    directory is given later.

    Nothing is granted to SYSTEM or to Administrators: an administrator can
    take ownership if they ever need to, and leaving them out keeps the set of
    accounts that can read a bearer token down to the one that owns it.
    """

    import ctypes
    import msvcrt
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class SECURITY_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", ctypes.c_void_p),
            ("bInheritHandle", wintypes.BOOL),
        ]

    class SECURITY_DESCRIPTOR(ctypes.Structure):
        _fields_ = [
            ("Revision", ctypes.c_ubyte),
            ("Sbz1", ctypes.c_ubyte),
            ("Control", wintypes.WORD),
            ("Owner", ctypes.c_void_p),
            ("Group", ctypes.c_void_p),
            ("Sacl", ctypes.c_void_p),
            ("Dacl", ctypes.c_void_p),
        ]

    class SID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("Sid", ctypes.c_void_p),
            ("Attributes", wintypes.DWORD),
        ]

    class TOKEN_USER(ctypes.Structure):
        _fields_ = [("User", SID_AND_ATTRIBUTES)]

    advapi32.OpenProcessToken.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    )
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.InitializeAcl.argtypes = (
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
    )
    advapi32.InitializeAcl.restype = wintypes.BOOL
    advapi32.AddAccessAllowedAce.argtypes = (
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
    )
    advapi32.AddAccessAllowedAce.restype = wintypes.BOOL
    advapi32.InitializeSecurityDescriptor.argtypes = (
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    advapi32.InitializeSecurityDescriptor.restype = wintypes.BOOL
    advapi32.SetSecurityDescriptorDacl.argtypes = (
        ctypes.c_void_p,
        wintypes.BOOL,
        ctypes.c_void_p,
        wintypes.BOOL,
    )
    advapi32.SetSecurityDescriptorDacl.restype = wintypes.BOOL
    advapi32.SetSecurityDescriptorControl.argtypes = (
        ctypes.c_void_p,
        wintypes.WORD,
        wintypes.WORD,
    )
    advapi32.SetSecurityDescriptorControl.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel32.CreateFileW.restype = wintypes.HANDLE

    invalid_handle = ctypes.c_void_p(-1).value

    def fail():
        raise ctypes.WinError(ctypes.get_last_error())

    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(),
        _TOKEN_QUERY,
        ctypes.byref(token),
    ):
        fail()
    try:
        needed = wintypes.DWORD(0)
        advapi32.GetTokenInformation(
            token,
            _TOKEN_USER_CLASS,
            None,
            0,
            ctypes.byref(needed),
        )
        # The first call is expected to fail for want of a buffer; only an
        # unexpected reason is a failure.
        error = ctypes.get_last_error()
        if not needed.value:
            if error and error != _ERROR_INSUFFICIENT_BUFFER:
                raise ctypes.WinError(error)
            raise OSError("Windows did not report this account's identity.")
        buffer = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetTokenInformation(
            token,
            _TOKEN_USER_CLASS,
            ctypes.byref(buffer),
            needed,
            ctypes.byref(needed),
        ):
            fail()
        sid = ctypes.cast(buffer, ctypes.POINTER(TOKEN_USER)).contents.User.Sid

        # A DWORD array rather than a byte buffer: an ACL has to be aligned,
        # and this is aligned by construction.
        acl = (wintypes.DWORD * (_ACL_BYTES // ctypes.sizeof(wintypes.DWORD)))()
        if not advapi32.InitializeAcl(
            ctypes.byref(acl),
            _ACL_BYTES,
            _ACL_REVISION,
        ):
            fail()
        if not advapi32.AddAccessAllowedAce(
            ctypes.byref(acl),
            _ACL_REVISION,
            _FILE_ALL_ACCESS,
            sid,
        ):
            fail()
        descriptor = SECURITY_DESCRIPTOR()
        if not advapi32.InitializeSecurityDescriptor(
            ctypes.byref(descriptor),
            _SECURITY_DESCRIPTOR_REVISION,
        ):
            fail()
        if not advapi32.SetSecurityDescriptorDacl(
            ctypes.byref(descriptor),
            True,
            ctypes.byref(acl),
            False,
        ):
            fail()
        if not advapi32.SetSecurityDescriptorControl(
            ctypes.byref(descriptor),
            _SE_DACL_PROTECTED,
            _SE_DACL_PROTECTED,
        ):
            fail()
        attributes = SECURITY_ATTRIBUTES(
            nLength=ctypes.sizeof(SECURITY_ATTRIBUTES),
            lpSecurityDescriptor=ctypes.addressof(descriptor),
            bInheritHandle=False,
        )
        handle = kernel32.CreateFileW(
            str(path),
            _GENERIC_WRITE,
            0,
            ctypes.byref(attributes),
            _CREATE_NEW,
            _FILE_ATTRIBUTE_NORMAL,
            None,
        )
        if not handle or handle == invalid_handle:
            # Reported as the OSError the POSIX path would have raised, so a
            # caller retrying a taken name does not have to know the platform.
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(token)
    try:
        # The descriptor owns the handle from here; closing it closes the
        # handle.
        return msvcrt.open_osfhandle(
            handle,
            os.O_WRONLY | getattr(os, "O_BINARY", 0),
        )
    except OSError:
        kernel32.CloseHandle(handle)
        raise


def _fsync_directory(path):
    """Make the rename itself durable. Best effort: not every platform can."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
