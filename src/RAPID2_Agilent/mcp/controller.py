# -*- coding: utf-8 -*-

"""Owns the MCP server for one RAPID2 window and follows its lifetime."""

from __future__ import annotations

import logging
import threading

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from .. import __version__ as APP_VERSION
from ..utils.private_files import UnsafeLocationError
from . import catalog
from . import endpoint as endpoint_file
from . import guide
from . import paths
from . import registry
from .bridge import MainThreadBridge, MainThreadTimeout
from .protocol import Dispatcher, ToolError
from .relay import launcher
from .tools import TOOLSET_VERSION, AnalysisSession, ViewerContext, analysis_tools, build_registry
from .transport import McpHttpServer


log = logging.getLogger(__name__)

SERVER_NAME = "rapid2-agilent"
ACCESS_REVOKED_MESSAGE = (
    "AI access to RAPID2 has been turned off. Ask the user to turn it back on "
    "(Tools > AI Assistant Access), then retry."
)


class _AccessSession:
    """One run of the server, revoked the moment access is turned off.

    Every viewer tool call carries the session it was admitted under. Closing
    the socket is not enough on its own: a call handed to the main thread is
    already queued behind whatever the event loop is doing, so it would run
    after the stop unless it checks first.
    """

    def __init__(self):
        self._revoked = threading.Event()

    @property
    def active(self):
        return not self._revoked.is_set()

    def revoke(self):
        self._revoked.set()


class McpController(QObject):
    """Start/stop the local MCP server and report where it is listening."""

    state_changed = pyqtSignal()

    def __init__(self, presenter, *, parent=None):
        super().__init__(parent)
        self._bridge = MainThreadBridge(parent=self)
        self._context = ViewerContext(presenter)
        self._analysis = AnalysisSession()
        # Claimed once, on the first start, and held until the process exits --
        # never handed back when the server stops. The letter is printed in the
        # window title and is how a user tells an assistant which window to
        # work in, so turning access off and on again must not rename it.
        self._claim = None
        self._session = None
        self._server = None
        self._token = None
        self._instance_id = None
        self._endpoint_path = None
        self._launcher_path = None
        self._last_error = None

    # -- state -----------------------------------------------------------
    @property
    def running(self):
        return self._server is not None and self._server.running

    @property
    def url(self):
        return None if self._server is None else self._server.url

    @property
    def token(self):
        return self._token

    @property
    def instance_id(self):
        return self._instance_id

    @property
    def label(self):
        """The letter this window answers to, or None before the first start."""

        return None if self._claim is None else self._claim.label

    @property
    def last_error(self):
        return self._last_error

    @property
    def launcher_path(self):
        """Where an AI client should be pointed. Stable across restarts."""

        return self._launcher_path

    def client_command(self):
        """The line a user runs once to register RAPID2 with Claude Code.

        Unlike the address and the token, this does not go stale: it names the
        launcher, and the launcher finds whichever RAPID2 is running at the time.
        """

        if not self.running or self._launcher_path is None:
            return None
        return launcher.registration_command(self._launcher_path, SERVER_NAME)

    # -- guards ----------------------------------------------------------
    def _guarded(self, call, access):
        """Refuse work the live window is in no position to do. Main thread."""

        if access is not None and not access.active:
            # Queued before access went off, run after. The window is not
            # touched.
            raise ToolError(ACCESS_REVOKED_MESSAGE)
        # A modal dialog runs a nested event loop, which still delivers queued
        # calls -- so without this the agent could change the window behind a
        # dialog the user is currently answering.
        if QApplication.activeModalWidget() is not None:
            raise ToolError(
                "RAPID2 is waiting for an answer in a dialog. Ask the user to "
                "dismiss it, then retry."
            )
        return call()

    def _run_on_main_thread(self, call, access=None):
        # Checked here and again on the main thread: this catches a call that
        # arrives after the stop, the other one catches a call that was
        # already queued when the stop happened.
        if access is not None and not access.active:
            raise ToolError(ACCESS_REVOKED_MESSAGE)
        try:
            return self._bridge.invoke(lambda: self._guarded(call, access))
        except MainThreadTimeout as exc:
            raise ToolError(str(exc)) from exc

    def _leave_catalog_for_the_relay(self, tools):
        """What the relay answers the handshake with while no window is open,
        and which tools it may run itself then.

        Not worth refusing access over: without it the relay forwards the
        handshake and waits for a window, as it always did.
        """

        try:
            catalog.write_runtime_catalog(
                catalog.catalog_document(
                    tools.describe_all(),
                    server_name=SERVER_NAME,
                    server_version=APP_VERSION,
                    toolset_version=TOOLSET_VERSION,
                    analysis_tools=[tool.name for tool in analysis_tools(self._analysis)],
                )
            )
        except (OSError, paths.DataPathError) as exc:
            log.warning("The tool catalog for the relay could not be written: %s", exc)

    # -- lifecycle -------------------------------------------------------
    def start(self):
        if self.running:
            return True
        self._last_error = None
        if self._claim is None:
            try:
                registry.sweep_stale()
                self._claim = registry.claim_label()
            except (registry.RegistryError, paths.DataPathError, OSError) as exc:
                self._last_error = str(exc)
                log.warning("MCP window letter could not be claimed: %s", exc)
                self.state_changed.emit()
                return False
        token = endpoint_file.new_token()
        instance_id = endpoint_file.new_instance_id()
        access = _AccessSession()
        tools = build_registry(
            self._context,
            self._analysis,
            run_on_main_thread=lambda call: self._run_on_main_thread(call, access),
            identity={
                "label": self._claim.label,
                "instance_id": instance_id,
                "started_at": self._claim.started_at,
                "app_version": APP_VERSION,
                "toolset_version": TOOLSET_VERSION,
            },
        )
        dispatcher = Dispatcher(
            tools,
            server_name=SERVER_NAME,
            server_version=APP_VERSION,
            instructions=guide.server_instructions(),
        )
        server = McpHttpServer(dispatcher.dispatch_payload, token)
        try:
            server.start()
        except OSError as exc:
            self._last_error = str(exc)
            log.warning("MCP server could not start: %s", exc)
            self.state_changed.emit()
            return False

        self._server = server
        self._session = access
        self._token = token
        self._instance_id = instance_id
        try:
            self._launcher_path = launcher.write_launcher()
        except UnsafeLocationError as exc:
            # The launcher is what the client executes, so a directory another
            # account can write to means whatever is sitting there cannot be
            # trusted either -- including a launcher an earlier run left.
            return self._abort_start(
                "AI assistant access could not start: the folder RAPID2 keeps "
                f"its launcher in is not private to your account. {exc}"
            )
        except (OSError, paths.DataPathError) as exc:
            # The server still works; only the registration command cannot be
            # offered.
            self._launcher_path = None
            log.warning("MCP launcher could not be written: %s", exc)
        self._leave_catalog_for_the_relay(tools)
        try:
            self._endpoint_path = endpoint_file.write_endpoint(
                server.host,
                server.port,
                token,
                path=self._claim.document_path,
                instance_id=instance_id,
                app_version=APP_VERSION,
                label=self._claim.label,
                started_at=self._claim.started_at,
                toolset_version=TOOLSET_VERSION,
            )
        except endpoint_file.EndpointError as exc:
            # The file carries the token and the port, and the relay re-reads
            # it on every request, so a server whose address was never
            # published is not one to leave listening.
            return self._abort_start(f"AI assistant access could not start: {exc}")
        self.state_changed.emit()
        return True

    def _abort_start(self, message):
        """Undo a start that got as far as listening, and say why."""

        log.warning("%s", message)
        self.stop()
        self._last_error = message
        self.state_changed.emit()
        return False

    def stop(self):
        server = self._server
        session, self._session = self._session, None
        if session is not None:
            # Before the socket goes: any tool call already queued for the main
            # thread must refuse rather than run once this returns.
            session.revoke()
        self._server = None
        self._token = None
        self._instance_id = None
        self._launcher_path = None
        # Only ever our own file. The letter stays claimed.
        if self._endpoint_path is not None:
            endpoint_file.remove_endpoint(self._endpoint_path)
        self._endpoint_path = None
        if server is not None:
            server.stop()
        self.state_changed.emit()
        return True

    def release(self):
        """Stop, close the analysis runs and give up the window letter. For exit."""

        self.stop()
        self._analysis.close()
        claim, self._claim = self._claim, None
        if claim is not None:
            claim.release()

    def apply_enabled(self, enabled):
        """Follow the AI Assistant Access switch."""

        if enabled and not self.running:
            return self.start()
        if not enabled and self.running:
            return self.stop()
        return True
