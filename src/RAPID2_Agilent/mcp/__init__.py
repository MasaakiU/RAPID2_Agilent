# -*- coding: utf-8 -*-

"""Local MCP control surface for the running RAPID2 window.

Ported from MyWB's, and layered the same way so that only the last files need
Qt:

``protocol``     JSON-RPC 2.0 / MCP dispatch over a tool registry.
``transport``    Loopback HTTP server, bearer token, Origin checks.
``paths``        Where application data lives, without asking Qt.
``endpoint``     The 0600 file that publishes host, port and token.
``registry``     Which windows exist, and which letter each answers to.
``presence``     Whether a window is open at all, AI access on or off.
``catalog``      What the relay answers with while RAPID2 is closed, and which
                 tools it may run itself.
``relay``        Transport adapter: carries a client's stdio messages to a
                 window's address. Spawned by the AI client.
``tools``        The tool surface: viewer tools over an adapter onto the live
                 window, analysis tools over `RAPID2_Agilent.services`.
``bridge``       Marshals a call from a request thread onto the Qt main thread.
``controller``   Qt glue: owns the pieces and follows the window lifetime.

Everything except ``bridge`` and ``controller`` is import-safe without a
QApplication, and ``tools`` reaches Qt only inside the method that captures.
``paths``, ``endpoint``, ``registry``, ``presence``, ``catalog`` and ``relay``
never import PyQt6 at all: the relay is spawned per client session and must
not pay for a toolkit it has no window to show.

NOTE on keeping in step with MyWB (env_WB, `mywb/modules/mcp`). This package
is a copy, not a shared dependency, so a fix on one side has to be carried to
the other by hand. Until the common part becomes a package both import:

- Copied with only names and texts changed: ``protocol``, ``transport``,
  ``endpoint``, ``registry``, ``trail``, ``relay/stdio``, ``relay/upstream``,
  ``relay/announce``, ``relay/__init__``, and ``utils/private_files``. Diff
  them against MyWB's to carry a fix across.
- Adapted: ``bridge`` (PySide6 -> PyQt6), ``controller`` (no approvals, work
  lock, dialog policy or user note), ``catalog`` (no user note), ``paths``,
  ``relay/launch`` (presence locks instead of MyWB's log locks; no app bundle),
  ``relay/launcher`` and ``relay/local`` (no compiled builds; ``local`` also
  runs the analysis tools while no window listens).
- RAPID2's own: ``tools``, ``guide``, ``presence``.
"""
