"""Test-wide setup, applied before any test module is imported.

Workaround: the Qt/pyqtgraph objects of earlier GUI tests are left to Python's cyclic garbage
collector, and collecting them during a later allocation crashes the process (segfault or bus
error; the original code does too). Until those lifetimes are managed, e.g. with pytest-qt,
the tests run without automatic collection.

The MCP code keeps window letters, tokens and a launcher under the application data directory.
It is redirected to a temporary one, through the environment so that subprocesses (the relay)
follow it, and no test writes into the profile of whoever runs the suite.

The settings (SettingsModel's QSettings) go to an INI file in that directory as well: QSettings'
(organization, application) constructor always takes the native store (the macOS preferences),
whatever the default format, so the model is handed one made for the file instead.
"""

import atexit
import gc
import os
import shutil
import tempfile

from PyQt6.QtCore import QSettings

from RAPID2_Agilent.mcp import paths as mcp_paths
from RAPID2_Agilent.models import settings_model


gc.disable()

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TEST_APP_DATA_DIR = tempfile.mkdtemp(prefix="rapid2-test-data-")
os.environ[mcp_paths.DATA_DIR_ENV] = TEST_APP_DATA_DIR
atexit.register(shutil.rmtree, TEST_APP_DATA_DIR, ignore_errors=True)

QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, TEST_APP_DATA_DIR)
settings_model.QSettings = lambda organization, application: QSettings(
    QSettings.Format.IniFormat, QSettings.Scope.UserScope, organization, application
)
