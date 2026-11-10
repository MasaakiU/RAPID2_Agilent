# -*- coding: utf-8 -*-

"""What the client-integration builds share, as MyWB's do.

Two AI clients are handed the same thing -- the launcher each RAPID2 window
writes under the user's home directory -- through two unrelated packaging
formats: a Claude Desktop `.mcpb` bundle and a Codex plugin. Neither carries
any server code. The facts they encode are the same: the version, the
launcher's path on each platform, the icon, and the wording a listing shows.
They live here so the two builds cannot disagree, and so renaming the
launcher breaks both builds rather than shipping one that points at nothing.

Unlike MyWB, whose integrations live in a separate public repository beside
the source, RAPID2 is installed from its own repository (`pip install -e .`),
so the integrations are committed here: a clone is the Codex marketplace, and
the `.mcpb` sits in it ready to open.

Run with the interpreter RAPID2 is installed in; the package refuses to import
from a bare source tree.
"""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

from RAPID2_Agilent.mcp import paths as mcp_paths
from RAPID2_Agilent.mcp.relay import launcher


REPO_ROOT = Path(__file__).resolve().parents[1]
INTEGRATIONS_RELATIVE = Path("integrations")

# Node's platform identifiers, which the .mcpb manifest uses; used for both
# builds so there is one spelling of each platform here.
DARWIN, WIN32 = "darwin", "win32"
PLATFORMS = (DARWIN, WIN32)

SERVER_NAME = launcher.SERVER_NAME

REPOSITORY_URL = "https://github.com/MasaakiU/RAPID2_Agilent"
AUTHOR_NAME = "MasaakiU"  # as LICENSE has it
LICENSE = "MIT"

DISPLAY_NAME = "RAPID2 Agilent"

SHORT_DESCRIPTION = "Read Agilent LC-MS runs and drive the RAPID2 viewer."

LONG_DESCRIPTION = """\
Lets an assistant read Agilent LC-MS .d folders and RAPID2 .rpd files without
vendor software -- XICs, TIC/BPC, spectra, the strongest spots of a run and
where two runs differ, the area, height and peak top of targets in many runs,
written as a .rpd.csv file, and the RT alignment of two runs -- and work in the
RAPID2 window you have open: open runs, show an RT and m/z range, add and
select targets, and capture the window.

RAPID2 has to be installed, and Tools > AI Assistant Access switched on once:
that writes the launcher this integration starts. After that the analysis
tools work while RAPID2 is closed, and the viewer tools need a window, which
the assistant can start. RAPID2 listens on 127.0.0.1 only, and every request
carries an access token made each time it starts. An assistant that runs its
model on a remote service sends what it reads from RAPID2 there.

The assistant writes only new files (a .rpd.csv that must not exist yet, a new
alignment folder). Its changes to the window's view and targets can be undone
from the View and Edit menus.
"""

KEYWORDS = ["LC-MS", "mass spectrometry", "Agilent", "metabolomics"]

ICON_PIXELS = 512
ICON_SOURCE = REPO_ROOT / "src" / "RAPID2_Agilent" / "resources" / "icons" / "app_icon.png"


def project_version():
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def launcher_relative(platform):
    """Where the launcher sits on `platform`, relative to the home directory, as POSIX.

    Read from the code that writes the launcher rather than spelled out, so a
    rename breaks the build. The home directory is the anchor because it is
    the only per-user location a client configuration can name -- see
    `mcp.paths.launcher_dir`.
    """

    return "{}/{}".format(mcp_paths.home_relative_launcher_dir(platform), launcher.launcher_name(platform))


def render_icon(destination, pixels=ICON_PIXELS):
    """The app icon at a size a listing shows, with Pillow if it is here, else the 1024 px PNG as it is."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        with Image.open(ICON_SOURCE) as source:
            source.convert("RGBA").resize((pixels, pixels), Image.LANCZOS).save(destination, format="PNG")
        return destination
    except ImportError:
        print("Note: Pillow is not installed; the icon is the 1024 px PNG as it is.")
    shutil.copyfile(ICON_SOURCE, destination)
    return destination
