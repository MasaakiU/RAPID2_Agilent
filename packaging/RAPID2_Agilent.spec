# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build of RAPID2: the window, the CLI and the MCP relay in one folder.

    python -m PyInstaller --noconfirm packaging/RAPID2_Agilent.spec

writes dist/RAPID2_Agilent.app on macOS and dist/RAPID2_Agilent/ elsewhere,
holding the three programs `pip install` gives (`RAPID2_Agilent.frozen`):
rapid2-agilent (the window), rapid2-agilent-cli and rapid2-agilent-mcp. On
macOS they are in RAPID2_Agilent.app/Contents/MacOS.

Build with the interpreter RAPID2 is installed in, on the platform the build
is for, after the native extensions are built (pip install -e .).
"""

import sys
from importlib.metadata import version
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata

from RAPID2_Agilent import frozen


PACKAGE = Path(SPECPATH).parent / "src" / "RAPID2_Agilent"
ICON = PACKAGE / "resources" / "icons" / "app_icon.png"
NAME = "RAPID2_Agilent"


def shipped(relative):
    """Whether a package data file goes into the build.

    Of the models, only the one of the detector the app runs
    (services.peak_detection.DETECTOR, "deconvolution"): the other detectors'
    weights, the clone of Kensert et al.'s repository and peakonly's PyTorch
    checkpoints are never read.
    """

    if relative.name == ".DS_Store":
        return False
    if relative.parts[:2] == ("resources", "model"):
        return relative.parts[:3] == ("resources", "model", "peak_deconvolution")
    return True


def package_data():
    """pyproject's package-data: resources/ and the alignment examples."""

    sources = [*(PACKAGE / "resources").rglob("*"), *(PACKAGE / "alignment" / "examples").glob("*.json")]
    datas = []
    for path in sources:
        relative = path.relative_to(PACKAGE)
        if path.is_file() and shipped(relative):
            datas.append((str(path), str(Path(NAME) / relative.parent)))
    return datas


# RAPID2_Agilent's own: the package reads its version from it. The rest are
# recorded in an alignment's metadata.json.
metadata = [entry for name in ("RAPID2_Agilent", "numpy", "scipy") for entry in copy_metadata(name)]

a = Analysis(
    [str(Path(SPECPATH) / "rapid2_entry.py")],
    datas=package_data() + metadata,
    excludes=[
        "torch",  # the gpu extra: the alignment runs on numpy without it
        "matplotlib",  # the diagnostics extra: an alignment is saved without its figures
        "PIL",  # matplotlib's
        "tkinter",
        "PyQt5",
        "PySide2",
        "PySide6",
        "IPython",
        "pytest",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)


def program(name, console):
    return EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=name,
        console=console,
        icon=[str(ICON)],
        strip=False,
        upx=False,
        argv_emulation=False,
        codesign_identity=None,
        entitlements_file=None,
    )


# The window first: on macOS the first executable is the one the .app opens.
programs = [
    program(frozen.GUI, console=False),
    program(frozen.CLI, console=True),
    program(frozen.RELAY, console=True),
]

coll = COLLECT(*programs, a.binaries, a.datas, strip=False, upx=False, name=NAME)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{NAME}.app",
        icon=str(ICON),
        bundle_identifier="io.github.masaakiu.rapid2-agilent",
        version=version("RAPID2_Agilent"),
        info_plist={
            "CFBundleDisplayName": "RAPID2 Agilent",
            "NSHighResolutionCapable": True,
        },
    )
