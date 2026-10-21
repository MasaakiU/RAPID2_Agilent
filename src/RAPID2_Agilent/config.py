from pathlib import Path
from . import __version__, __package_name__

APP_VERSION = __version__
APP_NAME = __package_name__
ORG_NAME = "MU_Apps"
SCHEMA_VERSION = 1
DEFAULT_NOTE = "Created with MyChemApp"
DEFAULT_HOME = str(Path.home())
