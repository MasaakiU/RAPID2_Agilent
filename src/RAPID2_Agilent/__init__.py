import importlib.metadata

__package_name__ = "RAPID2_Agilent" 

try:
    __version__ = importlib.metadata.version(__package_name__)
except importlib.metadata.PackageNotFoundError:
    raise Exception(f"try running 'pip install -e .' in the root directory of the project")
