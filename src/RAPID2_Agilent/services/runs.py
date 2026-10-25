"""Finding and opening .d runs by path."""

from collections import OrderedDict
from pathlib import Path

from ..io import AgilentReader


def run_path(path: str | Path) -> Path:
    """The resolved path of a .d folder; ValueError if it is not one."""
    path = Path(path).expanduser().resolve()
    if not (path / "AcqData").is_dir():
        raise ValueError(f"Not an Agilent .d folder (no AcqData): {path}")
    return path


class RunCache:
    """Keeps the most recently used runs loaded, keyed by resolved path."""

    def __init__(self, max_runs: int = 4):
        self.max_runs = max_runs
        self._readers: OrderedDict[Path, AgilentReader] = OrderedDict()

    def get(self, path: str | Path) -> AgilentReader:
        path = run_path(path)
        if path in self._readers:
            self._readers.move_to_end(path)
            return self._readers[path]
        reader = AgilentReader()
        reader.load(path)
        self._readers[path] = reader
        while len(self._readers) > self.max_runs:
            self._readers.popitem(last=False)[1].close()
        return reader

    def close(self) -> None:
        while self._readers:
            self._readers.popitem()[1].close()


def find_runs(directory: str | Path) -> list[dict]:
    """The .d folders under a directory, without descending into them."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    runs, stack = [], [directory]
    while stack:
        for path in sorted(stack.pop().iterdir(), reverse=True):
            if not path.is_dir():
                continue
            if path.suffix.lower() == ".d" and (path / "AcqData").is_dir():
                runs.append({"name": path.name, "path": str(path)})
            else:
                stack.append(path)
    return sorted(runs, key=lambda run: run["path"])
