"""Finding and opening runs by path: Agilent .d folders and RAPID2 .rpd files."""

from collections import OrderedDict
from pathlib import Path

from ..io import AgilentCentroidReader, AgilentReader, RPDReader, sample_info
from ..io.agilent_centroid_reader import stored_spectra as d_folder_spectra
from ..io.rpd_file import stored_spectrum_type
from ..io.scan_points_reader import SPECTRA_OF_TYPE

# the spectra of a run to read: its profile (MSProfile.bin), or the centroid spectra stored at acquisition
# (MSPeak.bin), about 1/17 of the data, for runs on a server; a .rpd file holds one of them
SPECTRA = ("profile", "centroid")
RPD_SUFFIX = ".rpd"

Reader = AgilentReader | AgilentCentroidReader | RPDReader


def is_rpd(path: str | Path) -> bool:
    return Path(path).suffix.lower() == RPD_SUFFIX


def run_path(path: str | Path) -> Path:
    """The resolved path of a run, a .d folder or a .rpd file; ValueError if it is neither."""
    path = Path(path).expanduser().resolve()
    if is_rpd(path):
        if not path.is_file():
            raise ValueError(f"No .rpd file at {path}")
        return path
    if not (path / "AcqData").is_dir():
        raise ValueError(f"Not an Agilent .d folder (no AcqData), nor a .rpd file: {path}")
    return path


def stored_spectra(path: str | Path) -> dict[str, int]:
    """Of "profile" and "centroid", those a run holds, by their bytes: of a .d folder its files' that are not empty,
    of a .rpd file the one it holds, by the file's."""
    path = run_path(path)
    if is_rpd(path):
        return {SPECTRA_OF_TYPE[stored_spectrum_type(path)]: path.stat().st_size}
    return d_folder_spectra(path)


def chosen_spectra(path: str | Path, spectra: str | None = None) -> str:
    """The spectra to read a run of: those asked for, else its profile if it holds one, else what it holds;
    ValueError if it has not those asked for."""
    if spectra is not None and spectra not in SPECTRA:
        raise ValueError(f"spectra is profile or centroid, not {spectra!r}")
    held = stored_spectra(path)
    if spectra is None:
        spectra = "profile" if "profile" in held or not held else next(iter(held))
    if spectra not in held:
        raise ValueError(f"{Path(path).name} has no {spectra} spectra, only {' and '.join(held) or 'no'} spectra")
    return spectra


def open_run(path: str | Path, spectra: str | None = None) -> Reader:
    """A run loaded, of the spectra asked for, else as chosen_spectra chooses; ValueError if it has not them."""
    path = run_path(path)
    spectra = chosen_spectra(path, spectra)
    if is_rpd(path):
        reader = RPDReader()
    else:
        reader = AgilentCentroidReader() if spectra == "centroid" else AgilentReader()
    reader.load(path)
    return reader


class RunCache:
    """Keeps the most recently used runs loaded, keyed by resolved path and the spectra read."""

    def __init__(self, max_runs: int = 4):
        self.max_runs = max_runs
        self._readers: OrderedDict[tuple[Path, str], Reader] = OrderedDict()

    def get(self, path: str | Path, spectra: str | None = None) -> Reader:
        path = run_path(path)
        key = (path, chosen_spectra(path) if spectra is None else spectra)
        if key in self._readers:
            self._readers.move_to_end(key)
            return self._readers[key]
        reader = open_run(*key)
        self._readers[key] = reader
        while len(self._readers) > self.max_runs:
            self._readers.popitem(last=False)[1].close()
        return reader

    def close(self) -> None:
        while self._readers:
            self._readers.popitem()[1].close()


RUN_ORDERS = ("path", "acquired")


def find_runs(directory: str | Path, order: str = "path") -> list[dict]:
    """The runs under a directory: the .d folders, without descending into them, and the .rpd files; of a .d folder
    also its sample name, type and position and when it was acquired, as its sample information has them. In the
    order of their paths, or (order "acquired") of their acquisition, those not told of it last."""
    if order not in RUN_ORDERS:
        raise ValueError(f"order is path or acquired, not {order!r}")
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    runs, stack = [], [directory]
    while stack:
        for path in sorted(stack.pop().iterdir(), reverse=True):
            if not path.is_dir():
                if is_rpd(path):
                    runs.append({"name": path.name, "path": str(path)})
                continue
            if path.suffix.lower() == ".d" and (path / "AcqData").is_dir():
                runs.append({"name": path.name, "path": str(path), **sample_info.summary(sample_info.read_sample_info(path))})
            else:
                stack.append(path)
    runs.sort(key=lambda run: run["path"])
    if order == "acquired":
        runs.sort(key=_acquisition_order)
    return runs


def _acquisition_order(run: dict) -> tuple:
    acquired = sample_info.acquired_at({"AcqTime": run["acquired_at"]}) if "acquired_at" in run else None
    return (0, acquired.timestamp()) if acquired is not None else (1, 0.0)
