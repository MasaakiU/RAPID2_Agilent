"""The two SHA-256 of a .d run that the .rpd.csv export writes; see docs/rpd_csv_format.md, section 3.3.

``scan_sha256``, of the scan records alone, is quick: the key to look for a run by. ``data_sha256`` reads all the
data and tells it is the very run. Each is lower-case hex.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path

from .rpd_file import stored_hash

# hashed in this order, those present. Not hashed: files added later (*.tags, Results), the paths and the times.
# DefaultMassCal.xml is, as the m/z depends on it, and only an edit by hand changes it.
DATA_FILES = ("DefaultMassCal.xml", "MSMassCal.bin", "MSPeak.bin", "MSProfile.bin", "MSScan.bin")
CHUNK_SIZE = 1 << 20

# by run: (the sizes and times of its files, scan_sha256, data_sha256), so that each is read once while it is unchanged
_hashes: dict[Path, tuple[tuple, str, str]] = {}


def scan_sha256(run: str | Path) -> str:
    return hashlib.sha256((Path(run) / "AcqData" / "MSScan.bin").read_bytes()).hexdigest()


def data_sha256(run: str | Path, on_chunk: Callable[[int], None] | None = None) -> str:
    """Each present file's name, a NUL and its size (8 bytes, little-endian), then its content, read in chunks;
    on_chunk is told the size of each chunk read, and may raise to stop."""
    digest = hashlib.sha256()
    for path in _data_paths(run):
        digest.update(path.name.encode() + b"\0" + path.stat().st_size.to_bytes(8, "little"))
        with open(path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                digest.update(chunk)
                if on_chunk is not None:
                    on_chunk(len(chunk))
    return digest.hexdigest()


def data_size(run: str | Path) -> int:
    """The bytes data_sha256 reads."""
    return sum(path.stat().st_size for path in _data_paths(run))


def run_hashes(run: str | Path, on_chunk: Callable[[int], None] | None = None) -> tuple[str, str]:
    """(scan_sha256, data_sha256) of the run, read again only once one of its files has changed. Of a RAPID2 .rpd
    file, ("", the SHA-256 it holds), the data_hash RAPID2 exports: it has no scan records of its own."""
    if Path(run).suffix.lower() == ".rpd":
        return "", stored_hash(run)
    run = Path(run).resolve()
    stamp = _stamp(run)
    kept = _hashes.get(run)
    if kept is not None and kept[0] == stamp:
        return kept[1], kept[2]
    hashes = scan_sha256(run), data_sha256(run, on_chunk)
    _hashes[run] = (stamp, *hashes)
    return hashes


def is_known(run: str | Path) -> bool:
    """Whether run_hashes has the run's hashes, its files unchanged since: nothing to read."""
    run = Path(run).resolve()
    kept = _hashes.get(run)
    return kept is not None and kept[0] == _stamp(run)


def _stamp(run: Path) -> tuple:
    return tuple((path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in _data_paths(run))


def _data_paths(run: str | Path) -> list[Path]:
    acq_data = Path(run) / "AcqData"
    return [acq_data / name for name in DATA_FILES if (acq_data / name).is_file()]
