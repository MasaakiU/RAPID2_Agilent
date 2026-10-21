"""Adapt the existing reader; do not duplicate Agilent calibration or decoding."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Protocol
import numpy as np

from .config import ScanSelection


class ScanSource(Protocol):
    rt_seconds: np.ndarray
    scan_ids: np.ndarray
    metadata: dict

    def read_scan(self, position: int) -> tuple[np.ndarray, np.ndarray]: ...


def validate_times(rt: np.ndarray, name: str = "RT") -> np.ndarray:
    rt = np.asarray(rt, dtype=np.float64)
    if rt.ndim != 1 or len(rt) < 2 or not np.all(np.isfinite(rt)) or not np.all(np.diff(rt) > 0):
        raise ValueError(f"{name} must contain at least two finite, strictly increasing times")
    return rt


@dataclass
class ReaderSource:
    reader: Any
    rt_seconds: np.ndarray
    scan_ids: np.ndarray
    metadata: dict

    def read_scan(self, position: int) -> tuple[np.ndarray, np.ndarray]:
        return self.reader.extract_spectrum(int(self.scan_ids[position]))


def source_from_reader(reader: Any, selection: ScanSelection = ScanSelection()) -> ReaderSource:
    """Keep original scan IDs; reject mixed polarities unless explicitly selected.

    Agilent acquisition times default to minutes (as documented in the local
    MSTS.xml files). The conversion is explicit and recorded; core code uses seconds.
    """
    ids = reader.scan_indices_for_ms_level(selection.ms_level)
    polarities = np.array([reader.ms_scan_info.get_scan_info(int(i), "IonPolarity") for i in ids])
    if selection.polarity is not None:
        wanted = {"positive": 0, "negative": 1}[selection.polarity]
        mask = polarities == wanted
        ids, polarities = ids[mask], polarities[mask]
    if len(ids) == 0:
        raise ValueError("No scans match the requested MS level and polarity")
    if len(np.unique(polarities)) != 1:
        raise ValueError("Mixed-polarity run: select positive or negative polarity explicitly")
    factor = 60.0 if selection.reader_rt_unit == "minutes" else 1.0
    rt = np.asarray(reader.RT_list[ids], dtype=np.float64) * factor
    if selection.rt_range_seconds is not None:
        lo, hi = selection.rt_range_seconds
        keep = (lo <= rt) & (rt <= hi)
        ids, rt = ids[keep], rt[keep]
    ids, rt = ids[::selection.stride], rt[::selection.stride]
    if selection.max_scans is not None:
        ids, rt = ids[:selection.max_scans], rt[:selection.max_scans]
    validate_times(rt)
    path = Path(reader.file_path)
    # Metadata fingerprint avoids reading gigabytes just for provenance. It is
    # explicitly not a cryptographic identity check of the entire raw dataset.
    digest = hashlib.sha256()
    file_sizes = {}
    for name in ("MSTS.xml", "MSScan.bin", "DefaultMassCal.xml", "MSMassCal.bin"):
        p = path / "AcqData" / name
        digest.update(name.encode())
        digest.update(p.read_bytes())
        file_sizes[name] = p.stat().st_size
    file_sizes["MSProfile.bin"] = (path / "AcqData" / "MSProfile.bin").stat().st_size
    metadata = {
        "kind": "agilent_d", "path": str(path.resolve()), "name": path.name,
        "metadata_sha256": digest.hexdigest(), "file_sizes": file_sizes,
        "fingerprint_scope": "metadata files only; profile content is not hashed",
        "selection": asdict(selection), "ms_level": selection.ms_level,
        "polarity": int(polarities[0]), "spectrum_param_index": reader.spectrum_param_index,
        "rt_unit": "seconds", "reader_rt_unit": selection.reader_rt_unit,
    }
    return ReaderSource(reader, rt, ids, metadata)
