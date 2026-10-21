"""Versioned, pickle-free results; no raw spectra or profile matrices are saved."""

from datetime import datetime, timezone
from importlib import metadata as package_metadata
import json
from pathlib import Path
import platform
import numpy as np

from .dtw import DTWResult
from .mapping import RTMapping
from .pipeline import AlignmentResult


def _check_schema(metadata: dict) -> None:
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported alignment result schema")
    if metadata.get("direction") != "query_to_reference" or metadata.get("rt_unit") != "seconds":
        raise ValueError("Unsupported mapping direction or RT unit")


def save_result(result: AlignmentResult, directory: str | Path) -> Path:
    """Save full scan-space diagnostics and a small independently loadable mapping.

    Existing result files are not overwritten. The two .d paths and metadata
    fingerprints are included for the future GUI to check the target datasets.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("mapping.json", "metadata.json", "diagnostics.npz"):
        if (directory / name).exists():
            raise FileExistsError(f"Result already exists: {directory / name}")
    mapping_data = {
        "schema_version": 1, "direction": "query_to_reference", "rt_unit": "seconds",
        "interpolation": "pchip", "extrapolation": result.mapping.extrapolation,
        "query_seconds": result.mapping.query_seconds.tolist(),
        "reference_seconds": result.mapping.reference_seconds.tolist(),
        "reference": result.metadata["reference"], "query": result.metadata["query"],
    }
    versions = {}
    for name in ("numpy", "scipy", "torch", "matplotlib"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    metadata = {
        **result.metadata, "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": versions, "total_dtw_cost": result.dtw.total_cost,
    }
    (directory / "mapping.json").write_text(json.dumps(mapping_data, indent=2, allow_nan=False) + "\n")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        directory / "diagnostics.npz", similarity=result.similarity,
        raw_path=result.dtw.path, cumulative_cost=result.dtw.cumulative_cost,
        predecessors=result.dtw.predecessors, anchors=result.anchors,
        reference_rt_seconds=result.reference_rt_seconds, query_rt_seconds=result.query_rt_seconds,
        reference_scan_ids=result.reference_scan_ids, query_scan_ids=result.query_scan_ids,
    )
    return directory


def load_mapping(path: str | Path) -> RTMapping:
    """Only mapping.json is needed to apply RT corrections in the main code."""
    path = Path(path)
    if path.is_dir():
        path = path / "mapping.json"
    data = json.loads(path.read_text())
    _check_schema(data)
    if data.get("interpolation") != "pchip":
        raise ValueError("Unsupported interpolation method")
    return RTMapping(np.asarray(data["query_seconds"]), np.asarray(data["reference_seconds"]), data["extrapolation"])


def load_result(directory: str | Path) -> AlignmentResult:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    _check_schema(metadata)
    mapping = load_mapping(directory)
    with np.load(directory / "diagnostics.npz", allow_pickle=False) as arrays:
        dtw = DTWResult(arrays["raw_path"], arrays["cumulative_cost"], arrays["predecessors"],
                        float(metadata["total_dtw_cost"]), metadata["config"]["solver"])
        return AlignmentResult(
            arrays["reference_rt_seconds"], arrays["query_rt_seconds"],
            arrays["reference_scan_ids"], arrays["query_scan_ids"], arrays["similarity"],
            dtw, arrays["anchors"], mapping, metadata,
        )
