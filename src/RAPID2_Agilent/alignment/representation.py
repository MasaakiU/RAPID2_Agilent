"""Build inspectable float32 profiles one scan at a time."""

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .binning import PPMGrid, aggregate_scan
from .config import BinningConfig
from .io import ScanSource, validate_times


@dataclass
class ProfileMatrix:
    values: np.ndarray
    rt_seconds: np.ndarray
    scan_ids: np.ndarray
    grid: PPMGrid
    metadata: dict


def build_profile_matrix(
    source: ScanSource, config: BinningConfig = BinningConfig(), *,
    out: np.ndarray | None = None, progress: Callable[[int, int], None] | None = None,
) -> ProfileMatrix:
    """No chromatographic peak picking and no retained per-scan m/z arrays.

    An optional caller-owned ndarray or memmap allows local storage control.
    Neither this function nor the pipeline exports raw profiles for transfer.
    """
    rt = validate_times(source.rt_seconds)
    if len(source.scan_ids) != len(rt):
        raise ValueError("scan IDs and RTs must have equal lengths")
    grid = PPMGrid(config)
    shape = (len(rt), grid.n_bins)
    if out is None:
        out = np.empty(shape, dtype=np.float32)
    elif out.shape != shape or out.dtype != np.float32:
        raise ValueError(f"out must have shape {shape} and dtype float32")
    for position in range(len(rt)):
        mz, intensity = source.read_scan(position)
        aggregate_scan(mz, intensity, grid, out=out[position])
        if progress is not None:
            progress(position + 1, len(rt))
    return ProfileMatrix(out, rt.copy(), np.asarray(source.scan_ids).copy(), grid, dict(source.metadata))
