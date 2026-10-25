"""The jobs the command lines and the MCP tools run, written once each.

`python -m RAPID2_Agilent.alignment`, `rapid2-agilent-cli` and the rapid2_* MCP
tools call these rather than chaining the stages themselves, so a job reads,
saves and draws the same way whichever front end asked for it.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import AlignmentConfig
from .io import ScanSource, source_from_reader
from .pipeline import AlignmentResult, align_agilent
from .representation import build_profile_matrix
from .result_io import save_result
from .transforms import suggest_scales


log = logging.getLogger(__name__)

RESULT_FILES = ("mapping.json", "metadata.json", "diagnostics.npz")


def scale_candidates(reader: Any, config: AlignmentConfig) -> dict:
    """Candidate asinh scales of one loaded run, under config's binning and scan selection."""
    source = source_from_reader(reader, config.selection)
    profile = build_profile_matrix(source, config.binning)
    return {"run": source.metadata, "n_scans": len(source.scan_ids), "scale_candidates": suggest_scales(profile.values)}


def check_output_dir(output_dir: str | Path) -> Path:
    """output_dir, refused if an alignment result is already there."""
    output_dir = Path(output_dir)
    if any((output_dir / name).exists() for name in RESULT_FILES):
        raise FileExistsError(f"Choose a new output directory; an alignment result already exists in {output_dir}")
    return output_dir


def save_figures(
    result: AlignmentResult,
    output_dir: Path,
    *,
    truth: np.ndarray | None = None,
    xic: tuple[ScanSource, ScanSource, Sequence[tuple[float, float]]] | None = None,
) -> list[Path]:
    """diagnostics.png, and xic_overlay.png for xic = (reference, query, m/z windows).

    None of them without matplotlib (the "diagnostics" extra): the result itself
    is already saved, and the figures only illustrate it.
    """
    try:
        from .diagnostics import save_diagnostics, save_xic_overlay
    except ImportError:
        log.warning("matplotlib is not installed; the alignment figures are not drawn")
        return []
    figures = [output_dir / "diagnostics.png"]
    save_diagnostics(result, figures[0], truth=truth)
    if xic is not None:
        reference, query, windows = xic
        figures.append(output_dir / "xic_overlay.png")
        save_xic_overlay(reference, query, result.mapping, windows, figures[-1])
    return figures


def align_and_save(
    reference_path: str | Path,
    query_path: str | Path,
    output_dir: str | Path,
    config: AlignmentConfig,
    *,
    xic_windows: Sequence[tuple[float, float]] = (),
    progress: Callable[[str], None] | None = None,
) -> tuple[AlignmentResult, list[Path]]:
    """Align two .d runs and save the result, and its figures, in a new output_dir."""
    config.transform.validate_for_use()
    output_dir = check_output_dir(output_dir)
    result = align_agilent(reference_path, query_path, config, progress=progress)
    save_result(result, output_dir)
    if not xic_windows:
        return result, save_figures(result, output_dir)
    from ..io import AgilentReader

    with AgilentReader() as reference, AgilentReader() as query:
        reference.load(Path(reference_path))
        query.load(Path(query_path))
        sources = (source_from_reader(reference, config.selection), source_from_reader(query, config.selection))
        return result, save_figures(result, output_dir, xic=(*sources, xic_windows))
