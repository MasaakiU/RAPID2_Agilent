"""What the diagnostics panel beside the B - A map plots, as plain arrays.

The x axis is always A's RT, shared with the map.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from RAPID2_Agilent.models.mz_rt_diff import DiffOverview, MzRtDiff


class Diagnostics(Enum):
    NONE = "None"
    DTW_PATH = "DTW path"
    RT_SHIFT = "RT shift"
    PATH_SIMILARITY = "Path similarity"
    TIC = "TIC"
    SQUARED_DIFFERENCE = "D² per scan"


@dataclass(frozen=True)
class Curve:
    role: str  # how the window draws it: a, b, b_original, main, path, anchors or guide
    x: np.ndarray
    y: np.ndarray
    name: str | None = None  # legend entry


@dataclass(frozen=True)
class DiagnosticsImage:
    values: np.ndarray                        # (n_x, n_y), on uniform grids
    rect: tuple[float, float, float, float]   # x, y, width, height
    levels: tuple[float, float]


@dataclass(frozen=True)
class DiagnosticsPlot:
    y_label: str
    curves: list[Curve]
    image: DiagnosticsImage | None = None


def dtw_path(diff: MzRtDiff) -> DiagnosticsPlot:
    """Scan similarity (A's RT x B's RT), the DTW path through it and the anchors on it."""
    alignment = diff.alignment
    rt_a, rt_b = diff.map_a.rt, diff.map_b.rt
    x, columns = _uniform(rt_a)
    y, rows = _uniform(rt_b)
    values = alignment.similarity[np.ix_(columns, rows)]
    dx, dy = x[1] - x[0], y[1] - y[0]
    image = DiagnosticsImage(
        values,
        (x[0] - dx / 2, y[0] - dy / 2, dx * len(x), dy * len(y)),
        tuple(float(level) for level in np.percentile(alignment.similarity, (1, 99))),
    )
    ends = np.array([max(rt_a[0], rt_b[0]), min(rt_a[-1], rt_b[-1])])
    curves = [
        Curve("guide", ends, ends, "same RT"),
        Curve("path", rt_a[alignment.path[:, 0]], rt_b[alignment.path[:, 1]], "DTW path"),
        Curve("anchors", rt_a[alignment.anchors[:, 0]], rt_b[alignment.anchors[:, 1]], "anchors"),
    ]
    return DiagnosticsPlot("B's RT (min)", curves, image)


def rt_shift(diff: MzRtDiff) -> DiagnosticsPlot:
    """How far each scan of B is moved onto A's RT axis."""
    alignment = diff.alignment
    rt_a, rt_b = diff.map_a.rt, diff.map_b.rt
    anchor_a, anchor_b = rt_a[alignment.anchors[:, 0]], rt_b[alignment.anchors[:, 1]]
    curves = [
        Curve("guide", rt_a[[0, -1]], np.zeros(2)),
        Curve("main", alignment.warped_rt, alignment.warped_rt - rt_b, "PCHIP mapping"),
        Curve("anchors", anchor_a, anchor_a - anchor_b, "anchors"),
    ]
    return DiagnosticsPlot("B's RT shift (min)", curves)


def path_similarity(diff: MzRtDiff) -> DiagnosticsPlot:
    """Similarity of the matched scans along the DTW path; low where the match is doubtful."""
    alignment = diff.alignment
    rt_a = diff.map_a.rt
    path, anchors = alignment.path, alignment.anchors
    curves = [
        Curve("main", rt_a[path[:, 0]], alignment.similarity[path[:, 0], path[:, 1]], "DTW path"),
        Curve("anchors", rt_a[anchors[:, 0]], alignment.similarity[anchors[:, 0], anchors[:, 1]], "anchors"),
    ]
    return DiagnosticsPlot("Similarity (cosine)", curves)


def tic(diff: MzRtDiff, b_factor: float) -> DiagnosticsPlot:
    """TIC of A, and of B (times b_factor) before and after the alignment."""
    rt_a, tic_a = diff.map_a.ms_data_model.extract_tic(scan_indices=diff.map_a.scan_indices)
    rt_b, tic_b = diff.map_b.ms_data_model.extract_tic(scan_indices=diff.map_b.scan_indices)
    b_name = "B" if b_factor == 1.0 else f"B ×{b_factor:.3g}"
    curves = [
        Curve("a", rt_a, tic_a, "A"),
        Curve("b_original", rt_b, tic_b * b_factor, f"{b_name} (original RT)"),
        Curve("b", diff.alignment.warped_rt, tic_b * b_factor, f"{b_name} (aligned)"),
    ]
    return DiagnosticsPlot("TIC", curves)


def squared_difference(diff: MzRtDiff, overview: DiffOverview) -> DiagnosticsPlot:
    """Sum of D^2 over the mid bins of each scan of A, separately where B and where A is higher."""
    rt_a = diff.map_a.rt
    curves = [
        Curve("b", rt_a, overview.squared_higher_in_b, "B higher"),
        Curve("a", rt_a, overview.squared_higher_in_a, "A higher"),
    ]
    return DiagnosticsPlot("Σ D² over m/z", curves)


def _uniform(rt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """As many evenly spaced RTs over the range of rt, and the nearest scan to each."""
    grid = np.linspace(rt[0], rt[-1], len(rt))
    after = np.clip(np.searchsorted(rt, grid), 1, len(rt) - 1)
    nearest = np.where(grid - rt[after - 1] <= rt[after] - grid, after - 1, after)
    return grid, nearest
