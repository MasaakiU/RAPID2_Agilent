"""Global DTW baseline consuming only a scan-by-scan similarity matrix."""

from dataclasses import dataclass
import numpy as np

from .config import DTWConfig
from .io import validate_times


@dataclass
class DTWResult:
    path: np.ndarray
    cumulative_cost: np.ndarray
    predecessors: np.ndarray
    total_cost: float
    method: str = "obiwarp"


def solve_dtw(
    similarity: np.ndarray, config: DTWConfig = DTWConfig(), *,
    reference_rt: np.ndarray | None = None, query_rt: np.ndarray | None = None,
) -> DTWResult:
    """Minimize weighted (1-S) plus a linear penalty for each axial step.

    D[i,j] = min(D[i-1,j-1] + wd*(1-S[i,j]),
                 D[i-1,j]   + wv*(1-S[i,j]) + gap,
                 D[i,j-1]   + wh*(1-S[i,j]) + gap).

    D[0,0] = wd*(1-S[0,0]). Default wd=2, wh=wv=1 balances one
    diagonal against two axial steps. Ties prefer diagonal, then vertical.
    Endpoints are fixed to (0,0) and (n-1,m-1); there is no local/free-end mode.
    Banding is a correctness mask, not yet a sparse-computation optimization.
    This is an Obiwarp-like baseline, not a numerical clone of OBI-Warp/xcms.
    """
    similarity = np.asarray(similarity)
    if similarity.ndim != 2 or min(similarity.shape) < 2:
        raise ValueError("DTW requires at least two scans in each run")
    if not np.all(np.isfinite(similarity)) or np.any(np.abs(similarity) > 1 + 1e-5):
        raise ValueError("Similarity must be finite and in [-1, 1]")
    n, m = similarity.shape
    allowed = np.ones((n, m), dtype=bool)
    if config.rt_band_seconds is not None:
        if reference_rt is None or query_rt is None:
            raise ValueError("An RT band requires both RT arrays in seconds")
        reference_rt, query_rt = validate_times(reference_rt), validate_times(query_rt)
        if reference_rt.shape != (n,) or query_rt.shape != (m,):
            raise ValueError("RT arrays must match the similarity axes")
        allowed = np.abs(reference_rt[:, None] - query_rt[None, :]) <= config.rt_band_seconds
    if not allowed[0, 0] or not allowed[-1, -1]:
        raise ValueError("RT band excludes a required global endpoint; widen the band or select common RT ranges")
    costs = np.full((n, m), np.inf, dtype=np.float64)
    predecessors = np.full((n, m), 255, dtype=np.uint8)
    costs[0, 0] = config.diagonal_weight * max(0.0, 1.0 - float(similarity[0, 0]))
    for i in range(n):
        for j in range(m):
            if (i == 0 and j == 0) or not allowed[i, j]:
                continue
            local = max(0.0, 1.0 - float(similarity[i, j]))
            diagonal = costs[i - 1, j - 1] + config.diagonal_weight * local if i and j else np.inf
            vertical = costs[i - 1, j] + config.vertical_weight * local + config.gap_penalty if i else np.inf
            horizontal = costs[i, j - 1] + config.horizontal_weight * local + config.gap_penalty if j else np.inf
            if diagonal <= vertical and diagonal <= horizontal:
                costs[i, j], predecessors[i, j] = diagonal, 0
            elif vertical <= horizontal:
                costs[i, j], predecessors[i, j] = vertical, 1
            else:
                costs[i, j], predecessors[i, j] = horizontal, 2
    if not np.isfinite(costs[-1, -1]):
        raise ValueError("No connected DTW path inside the RT band")
    path = [(n - 1, m - 1)]
    i, j = path[-1]
    while i or j:
        step = predecessors[i, j]
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        elif step == 2:
            j -= 1
        else:
            raise RuntimeError("Invalid DTW predecessor during backtracking")
        path.append((i, j))
    return DTWResult(np.asarray(path[::-1], dtype=np.int64), costs, predecessors, float(costs[-1, -1]))


def solve_alignment(
    similarity_matrix: np.ndarray, method: str = "obiwarp", *,
    config: DTWConfig = DTWConfig(), reference_rt: np.ndarray | None = None,
    query_rt: np.ndarray | None = None,
) -> DTWResult:
    """Dispatch path solvers; anchor selection and RT interpolation remain separate.

    A future solver returns the same DTWResult shape (or extends its diagnostics)
    using this exact similarity input, without rebuilding spectral representations.
    """
    if method == "obiwarp":
        return solve_dtw(similarity_matrix, config, reference_rt=reference_rt, query_rt=query_rt)
    if method == "smooth_viterbi":
        raise NotImplementedError("smooth_viterbi is an extension point; only the obiwarp baseline is implemented")
    raise ValueError(f"Unknown alignment solver: {method}")
