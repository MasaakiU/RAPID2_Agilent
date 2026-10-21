"""Convert non-bijective DTW transitions to ordered one-to-one scan anchors."""

import numpy as np

from .config import AnchorConfig


def extract_bijective_anchors(
    path: np.ndarray, similarity: np.ndarray, config: AnchorConfig = AnchorConfig(),
) -> np.ndarray:
    """Select the highest-similarity candidate per maximal axial-transition block.

    A diagonal edge separates blocks. Within a block, horizontal/vertical steps
    create shared scan indices; one representative removes these ambiguities.
    Blocks separated by a diagonal have strictly ordered coordinates regardless
    of which candidate is selected. Ties select the earliest path point.

    By default global endpoints are retained as explicit boundary constraints;
    candidates sharing either endpoint's row/column are excluded. This supports
    a full-run mapping, but does not claim reliable matches at truncated run ends.
    """
    path = np.asarray(path)
    similarity = np.asarray(similarity)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or path.dtype.kind not in "iu":
        raise ValueError("Path must be an integer (n_points, 2) array")
    if similarity.ndim != 2 or not np.all(np.isfinite(similarity)):
        raise ValueError("Similarity must be a finite matrix")
    if np.any(path < 0) or np.any(path[:, 0] >= similarity.shape[0]) or np.any(path[:, 1] >= similarity.shape[1]):
        raise ValueError("Path indices fall outside the similarity matrix")
    steps = np.diff(path, axis=0)
    if np.any(steps < 0) or np.any(steps > 1) or np.any(steps.sum(axis=1) == 0):
        raise ValueError("Path must contain monotone unit diagonal/horizontal/vertical steps")
    first, last = path[0], path[-1]
    if np.any(last <= first):
        raise ValueError("Path must span both RT axes")
    splits = np.flatnonzero(np.all(steps == 1, axis=1)) + 1
    selected = []
    for block in np.split(path, splits):
        if config.preserve_endpoints:
            block = block[np.all(block > first, axis=1) & np.all(block < last, axis=1)]
        if len(block):
            score = similarity[block[:, 0], block[:, 1]]
            selected.append(block[int(np.argmax(score))])
    if config.preserve_endpoints:
        selected = [first, *selected[::config.stride], last]
    else:
        selected = selected[::config.stride]
    anchors = np.asarray(selected, dtype=np.int64).reshape(-1, 2)
    if config.max_anchors is not None and len(anchors) > config.max_anchors:
        keep = np.linspace(0, len(anchors) - 1, config.max_anchors).round().astype(int)
        anchors = anchors[keep]
    if len(anchors) < 2:
        raise ValueError("Fewer than two bijective anchors; enable endpoints or revise path/settings")
    if not np.all(np.diff(anchors, axis=0) > 0):
        raise RuntimeError("Anchor extraction failed strict one-to-one ordering")
    return anchors
