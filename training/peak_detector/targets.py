"""What the detector is trained to give, on its grid of cells, `stride` points each, from a sample's peaks (SPEC.md 5).
numpy only.

    heat:       1 at the cell of a positive's apex, a Gaussian about it, 0 elsewhere (negatives included)
    weight:     0 where it is not scored: about the apex of a peak not scored, and past the end of the sample
    offset:     where in its cell the apex is, 0 to 1; at the cells of positives' apexes (reg_mask)
    log_widths: log of the distances from the apex to 5 % of its height, left and right, in points; there too

A position x (points) is in cell floor((x + 0.5) / stride); the cell's place of x is (x + 0.5) / stride less that.
"""

import numpy as np

HEAT_SIGMA = (0.5, 0.1)  # the Gaussian's sd, in cells: the larger of 0.5 and 0.1 FWHM
IGNORE_RADIUS = (1.0, 0.25)  # not scored about an apex: the larger of 1 cell and 0.25 FWHM


def cells_of(length: int, stride: int) -> int:
    return -(-length // stride)


def make_targets(peaks, length: int, stride: int, valid: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The targets of the peaks (objects with apex, fwhm, left, right, snr, label) of a sample of length points,
    those of valid False past its end."""
    cells = cells_of(length, stride)
    index = np.arange(cells) + 0.5
    heat = np.zeros(cells, dtype=np.float32)
    weight = np.ones(cells, dtype=np.float32)
    if valid is not None:
        padded = np.zeros(cells * stride, dtype=bool)
        padded[: len(valid)] = valid
        weight[~padded.reshape(cells, stride).any(axis=1)] = 0.0
    offset = np.zeros(cells, dtype=np.float32)
    log_widths = np.zeros((2, cells), dtype=np.float32)
    reg_mask = np.zeros(cells, dtype=np.float32)
    # the stronger last, so that of two in a cell it is the one taken
    for peak in sorted((p for p in peaks if p.label == "positive"), key=lambda p: p.snr):
        place = (peak.apex + 0.5) / stride
        cell = min(max(int(np.floor(place)), 0), cells - 1)
        sigma = max(HEAT_SIGMA[0], HEAT_SIGMA[1] * peak.fwhm / stride)
        heat = np.maximum(heat, np.exp(-((index - place) ** 2) / (2 * sigma ** 2)).astype(np.float32))
        heat[cell] = 1.0
        offset[cell] = place - cell
        log_widths[:, cell] = np.log([max(peak.left, 0.5), max(peak.right, 0.5)])
        reg_mask[cell] = 1.0
    for peak in peaks:
        if peak.label == "ignore":
            place = (peak.apex + 0.5) / stride
            radius = max(IGNORE_RADIUS[0], IGNORE_RADIUS[1] * peak.fwhm / stride)
            weight[(np.abs(index - place) <= radius) & (heat < 1.0)] = 0.0
    return {"heat": heat, "weight": weight, "offset": offset, "log_widths": log_widths, "reg_mask": reg_mask}


def positions_of(cells: np.ndarray, offsets: np.ndarray, stride: int) -> np.ndarray:
    """The positions (points) of apexes at these cells and offsets."""
    return (cells + offsets) * stride - 0.5
