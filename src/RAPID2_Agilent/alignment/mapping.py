"""Portable query-to-reference RT mappings; no reader, GUI or torch dependency."""

from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import PchipInterpolator

from .io import validate_times


@dataclass(frozen=True)
class RTMapping:
    query_seconds: np.ndarray
    reference_seconds: np.ndarray
    extrapolation: str = "nan"
    _interpolator: PchipInterpolator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        query = validate_times(self.query_seconds, "query anchor RT").copy()
        reference = validate_times(self.reference_seconds, "reference anchor RT").copy()
        if query.shape != reference.shape:
            raise ValueError("Anchor RT arrays must have equal lengths")
        if self.extrapolation not in {"nan", "error", "linear"}:
            raise ValueError("Unknown extrapolation policy")
        query.flags.writeable = reference.flags.writeable = False
        object.__setattr__(self, "query_seconds", query)
        object.__setattr__(self, "reference_seconds", reference)
        object.__setattr__(self, "_interpolator", PchipInterpolator(query, reference, extrapolate=False))

    def transform(self, rt: np.ndarray | float, *, unit: str = "seconds") -> np.ndarray | float:
        return apply_rt_mapping(self, rt, unit=unit)


def fit_rt_mapping(
    anchors: np.ndarray, reference_rt: np.ndarray, query_rt: np.ndarray, *,
    extrapolation: str = "nan",
) -> RTMapping:
    """Fit reference_RT = f(query_RT) using scan-index anchors (reference, query)."""
    reference_rt, query_rt = validate_times(reference_rt), validate_times(query_rt)
    anchors = np.asarray(anchors)
    if anchors.ndim != 2 or anchors.shape[1] != 2 or anchors.dtype.kind not in "iu":
        raise ValueError("Anchors must be integer reference/query scan-index pairs")
    if np.any(anchors < 0) or np.any(anchors[:, 0] >= len(reference_rt)) or np.any(anchors[:, 1] >= len(query_rt)):
        raise ValueError("Anchor indices out of bounds")
    return RTMapping(query_rt[anchors[:, 1]], reference_rt[anchors[:, 0]], extrapolation)


def apply_rt_mapping(mapping: RTMapping, rt: np.ndarray | float, *, unit: str = "seconds") -> np.ndarray | float:
    """Input and output have the same declared unit; stored anchors use seconds.

    PCHIP preserves increasing anchor order inside its domain. Optional outside
    extrapolation uses positive endpoint secants, not unconstrained cubic tails.
    """
    if unit not in {"seconds", "minutes"}:
        raise ValueError("unit must be seconds or minutes")
    factor = 60.0 if unit == "minutes" else 1.0
    values = np.asarray(rt, dtype=np.float64) * factor
    if not np.all(np.isfinite(values)):
        raise ValueError("Input RTs must be finite")
    x, y = mapping.query_seconds, mapping.reference_seconds
    left, right = values < x[0], values > x[-1]
    if mapping.extrapolation == "error" and np.any(left | right):
        raise ValueError("RT falls outside mapping's supported anchor range")
    result = mapping._interpolator(values)
    if mapping.extrapolation == "linear":
        left_slope = (y[1] - y[0]) / (x[1] - x[0])
        right_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        result = np.where(left, y[0] + (values - x[0]) * left_slope, result)
        result = np.where(right, y[-1] + (values - x[-1]) * right_slope, result)
    result = result / factor
    return float(result) if values.ndim == 0 else result
