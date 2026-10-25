"""A simulator of 1-D signals -- peaks on a baseline, with noise and artifacts -- and of their peaks, labeled as
SPEC.md defines them. numpy and scipy only, so that it runs on the GPU machine as here; nothing of RAPID2_Agilent.

The x axis is in points: a sample is of `length` points, a peak's width its FWHM in points.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfcx

SHAPES = ("gauss", "emg", "bigauss", "voigt", "unimodal")
# of the samples and their labels, as SPEC.md's: one more at each change of them, which every run records
VERSION = 2


@dataclass(frozen=True)
class Config:
    length: tuple[int, int] = (512, 4096)
    fwhm: tuple[float, float] = (2.0, 300.0)  # W_min, W_max; a hump of the baseline is wider than W_max
    width_spread: float = 0.3  # log sd of the widths of a sample's components about its width
    width_outlier: float = 0.1  # of the components, those of a width anywhere in the range
    snr: tuple[float, float] = (0.5, 1e6)  # of the components, log-uniform
    snr_positive: float = 5.0
    snr_negative: float = 2.0
    rs_split: float = 0.5
    rs_merge: float = 0.25
    merge_width_ratio: float = 3.0  # of FWHMs: components of more apart are not merged, nor a pair not scored
    empty: float = 0.1  # of the samples, those of no component
    cluster: float = 0.4  # of the components, those that start a cluster of 2 to 4
    cluster_rs: tuple[float, float] = (0.1, 2.0)
    shapes: dict = field(default_factory=lambda: {"gauss": 0.15, "emg": 0.25, "bigauss": 0.15, "voigt": 0.10,
                                                  "unimodal": 0.35})
    saturation: float = 0.1  # of the samples with a component of an SNR of 100 or more, the strongest one cut
    reverse: float = 0.5
    twin: float = 0.3  # of the samples, those given a second noise on the same clean signal
    # of the samples, those made on a grid this many times coarser, of peaks of few points (coarse_width, FWHM in its
    # points), and interpolated linearly, as real data of few points are for the model
    coarse: float = 0.3
    coarse_factor: tuple[float, float] = (2.0, 8.0)
    coarse_width: tuple[float, float] = (2.0, 8.0)


@dataclass
class Peak:
    apex: float  # points
    height: float  # of the component above the baseline
    fwhm: float
    left: float  # from the apex to where it is down to 5 % of its height, on the left
    right: float  # and on the right
    shape: str
    snr: float
    label: str  # "positive", "ignore" (not scored) or "negative"
    saturated: bool = False
    components: int = 1  # more for components too close to be told apart, taken for one


@dataclass
class Sample:
    signal: np.ndarray  # float32
    clean: np.ndarray  # the signal without the noise (but with the saturation, which the detector does)
    missing: np.ndarray  # bool, points not measured (0 in signal)
    peaks: list[Peak]
    twin: np.ndarray | None = None  # the clean signal with another noise
    meta: dict = field(default_factory=dict)


def simulate(rng: np.random.Generator, config: Config = Config()) -> Sample:
    length = int(rng.integers(config.length[0], config.length[1] + 1))
    factor = int(round(_log_uniform(rng, *config.coarse_factor))) if rng.random() < config.coarse else 1
    if factor > 1:
        grid_length = length // factor + 1
        width = _log_uniform(rng, *config.coarse_width)  # the sample's, in points of the grid
    else:
        grid_length = length
        width = _log_uniform(rng, config.fwhm[0] * 1.25, config.fwhm[1] / 1.5)
    # W_min in points of the grid, W_max in those of the sample
    fwhm_range = (config.fwhm[0], config.fwhm[1] / factor)
    sample = _simulate_grid(rng, config, grid_length, width, fwhm_range)
    if factor > 1:
        sample = _upsampled(sample, factor)
    return _reversed(sample) if rng.random() < config.reverse else sample


def _simulate_grid(rng, config, length, width, fwhm_range) -> Sample:
    x = np.arange(length, dtype=np.float64)
    components = [] if rng.random() < config.empty else _components(rng, config, length, width, fwhm_range)
    clean_peaks = np.zeros(length)
    for component in components:
        clean_peaks += component["height"] * _shape(component, x)
    baseline, baseline_kinds = _baseline(rng, x, fwhm_range[1])
    clean = baseline + clean_peaks
    saturation = _saturation(rng, config, x, components, baseline, clean)
    if saturation is not None:
        clean = saturation["clean"]
    signal, noise_kinds = _noisy(rng, clean, saturation)
    twin = _noisy(rng, clean, saturation)[0] if rng.random() < config.twin else None
    missing = _missing(rng, length)
    peaks = _labels(config, x, components, signal - clean, missing, saturation, length)
    # the points not measured: across them, as the scans not measured are left out and the rest interpolated
    signal, twin = _across(signal, missing), None if twin is None else _across(twin, missing)
    return Sample(signal.astype(np.float32), clean.astype(np.float32), missing, peaks,
                  None if twin is None else twin.astype(np.float32),
                  {"width": width, "baseline": baseline_kinds, "noise": noise_kinds, "coarse": 1})


def _across(values: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """values, those of the points missing interpolated linearly from the others."""
    if not missing.any() or missing.all():
        return values
    x = np.arange(len(values))
    return np.interp(x, x[~missing], values[~missing])


def _upsampled(sample: Sample, factor: int) -> Sample:
    """A sample of a grid, interpolated linearly to one factor times finer."""
    grid = np.arange(len(sample.signal), dtype=np.float64)
    fine = np.arange((len(grid) - 1) * factor + 1) / factor
    measured = ~sample.missing

    def up(values):
        return np.interp(fine, grid[measured], values[measured]).astype(np.float32) if measured.any() else \
            np.interp(fine, grid, values).astype(np.float32)

    peaks = [Peak(p.apex * factor, p.height, p.fwhm * factor, p.left * factor, p.right * factor, p.shape, p.snr,
                  p.label, p.saturated, p.components) for p in sample.peaks]
    return Sample(up(sample.signal), np.interp(fine, grid, sample.clean).astype(np.float32),
                  np.interp(fine, grid, sample.missing.astype(float)) > 0, peaks,
                  None if sample.twin is None else up(sample.twin),
                  sample.meta | {"width": sample.meta["width"] * factor, "coarse": factor})


### components ###
def _components(rng, config, length, width, fwhm_range) -> list[dict]:
    # as many as would fill the sample, peak by peak, times a log-uniform part: most samples sparse, some crowded
    expected = _log_uniform(rng, 0.05, 1.0) * length / (4 * width)
    count = int(rng.poisson(max(expected, 0.5)))
    shapes, weights = list(config.shapes), np.array(list(config.shapes.values()))
    components = []

    def new(center):
        fwhm = (_log_uniform(rng, *fwhm_range) if rng.random() < config.width_outlier
                else float(np.clip(width * np.exp(rng.normal(0, config.width_spread)), *fwhm_range)))
        shape = shapes[rng.choice(len(shapes), p=weights / weights.sum())]
        component = {"center": center, "fwhm": fwhm, "shape": shape, "params": _shape_params(rng, shape),
                     "height": _log_uniform(rng, *config.snr)}  # in units of the white noise, 1
        component |= _measure(component)
        components.append(component)
        return component

    while len(components) < count:
        first = new(rng.uniform(-0.05, 1.05) * length)
        if rng.random() < config.cluster:
            last = first
            for _ in range(int(rng.integers(1, 4))):
                fwhm = width * np.exp(rng.normal(0, config.width_spread))
                rs = rng.uniform(*config.cluster_rs)
                apart = rs * 2 * (last["fwhm"] + fwhm) / 2.3548
                last = new(last["center"] + apart * rng.choice([-1, 1]))
    return components


def _shape_params(rng, shape) -> dict:
    if shape == "emg":
        return {"ratio": _log_uniform(rng, 0.1, 5.0)}  # tau / sigma
    if shape == "bigauss":
        return {"ratio": _log_uniform(rng, 1 / 3, 3)}  # right sigma / left sigma
    if shape == "voigt":
        return {"eta": rng.uniform(0.2, 1.0)}  # of the Lorentzian
    if shape == "unimodal":
        # exp(-g(v)) each side, g(v) = sum a v ** p: from a cusp (p near 1) to a flat top (p near 4)
        return {side: {"a": [_log_uniform(rng, 0.1, 2.0) for _ in range(3)], "p": list(rng.uniform(1, 4, 3))}
                for side in ("left", "right")} | {"ratio": _log_uniform(rng, 0.5, 2.0)}
    return {}


def _unit_shape(shape: str, params: dict, u: np.ndarray) -> np.ndarray:
    """The shape about 0, of an arbitrary scale; its top need not be at 0, nor 1."""
    if shape == "gauss":
        return np.exp(-u ** 2 / 2)
    if shape == "emg":
        # exp(-u²/2) erfcx(z), z = (1/ratio - u)/√2; for z < 0, erfcx(z) = 2 exp(z²) - erfcx(-z), which keeps it finite
        ratio = params["ratio"]
        z = (1 / ratio - u) / np.sqrt(2)
        out = np.exp(-u ** 2 / 2) * erfcx(np.abs(z))
        tail = z < 0
        out[tail] = 2 * np.exp(1 / (2 * ratio ** 2) - u[tail] / ratio) - out[tail]
        return out
    if shape == "bigauss":
        return np.where(u < 0, np.exp(-u ** 2 / 2), np.exp(-u ** 2 / (2 * params["ratio"] ** 2)))
    if shape == "voigt":
        eta = params["eta"]
        return eta / (1 + u ** 2) + (1 - eta) * np.exp(-u ** 2 * np.log(2))
    if shape == "unimodal":
        v = np.abs(u)
        out = np.empty_like(u)
        for side, mask in (("left", u < 0), ("right", u >= 0)):
            scale = 1.0 if side == "left" else params["ratio"]
            g = sum(a * (v[mask] / scale) ** p for a, p in zip(params[side]["a"], params[side]["p"]))
            out[mask] = np.exp(-g)
        return out
    raise ValueError(shape)


FINE = np.linspace(-40, 80, 2401)  # a unit shape's support, in steps of 0.05; the crossings interpolated within them


def _measure(component: dict) -> dict:
    """The scale that gives the component its FWHM, and where its top is from its center, and its 5 % widths."""
    shape, params = component["shape"], component["params"]
    values = _unit_shape(shape, params, FINE)
    top = int(np.argmax(values))
    # the top between the points beside it, finely: a parabola would miss that of a cusp, or of sides unlike
    near = np.linspace(FINE[max(top - 1, 0)], FINE[min(top + 1, len(FINE) - 1)], 201)
    near_values = _unit_shape(shape, params, near)
    u_top, peak = float(near[np.argmax(near_values)]), float(near_values.max())
    values = values / peak
    left_half, right_half = _crossings(values, top, 0.5)
    left_five, right_five = _crossings(values, top, 0.05)
    scale = component["fwhm"] / (right_half - left_half)
    return {"scale": scale, "peak": peak, "apex": component["center"] + u_top * scale,
            "left": (u_top - left_five) * scale, "right": (right_five - u_top) * scale}


def _crossings(values: np.ndarray, top: int, level: float) -> tuple[float, float]:
    """Where the shape, its top at index top, comes down to level on each side, interpolated linearly."""
    below_left = np.flatnonzero(values[:top] < level)
    below_right = np.flatnonzero(values[top:] < level)
    if len(below_left):
        i = below_left[-1]
        left = FINE[i] + (level - values[i]) / (values[i + 1] - values[i]) * (FINE[i + 1] - FINE[i])
    else:
        left = FINE[0]
    if len(below_right):
        j = top + below_right[0]
        right = FINE[j - 1] + (values[j - 1] - level) / (values[j - 1] - values[j]) * (FINE[j] - FINE[j - 1])
    else:
        right = FINE[-1]
    return float(left), float(right)


def _shape(component: dict, x: np.ndarray) -> np.ndarray:
    """The component's shape on x, its top 1."""
    u = (x - component["center"]) / component["scale"]
    return _unit_shape(component["shape"], component["params"], u) / component["peak"]


### baseline, saturation, noise, missing points ###
def _baseline(rng, x, widest) -> tuple[np.ndarray, list[str]]:
    """An offset, a drift, humps and steps: of what changes slower than a peak of the widest FWHM. A hump no wider
    is a peak."""
    length, sigma = len(x), widest / 2.3548
    baseline, kinds = np.full(length, _log_uniform(rng, 0.1, 1e4) if rng.random() < 0.7 else 0.0), ["offset"]
    if rng.random() < 0.4:
        baseline += rng.normal(0, 1) * _log_uniform(rng, 0.1, 1e3) * (x / length - 0.5)
        kinds.append("drift")
    if rng.random() < 0.4:
        for _ in range(int(rng.integers(1, 4))):
            width = sigma * _log_uniform(rng, 1.2, 4)
            baseline += _log_uniform(rng, 0.3, 1e4) * np.exp(-((x - rng.uniform(0, length)) / width) ** 2 / 2)
        kinds.append("humps")
    if rng.random() < 0.2:
        width = sigma * _log_uniform(rng, 1, 3)
        baseline += rng.choice([-1, 1]) * _log_uniform(rng, 0.3, 1e3) / (1 + np.exp(-(x - rng.uniform(0, length)) / width))
        kinds.append("step")
    return baseline, kinds


def _saturation(rng, config, x, components, baseline, clean) -> dict | None:
    strong = [c for c in components if c["height"] >= 100 and 0 <= c["apex"] < len(x)]
    if not strong or rng.random() >= config.saturation:
        return None
    component = max(strong, key=lambda c: c["height"])
    top = min(int(round(component["apex"])), len(x) - 1)  # of a top within half a point of the end, the last point
    ceiling = baseline[top] + component["height"] * rng.uniform(0.4, 0.9)
    cut = clean > ceiling
    region = np.flatnonzero(cut)
    # the run of the cut points about the top
    start = end = top
    while start > 0 and cut[start - 1]:
        start -= 1
    while end < len(x) - 1 and cut[end + 1]:
        end += 1
    clipped = np.minimum(clean, ceiling)
    dip = None
    if rng.random() < 0.5 and end > start:
        # an M: the top lower in its middle, as a narrow m/z window misses the m/z the detector shifts there
        middle, half = (start + end) / 2, max((end - start) / 4, 1.0)
        dip = rng.uniform(0.1, 0.5)
        clipped = clipped - dip * (ceiling - baseline) * np.exp(-((x - middle) / half) ** 2 / 2) * (x >= start) * (x <= end)
    return {"component": component, "start": start, "end": end, "ceiling": ceiling, "dip": dip, "clean": clipped,
            "points": len(region)}


def _noisy(rng, clean, saturation) -> tuple[np.ndarray, list[str]]:
    length = len(clean)
    noise, kinds = rng.normal(0, 1, length), ["white"]
    if rng.random() < 0.3:
        frequencies = np.fft.rfftfreq(length)
        frequencies[0] = frequencies[1]
        colored = np.fft.irfft(np.fft.rfft(rng.normal(0, 1, length)) * frequencies ** (-rng.uniform(0.5, 1.5) / 2), length)
        noise += colored / colored.std() * rng.uniform(0.5, 2)
        kinds.append("colored")
    level = np.maximum(clean - clean.min(), 0)
    if rng.random() < 0.5:
        noise += rng.normal(0, 1, length) * np.sqrt(_log_uniform(rng, 0.01, 10) * level)
        kinds.append("shot")
    if rng.random() < 0.3:
        noise += rng.normal(0, 1, length) * rng.uniform(0.01, 0.1) * level
        kinds.append("multiplicative")
    if saturation is not None:
        # little noise where the detector is saturated
        noise[saturation["start"]:saturation["end"] + 1] *= 0.1
    signal = clean + noise
    if rng.random() < 0.3:
        count = int(rng.poisson(length * _log_uniform(rng, 1e-4, 5e-3)))
        signal[rng.integers(0, length, count)] += _log_uniform(rng, 5, 200, count)
        kinds.append("spikes")
    if rng.random() < 0.3:
        signal = np.maximum(signal, 0.0)
        kinds.append("floor")
    return signal, kinds


def _missing(rng, length) -> np.ndarray:
    missing = np.zeros(length, dtype=bool)
    if rng.random() < 0.2:
        for start in rng.integers(0, length, int(rng.poisson(length * 0.005))):
            missing[start:start + int(rng.integers(1, 6))] = True
    return missing


### labels ###
def _labels(config, x, components, noise, missing, saturation, length) -> list[Peak]:
    # components too close to tell apart are one peak, and of those between the smaller is not scored; but of widths
    # far apart they are two, one on the other, and the wider is not scored where its top is under the other
    groups = list(range(len(components)))

    def group_of(i):
        while groups[i] != i:
            groups[i] = groups[groups[i]]
            i = groups[i]
        return i

    ambiguous = []  # pairs between merge and split: whether the smaller is a second peak, not scored
    hidden = set()  # wide components whose tops are under narrower ones
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            a, b = components[i], components[j]
            if max(a["fwhm"], b["fwhm"]) > config.merge_width_ratio * min(a["fwhm"], b["fwhm"]):
                narrow, wide = (i, j) if a["fwhm"] < b["fwhm"] else (j, i)
                if abs(a["apex"] - b["apex"]) < components[narrow]["fwhm"]:
                    hidden.add(wide)
                continue
            rs = abs(a["apex"] - b["apex"]) / (2 * (a["fwhm"] + b["fwhm"]) / 2.3548)
            if rs < config.rs_merge:
                groups[group_of(i)] = group_of(j)
            elif rs < config.rs_split:
                ambiguous.append((i, j))
    members: dict[int, list[int]] = {}
    for i in range(len(components)):
        members.setdefault(group_of(i), []).append(i)
    peaks, peak_of = [], {}
    for group in members.values():
        parts = [components[i] for i in group]
        if len(parts) == 1:
            part = parts[0]
            apex, height, fwhm, left, right, shape = (part["apex"], part["height"], part["fwhm"], part["left"],
                                                      part["right"], part["shape"])
        else:
            # the top of their sum, its widths numerically
            span = np.arange(min(p["apex"] - 3 * p["left"] for p in parts) - 5,
                             max(p["apex"] + 3 * p["right"] for p in parts) + 5, 0.05)
            total = sum(p["height"] * _shape(p, span) for p in parts)
            top = int(np.argmax(total))
            apex, height = float(span[top]), float(total[top])
            half, five = np.flatnonzero(total >= height / 2), np.flatnonzero(total >= height * 0.05)
            fwhm = float(span[half[-1]] - span[half[0]])
            left, right, shape = float(span[top] - span[five[0]]), float(span[five[-1]] - span[top]), "merged"
        saturated = saturation is not None and saturation["component"] in parts
        if saturated:
            apex = (saturation["start"] + saturation["end"]) / 2  # the middle of its flat top
        if not 0 <= apex <= length - 1:
            continue
        near = np.abs(x - apex) <= max(fwhm, 5)
        quiet = noise[near & ~missing]
        local = 1.4826 * np.median(np.abs(quiet - np.median(quiet))) if len(quiet) > 3 else 1.0
        snr = height / max(local, 1e-9)
        if saturated or snr >= config.snr_positive:
            label = "positive"
        elif snr <= config.snr_negative:
            label = "negative"
        else:
            label = "ignore"
        # a peak's top cut by an end of the sample, or under a narrower peak, is not scored
        if label == "positive" and (min(apex, length - 1 - apex) < fwhm / 2 or (len(group) == 1 and group[0] in hidden)):
            label = "ignore"
        peak = Peak(float(apex), float(height), float(fwhm), float(left), float(right), shape, float(snr), label,
                    saturated, len(parts))
        peaks.append(peak)
        for i in group:
            peak_of[i] = peak
    # of a pair between merge and split, the smaller is not scored, as it may be taken for a shoulder of the other;
    # the larger is a peak either way. One lost in the noise makes no pair
    for i, j in ambiguous:
        a, b = peak_of.get(i), peak_of.get(j)
        if a is None or b is None or a is b or min(a.snr, b.snr) <= config.snr_negative:
            continue
        smaller = a if a.height < b.height else b
        if smaller.label == "positive":
            smaller.label = "ignore"
    return sorted(peaks, key=lambda peak: peak.apex)


def _reversed(sample: Sample) -> Sample:
    last = len(sample.signal) - 1
    peaks = [Peak(last - p.apex, p.height, p.fwhm, p.right, p.left, p.shape, p.snr, p.label, p.saturated, p.components)
             for p in reversed(sample.peaks)]
    return Sample(sample.signal[::-1].copy(), sample.clean[::-1].copy(), sample.missing[::-1].copy(), peaks,
                  None if sample.twin is None else sample.twin[::-1].copy(), sample.meta | {"reversed": True})


def _log_uniform(rng, low, high, size=None):
    return np.exp(rng.uniform(np.log(low), np.log(high), size))
