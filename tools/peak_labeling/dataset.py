"""XICs to label the peaks of, as extract_xics.py writes them and label_peaks.py reads them, and the candidate tops
offered there. A tool apart from the app: it uses RAPID2_Agilent, and the app nothing of it.

A dataset is one .npz: "meta", a JSON list of the traces (id, run, kind, name, m/z window, polarity, MS level, and
start and end, their points in "rt" and "intensity", which hold all the traces one after another).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from RAPID2_Agilent.services import kensert_cnn
from RAPID2_Agilent.services import peak_detection

# the CNN's tops offered, of a probability of this or more: low, so that few peaks are missed for want of one
CANDIDATE_PROBABILITY = 0.3
# a top of the scipy stand-in within this many scans of one of the CNN's is the same
SAME_TOP_SCANS = 3
# the scipy stand-in's tops offered stand out of the trace by this many times its noise (their prominence): in a
# trace of noise alone it takes some tens of bumps for peaks
SCIPY_NOISE_TIMES = 5


@dataclass(frozen=True)
class Trace:
    id: str
    run: str  # its .d folder
    kind: str  # "target", "spot" (a strong spot of the run's m/z-RT map) or "random" (an m/z at random)
    name: str
    mz_window: tuple[float, float] | None
    polarity: int | None
    ms_level: int | None
    rt: np.ndarray
    intensity: np.ndarray


def write_dataset(path: str | Path, traces: list[Trace]) -> None:
    meta, start = [], 0
    for trace in traces:
        record = {key: value for key, value in asdict(trace).items() if key not in ("rt", "intensity")}
        meta.append(record | {"start": start, "end": start + len(trace.rt)})
        start += len(trace.rt)
    np.savez_compressed(
        path,
        meta=json.dumps(meta),
        rt=np.concatenate([trace.rt for trace in traces]) if traces else np.array([]),
        intensity=np.concatenate([trace.intensity for trace in traces]) if traces else np.array([]),
    )


def read_dataset(path: str | Path) -> list[Trace]:
    with np.load(path) as data:
        meta, rt, intensity = json.loads(str(data["meta"])), data["rt"], data["intensity"]
    return [
        Trace(
            record["id"], record["run"], record["kind"], record["name"],
            None if record["mz_window"] is None else tuple(record["mz_window"]),
            record["polarity"], record["ms_level"],
            rt[record["start"]:record["end"]], intensity[record["start"]:record["end"]],
        )
        for record in meta
    ]


def candidate_tops(rt: np.ndarray, intensity: np.ndarray) -> list[dict]:
    """The tops to offer, in RT order: the CNN's of Kensert et al. of a probability of CANDIDATE_PROBABILITY or more,
    as peak_detection.kensert_peaks draws the trace for it, each with its probability; and the scipy stand-in's that
    are not near one of them, with none. {"rt", "source", "confidence"} each."""
    rt, intensity = np.asarray(rt, dtype=np.float64), np.asarray(intensity, dtype=np.float64)
    if len(rt) < 3 or intensity.max() <= intensity.min():
        return []
    signal = intensity - np.percentile(intensity, peak_detection.BASELINE_PERCENTILE)
    frame = max(peak_detection.FRAME_MINUTES, rt[-1] - rt[0])
    x = rt[0] + frame * np.linspace(0, 1, kensert_cnn.INPUT_SIZE)
    drawn = np.interp(x, rt, signal)
    probabilities, places, _areas = kensert_cnn.decode(
        kensert_cnn.predict(drawn[None] / drawn[x <= rt[-1]].max())[0], CANDIDATE_PROBABILITY
    )
    apexes = rt[0] + places * frame
    window = frame / kensert_cnn.NUM_WINDOWS
    # of tops nearer than one of the CNN's windows, the likelier, as kensert_peaks keeps them
    kept = []
    for i in np.argsort(-probabilities):
        if apexes[i] <= rt[-1] and all(abs(apexes[i] - apexes[j]) >= window for j in kept):
            kept.append(i)
    candidates = [{"rt": float(apexes[i]), "source": "kensert", "confidence": float(probabilities[i])} for i in kept]
    from scipy.signal import peak_prominences

    step = float(np.median(np.diff(rt)))
    noise = _noise(intensity)
    for peak in peak_detection.scipy_peaks(rt, intensity):
        top = int(np.argmin(np.abs(rt - peak.apex)))
        if peak_prominences(intensity, [top])[0][0] >= SCIPY_NOISE_TIMES * noise and all(
            abs(peak.apex - candidate["rt"]) > SAME_TOP_SCANS * step for candidate in candidates
        ):
            candidates.append({"rt": peak.apex, "source": "scipy", "confidence": None})
    return sorted(candidates, key=lambda candidate: candidate["rt"])


def _noise(intensity: np.ndarray) -> float:
    """The standard deviation of the trace's noise, of the median absolute difference of a point from the next."""
    return float(1.4826 * np.median(np.abs(np.diff(intensity) - np.median(np.diff(intensity)))) / np.sqrt(2))
