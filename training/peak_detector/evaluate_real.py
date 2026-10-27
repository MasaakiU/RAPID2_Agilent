"""Check a detector on real XICs labeled with tools/peak_labeling (datasets .npz and their .labels.json), each drawn
at one step of RT as the app draws them, the traces done and of use only: a detector trained here, or one of the
app's (RAPID2_Agilent's peak_detection, to compare with), on the traces as measured, as the app runs it.

    python evaluate_real.py (--checkpoint runs/transformer/best.pt | --app kensert) --dataset xics.npz [more.npz ...]
        [--labels xics.labels.json] [--step-seconds 0.5] [--tolerance-seconds 3] [--tolerance-fwhm 0.5]
        [--window 4096] [--out real.json] [--details found.json]

A trace's peaks are its tops taken and added; those unsure are not scored, as the simulator's "ignore"; the rest of
it is of no peak. An apex found matches an unmatched peak within the tolerance, the most confident first; one that
matches none, and is not within it of an unsure top, is a false positive. The tolerance of a peak is the larger of
--tolerance-seconds and --tolerance-fwhm times its width at half height, as the simulated peaks' (decode.py): the top
of a broad peak, its noise on it, is placed less closely, by a labeler as by a detector.

The scores are of all the traces, of each kind and of each dataset (its file's name), at each of THRESHOLDS of the
confidence. The app's Kensert CNN gives each top its probability, from KENSERT_FLOOR up (the app takes those over
0.5); peakonly and scipy give none, each top 1. --details writes each trace's peaks, with the confidence they are
found at (null if not found), and the apexes found, each "true", "false" or "ignored" (by an unsure top).
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from decode import assign, average_precision, decode
from features import features
from models import build

OVERLAP = 512  # points of a window over the next, of a trace longer than one
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
KENSERT_FLOOR = 0.05  # of the app's Kensert CNN's tops, the least probable looked at


def labels_of(dataset: str) -> str:
    """The labels of a dataset, as label_peaks.py keeps them: beside it."""
    return str(Path(dataset).with_name(Path(dataset).name.removesuffix(".npz") + ".labels.json"))


def read_labeled(dataset: str, labels: str) -> list[dict]:
    """The traces done and of use: their run, RT (min), intensity, kind, name, and the RTs of their peaks and of their
    tops unsure."""
    with np.load(dataset) as data:
        meta, rt, intensity = json.loads(str(data["meta"])), data["rt"], data["intensity"]
    traces = json.loads(Path(labels).read_text())["traces"]
    labeled = []
    for record in meta:
        label = traces.get(record["id"])
        if not label or not label["done"] or label["unusable"]:
            continue
        tops = [c["rt"] for c in label["candidates"] if c["state"] == "taken"] + list(label["added"])
        unsure = [c["rt"] for c in label["candidates"] if c["state"] == "unsure"] + list(label.get("added_unsure", []))
        labeled.append({"id": record["id"], "run": record["run"], "kind": record["kind"], "name": record["name"],
                        "rt": rt[record["start"]:record["end"]], "intensity": intensity[record["start"]:record["end"]],
                        "peaks": sorted(tops), "unsure": sorted(unsure)})
    return labeled


BASE_MINUTES = 3.0  # each side of a top, where its baseline is looked for


def half_widths(rt: np.ndarray, intensity: np.ndarray, tops: list[float], peaks: list[float] = ()) -> list[float]:
    """The width at half height (min) of the peak of each top, of a 3-scan mean: from its highest within 3 scans of
    the top, out each side while over halfway down to the baseline -- the higher of the lowest points within
    BASE_MINUTES either side -- and not past halfway to the next of the peaks (tops taken; not those unsure, which
    may be no more than a shoulder or noise), so that noise on a broad peak does not cut it short, nor a peak beside
    it make it long."""
    mean = np.convolve(np.pad(np.asarray(intensity, dtype=float), 1, mode="edge"), np.ones(3) / 3, "valid")
    interval = float(np.median(np.diff(rt)))
    reach = int(round(BASE_MINUTES / interval))
    every = sorted(set(peaks))
    widths = []
    for top in tops:
        nearest = int(np.argmin(np.abs(rt - top)))
        near = np.arange(max(nearest - 3, 0), min(nearest + 4, len(rt)))
        peak = int(near[np.argmax(mean[near])])
        low, high = max(peak - reach, 0), min(peak + reach, len(rt) - 1)
        before, after = [t for t in every if t < top - interval / 2], [t for t in every if t > top + interval / 2]
        if before:
            low = max(low, int(np.searchsorted(rt, (before[-1] + top) / 2)))
        if after:
            high = min(high, int(np.searchsorted(rt, (after[0] + top) / 2)) - 1)
        low, high = min(low, peak), max(high, peak)
        half = (mean[peak] + max(mean[low:peak + 1].min(), mean[peak:high + 1].min())) / 2
        left, right = peak, peak
        while left > low and mean[left - 1] >= half:
            left -= 1
        while right < high and mean[right + 1] >= half:
            right += 1
        widths.append((right - left + 1) * interval if mean[peak] > half else 0.0)
    return widths


def drawn(rt: np.ndarray, intensity: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The trace at one step of RT, linearly; missing True where scans are wanting (a gap over 1.5 times the usual
    interval)."""
    grid = rt[0] + step * np.arange(int(np.floor((rt[-1] - rt[0]) / step)) + 1)
    values = np.interp(grid, rt, intensity)
    missing = np.zeros(len(grid), dtype=bool)
    usual = np.median(np.diff(rt))
    for gap in np.flatnonzero(np.diff(rt) > 1.5 * usual):
        missing |= (grid > rt[gap]) & (grid < rt[gap + 1])
    return grid, values, missing


@torch.no_grad()
def detect(model, values: np.ndarray, missing: np.ndarray, window: int, device) -> list[dict]:
    """The apexes of a whole trace, in windows of window points each over the next by OVERLAP at least, each apex of
    the window whose middle is nearest it: away from the ends of the windows, and once."""
    n = len(values)
    starts = [0] if n <= window else list(range(0, n - window, window - OVERLAP)) + [n - window]
    middles = np.array([start + min(window, n) / 2 for start in starts])
    found = []
    for i, start in enumerate(starts):
        part = slice(start, min(start + window, n))
        x, _ = features(values[part], missing[part])
        outputs = model(torch.from_numpy(x)[None].to(device))
        heat = torch.sigmoid(outputs["heat"][0].float()).cpu().numpy()
        apexes = decode(heat, outputs["offset"][0].float().cpu().numpy(),
                        outputs["log_widths"][0].float().cpu().numpy(), model.stride, length=part.stop - start)
        for apex in apexes:
            apex = apex | {"apex": apex["apex"] + start}
            if int(np.argmin(np.abs(middles - apex["apex"]))) == i:
                found.append(apex)
    return found


def app_tops(name: str, rt: np.ndarray, intensity: np.ndarray) -> list[dict]:
    """The tops one of the app's detectors finds in a trace as measured, {"rt", "confidence"} each. Kensert's as
    peak_detection.kensert_peaks has them, of a probability of KENSERT_FLOOR or more, each with it: the likelier
    first, so that those of a probability over a threshold are the app's at that threshold."""
    from RAPID2_Agilent.services import kensert_cnn, peak_detection

    if len(rt) < 3 or intensity.max() <= intensity.min():
        return []
    if name != "kensert":
        detector = {"peakonly": peak_detection.peakonly_peaks, "scipy": peak_detection.scipy_peaks}[name]
        return [{"rt": peak.apex, "confidence": 1.0} for peak in detector(rt, intensity)]
    signal = intensity - np.percentile(intensity, peak_detection.BASELINE_PERCENTILE)
    frame = max(peak_detection.FRAME_MINUTES, rt[-1] - rt[0])
    x = rt[0] + frame * np.linspace(0, 1, kensert_cnn.INPUT_SIZE)
    drawn_signal = np.interp(x, rt, signal)
    probabilities, places, _areas = kensert_cnn.decode(
        kensert_cnn.predict(drawn_signal[None] / drawn_signal[x <= rt[-1]].max())[0], KENSERT_FLOOR
    )
    apexes = rt[0] + places * frame
    window = frame / kensert_cnn.NUM_WINDOWS
    kept, tops = [], {}
    for i in np.argsort(-probabilities):
        if apexes[i] <= rt[-1] and all(abs(apexes[i] - apexes[j]) >= window for j in kept):
            kept.append(i)
            near = np.flatnonzero(np.abs(rt - apexes[i]) <= window / 2)
            top = int(near[np.argmax(intensity[near])]) if len(near) else int(np.argmin(np.abs(rt - apexes[i])))
            # as _peaks_at keeps them: of a top higher than beside it, once
            if 0 < top < len(rt) - 1 and intensity[top] > max(intensity[top - 1], intensity[top + 1]):
                tops.setdefault(top, float(probabilities[i]))
    return [{"rt": float(rt[top]), "confidence": probability} for top, probability in tops.items()]


def scores(records: list[tuple[float, bool]], positives: int, traces: int) -> dict:
    """The average precision, and the precision, recall and false positives at each of THRESHOLDS."""
    entry = {"traces": traces, "traces_peaks": positives, "ap": average_precision(records, positives)}
    for threshold in THRESHOLDS:
        true = sum(1 for confidence, is_true in records if confidence >= threshold and is_true)
        false = sum(1 for confidence, is_true in records if confidence >= threshold and not is_true)
        entry[f"precision@{threshold}"] = true / max(true + false, 1)
        entry[f"recall@{threshold}"] = true / max(positives, 1)
        entry[f"false@{threshold}"] = false
    return entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    detector = parser.add_mutually_exclusive_group(required=True)
    detector.add_argument("--checkpoint")
    detector.add_argument("--app", choices=("kensert", "peakonly", "scipy"), help="one of the app's detectors")
    parser.add_argument("--dataset", required=True, nargs="+")
    parser.add_argument("--labels", help="of a single dataset: its labels, if not beside it")
    parser.add_argument("--step-seconds", type=float, default=0.5)
    parser.add_argument("--tolerance-seconds", type=float, default=3.0)
    parser.add_argument("--tolerance-fwhm", type=float, default=0.5, help="of the peak's width at half height")
    parser.add_argument("--window", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out")
    parser.add_argument("--details", help="a .json of each trace's peaks and apexes found")
    args = parser.parse_args(argv)
    if args.labels and len(args.dataset) > 1:
        parser.error("--labels is of a single --dataset")
    model, checkpoint = None, None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = build(checkpoint["config"]["model"], **checkpoint["config"]["model_options"]).to(args.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    step = args.step_seconds / 60
    tolerance = args.tolerance_seconds / args.step_seconds  # in points of the step

    groups: dict[tuple[str, str], list] = {}  # ("all", "") or ("kind" or "dataset", its name): records, peaks, traces
    details = []
    for dataset in args.dataset:
        dataset_name = Path(dataset).name.removesuffix(".npz")
        for trace in read_labeled(dataset, args.labels or labels_of(dataset)):
            grid, values, missing = drawn(trace["rt"], trace["intensity"], step)
            if model is None:
                found = [{"apex": (top["rt"] - grid[0]) / step, "confidence": top["confidence"]}
                         for top in app_tops(args.app, trace["rt"], trace["intensity"])]
            else:
                found = detect(model, values, missing, args.window, args.device)
            peaks = [SimpleNamespace(rt=rt, apex=(rt - grid[0]) / step, fwhm=width / step, label=label)
                     for key, label in (("peaks", "positive"), ("unsure", "ignore"))
                     for rt, width in zip(trace[key], half_widths(trace["rt"], trace["intensity"], trace[key],
                                                                  trace["peaks"]))]
            assigned = assign(found, peaks, tolerance=(tolerance, args.tolerance_fwhm))
            records = [(apex["confidence"], peak is not None) for apex, peak, counts in assigned if counts]
            for key in (("all", ""), ("kind", trace["kind"]), ("dataset", dataset_name)):
                group = groups.setdefault(key, [[], 0, 0])
                group[0] += records
                group[1] += len(trace["peaks"])
                group[2] += 1
            if args.details:
                found_at = {id(peak): apex["confidence"] for apex, peak, _counts in assigned if peak is not None}
                details.append({
                    "dataset": dataset_name, "id": trace["id"], "run": trace["run"], "name": trace["name"],
                    "peaks": [{"rt": peak.rt, "fwhm": peak.fwhm * step, "unsure": peak.label == "ignore",
                               "found_at": found_at.get(id(peak))} for peak in peaks],
                    "found": [{"rt": grid[0] + apex["apex"] * step, "confidence": apex["confidence"],
                               "outcome": "true" if peak is not None else "false" if counts else "ignored"}
                              for apex, peak, counts in assigned],
                })
    summary = {"detector": args.checkpoint or f"app:{args.app}", "datasets": args.dataset}
    if checkpoint is not None:
        summary |= {"step": checkpoint["step"], "provenance": checkpoint["config"].get("provenance")}
    summary["all"] = scores(*groups[("all", "")]) if groups else scores([], 0, 0)
    for kind in ("kind", "dataset"):
        summary[f"by_{kind}"] = {name: scores(*group) for (key, name), group in groups.items() if key == kind}
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
    if args.details:
        Path(args.details).write_text(json.dumps(details, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
