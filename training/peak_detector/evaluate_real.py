"""Check a detector on real XICs labeled with tools/peak_labeling (its dataset .npz and .labels.json), each drawn at
one step of RT as the app draws them, the traces done and of use only.

    python evaluate_real.py --checkpoint runs/transformer/best.pt --dataset xics.npz [--labels xics.labels.json]
        [--step-seconds 0.5] [--tolerance-seconds 3] [--window 4096] [--out real.json]

A trace's peaks are its tops taken and added; the rest of it is of no peak. An apex found matches an unmatched peak
within the tolerance, the most confident first; one that matches none is a false positive.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from decode import THRESHOLDS, average_precision, decode, match
from features import features
from models import build

OVERLAP = 512  # points of a window over the next, of a trace longer than one


def read_labeled(dataset: str, labels: str) -> list[dict]:
    """The traces done and of use: their RT (min), intensity, kind, name and the RTs of their peaks."""
    with np.load(dataset) as data:
        meta, rt, intensity = json.loads(str(data["meta"])), data["rt"], data["intensity"]
    traces = json.loads(Path(labels).read_text())["traces"]
    labeled = []
    for record in meta:
        label = traces.get(record["id"])
        if not label or not label["done"] or label["unusable"]:
            continue
        tops = [c["rt"] for c in label["candidates"] if c["state"] == "taken"] + list(label["added"])
        labeled.append({"id": record["id"], "kind": record["kind"], "name": record["name"],
                        "rt": rt[record["start"]:record["end"]], "intensity": intensity[record["start"]:record["end"]],
                        "peaks": sorted(tops)})
    return labeled


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--step-seconds", type=float, default=0.5)
    parser.add_argument("--tolerance-seconds", type=float, default=3.0)
    parser.add_argument("--window", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    labels = args.labels or str(Path(args.dataset).with_name(Path(args.dataset).name.removesuffix(".npz") + ".labels.json"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build(checkpoint["config"]["model"], **checkpoint["config"]["model_options"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    step = args.step_seconds / 60
    tolerance = args.tolerance_seconds / args.step_seconds  # in points of the step

    records_by: dict[str, list] = {}
    positives_by: dict[str, int] = {}
    for trace in read_labeled(args.dataset, labels):
        grid, values, missing = drawn(trace["rt"], trace["intensity"], step)
        found = detect(model, values, missing, args.window, args.device)
        peaks = [SimpleNamespace(apex=(rt - grid[0]) / step, fwhm=0.0, label="positive") for rt in trace["peaks"]]
        records, _ = match(found, peaks, tolerance=(tolerance, 0.0))
        for key in ("all", trace["kind"]):
            records_by.setdefault(key, []).extend(records)
            positives_by[key] = positives_by.get(key, 0) + len(peaks)
    summary = {"checkpoint": args.checkpoint, "step": checkpoint["step"],
               "provenance": checkpoint["config"].get("provenance")}
    for key, records in records_by.items():
        positives = positives_by[key]
        entry = {"traces_peaks": positives, "ap": average_precision(records, positives)}
        for threshold in THRESHOLDS:
            true = sum(1 for confidence, is_true in records if confidence >= threshold and is_true)
            false = sum(1 for confidence, is_true in records if confidence >= threshold and not is_true)
            entry[f"precision@{threshold}"] = true / max(true + false, 1)
            entry[f"recall@{threshold}"] = true / max(positives, 1)
        summary[key] = entry
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
