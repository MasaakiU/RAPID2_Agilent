"""The calibration of the later stage's areas' sigma (SPEC_deconvolution.md 5). Of the peaks it finds in simulated
windows of a seed of their own (apart from the training's, the validation's and the references'), with a front's
candidates: the relative error of each one's area, and the features of its fit (deconvolve.fit_features); of those,
the quantiles of the |error| as the features have them (deconvolve.fit_calibration). Written as JSON, for
deconvolve_models.estimates (evaluate_deconvolution.py --calibration); and of --test of evaluate_deconvolution's
windows, how well it covers their errors, of each category.

    python calibrate_deconvolution.py --checkpoint runs/deconvolution/best.pt --front runs/unet/best.pt \
        --count 3000 --test 2000 --out runs/deconvolution/calibration.json

Of a later stage, a front, or a real, strength or low of its own, a calibration of its own: the errors are of them.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from decode import assign
from deconvolve import Scores, fit_calibration, full, shaped
from evaluate_deconvolution import windows

CALIBRATION_SEED = 1_000_009  # of the windows: apart from the references' (1_000_005) and the validation's (1_000_007)
SHOWN = ("all", "scored", "not scored", "isolated", "rs 0.25-0.5", "rs 0.5-1.0", "neighbor x 10.0-", "snr -20.0",
         "snr 100.0-", "coarse")


def errors_of(window, estimates) -> list[tuple[np.ndarray, float]]:
    """Of the estimates taken of a window (with "features" and "scale"), those matched to a peak with a shape as
    deconvolve.Scores matches them: (features, relative error of the area, over the scale) each."""
    valid = window["valid"]
    peaks = shaped(window)
    as_taken = [replace(peak, label="positive") for peak in peaks]
    of = {id(taken): peak for taken, peak in zip(as_taken, peaks)}
    out = []
    for estimate, peak, _counts in assign(estimates, as_taken):
        if peak is None:
            continue
        true = float(np.where(valid, full(of[id(peak)], len(valid)), 0.0).sum())
        if true > 0:
            out.append((estimate["features"], (estimate["area"] - true) / estimate["scale"]))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="of the later stage (train_deconvolution.py)")
    parser.add_argument("--front", required=True, help="of the candidates: a checkpoint of train.py")
    parser.add_argument("--count", type=int, default=3000, help="windows to calibrate on")
    parser.add_argument("--test", type=int, default=0, help="windows of evaluate_deconvolution's to test on")
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--real", type=float)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--low", type=float)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    import torch

    from deconvolve_data import Front, inputs
    from deconvolve_models import LOW, deconvolve, load

    model, front = load(args.checkpoint, args.device), Front(args.front, args.device)
    config = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
    real = config["real"] if args.real is None else args.real
    strength = config["strength"] if args.strength is None else args.strength
    low = config.get("low", LOW) if args.low is None else args.low

    def found(window, calibration=None) -> list[dict]:
        valid = window["valid"]
        candidates = front(inputs(window["signal"], window["missing"], valid, [])["x"], int(valid.sum()))
        out = deconvolve(model, window["signal"], window["missing"], valid, candidates, args.device, real, strength,
                         low, calibration)
        return [peak for peak in out["peaks"] if peak["real"]]

    started, features, errors = time.time(), [], []
    for window, _rng in windows(args.count, args.length, seed=CALIBRATION_SEED):
        for f, e in errors_of(window, found(window)):
            features.append(f)
            errors.append(e)
    calibration = fit_calibration(np.array(features), np.array(errors))
    calibration |= {"checkpoint": args.checkpoint, "front": args.front, "real": real, "strength": strength,
                    "low": low, "seed": CALIBRATION_SEED, "windows": args.count, "peaks": len(errors)}
    print(f"calibrated on {len(errors)} peaks of {args.count} windows ({time.time() - started:.0f} s)")
    if args.test:
        scores = Scores()
        for window, _rng in windows(args.test, args.length):
            scores.add(window, found(window, calibration))
        by = scores.summary()["by category"]
        keys = ("peaks", "z sd (robust)", "within 1 sigma", "within 2 sigma", "within the interval")
        calibration["test"] = {"windows": args.test,
                               "by category": {name: {key: by[name][key] for key in keys if key in by[name]}
                                               for name in SHOWN if name in by}}
        for name, group in calibration["test"]["by category"].items():
            print(f"{name:18s} {group['peaks']:6d}  z sd {group.get('z sd (robust)', float('nan')):5.2f}  "
                  f"within 1 sigma {group.get('within 1 sigma', float('nan')):.2f}  "
                  f"within the interval {group.get('within the interval', float('nan')):.2f}")
    Path(args.out).write_text(json.dumps(calibration, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
