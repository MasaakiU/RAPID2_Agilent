"""Write the front (train.py) and the later stage (train_deconvolution.py) as ONNX, for the app to run with
onnxruntime (no torch), and check them against torch on simulated windows: of each, the outputs, the front's
candidates (decode), and the later stage's areas (deconvolve_models.estimates).

    python export_onnx.py --front runs/2026-10-29_unet_v5_200k/best.pt \
        --later runs/2026-11-02_deconv_v6_sure_100k_unet_v5/best.pt --out runs/onnx_unet_v5 [--check 50]

Written to --out: front.onnx, later.onnx, and manifest.json (the checkpoints, the inputs and outputs, the window's
length, the front's stride, the opset, and the check). Of torch's exporter of dynamo (onnx and onnxscript), on the
CPU, in float32.

- front.onnx: x (1, 3, LENGTH), the input channels (deconvolve_data.inputs) -> heat (1, cells), offset (1, cells),
  log_widths (1, 2, cells): of the heat, its logits; decode takes sigmoid(heat).
- later.onnx: x (1, 3, LENGTH), candidates (1, K, 4) of K >= 1 (deconvolve_data.table), present (1, LENGTH),
  scaling (1, 3) -> baseline, others, log_sigma (1, LENGTH), real, log_height (1, K), shape (K, LENGTH): as the
  model gives them of a candidate_mask of all there.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from decode import decode
from deconvolve_data import Front, collate, inputs
from deconvolve_models import estimates, load
from evaluate_deconvolution import windows
from train import provenance

LENGTH = 2048  # of a window, as the later stage is trained and the real traces drawn (plot_real.py)
MOST_CANDIDATES = 1024
FRONT_OUTPUTS = ("heat", "offset", "log_widths")
LATER_INPUTS = ("x", "candidates", "present", "scaling")
LATER_OUTPUTS = ("baseline", "others", "log_sigma", "real", "log_height", "shape")


class FrontOut(nn.Module):
    """The front, of its heads as a tuple."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return tuple(out[name] for name in FRONT_OUTPUTS)


class LaterOut(nn.Module):
    """The later stage of a window and all its candidates, of its outputs as a tuple."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, candidates, present, scaling):
        out = self.model(x, candidates, None, present, scaling)
        return tuple(out[name] for name in LATER_OUTPUTS)


def example(front: Front, count: int = 50):
    """Of the simulated windows (evaluate_deconvolution's), of the first with a few candidates, its batch."""
    for window, _rng in windows(count, LENGTH):
        valid = window["valid"]
        candidates = front(inputs(window["signal"], window["missing"], valid, [])["x"], int(valid.sum()))
        if len(candidates) >= 3:
            return collate([inputs(window["signal"], window["missing"], valid, candidates)])
    raise RuntimeError("no window of 3 candidates or more")


def exported(module, args, names, outputs, path, dynamic_shapes=None):
    torch.onnx.export(module, args, str(path), input_names=list(names), output_names=list(outputs),
                      dynamic_shapes=dynamic_shapes, dynamo=True)


def check(front: Front, later, front_session, later_session, count: int) -> dict:
    """Of count windows, the most difference between torch and onnxruntime: of the front's heads, of its candidates
    (all the same, or how many windows not), of the later stage's outputs, and of its areas (relative)."""
    most = {name: 0.0 for name in FRONT_OUTPUTS + LATER_OUTPUTS}
    candidates_differ, areas, checked = 0, [], 0
    for window, _rng in windows(count, LENGTH):
        valid = window["valid"]
        x = inputs(window["signal"], window["missing"], valid, [])["x"]
        with torch.no_grad():
            heads = front.model(torch.from_numpy(x)[None])
        given = front_session.run(None, {"x": x[None].astype(np.float32)})
        for name, values in zip(FRONT_OUTPUTS, given):
            most[name] = max(most[name], float(np.abs(heads[name].numpy() - values).max()))
        candidates = front(x, int(valid.sum()))
        heat = 1 / (1 + np.exp(-given[0][0]))
        theirs = decode(heat, given[1][0], given[2][0], front.model.stride, length=int(valid.sum()))
        candidates_differ += [round(c["apex"], 3) for c in theirs] != [round(c["apex"], 3) for c in candidates]
        if not candidates:
            continue
        checked += 1
        batch = collate([inputs(window["signal"], window["missing"], valid, candidates)])
        with torch.no_grad():
            ours = later(batch["x"], batch["candidates"], batch["candidate_mask"], batch["present"], batch["scaling"])
        feed = {name: batch[name].numpy().astype(np.float32) for name in LATER_INPUTS}
        theirs = dict(zip(LATER_OUTPUTS, later_session.run(None, feed)))
        for name in LATER_OUTPUTS:
            most[name] = max(most[name], float(np.abs(ours[name].numpy() - theirs[name]).max()))
        mine = estimates(ours, batch, 0.3, 0.1, 0.02)[0]
        given = estimates({name: torch.from_numpy(values) for name, values in theirs.items()}, batch, 0.3, 0.1,
                          0.02)[0]
        for a, b in zip(mine["peaks"], given["peaks"]):
            if a["real"] and b["real"] and a["area"] > 0:
                areas.append(abs(b["area"] - a["area"]) / a["area"])
            elif a["real"] != b["real"]:
                areas.append(1.0)
    areas = np.array(areas) if areas else np.zeros(1)
    return {"windows": count, "windows with candidates": checked, "most difference": most,
            "windows of other candidates": candidates_differ,
            "areas' relative difference": {"median": float(np.median(areas)), "p99": float(np.percentile(areas, 99)),
                                           "most": float(areas.max())}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--front", required=True, help="a checkpoint of train.py")
    parser.add_argument("--later", required=True, help="a checkpoint of train_deconvolution.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--check", type=int, default=50, help="windows to check onnxruntime against torch on")
    args = parser.parse_args(argv)
    import onnx
    import onnxruntime

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    front, later = Front(args.front), load(args.later, "cpu")
    batch = example(front)  # loads the front's model too
    front_model = front.model.eval()
    exported(FrontOut(front_model), (batch["x"],), ("x",), FRONT_OUTPUTS, out / "front.onnx")
    exported(LaterOut(later), tuple(batch[name] for name in LATER_INPUTS), LATER_INPUTS, LATER_OUTPUTS,
             out / "later.onnx",
             dynamic_shapes={"x": None, "candidates": {1: torch.export.Dim("candidates", min=1,
                                                                             max=MOST_CANDIDATES)},
                             "present": None, "scaling": None})
    sessions = [onnxruntime.InferenceSession(str(out / name), providers=["CPUExecutionProvider"])
                for name in ("front.onnx", "later.onnx")]
    checked = check(front, later, *sessions, args.check)
    front_saved = torch.load(args.front, map_location="cpu", weights_only=False)
    later_saved = torch.load(args.later, map_location="cpu", weights_only=False)
    manifest = {
        "front": {"checkpoint": args.front, "step": front_saved["step"], "model": front_saved["config"]["model"],
                  "provenance": front_saved["config"].get("provenance"), "stride": front_model.stride,
                  "inputs": ["x"], "outputs": list(FRONT_OUTPUTS)},
        "later": {"checkpoint": args.later, "step": later_saved["step"],
                  "provenance": later_saved["config"].get("provenance"),
                  "inputs": list(LATER_INPUTS), "outputs": list(LATER_OUTPUTS), "most candidates": MOST_CANDIDATES},
        "length": LENGTH, "opset": [{"domain": o.domain, "version": o.version}
                                    for o in onnx.load(str(out / "later.onnx")).opset_import],
        "exported": provenance() | {"torch": torch.__version__, "onnx": onnx.__version__,
                                    "onnxruntime": onnxruntime.__version__},
        "check": checked}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(checked, indent=1))
    differences = checked["most difference"]
    good = (max(differences.values()) < 1e-2 and checked["windows of other candidates"] == 0
            and checked["areas' relative difference"]["p99"] < 1e-2)
    print("as torch: " + ("yes" if good else "NO: look at the differences above"))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
