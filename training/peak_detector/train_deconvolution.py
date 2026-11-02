"""Train the later stage (SPEC_deconvolution.md 4) on windows simulated as it goes, with the candidates a front gives
of them (deconvolve_data.py), and check it on a fixed set of them as it is to be used: the candidates of realness from
`real` taken, and their heights by least squares (deconvolve_models.estimates; those from `low` fitted to the others).

    python train_deconvolution.py --config configs/deconvolution.json --out runs/deconvolution
        [--resume runs/deconvolution/last.pt] [--device cuda] --set front=runs/unet/best.pt [steps=1000 ...]

The front (a checkpoint of train.py) runs in each worker, on the CPU; in `rough` of the windows, its candidates are
made worse (deconvolve.roughened). Of a list of fronts (front=[a.pt, b.pt]), each window's candidates are of one of
them at random, and the validation set's of each in turn. Without one, the candidates are made of the true peaks
(perturbed_candidates).

Written to --out: config.json, metrics.jsonl (a line each log and each check), last.pt and best.pt (of the least
objective on the validation set: the mean of log(1 + |relative error|) of the peaks' areas, those missed as 1, and
the area of the candidates matched to no peak, over all the peaks').
"""

from train import autocast, load_config, provenance, schedule  # first: of numpy's threads (train.py)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from decode import average_precision, match  # noqa: E402
from deconvolve import CandidateErrors, Scores  # noqa: E402
from deconvolve_data import Front, SimulatedWindows, collate, validation_set  # noqa: E402
from deconvolve_losses import DEFAULT_WEIGHTS, losses  # noqa: E402
from deconvolve_models import BACKGROUND, LOW, REAL, STRENGTH, build, estimates, options_of, run  # noqa: E402
from simulator import Config  # noqa: E402

DEFAULTS = {
    "model_options": {},
    "length": 2048,
    "batch": 32,
    "steps": 100_000,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup": 2_000,
    "grad_clip": 1.0,
    "loss_weights": DEFAULT_WEIGHTS,
    "strength": STRENGTH,
    "real": REAL,
    "low": LOW,
    "workers": 8,
    "validation": 2_000,
    "eval_every": 5_000,
    "log_every": 100,
    "amp": "bf16",
    "seed": 0,
    "front": None,  # the checkpoint of train.py whose candidates are taken (or a list); none, perturbed_candidates'
    "rough": 0.3,  # of the windows, those of the front's candidates made worse (deconvolve.roughened)
    "simulator": {},  # Config fields to change
    "candidates": {},  # CandidateErrors fields to change, of roughened (or perturbed_candidates)
}
SHOWN = ("all", "isolated", "rs 0.25-0.5", "rs 0.5-1.0", "neighbor x 10.0-", "snr -20.0")  # printed of each check


@torch.no_grad()
def check(model, validation, config: dict, device: torch.device) -> dict:
    """Of the validation set: deconvolve.Scores of the candidates taken; the realness's average precision, and its
    precision and recall at `real`; the objective; and the mean losses."""
    model.eval()
    scores, records, positives, totals = Scores(), [], 0, {}
    for start in range(0, len(validation), config["batch"]):
        chunk = validation[start:start + config["batch"]]
        batch = collate([item for item, _window, _candidates in chunk])
        on = {key: value.to(device) for key, value in batch.items()}
        with autocast(config, device):
            outputs = run(model, on, device)
        for name, value in losses(outputs, on, config["loss_weights"], config["strength"]).items():
            totals[name] = totals.get(name, 0.0) + float(value) * len(chunk)
        for (_item, window, _candidates), found in zip(chunk, estimates(outputs, batch, config["real"],
                                                                        config["strength"],
                                                                        config.get("low", config["real"]))):
            scores.add(window, [peak for peak in found["peaks"] if peak["real"]])
            records += match(found["peaks"], window["peaks"])[0]
            positives += sum(peak.label == "positive" for peak in window["peaks"])
    model.train()
    summary = scores.summary()
    errors = [math.log(2) if record.get("missed") else math.log1p(abs(record["area"])) for record in scores.records]
    summary["area log error"] = float(np.mean(errors)) if errors else float("nan")
    summary["objective"] = summary["area log error"] + summary["false area"]
    true = sum(1 for confidence, is_true in records if confidence >= config["real"] and is_true)
    false = sum(1 for confidence, is_true in records if confidence >= config["real"] and not is_true)
    summary |= {"real ap": average_precision(records, positives), "real precision": true / max(true + false, 1),
                "real recall": true / max(positives, 1)}
    summary["loss"] = {name: value / len(validation) for name, value in totals.items()}
    return summary


def fronts_of(config: dict) -> list[str]:
    """The checkpoints of the fronts of a config: of front, one, a list, or none."""
    front = config["front"]
    return [front] if isinstance(front, str) else list(front or [])


def shown(summary: dict) -> dict:
    """Of a check, what is printed."""
    out = {key: value for key, value in summary.items() if key != "by category"}
    for name in SHOWN:
        if name in summary["by category"]:
            group = summary["by category"][name]
            out[name] = {key: round(group[key], 4) for key in ("peaks", "area |error| median", "area |error| p90",
                                                                "missed") if key in group}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, key=value (value as JSON)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    config = checkpoint["config"] if checkpoint else load_config(args.config, args.set, DEFAULTS)
    if checkpoint:
        config["resumed"] = config.get("resumed", []) + [provenance() | {"step": checkpoint["step"]}]
    else:
        config["provenance"] = provenance()
        config["model_options"] = {"background": BACKGROUND} | config["model_options"]  # as load reads it again
        made = [torch.load(front, map_location="cpu", weights_only=False) for front in fronts_of(config)]
        made = [{"step": m["step"], "provenance": m["config"].get("provenance")} for m in made]
        if made:
            config["front_made"] = made[0] if isinstance(config["front"], str) else made
    (out / "config.json").write_text(json.dumps(config, indent=1))
    torch.manual_seed(config["seed"])
    simulator = replace(Config(), **config["simulator"])
    errors = replace(CandidateErrors(), **config["candidates"])
    model = build(**options_of(config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule(step, config))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config["amp"] == "fp16")
    step, best = 0, math.inf
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step, best = checkpoint["step"], checkpoint["best"]
    print(f"deconvolver: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, on {device}; candidates "
          + (f"of the front {config['front']}" if config["front"] else "of the true peaks, made off (no front)"))

    def fronts(device="cpu"):
        given = [Front(front, device) for front in fronts_of(config)]
        return given[0] if isinstance(config["front"], str) else given or None

    validation = validation_set(config["validation"], config["length"], simulator, errors, fronts(device))
    # the data go on from the step resumed at, not over again
    dataset = SimulatedWindows(config["length"], seed=config["seed"] * 1_000_000 + step, config=simulator,
                               errors=errors, front=fronts(), rough=config["rough"])
    loader = DataLoader(dataset, batch_size=config["batch"], num_workers=config["workers"], collate_fn=collate,
                        pin_memory=device.type == "cuda", persistent_workers=config["workers"] > 0)

    def save(name: str):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step, "best": best, "config": config}, out / name)

    log = open(out / "metrics.jsonl", "a")
    running, started, skipped = {}, time.perf_counter(), 0
    model.train()
    for batch in loader:
        if step >= config["steps"]:
            break
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with autocast(config, device):
            outputs = run(model, batch, device)
        values = losses(outputs, batch, config["loss_weights"], config["strength"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(values["total"]).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        if torch.isfinite(norm):
            scaler.step(optimizer)
        else:  # of a step of gradients not finite, none: once, at 2000 of 3000 steps on the CPU, all went NaN after
            skipped += 1
        scaler.update()
        scheduler.step()
        step += 1
        for name, value in values.items():
            running[name] = running.get(name, 0.0) + value.item()
        if step % config["log_every"] == 0:
            rate = config["log_every"] / (time.perf_counter() - started)
            record = {"step": step, "lr": scheduler.get_last_lr()[0], "steps_per_s": rate, "skipped": skipped}
            record |= {name: value / config["log_every"] for name, value in running.items()}
            print(json.dumps(record))
            log.write(json.dumps(record) + "\n")
            log.flush()
            running, started = {}, time.perf_counter()
        if step % config["eval_every"] == 0 or step == config["steps"]:
            summary = check(model, validation, config, device)
            print(json.dumps({"step": step, "check": shown(summary)}))
            log.write(json.dumps({"step": step, "check": summary}) + "\n")
            log.flush()
            if summary["objective"] < best:
                best = summary["objective"]
                save("best.pt")
            save("last.pt")
            started = time.perf_counter()
    log.close()
    save("last.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
