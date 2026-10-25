"""Train a detector on samples simulated as it goes (data.py), and check it on a fixed set of them.

    python train.py --config configs/transformer.json --out runs/transformer [--resume runs/transformer/last.pt]
        [--device cuda] [--set steps=1000 batch=32 ...]

Written to --out: config.json, metrics.jsonl (a line each log and each check), last.pt and best.pt (of the best
average precision on the validation set).
"""

import argparse
import json
import math
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import SimulatedPeaks, validation_set
from decode import decode, evaluate
from losses import DEFAULT_WEIGHTS, losses
from models import build
from simulator import VERSION, Config

DEFAULTS = {
    "model": "transformer",
    "model_options": {},
    "length": 2048,  # points of a training item
    "batch": 64,
    "steps": 200_000,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup": 2_000,
    "grad_clip": 1.0,
    "loss_weights": DEFAULT_WEIGHTS,
    "workers": 8,
    "validation": 1_000,  # samples
    "eval_every": 5_000,
    "log_every": 100,
    "amp": "bf16",  # "bf16", "fp16" or "none"; on CUDA only
    "seed": 0,
    "simulator": {},  # Config fields to change
}


def load_config(path: str | None, overrides: list[str]) -> dict:
    config = json.loads(json.dumps(DEFAULTS))
    if path:
        config |= json.loads(Path(path).read_text())
    for override in overrides:
        key, value = override.split("=", 1)
        try:
            config[key] = json.loads(value)
        except ValueError:
            config[key] = value  # a string
    return config


def provenance() -> dict:
    """What made a run: the git commit of this folder, whether it had changes not committed, the simulator's
    version, when and where."""
    here = Path(__file__).resolve().parent

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=here, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = git("rev-parse", "HEAD") or None
    changes = git("status", "--porcelain", "--", ".")
    return {"git_commit": commit, "git_dirty": None if commit is None else bool(changes),
            "simulator_version": VERSION, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": platform.node()}


def schedule(step: int, config: dict) -> float:
    """The learning rate's factor: linear up over warmup, then a cosine down to 0 at steps."""
    if step < config["warmup"]:
        return (step + 1) / config["warmup"]
    progress = (step - config["warmup"]) / max(config["steps"] - config["warmup"], 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def autocast(config: dict, device: torch.device):
    if device.type != "cuda" or config["amp"] == "none":
        return torch.autocast(device.type, enabled=False)
    return torch.autocast("cuda", dtype=torch.bfloat16 if config["amp"] == "bf16" else torch.float16)


@torch.no_grad()
def check(model, validation, config: dict, device: torch.device) -> dict:
    """The average precision and the rest on the validation set, and its mean losses."""
    model.eval()
    results, totals = [], {}
    for start in range(0, len(validation), config["batch"]):
        chunk = validation[start:start + config["batch"]]
        batch = {key: torch.stack([item[key] for item, _ in chunk]).to(device) for key in chunk[0][0]}
        with autocast(config, device):
            outputs = model(batch["x"])
        for name, value in losses(outputs, batch, config["loss_weights"]).items():
            totals[name] = totals.get(name, 0.0) + float(value) * len(chunk)
        heat = torch.sigmoid(outputs["heat"].float()).cpu().numpy()
        offset = outputs["offset"].float().cpu().numpy()
        widths = outputs["log_widths"].float().cpu().numpy()
        for i, (item, window) in enumerate(chunk):
            length = int(item["valid"].sum())
            found = decode(heat[i], offset[i], widths[i], model.stride, length=length)
            results.append((found, window["peaks"], window["meta"]))
    model.train()
    summary = evaluate(results)
    summary["loss"] = {name: value / len(validation) for name, value in totals.items()}
    return summary


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
    config = checkpoint["config"] if checkpoint else load_config(args.config, args.set)
    if checkpoint:
        config["resumed"] = config.get("resumed", []) + [provenance() | {"step": checkpoint["step"]}]
    else:
        config["provenance"] = provenance()
    (out / "config.json").write_text(json.dumps(config, indent=1))
    torch.manual_seed(config["seed"])
    simulator = replace(Config(), **config["simulator"])
    model = build(config["model"], **config["model_options"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule(step, config))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config["amp"] == "fp16")
    step, best = 0, -1.0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step, best = checkpoint["step"], checkpoint["best"]
    print(f"{config['model']}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, on {device}")

    validation = validation_set(config["validation"], config["length"], model.stride, simulator)
    # the data go on from the step resumed at, not over again
    dataset = SimulatedPeaks(config["length"], model.stride, seed=config["seed"] * 1_000_000 + step, config=simulator)
    loader = DataLoader(dataset, batch_size=config["batch"], num_workers=config["workers"],
                        pin_memory=device.type == "cuda", persistent_workers=config["workers"] > 0)

    def save(name: str):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step, "best": best, "config": config}, out / name)

    log = open(out / "metrics.jsonl", "a")
    running, started = {}, time.perf_counter()
    model.train()
    for batch in loader:
        if step >= config["steps"]:
            break
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with autocast(config, device):
            outputs = model(batch["x"])
        values = losses(outputs, batch, config["loss_weights"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(values["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        for name, value in values.items():
            running[name] = running.get(name, 0.0) + value.item()
        if step % config["log_every"] == 0:
            rate = config["log_every"] / (time.perf_counter() - started)
            record = {"step": step, "lr": scheduler.get_last_lr()[0], "steps_per_s": rate}
            record |= {name: value / config["log_every"] for name, value in running.items()}
            print(json.dumps(record))
            log.write(json.dumps(record) + "\n")
            log.flush()
            running, started = {}, time.perf_counter()
        if step % config["eval_every"] == 0 or step == config["steps"]:
            summary = check(model, validation, config, device)
            print(json.dumps({"step": step, "check": {k: v for k, v in summary.items() if k != "recall_by"}}))
            log.write(json.dumps({"step": step, "check": summary}) + "\n")
            log.flush()
            if not np.isnan(summary["ap"]) and summary["ap"] > best:
                best = summary["ap"]
                save("best.pt")
            save("last.pt")
            started = time.perf_counter()
    log.close()
    save("last.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
