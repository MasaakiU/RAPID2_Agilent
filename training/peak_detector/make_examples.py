"""Simulate samples (simulator.py), write them as one .npz, and draw some of them.

    python training/peak_detector/make_examples.py --count 100 --out examples [--seed 0] [--drawn 30]

The .npz holds "signal", "clean", "twin" (NaN where a sample has none) and "missing", all the samples one after
another, and "meta", a JSON list of each sample's start, end, peaks and the rest.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from simulator import Config, simulate

MARKERS = {"positive": ("^", "#00A000"), "ignore": ("D", "#E08000"), "negative": ("x", "#D00000")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out", default="examples", help="a folder for samples.npz, samples.png and samples_log.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drawn", type=int, default=30)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = Config()
    samples = [simulate(np.random.default_rng([args.seed, i]), config) for i in range(args.count)]
    write_samples(out / "samples.npz", samples)
    draw(out / "samples.png", samples[: args.drawn])
    draw(out / "samples_log.png", samples[: args.drawn], log=True)
    labels = [peak.label for sample in samples for peak in sample.peaks]
    print(f"{len(samples)} samples, {sum(len(s.signal) for s in samples)} points; peaks: "
          + ", ".join(f"{labels.count(label)} {label}" for label in MARKERS) + f"; written to {out}")
    return 0


def write_samples(path: Path, samples) -> None:
    meta, start = [], 0
    for sample in samples:
        peaks = [{k: v for k, v in asdict(p).items() if k not in ("start", "profile")} for p in sample.peaks]
        meta.append({"start": start, "end": start + len(sample.signal), "peaks": peaks, "meta": sample.meta})
        start += len(sample.signal)
    np.savez_compressed(
        path,
        meta=json.dumps(meta),
        signal=np.concatenate([s.signal for s in samples]),
        clean=np.concatenate([s.clean for s in samples]),
        twin=np.concatenate([np.full(len(s.signal), np.nan, np.float32) if s.twin is None else s.twin for s in samples]),
        missing=np.concatenate([s.missing for s in samples]),
    )


def draw(path: Path, samples, *, log: bool = False) -> None:
    """A panel of each sample; on a symmetric log scale, which shows small peaks beside large ones, if log."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 3
    rows = -(-len(samples) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(18, 2.6 * rows), squeeze=False)
    for ax, (i, sample) in zip(axes.flat, enumerate(samples)):
        x = np.arange(len(sample.signal))
        ax.plot(x, np.where(sample.missing, np.nan, sample.signal), color="#B0B0B0", linewidth=0.6)
        ax.plot(x, sample.clean, color="#1f77b4", linewidth=0.9)
        for peak in sample.peaks:
            marker, color = MARKERS[peak.label]
            y = float(np.interp(peak.apex, x, sample.clean))
            ax.plot(peak.apex, y, marker=marker, color=color, markersize=6, linestyle="none")
            if peak.saturated:
                ax.annotate("S", (peak.apex, y), textcoords="offset points", xytext=(4, 4), fontsize=7, color=color)
        if sample.missing.any():
            ax.plot(x[sample.missing], np.full(sample.missing.sum(), sample.clean.min()), "|", color="#D00000",
                    markersize=4)
        counts = {label: sum(p.label == label for p in sample.peaks) for label in MARKERS}
        ax.set_title(
            f"#{i}  {len(x)} pts, width {sample.meta['width']:.0f}  "
            f"+{counts['positive']} ?{counts['ignore']} -{counts['negative']}  "
            f"noise: {'/'.join(sample.meta['noise'])}  base: {'/'.join(sample.meta['baseline'])}"
            + (f"  coarse x{sample.meta['coarse']}" if sample.meta.get("coarse", 1) > 1 else "")
            + ("  (reversed)" if sample.meta.get("reversed") else ""),
            fontsize=7, loc="left",
        )
        ax.tick_params(labelsize=6)
        if log:
            ax.set_yscale("symlog", linthresh=10)
    for ax in axes.flat[len(samples):]:
        ax.axis("off")
    from matplotlib.lines import Line2D

    legend = [Line2D([], [], color="#B0B0B0", label="signal (with noise)"),
              Line2D([], [], color="#1f77b4", label="clean (without noise)"),
              Line2D([], [], marker="^", color="#00A000", linestyle="none", label="peak (positive)"),
              Line2D([], [], marker="D", color="#E08000", linestyle="none", label="not scored (ignore)"),
              Line2D([], [], marker="x", color="#D00000", linestyle="none", label="no peak (negative: SNR <= 2)"),
              Line2D([], [], marker="|", color="#D00000", linestyle="none", label="missing point"),
              Line2D([], [], marker="$S$", color="#00A000", linestyle="none", label="saturated")]
    figure.legend(handles=legend, loc="upper center", ncol=len(legend), fontsize=9, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.3 / rows))
    figure.savefig(path, dpi=110)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
