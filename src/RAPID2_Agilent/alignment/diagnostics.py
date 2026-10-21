"""Headless scientific figures; importing this module does not load Qt."""

from pathlib import Path
import numpy as np

from .io import ScanSource
from .mapping import RTMapping
from .pipeline import AlignmentResult


def _figure(rows: int, columns: int, size: tuple[float, float]):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(figure)
    return figure, figure.subplots(rows, columns, squeeze=False)


def save_diagnostics(
    result: AlignmentResult, path: str | Path, *, truth: np.ndarray | None = None,
) -> Path:
    """Show matrix, path+anchors, mapping, correction, local slopes and scores."""
    figure, axes = _figure(2, 3, (16, 9))
    reference, query = result.reference_rt_seconds, result.query_rt_seconds
    path_points, anchors = result.dtw.path, result.anchors
    metric = result.metadata["config"]["compute"]["metric"]
    for ax in axes[0, :2]:
        mesh = ax.pcolormesh(query, reference, result.similarity, shading="nearest",
                             cmap="viridis", vmin=-1 if metric == "pearson" else 0, vmax=1,
                             rasterized=True)
        ax.set(xlabel="Query RT (s)", ylabel="Reference RT (s)")
        figure.colorbar(mesh, ax=ax, label=metric, shrink=0.85)
    axes[0, 0].set_title("Spectral similarity")
    ax = axes[0, 1]
    ax.plot(query[path_points[:, 1]], reference[path_points[:, 0]], color="white", lw=1, label="Raw DTW path")
    ax.scatter(query[anchors[:, 1]], reference[anchors[:, 0]], c="#f05a47", s=13, label="Bijective anchors", zorder=3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("DTW path and selected anchors")
    dense_rt = np.linspace(query[0], query[-1], 1000)
    mapped = result.mapping.transform(dense_rt)
    ax = axes[0, 2]
    ax.plot(dense_rt, dense_rt, "--", color="0.6", lw=1, label="Identity")
    ax.plot(dense_rt, mapped, color="#2176ae", label="PCHIP")
    ax.scatter(result.mapping.query_seconds, result.mapping.reference_seconds, s=10, c="#f05a47", label="Anchors")
    if truth is not None:
        ax.plot(query, truth, color="#218c54", ls="--", label="Ground truth")
    ax.set(title="Query → reference mapping", xlabel="Original query RT (s)", ylabel="Corrected RT (s)")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(dense_rt, mapped - dense_rt, color="#2176ae", label="Estimated")
    if truth is not None:
        ax.plot(query, truth - query, color="#218c54", ls="--", label="Ground truth")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set(title="RT correction", xlabel="Original query RT (s)", ylabel="Corrected − original RT (s)")
    ax.legend(fontsize=8)
    slope = np.gradient(mapped, dense_rt)
    axes[1, 1].plot(dense_rt, slope, color="#2176ae")
    axes[1, 1].axhline(1, color="0.6", ls="--", lw=1)
    axes[1, 1].set(title="Local mapping slope", xlabel="Original query RT (s)", ylabel="d(reference RT) / d(query RT)")
    finite_slope = slope[np.isfinite(slope)]
    if len(finite_slope):
        lo, hi = min(1.0, float(finite_slope.min())), max(1.0, float(finite_slope.max()))
        pad = max(0.1, 0.05 * (hi - lo))
        axes[1, 1].set_ylim(lo - pad, hi + pad)
    axes[1, 1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1, 2].plot(query[path_points[:, 1]], result.similarity[path_points[:, 0], path_points[:, 1]], color="#2176ae", lw=1)
    axes[1, 2].set(title="Similarity along raw path", xlabel="Original query RT (s)", ylabel=metric)
    axes[1, 2].set_ylim(-1.02 if metric == "pearson" else -0.02, 1.02)
    config = result.metadata["config"]
    label = (f"{result.metadata['reference'].get('name', 'reference')}  ←  {result.metadata['query'].get('name', 'query')}\n"
             f"{config['binning']['ppm']:g} ppm · {config['binning']['aggregation']} · "
             f"{config['transform']['method']} (scale={config['transform']['scale']}) · {metric} · "
             f"{len(anchors)} anchors")
    figure.suptitle(label, fontsize=13)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    return path


def extract_xics(source: ScanSource, mz_ranges: list[tuple[float, float]]) -> np.ndarray:
    """Sum raw intensities in half-open m/z intervals, without peak picking."""
    if not mz_ranges or any(not (0 < lo < hi) for lo, hi in mz_ranges):
        raise ValueError("Choose increasing positive m/z intervals")
    values = np.empty((len(source.rt_seconds), len(mz_ranges)), dtype=np.float64)
    for i in range(len(source.rt_seconds)):
        mz, intensity = source.read_scan(i)
        for k, (lo, hi) in enumerate(mz_ranges):
            values[i, k] = np.sum(intensity[(lo <= mz) & (mz < hi)], dtype=np.float64)
    return values


def save_xic_overlay(
    reference: ScanSource, query: ScanSource, mapping: RTMapping,
    mz_ranges: list[tuple[float, float]], path: str | Path,
) -> Path:
    """Change query x-coordinates only; intensities/peak areas are not resampled."""
    a, b = extract_xics(reference, mz_ranges), extract_xics(query, mz_ranges)
    figure, axes = _figure(len(mz_ranges), 2, (12, 3.2 * len(mz_ranges)))
    for k, (lo, hi) in enumerate(mz_ranges):
        for column, rt in enumerate((query.rt_seconds, mapping.transform(query.rt_seconds))):
            ax = axes[k, column]
            ax.plot(reference.rt_seconds, a[:, k], label="Reference", color="#2176ae")
            ax.plot(rt, b[:, k], label="Query", color="#f05a47", alpha=0.8)
            ax.set(title=f"m/z {lo:g}–{hi:g} · {'before' if column == 0 else 'after'}",
                   xlabel="RT (s)", ylabel="Summed raw intensity")
            ax.legend(fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    return path
