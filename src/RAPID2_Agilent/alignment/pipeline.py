"""Small orchestration layer connecting independently usable analysis stages."""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable
import warnings
import numpy as np

from .anchors import extract_bijective_anchors
from .backend import resolve_backend
from .config import AlignmentConfig, ScanSelection
from .dtw import DTWResult, solve_alignment
from .io import ScanSource, source_from_reader
from .mapping import RTMapping, fit_rt_mapping
from .representation import build_profile_matrix
from .similarity import compute_similarity
from .transforms import transform_intensity


@dataclass
class AlignmentResult:
    reference_rt_seconds: np.ndarray
    query_rt_seconds: np.ndarray
    reference_scan_ids: np.ndarray
    query_scan_ids: np.ndarray
    similarity: np.ndarray
    dtw: DTWResult
    anchors: np.ndarray
    mapping: RTMapping
    metadata: dict
    normalized_reference: np.ndarray | None = None
    normalized_query: np.ndarray | None = None


def map_from_similarity(
    similarity: np.ndarray, reference_rt_seconds: np.ndarray, query_rt_seconds: np.ndarray,
    config: AlignmentConfig,
) -> tuple[DTWResult, np.ndarray, RTMapping]:
    """The DTW path, its one-to-one anchors and the query -> reference RT mapping.

    The stages after the similarity matrix, shared by `align_sources` (raw scans)
    and the GUI's B - A map (`models.mz_rt_diff.align_maps`, binned map layers):
    only how the similarity is computed differs between the two.
    """
    dtw = solve_alignment(similarity, config.solver, config=config.dtw,
                          reference_rt=reference_rt_seconds, query_rt=query_rt_seconds)
    anchors = extract_bijective_anchors(dtw.path, similarity, config.anchors)
    mapping = fit_rt_mapping(anchors, reference_rt_seconds, query_rt_seconds,
                             extrapolation=config.extrapolation)
    return dtw, anchors, mapping


def align_sources(
    reference: ScanSource, query: ScanSource, config: AlignmentConfig, *,
    keep_profiles: bool = False, progress: Callable[[str], None] | None = None,
) -> AlignmentResult:
    """Reference=A, query=B throughout; output maps B's original RT to A's RT.

    Dense profiles are transformed and normalized in place after aggregation.
    Use the individual stage functions to retain raw/transformed representations;
    keep_profiles retains the final normalized matrices for inspection only.
    """
    config.transform.validate_for_use()
    if config.solver != "obiwarp":
        if config.solver == "smooth_viterbi":
            raise NotImplementedError("smooth_viterbi is not implemented yet")
        raise ValueError(f"Unknown solver: {config.solver}")
    for key in ("ms_level", "polarity"):
        if key in reference.metadata and key in query.metadata and reference.metadata[key] != query.metadata[key]:
            raise ValueError(f"Run selections have different {key}")
    timings = {}

    def announce(message: str) -> None:
        if progress is not None:
            progress(message)

    backend = resolve_backend(config.compute.device)
    announce(f"Binning reference scans ({len(reference.rt_seconds)})")
    start = perf_counter()
    a = build_profile_matrix(reference, config.binning)
    announce(f"Binning query scans ({len(query.rt_seconds)})")
    b = build_profile_matrix(query, config.binning)
    timings["binning_seconds"] = perf_counter() - start
    announce(f"Transforming profiles on {backend.name}/{backend.device}")
    start = perf_counter()
    for profile in (a, b):
        transform_intensity(profile.values, config.transform, out=profile.values,
                            backend=backend, chunk_size=config.compute.chunk_size)
    del profile
    timings["transform_seconds"] = perf_counter() - start
    announce("Normalizing profiles and calculating scan similarities")
    start = perf_counter()
    similarity, compute_info = compute_similarity(
        a.values, b.values, config.compute, backend=backend, inplace=True, return_info=True,
    )
    timings["similarity_seconds"] = perf_counter() - start
    zero_a = compute_info["zero_norm_reference_scans"]
    zero_b = compute_info["zero_norm_query_scans"]
    if zero_a == len(a.values) or zero_b == len(b.values):
        raise ValueError("At least one run has no informative scan vectors under these settings")
    if zero_a or zero_b:
        warnings.warn(f"Uninformative scan vectors: reference={zero_a}, query={zero_b}; their similarities are zero", RuntimeWarning)
    metadata = {
        "reference": a.metadata, "query": b.metadata, "config": asdict(config),
        "compute": compute_info, "n_mz_bins": a.grid.n_bins,
        "profile_dtype": "float32", "timings": timings,
        "direction": "query_to_reference", "rt_unit": "seconds",
        "solver_description": "Global weighted DTW baseline; not an exact OBI-Warp/xcms port",
    }
    retained_a = a.values if keep_profiles else None
    retained_b = b.values if keep_profiles else None
    del a, b
    announce("Solving DTW and extracting bijective anchors")
    start = perf_counter()
    dtw, anchors, mapping = map_from_similarity(similarity, reference.rt_seconds, query.rt_seconds, config)
    timings["solver_mapping_seconds"] = perf_counter() - start
    return AlignmentResult(
        np.asarray(reference.rt_seconds).copy(), np.asarray(query.rt_seconds).copy(),
        np.asarray(reference.scan_ids).copy(), np.asarray(query.scan_ids).copy(),
        similarity, dtw, anchors, mapping, metadata, retained_a, retained_b,
    )


def align_agilent(
    reference_path: str | Path, query_path: str | Path, config: AlignmentConfig, *,
    reference_selection: ScanSelection | None = None,
    query_selection: ScanSelection | None = None,
    keep_profiles: bool = False, progress: Callable[[str], None] | None = None,
) -> AlignmentResult:
    """Read .d directories directly with the existing shared AgilentReader."""
    from ..io import AgilentReader

    config.transform.validate_for_use()
    with AgilentReader() as reference_reader, AgilentReader() as query_reader:
        reference_reader.load(Path(reference_path))
        query_reader.load(Path(query_path))
        reference = source_from_reader(reference_reader, reference_selection or config.selection)
        query = source_from_reader(query_reader, query_selection or config.selection)
        return align_sources(reference, query, config, keep_profiles=keep_profiles, progress=progress)
