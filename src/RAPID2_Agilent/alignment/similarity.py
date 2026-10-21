"""Exchangeable scan-vector similarities with row-block CPU/CUDA execution."""

import warnings
import numpy as np

from .backend import Backend, process_rows, resolve_backend
from .config import ComputeConfig


def normalize_profiles(
    values: np.ndarray, metric: str = "cosine", *, out: np.ndarray | None = None,
    backend: Backend | None = None, chunk_size: int = 64,
) -> np.ndarray:
    if metric not in {"cosine", "pearson"}:
        raise ValueError("Unknown similarity metric")
    values = np.asarray(values)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("Expected a nonempty scan-by-bin matrix")
    if out is None:
        out = np.empty(values.shape, dtype=np.float32)
    if out.shape != values.shape or out.dtype != np.float32:
        raise ValueError("out must match the input shape and be float32")
    backend = backend or resolve_backend("cpu")
    process_rows(values, out, lambda x: backend.normalize(x, metric), backend, chunk_size)
    return out


def compute_similarity(
    reference: np.ndarray, query: np.ndarray, config: ComputeConfig = ComputeConfig(), *,
    backend: Backend | None = None, inplace: bool = False, return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """S[i,j] compares reference scan i with query scan j, over every m/z bin.

    Inputs have already undergone intensity transformation. Pearson centers each
    scan over *all* bins, including empty bins. By default inputs are preserved;
    inplace=True normalizes float32 inputs in place to save two large copies.
    The full profiles stay on the CPU; only row blocks reside on the GPU.
    """
    reference, query = np.asarray(reference), np.asarray(query)
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError("Profiles must be matrices with a common m/z-bin dimension")
    if min(*reference.shape, *query.shape) < 1:
        raise ValueError("Empty profiles")
    if inplace and (reference.dtype != np.float32 or query.dtype != np.float32):
        raise ValueError("In-place normalization requires float32 profiles")
    if inplace and np.shares_memory(reference, query):
        raise ValueError("In-place reference and query matrices must not share memory")
    backend = backend or resolve_backend(config.device)
    a = normalize_profiles(reference, config.metric, out=reference if inplace else None,
                           backend=backend, chunk_size=config.chunk_size)
    b = normalize_profiles(query, config.metric, out=query if inplace else None,
                           backend=backend, chunk_size=config.chunk_size)
    # Only O(n_reference * n_query) output is allocated, independent of m/z bins.
    similarity = np.empty((len(a), len(b)), dtype=np.float32)
    chunk = config.chunk_size
    while True:
        def fill() -> None:
            for i in range(0, len(a), chunk):
                left = backend.copy_array(a[i:i + chunk])
                for j in range(0, len(b), chunk):
                    right = backend.copy_array(b[j:j + chunk])
                    tile = left @ right.T
                    similarity[i:i + chunk, j:j + chunk] = backend.numpy(tile)
        failed = False
        try:
            fill()
        except Exception as error:
            if not backend.is_oom(error) or chunk == 1:
                raise
            failed = True
        if not failed:
            break
        backend.clear_oom()
        chunk = max(1, chunk // 2)
        warnings.warn(f"CUDA memory limit: reducing similarity tile to {chunk}", RuntimeWarning)
    if not np.all(np.isfinite(similarity)):
        raise ValueError("Similarity contains nonfinite values")
    np.clip(similarity, -1.0, 1.0, out=similarity)
    info = {
        "backend": backend.name, "device": backend.device,
        "requested_device": config.device, "similarity_chunk_size": chunk,
        "zero_norm_reference_scans": int(np.count_nonzero(~np.any(a, axis=1))),
        "zero_norm_query_scans": int(np.count_nonzero(~np.any(b, axis=1))),
    }
    return (similarity, info) if return_info else similarity
