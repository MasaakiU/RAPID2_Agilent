"""Keep optional PyTorch and device-specific operations outside analysis logic."""

from dataclasses import dataclass
import gc
from typing import Any, Callable
import warnings
import numpy as np


@dataclass
class Backend:
    requested_device: str
    device: str = "cpu"
    torch: Any = None

    @property
    def name(self) -> str:
        return "torch" if self.torch is not None else "numpy"

    def copy_array(self, values: np.ndarray) -> Any:
        if self.torch is None:
            return np.array(values, dtype=np.float32, copy=True)
        return self.torch.tensor(values, dtype=self.torch.float32, device=self.device)

    def numpy(self, values: Any) -> np.ndarray:
        return values if self.torch is None else values.detach().cpu().numpy()

    def normalize(self, values: Any, metric: str) -> Any:
        """Float64 reductions avoid overflow; stored/multiplied vectors are float32.

        Zero cosine vectors and constant Pearson vectors map to zero similarity,
        including with themselves. They are not treated as perfect matches.
        """
        if self.torch is None:
            if metric == "pearson":
                values -= values.mean(axis=1, keepdims=True, dtype=np.float64)
            norm = np.sqrt(np.einsum("ij,ij->i", values, values, dtype=np.float64))
            np.divide(values, norm[:, None], out=values, where=norm[:, None] > 0)
        else:
            t = self.torch
            if metric == "pearson":
                values -= values.mean(dim=1, keepdim=True, dtype=t.float64).to(t.float32)
            norm = t.linalg.vector_norm(values, dim=1, keepdim=True, dtype=t.float64)
            values /= t.where(norm > 0, norm, 1)
        return values

    def is_oom(self, error: BaseException) -> bool:
        return self.device == "cuda" and self.torch is not None and isinstance(error, self.torch.cuda.OutOfMemoryError)

    def clear_oom(self) -> None:
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


def resolve_backend(device: str = "auto") -> Backend:
    if device not in {"auto", "cpu", "torch_cpu", "cuda"}:
        raise ValueError("Unknown device")
    if device == "cpu":
        return Backend(device)
    try:
        import torch
    except ImportError:
        if device != "auto":
            warnings.warn("PyTorch unavailable; falling back to NumPy CPU", RuntimeWarning)
        return Backend(device)
    if device == "torch_cpu":
        return Backend(device, "cpu", torch)
    if torch.cuda.is_available():
        return Backend(device, "cuda", torch)
    if device == "cuda":
        warnings.warn("CUDA unavailable; falling back to NumPy CPU", RuntimeWarning)
    return Backend(device)


def process_rows(
    source: np.ndarray, out: np.ndarray, operation: Callable[[Any], Any],
    backend: Backend, chunk_size: int,
) -> int:
    """Commit only successful blocks, allowing safe OOM retries with out=source."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    start = 0
    chunk = chunk_size
    while start < len(source):
        stop = min(start + chunk, len(source))
        # Scope temporary tensors in a function: no failed traceback is retained
        # when the caller reduces the tile size and clears CUDA's allocator cache.
        def attempt() -> np.ndarray:
            return backend.numpy(operation(backend.copy_array(source[start:stop])))
        failed = False
        try:
            block = attempt()
        except Exception as error:
            if not backend.is_oom(error) or chunk == 1:
                raise
            failed = True
        if failed:
            backend.clear_oom()
            chunk = max(1, chunk // 2)
            warnings.warn(f"CUDA memory limit: reducing row chunk to {chunk}", RuntimeWarning)
            continue
        if not np.all(np.isfinite(block)):
            raise ValueError("Transform/normalization produced nonfinite values")
        out[start:stop] = block
        start = stop
    return chunk
