import importlib.abc
import sys
import numpy as np
import pytest

from RAPID2_Agilent.alignment.backend import Backend, process_rows, resolve_backend
from RAPID2_Agilent.alignment.config import ComputeConfig, TransformConfig
from RAPID2_Agilent.alignment.similarity import compute_similarity
from RAPID2_Agilent.alignment.transforms import transform_intensity


@pytest.mark.parametrize("metric", ["cosine", "pearson"])
@pytest.mark.parametrize("chunk", [1, 3, 100])
def test_against_float64_pairwise_reference(metric, chunk):
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 10, (7, 13)).astype(np.float32)
    b = rng.uniform(0, 10, (9, 13)).astype(np.float32)
    original = a.copy()
    actual = compute_similarity(a, b, ComputeConfig(metric, "cpu", chunk))
    expected = np.empty_like(actual)
    for i, row in enumerate(a):
        for j, other in enumerate(b):
            x, y = row.astype(np.float64), other.astype(np.float64)
            if metric == "pearson":
                x, y = x - x.mean(), y - y.mean()
            expected[i, j] = x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)
    np.testing.assert_array_equal(a, original)
    in_place = compute_similarity(a, b, ComputeConfig(metric, "cpu", chunk), inplace=True)
    np.testing.assert_allclose(in_place, expected, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("metric", ["cosine", "pearson"])
def test_uninformative_vectors_are_zero_not_perfect_matches(metric):
    a = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 2]], dtype=np.float32)
    similarity = compute_similarity(a, a, ComputeConfig(metric, "cpu", 2))
    np.testing.assert_array_equal(similarity[0], [0, 0, 0])
    if metric == "pearson":
        np.testing.assert_array_equal(similarity[1], [0, 0, 0])
    assert similarity[2, 2] == pytest.approx(1)


@pytest.mark.parametrize("metric", ["cosine", "pearson"])
def test_torch_cpu_matches_numpy(metric):
    pytest.importorskip("torch")
    rng = np.random.default_rng(9)
    a, b = rng.uniform(0, 1e6, (12, 501)).astype(np.float32), rng.uniform(0, 1e6, (15, 501)).astype(np.float32)
    for method in ("none", "sqrt", "log1p", "asinh"):
        transformed = []
        for device in ("cpu", "torch_cpu"):
            backend = resolve_backend(device)
            x = transform_intensity(a, TransformConfig(method, 3), backend=backend, chunk_size=4)
            y = transform_intensity(b, TransformConfig(method, 3), backend=backend, chunk_size=4)
            transformed.append(compute_similarity(x, y, ComputeConfig(metric, device, 4)))
        np.testing.assert_allclose(*transformed, rtol=1e-4, atol=5e-7)


def test_cuda_matches_cpu_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware validation is separate")
    rng = np.random.default_rng(4)
    a, b = rng.random((9, 1024), dtype=np.float32), rng.random((11, 1024), dtype=np.float32)
    for metric in ("cosine", "pearson"):
        cpu = compute_similarity(a, b, ComputeConfig(metric, "cpu", 3))
        gpu = compute_similarity(a, b, ComputeConfig(metric, "cuda", 3))
        np.testing.assert_allclose(cpu, gpu, rtol=2e-4, atol=1e-5)


def test_missing_torch_automatically_falls_back(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_backend("auto").name == "numpy"
    with pytest.warns(RuntimeWarning, match="falling back"):
        assert resolve_backend("cuda").device == "cpu"


def test_oom_retry_does_not_transform_successful_rows_twice():
    class SimulatedOOM(Exception):
        pass

    class SmallBackend(Backend):
        def is_oom(self, error):
            return isinstance(error, SimulatedOOM)

        def clear_oom(self):
            pass

    values = np.arange(15, dtype=np.float32).reshape(5, 3)
    original = values.copy()

    def operation(block):
        if len(block) > 2:
            raise SimulatedOOM()
        return block + 1

    with pytest.warns(RuntimeWarning, match="reducing"):
        used = process_rows(values, values, operation, SmallBackend("cpu"), 4)
    assert used == 2
    np.testing.assert_array_equal(values, original + 1)
