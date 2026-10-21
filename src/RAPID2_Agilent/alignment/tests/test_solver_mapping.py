import numpy as np
import pytest

from RAPID2_Agilent.alignment.anchors import extract_bijective_anchors
from RAPID2_Agilent.alignment.config import AnchorConfig, DTWConfig
from RAPID2_Agilent.alignment.dtw import solve_alignment, solve_dtw
from RAPID2_Agilent.alignment.mapping import RTMapping, apply_rt_mapping, fit_rt_mapping


@pytest.mark.parametrize("seed", range(5))
def test_dtw_is_optimal_against_exhaustive_paths(seed):
    similarity = np.random.default_rng(seed).uniform(-1, 1, (3, 4))
    config = DTWConfig(1.3, 0.8, 1.1, 0.2)
    candidates = []

    def visit(i, j, cost, path):
        if (i, j) == (2, 3):
            candidates.append((cost, path))
        for di, dj, weight in ((1, 1, 1.3), (1, 0, 1.1), (0, 1, 0.8)):
            ni, nj = i + di, j + dj
            if ni < 3 and nj < 4:
                penalty = 0 if di == dj else 0.2
                visit(ni, nj, cost + weight * (1 - similarity[ni, nj]) + penalty, path + [(ni, nj)])

    visit(0, 0, 1.3 * (1 - similarity[0, 0]), [(0, 0)])
    result = solve_dtw(similarity, config)
    expected_cost, expected_path = min(candidates, key=lambda x: x[0])
    assert result.total_cost == pytest.approx(expected_cost)
    np.testing.assert_array_equal(result.path, expected_path)


def test_band_and_unreachable_path():
    similarity = np.eye(5, dtype=np.float32)
    times = np.arange(5, dtype=float)
    result = solve_dtw(similarity, DTWConfig(rt_band_seconds=0), reference_rt=times, query_rt=times)
    np.testing.assert_array_equal(result.path, np.column_stack((np.arange(5), np.arange(5))))
    with pytest.raises(ValueError, match="endpoint"):
        solve_dtw(similarity, DTWConfig(rt_band_seconds=0), reference_rt=times, query_rt=times + 1)
    with pytest.raises(ValueError, match="connected"):
        solve_dtw(np.ones((3, 3)), DTWConfig(rt_band_seconds=0.1),
                  reference_rt=np.array([0, 2, 4]), query_rt=np.array([0, 1, 4]))


def test_anchor_blocks_select_local_similarity_maxima():
    path = np.array([[0, 0], [1, 1], [1, 2], [2, 2], [3, 3], [4, 4], [5, 5]])
    similarity = np.zeros((6, 6))
    similarity[1, 2] = 0.9
    anchors = extract_bijective_anchors(path, similarity)
    np.testing.assert_array_equal(anchors, [[0, 0], [1, 2], [3, 3], [4, 4], [5, 5]])
    sparse = extract_bijective_anchors(path, similarity, AnchorConfig(max_anchors=3))
    assert len(sparse) == 3
    np.testing.assert_array_equal(sparse[[0, -1]], [[0, 0], [5, 5]])
    assert np.all(np.diff(sparse, axis=0) > 0)


def test_entire_axial_block_and_endpoint_policy():
    path = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3], [3, 3]])
    similarity = np.zeros((4, 4))
    similarity[1, 2] = 1
    np.testing.assert_array_equal(extract_bijective_anchors(path, similarity), [[0, 0], [1, 2], [3, 3]])
    with pytest.raises(ValueError, match="Fewer than two"):
        extract_bijective_anchors(path, similarity, AnchorConfig(preserve_endpoints=False))


def test_pchip_direction_units_and_monotonicity():
    reference_rt = np.array([0, 50, 100, 200.])
    query_rt = np.array([0, 60, 130, 220.])
    anchors = np.column_stack((np.arange(4), np.arange(4)))
    mapping = fit_rt_mapping(anchors, reference_rt, query_rt)
    np.testing.assert_allclose(mapping.transform(query_rt), reference_rt)
    np.testing.assert_allclose(apply_rt_mapping(mapping, query_rt / 60, unit="minutes"), reference_rt / 60)
    assert np.all(np.diff(mapping.transform(np.linspace(0, 220, 2000))) > 0)
    assert np.isnan(mapping.transform(-1))
    assert mapping.transform(60.) == pytest.approx(50.)
    with pytest.raises(ValueError, match="outside"):
        RTMapping(query_rt, reference_rt, "error").transform(-1)
    linear = RTMapping(query_rt, reference_rt, "linear")
    assert linear.transform(-60) == pytest.approx(-50)
    assert linear.transform(310) == pytest.approx(300)


@pytest.mark.parametrize("query,reference", [([0, 0], [0, 1]), ([0, 1], [1, 0]), ([0, np.nan], [0, 1])])
def test_invalid_anchor_times_fail(query, reference):
    with pytest.raises(ValueError):
        RTMapping(np.array(query), np.array(reference))


def test_future_solver_is_explicit_not_silently_dtw():
    with pytest.raises(NotImplementedError, match="extension point"):
        solve_alignment(np.eye(3), method="smooth_viterbi")
