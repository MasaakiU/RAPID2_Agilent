from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import numpy as np
import pytest

from RAPID2_Agilent.alignment import (
    AlignmentConfig, AnchorConfig, BinningConfig, ComputeConfig, ScanSelection,
    TransformConfig, align_agilent, align_sources, load_mapping, load_result, save_result,
)
from RAPID2_Agilent.alignment.diagnostics import save_diagnostics, save_xic_overlay
from RAPID2_Agilent.alignment.io import source_from_reader
from RAPID2_Agilent.alignment.synthetic import generate_synthetic_pair, mapping_errors


def cpu_config():
    return AlignmentConfig(binning=BinningConfig(ppm=50), transform=TransformConfig(scale=1),
                           compute=ComputeConfig(device="cpu", chunk_size=16))


@pytest.fixture(scope="module")
def recovered():
    pair = generate_synthetic_pair()
    return pair, align_sources(pair.reference, pair.query, cpu_config())


def test_recovers_smooth_drift_despite_strong_run_specific_ions(recovered):
    pair, result = recovered
    errors = mapping_errors(result.mapping.transform(pair.query.rt_seconds), pair.query_to_reference_truth)
    # Scan spacing is ~3 s; sub-scan mean accuracy with bounded local errors.
    assert errors["mae_seconds"] < 2.0
    assert errors["max_abs_seconds"] < 6.0
    assert np.all(np.diff(result.mapping.transform(pair.query.rt_seconds)) > 0)
    assert result.similarity.dtype == np.float32
    assert np.all(np.diff(result.anchors, axis=0) > 0)
    assert result.normalized_reference is None
    assert result.metadata["config"]["transform"]["scale"] == 1


def test_asinh_improves_this_specific_strong_ion_example(recovered):
    pair, robust = recovered
    plain = align_sources(pair.reference, pair.query, replace(cpu_config(), transform=TransformConfig("none")))
    robust_error = mapping_errors(robust.mapping.transform(pair.query.rt_seconds), pair.query_to_reference_truth)
    plain_error = mapping_errors(plain.mapping.transform(pair.query.rt_seconds), pair.query_to_reference_truth)
    assert robust_error["mae_seconds"] < plain_error["mae_seconds"]


def test_saved_result_round_trip_and_small_mapping(recovered, tmp_path):
    pair, result = recovered
    save_result(result, tmp_path)
    loaded = load_result(tmp_path)
    np.testing.assert_array_equal(loaded.similarity, result.similarity)
    np.testing.assert_array_equal(loaded.dtw.path, result.dtw.path)
    np.testing.assert_array_equal(loaded.dtw.predecessors, result.dtw.predecessors)
    np.testing.assert_array_equal(loaded.anchors, result.anchors)
    np.testing.assert_allclose(load_mapping(tmp_path).transform(pair.query.rt_seconds), result.mapping.transform(pair.query.rt_seconds))
    with np.load(tmp_path / "diagnostics.npz", allow_pickle=False) as arrays:
        assert "profile" not in arrays.files
    with pytest.raises(FileExistsError):
        save_result(result, tmp_path)
    metadata = json.loads((tmp_path / "mapping.json").read_text())
    metadata["direction"] = "reference_to_query"
    (tmp_path / "mapping.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="direction"):
        load_mapping(tmp_path)


def test_headless_diagnostic_figures(recovered, tmp_path):
    pair, result = recovered
    png = save_diagnostics(result, tmp_path / "diagnostics.png", truth=pair.query_to_reference_truth)
    xic = save_xic_overlay(pair.reference, pair.query, result.mapping, [(79, 81), (999, 1001)], tmp_path / "xic.png")
    for path in (png, xic):
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 10_000


def test_synthetic_core_has_no_torch_qt_or_agilent_reader_dependency():
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                for name in ('torch', 'PyQt6', 'pyqtgraph', 'RAPID2_Agilent.io', 'RAPID2_Agilent.models'):
                    if fullname == name or fullname.startswith(name + '.'):
                        raise ModuleNotFoundError(fullname)
        sys.meta_path.insert(0, BlockOptional())
        from RAPID2_Agilent.alignment import AlignmentConfig, BinningConfig, TransformConfig, align_sources
        from RAPID2_Agilent.alignment.synthetic import generate_synthetic_pair
        pair = generate_synthetic_pair(n_reference=20, n_query=23)
        result = align_sources(pair.reference, pair.query, AlignmentConfig(
            binning=BinningConfig(ppm=200), transform=TransformConfig(scale=1)))
        assert result.metadata['compute']['backend'] == 'numpy'
        assert result.similarity.shape == (20, 23)
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_agilent_subset_identity_at_ten_ppm():
    path = Path(__file__).resolve().parents[4] / "tests/data/dummy_RP_2min_neg.d"
    if not path.exists():
        pytest.skip("Local Agilent test dataset is unavailable")
    config = AlignmentConfig(transform=TransformConfig(scale=1), compute=ComputeConfig(device="cpu", chunk_size=4),
                             selection=ScanSelection(max_scans=8))
    result = align_agilent(path, path, config)
    np.testing.assert_array_equal(result.dtw.path, np.column_stack((np.arange(8), np.arange(8))))
    np.testing.assert_allclose(result.mapping.transform(result.query_rt_seconds), result.reference_rt_seconds)
    assert result.metadata["n_mz_bins"] == 352638
    assert result.metadata["reference"]["reader_rt_unit"] == "minutes"
    assert result.metadata["reference"]["polarity"] == 1


def test_scan_selection_rejects_mixed_polarity_and_converts_minutes():
    class ScanInfo:
        def get_scan_info(self, index, field):
            return index % 2

    class FakeReader:
        RT_list = np.arange(4.)
        ms_scan_info = ScanInfo()

        def scan_indices_for_ms_level(self, level):
            return np.arange(4)

    with pytest.raises(ValueError, match="Mixed-polarity"):
        source_from_reader(FakeReader())
