"""Regression checks for the shared reader and its Qt facade."""

from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from RAPID2_Agilent.io import AgilentReader
from RAPID2_Agilent.models.rpd_model import AgilentDModel
from tests.data_paths import DATA_DIR, require


DATA_PATH = DATA_DIR / "dummy_RP_2min_neg.d"
MSMS_DATA_PATH = DATA_DIR / "MSMS_data" / "9_PRJ_0-1-2_PA34-1_1uM_n1.d"


def test_reader_reads_real_data_without_qt_or_models():
    # A fresh interpreter prevents an earlier GUI import from hiding a dependency.
    script = textwrap.dedent("""
        import importlib.abc
        from pathlib import Path
        import sys

        class BlockGuiImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                blocked = ("PyQt6", "pyqtgraph", "RAPID2_Agilent.models",
                           "RAPID2_Agilent.views", "RAPID2_Agilent.presenters")
                if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
                    raise AssertionError("GUI dependency: " + fullname)

        sys.meta_path.insert(0, BlockGuiImports())
        import numpy as np
        from RAPID2_Agilent.io import AgilentReader

        with AgilentReader() as reader:
            reader.load(Path(sys.argv[1]))
            assert reader.N_scans == 118
            assert reader.get_scan_type_summary() == [("Negative MS1", 118)]
            mz, intensity = reader.extract_spectrum(reader.N_scans - 1)
            assert len(mz) == len(intensity) > 0
            for rt, values in (reader.extract_tic(), reader.extract_bpc(),
                               reader.extract_chromatogram(150, 151)):
                np.testing.assert_array_equal(rt, reader.RT_list)
                assert len(values) == reader.N_scans
            avg_mz, avg_intensity, index = reader.extract_averaged_spectrum(
                float(reader.RT_list[-1] + 1), float(reader.RT_list[-1] + 2)
            )
            assert index == reader.N_scans - 1
            np.testing.assert_allclose(avg_mz, mz)
            np.testing.assert_allclose(avg_intensity, intensity)
            profile = reader.ms_profile
        assert profile._fh.closed
        assert reader.N_scans == 0
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(require(DATA_PATH))],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("factory", [AgilentReader, AgilentDModel])
def test_ms_level_filters_traces_and_averaged_spectra(factory):
    data = factory()
    try:
        data.load(require(MSMS_DATA_PATH))
        assert data.get_scan_type_summary() == [
            ("Negative MS1", 209), ("Negative MS2", 1024),
        ]
        for level in (1, 2):
            indices = data.scan_indices_for_ms_level(level)
            np.testing.assert_array_equal(indices, np.flatnonzero(data.MS_level_list == level))
            for method, args in (
                (data.extract_tic, ()),
                (data.extract_bpc, ()),
                (data.extract_chromatogram, (150, 151)),
            ):
                all_rt, all_values = method(*args)
                rt, values = method(*args, scan_indices=indices)
                np.testing.assert_array_equal(rt, all_rt[indices])
                np.testing.assert_array_equal(values, all_values[indices])
            scan_index = int(indices[4])
            mz, intensity, representative = data.extract_averaged_spectrum(
                float(data.RT_list[scan_index]), float(data.RT_list[indices[5]]),
                scan_indices=indices,
            )
            expected_mz, expected_intensity = data.extract_spectrum(scan_index)
            assert representative == scan_index
            np.testing.assert_allclose(mz, expected_mz)
            np.testing.assert_allclose(intensity, expected_intensity)
        with pytest.raises(ValueError, match="No MS3 scans"):
            data.scan_indices_for_ms_level(3)
    finally:
        data.close()


@pytest.mark.parametrize("factory", [AgilentReader, AgilentDModel])
def test_reload_replaces_data_and_releases_previous_profile(factory):
    data = factory()
    try:
        data.load(require(DATA_PATH))
        assert data.N_scans == 118
        previous_profile = data.ms_profile
        data.load(require(MSMS_DATA_PATH))
        assert previous_profile._fh.closed
        assert previous_profile._mm.closed
        assert data.N_scans == 1233
        assert len(data.RT_list) == len(data.MS_level_list) == data.N_scans
        assert len(data.precursor_mz_list) == len(data.isolation_width_list) == len(data.collision_energy_list) == data.N_scans
        assert data.file_path == MSMS_DATA_PATH
        mz, intensity = data.extract_spectrum(data.N_scans - 1)
        assert len(mz) == len(intensity) > 0
        current_profile = data.ms_profile
    finally:
        data.close()
    assert current_profile._fh.closed
    assert current_profile._mm.closed
    assert data.file_path is None
    assert data.N_scans == 0
    assert data.RT_list is None
    assert data.MS_level_list is None
    assert data.precursor_mz_list is None
    data.close()  # Closing twice is harmless.


def test_model_metadata_tracks_reader_without_copied_state():
    model = AgilentDModel()
    try:
        model.reader.load(require(DATA_PATH))
        assert model.file_path == model.reader.file_path
        assert model.RT_list is model.reader.RT_list
        assert model.MS_level_list is model.reader.MS_level_list
        assert model.ms_scan_info is model.reader.ms_scan_info
        assert model.ms_calib_info is model.reader.ms_calib_info
        assert model.ms_profile is model.reader.ms_profile
        assert model.N_scans == 118
        model.reader.close()
        assert model.RT_list is None
        assert model.N_scans == 0
        model.spectrum_param_index = 1
        assert model.reader.spectrum_param_index == 1
        model.reader.spectrum_param_index = 0
        assert model.spectrum_param_index == 0
    finally:
        model.close()


def test_load_tells_of_the_scans_read_and_stops_where_that_raises():
    told = []
    with AgilentReader() as reader:
        reader.load(require(DATA_PATH), lambda done, total: told.append((done, total)))
        assert told == [(done, reader.N_scans) for done in range(1, reader.N_scans + 1)]

    class Stop(Exception):
        pass

    def stop_at_3(done, _total):
        if done == 3:
            raise Stop

    with AgilentReader() as reader, pytest.raises(Stop):
        reader.load(DATA_PATH, stop_at_3)
