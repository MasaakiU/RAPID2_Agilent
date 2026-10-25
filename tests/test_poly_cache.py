"""Polynomial reuse, file isolation and native buffer regression checks."""

import struct

import numpy as np
import pytest

from RAPID2_Agilent.io import AgilentReader
from RAPID2_Agilent.io import agilent_reader
from RAPID2_Agilent.io.agilent_reader import MSCalibInfo
from RAPID2_Agilent.io._native.calibrate_mz import calibrate_mz
from RAPID2_Agilent.io._native.average_spectrum import accumulate_decoded_calibrated_nearest_bins
from tests.data_paths import DATA_DIR, require


DATA_PATH = DATA_DIR / "dummy_RP_2min_neg.d"
OTHER_DATA_PATH = DATA_DIR / "9_P2-8-15_spl1_p0.d"


def test_real_file_builds_one_poly_for_spectra_and_fused_average(monkeypatch):
    calculate = agilent_reader.calculate_poly
    calls = []

    def counted_calculate(*args):
        calls.append(args)
        return calculate(*args)

    monkeypatch.setattr(agilent_reader, "calculate_poly", counted_calculate)
    with AgilentReader() as reader:
        reader.load(require(DATA_PATH))
        calibration = reader.ms_calib_info
        assert calibration.poly is None
        reader.extract_chromatogram(499.99, 500.01)
        assert calibration.poly is None  # XIC-only use does not allocate all bins.
        for index in (5, 25, 100):
            mz, _ = reader.extract_spectrum(index)
            offset = reader.ms_scan_info.get_spectrum_info(index, 0, "SpectrumOffset")
            start, delta, count, _ = reader.ms_profile.decode_header(offset)
            calib_id = reader.ms_scan_info.get_scan_info(index, "CalibrationID")
            reference = calibration.calibrate_mz_python(index, count, calib_id, start, delta)
            np.testing.assert_allclose(mz, reference, rtol=1e-13, atol=1e-12)
        poly = calibration.poly
        saved_poly = poly.copy()
        assert not poly.flags.writeable
        reader.extract_averaged_spectrum(float(reader.RT_list[4]), float(reader.RT_list[8]))
        assert calibration.poly is poly
        np.testing.assert_array_equal(poly, saved_poly)
        assert len(calls) == 1


def test_poly_is_per_file_and_replaced_on_reload():
    with AgilentReader() as reader, AgilentReader() as other:
        reader.load(require(DATA_PATH))
        reader.extract_spectrum(0)
        original_calibration = reader.ms_calib_info
        original_poly = original_calibration.poly
        other.load(require(OTHER_DATA_PATH))
        other.extract_spectrum(0)
        assert other.ms_calib_info.poly is not original_poly
        assert len(other.ms_calib_info.poly) != len(original_poly)

        reader.load(require(OTHER_DATA_PATH))
        assert reader.ms_calib_info is not original_calibration
        assert reader.ms_calib_info.poly is None
        actual_mz, _ = reader.extract_spectrum(0)
        expected_mz, _ = other.extract_spectrum(0)
        np.testing.assert_array_equal(actual_mz, expected_mz)
        assert reader.ms_calib_info.poly is not original_poly
    assert reader.ms_calib_info is None


@pytest.fixture
def small_calibration(tmp_path):
    xml = tmp_path / "DefaultMassCal.xml"
    xml.write_text("""<Root><DefaultCalibrations>
      <DefaultCalibration DefaultCalibrationID="1"><Step Number="2">
        <CalibrationTechnique>ExternalReference</CalibrationTechnique>
        <CalibrationFormula>Polynomial</CalibrationFormula>
        <ValueUseFlags>7</ValueUseFlags>
        <Values><Value>0.5</Value><Value>0.25</Value><Value>0.125</Value></Values>
      </Step></DefaultCalibration>
    </DefaultCalibrations></Root>""")
    binary = tmp_path / "MSMassCal.bin"
    records = [
        (2.0, 0.0, 2.0, 10.0, 0.5, 0.25, 0.125, 0.0, 0.0, 0.0),
        (3.0, 0.5, 2.0, 10.0, 0.5, 0.25, 0.125, 0.0, 0.0, 0.0),
    ]
    binary.write_bytes(b"".join(struct.pack("<I10d", 10, *row) for row in records))
    return MSCalibInfo(xml, binary, [0, 84])


@pytest.mark.parametrize("change", [None, "start", "delta", "count", "left", "right", "coefficient", "flags"])
def test_only_common_correction_conditions_enable_reuse(small_calibration, change):
    calibration = small_calibration
    grids = [(1.0, 0.25, 64), (1.0, 0.25, 64)]
    ids = [1, 1]
    if change in ("start", "delta", "count"):
        grid = list(grids[1])
        index = {"start": 0, "delta": 1, "count": 2}[change]
        grid[index] *= 2
        grids[1] = tuple(grid)
    elif change in ("left", "right", "coefficient"):
        index = {"left": 2, "right": 3, "coefficient": 4}[change]
        calibration.calib_info[1][index] *= 2
    elif change == "flags":
        flags = list(calibration.default_calib_info[1]["polynomial_flag"])
        flags[2], flags[3] = False, True
        calibration.default_calib_info[2] = {"polynomial_flag": flags}
        ids[1] = 2
    calibration.configure_poly_cache(grids, ids)
    for i in range(2):
        start, delta, count = grids[i]
        actual = calibration.calibrate_mz(i, count, ids[i], start, delta)
        expected = calibration.calibrate_mz_python(i, count, ids[i], start, delta)
        np.testing.assert_array_equal(actual, expected)
    assert (calibration.poly is not None) == (change is None)


def test_native_calibration_rejects_wrong_poly_length():
    with pytest.raises(ValueError, match="N_datapoints"):
        calibrate_mz(1.0, 0.0, 8, 1.0, 0.25, np.zeros(7))


def test_native_average_rejects_poly_shorter_than_decoded_range():
    with pytest.raises(ValueError, match="poly must cover"):
        accumulate_decoded_calibrated_nearest_bins(
            b"\x00" * 8, 0, 0, 2, 4, np.arange(2, dtype=np.float64),
            1.0, 0.0, np.zeros(1), 1.0, 0.25, np.zeros(2),
        )
