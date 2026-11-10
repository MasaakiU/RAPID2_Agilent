"""RAPID2's .rpd files: read and written as RAPID2 does them, read as MSData by RPDReader, converted from mzData."""

import base64
import hashlib
import io
import pickle
from pathlib import Path

import numpy as np
import pytest

from RAPID2_Agilent.io import mzdata, rpd_file, run_hashes
from RAPID2_Agilent.io.rpd_file import RPDData, read_rpd, stored_hash, write_rpd
from RAPID2_Agilent.io.rpd_reader import RPDReader


def _data(spectrum_type="continuous", counts=(60, 58, 60, 59), *, polarity="Positive", drift=1e-5) -> RPDData:
    """Scans of ascending m/z 0.02 apart, each shifted by drift from the one before, of large intensities, so that
    their differences wrap."""
    rng = np.random.default_rng(0)
    mz_set = np.full((len(counts), max(counts)), np.nan)
    inten_set = np.zeros(mz_set.shape, dtype=np.int32)
    for i, count in enumerate(counts):
        mz_set[i, :count] = 150.0 + 0.02 * np.arange(count) + drift * i
        inten_set[i, :count] = rng.integers(0, 2**31 - 1, count)
    return RPDData(
        spectrum_type, mz_set, inten_set, np.array(counts), 0.1 * np.arange(1, len(counts) + 1), "TimeInMinutes",
        {"ScanMode": "Scan", "Polarity": polarity}, "Esi", "TimeOfFlight",
    )


def _sections(path: Path) -> list[bytes]:
    """The sections of a file of 2.3: Header, m/z, intensities, the columns for XICs, NoCompressionData."""
    content = Path(path).read_bytes()
    position = len(rpd_file.MAGIC) + 2 + rpd_file.HASH_SIZE
    sections = []
    for size in (2, 10, 10, 5, 5):
        count = int.from_bytes(content[position:position + size], "little")
        sections.append(content[position + size:position + size + count])
        position += size + count
    assert position == len(content)
    return sections


def _rapid2_undiffed(array, initial_arrays, axes):
    """RAPID2's revert_deep_diff."""
    for initial_array, axis in zip(initial_arrays[::-1], axes[::-1]):
        stacked = np.concatenate((np.expand_dims(initial_array, axis), array), axis=axis)
        array = np.cumsum(stacked, axis=axis, dtype=array.dtype)
    return array


def _rapid2_chromatogram_columns(mz_set, ref_row):
    """RAPID2's RPD.get_mz_set_info_for_chromatogram_extraction_core."""
    info = np.empty((2, mz_set.shape[1]), dtype=int)
    for i in range(mz_set.shape[1]):
        j = 1
        while True:
            if i - j < 0:
                info[0, i] = 0
                break
            if (mz_set[:, i - j] < mz_set[ref_row, i]).all():
                info[0, i] = i - j + 1
                break
            j += 1
        j = 1
        while True:
            if i + j > mz_set.shape[1] - 1:
                info[1, i] = mz_set.shape[1] - 1
                break
            if (mz_set[ref_row, i] < mz_set[:, i + j]).all():
                info[1, i] = i + j - 1
                break
            j += 1
    return info


@pytest.mark.parametrize("spectrum_type", ["continuous", "discrete"])
def test_a_file_written_is_read_back_as_it_was(tmp_path, spectrum_type):
    data = _data(spectrum_type)
    mz_set, inten_set = data.mz_set.copy(), data.inten_set.copy()
    data_hash = write_rpd(tmp_path / "x.rpd", data)
    back = read_rpd(tmp_path / "x.rpd")

    assert np.array_equal(data.mz_set, mz_set, equal_nan=True) and np.array_equal(data.inten_set, inten_set)
    np.testing.assert_allclose(back.mz_set, data.mz_set, rtol=0, atol=1e-9)  # NaN where it was
    assert np.array_equal(back.inten_set, data.inten_set) and np.array_equal(back.point_counts, data.point_counts)
    assert np.array_equal(back.RT_list, data.RT_list)
    assert (back.spectrum_type, back.RT_unit, back.spectrum_settings_dict, back.ionization_type, back.analyzer_type) == (
        spectrum_type, "TimeInMinutes", {"ScanMode": "Scan", "Polarity": "Positive"}, "Esi", "TimeOfFlight"
    )
    assert back.version == (2, 3) and back.polarity_list is None
    header_size = len(rpd_file.MAGIC) + 2 + rpd_file.HASH_SIZE
    assert back.data_hash == data_hash == stored_hash(tmp_path / "x.rpd")
    assert data_hash == hashlib.sha256((tmp_path / "x.rpd").read_bytes()[header_size:]).hexdigest()
    assert not (tmp_path / "x.rpd.part").exists()


def test_the_differences_are_rapid2s_and_summed_back_in_its_order(tmp_path):
    data = _data()
    write_rpd(tmp_path / "x.rpd", data)
    _header, mz_section, inten_section, _columns, _settings = _sections(tmp_path / "x.rpd")
    back = read_rpd(tmp_path / "x.rpd")

    mz = np.load(io.BytesIO(mz_section))
    assert sorted(mz.files) == ["initial_mz_array0", "initial_mz_array1", "initial_mz_array2", "mz_set_diff"]
    as_rapid2 = _rapid2_undiffed(mz["mz_set_diff"], [mz[f"initial_mz_array{i}"] for i in range(3)], [1, 1, 0])
    points = ~np.isnan(data.mz_set)
    assert np.array_equal(back.mz_set[points], as_rapid2[points])
    inten = np.load(io.BytesIO(inten_section))
    assert np.array_equal(_rapid2_undiffed(inten["inten_set_diff"], [inten["initial_inten_array0"]], [0]), data.inten_set)


def test_the_columns_for_xics_are_those_rapid2_finds(tmp_path):
    # shifted more than their spacing, so that the columns of the scans overlap
    data = _data(counts=(60, 55, 60, 57, 60), drift=0.03)
    write_rpd(tmp_path / "x.rpd", data)
    columns = np.load(io.BytesIO(_sections(tmp_path / "x.rpd")[3]))
    assert int(columns["ref_row"]) == 0
    expected = _rapid2_chromatogram_columns(data.mz_set, 0)
    assert np.array_equal(columns["mz_set_info_for_chromatogram_extraction"], expected)
    assert (expected[0] < np.arange(60)).any() and (expected[1] > np.arange(60)).any()


def test_the_spectrum_type_is_read_without_the_scans(tmp_path):
    for spectrum_type in rpd_file.SPECTRUM_TYPES:
        write_rpd(tmp_path / f"{spectrum_type}.rpd", _data(spectrum_type))
        assert rpd_file.stored_spectrum_type(tmp_path / f"{spectrum_type}.rpd") == spectrum_type
    (tmp_path / "not.rpd").write_bytes(b"not a .rpd file")
    with pytest.raises(ValueError, match="not.rpd is not a .rpd file"):
        rpd_file.stored_spectrum_type(tmp_path / "not.rpd")


def test_centroid_data_has_no_columns_for_xics(tmp_path):
    write_rpd(tmp_path / "x.rpd", _data("discrete"))
    assert _sections(tmp_path / "x.rpd")[3] == b""


def test_the_pickles_name_numpy_1s_and_only_rapid2s_are_read():
    pickled = rpd_file._pickled("NoCompressionData", {"RT_list": np.array([0.1, 0.2]), "locs": (np.array([1]), np.array([5]))})
    # numpy 2 pickles arrays under numpy._core, which RAPID2's numpy 1 cannot read
    assert b"numpy._core" not in pickled
    assert pickled.startswith(b"\x80\x02cModules.process.convert_open\nNoCompressionData\n")
    record = rpd_file._unpickled(pickled)
    assert record.RT_list.tolist() == [0.1, 0.2] and record.RT_list.flags.writeable
    assert [array.tolist() for array in record.locs] == [[1], [5]]
    # as numpy 2 pickles them itself
    assert rpd_file._unpickled(pickle.dumps(np.arange(3), protocol=4)).tolist() == [0, 1, 2]
    with pytest.raises(pickle.UnpicklingError):
        rpd_file._unpickled(pickle.dumps(Path("x")))


def test_other_files_and_versions_are_refused(tmp_path):
    (tmp_path / "other.rpd").write_bytes(b"not a file of RAPID2")
    with pytest.raises(ValueError, match="not a .rpd file"):
        read_rpd(tmp_path / "other.rpd")
    write_rpd(tmp_path / "x.rpd", _data())
    content = bytearray((tmp_path / "x.rpd").read_bytes())
    (tmp_path / "cut.rpd").write_bytes(content[:-10])
    with pytest.raises(ValueError):
        read_rpd(tmp_path / "cut.rpd")
    content[len(rpd_file.MAGIC) + 1] = 4
    (tmp_path / "newer.rpd").write_bytes(content)
    with pytest.raises(ValueError, match="2.4"):
        read_rpd(tmp_path / "newer.rpd")


def test_a_write_stopped_leaves_nothing(tmp_path):
    def stop(what, done, total):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        write_rpd(tmp_path / "x.rpd", _data(counts=(60,) * 40), stop)
    assert list(tmp_path.iterdir()) == []


def test_the_hashes_of_a_rpd_file_are_the_one_it_holds(tmp_path):
    data_hash = write_rpd(tmp_path / "x.rpd", _data())
    assert run_hashes.run_hashes(tmp_path / "x.rpd") == ("", data_hash)


# -- RPDReader ---------------------------------------------------------------

def _reader(tmp_path, *args, **kwargs) -> RPDReader:
    write_rpd(tmp_path / "x.rpd", _data(*args, **kwargs))
    reader = RPDReader()
    reader.load(tmp_path / "x.rpd")
    return reader


def test_the_reader_has_the_scans_of_the_file(tmp_path):
    reader = _reader(tmp_path, polarity="Negative")
    data = _data()
    assert reader.N_scans == 4 and reader.file_path == tmp_path / "x.rpd"
    assert np.array_equal(reader.RT_list, data.RT_list)
    assert reader.MS_level_list.tolist() == [1] * 4 and reader.polarity_list.tolist() == [-1] * 4
    assert reader.ion_mode_list.tolist() == [64] * 4 and reader.scan_type_list.tolist() == [1] * 4
    assert reader.precursor_mz_list.tolist() == [0.0] * 4
    assert reader.get_scan_type_summary() == [("Negative MS1", 4)]
    assert reader.scan_indices_for_ms_level(1).tolist() == [0, 1, 2, 3]
    mz, inten = reader.extract_spectrum(1)
    np.testing.assert_allclose(mz, data.mz_set[1, :58], rtol=0, atol=1e-9)
    assert inten.tolist() == data.inten_set[1, :58].tolist()
    assert reader.extract_tic()[1].tolist() == data.inten_set.sum(axis=1, dtype=np.int64).tolist()
    assert reader.extract_bpc(scan_indices=np.array([2]))[1].tolist() == [data.inten_set[2].max()]
    assert reader.profile_mz_range(np.arange(4)) == pytest.approx((150.0, 150.0 + 0.02 * 59 + 2e-5))
    reader.close()
    assert reader.file_path is None and reader.RT_list is None


def test_an_xic_sums_the_points_in_its_window_and_tells_the_scans_with_none(tmp_path):
    reader = _reader(tmp_path)
    spectra = [reader.extract_spectrum(i) for i in range(4)]
    rt, xic = reader.extract_chromatogram(150.105, 150.205)
    assert xic.tolist() == [int(inten[(150.105 <= mz) & (mz < 150.205)].sum(dtype=np.int64)) for mz, inten in spectra]
    assert reader.mz_window_gaps(150.105, 150.205).tolist() == [0.0] * 4
    # narrower than the spacing of the points, between two of them: that spacing
    np.testing.assert_allclose(reader.mz_window_gaps(150.101, 150.105), 0.02)
    assert np.isnan(reader.mz_window_gaps(200, 201)).all()
    assert reader.extract_chromatogram(150.101, 150.105, scan_indices=np.array([0, 3]))[1].tolist() == [0, 0]


def test_a_scan_of_centroid_data_with_no_peak_in_a_window_has_0_there(tmp_path):
    reader = _reader(tmp_path, "discrete")
    assert reader.mz_window_gaps(150.101, 150.105).tolist() == [0.0] * 4
    assert reader.mz_window_gaps(200, 201).tolist() == [0.0] * 4


def test_spectra_are_averaged_on_the_points_of_the_scan_in_the_middle(tmp_path):
    reader = _reader(tmp_path)
    data = _data()
    mz, intensity, representative = reader.extract_averaged_spectrum(0.25, 0.42)
    # the points of the scans at 0.3 and 0.4 nearest to those of the one at 0.3, nearer the middle: each its own column
    assert representative == 2 and len(mz) == 60
    np.testing.assert_allclose(intensity, data.inten_set[2:].sum(axis=0, dtype=np.int64) / 2)
    assert reader.extract_averaged_spectrum(0.25, 0.42, "sum")[1].sum() == data.inten_set[2:].sum(dtype=np.int64)

    mz, intensity = reader.extract_averaged_sub_spectrum(150.105, 150.305, scan_indices=np.arange(4))
    columns = np.arange(6, 16)  # 150.12 to 150.30
    np.testing.assert_allclose(mz, data.mz_set[1, columns], atol=1e-9)  # of scan 1, the first nearest to 0.25
    np.testing.assert_allclose(intensity, data.inten_set[:, columns].sum(axis=0, dtype=np.int64) / 4)


def test_the_max_in_each_bin_is_kept_where_higher(tmp_path):
    reader = _reader(tmp_path)
    mz, inten = reader.extract_spectrum(0)
    edges = np.linspace(150.0, 150.5, 6)
    out = np.zeros(5, dtype=np.float32)
    out[4] = np.float32(2**31)
    reader.max_into_bins(0, edges, out)
    expected = [inten[(low <= mz) & (mz < high)].max() for low, high in zip(edges[:-1], edges[1:])]
    assert out[:4].tolist() == np.array(expected[:4], dtype=np.float32).tolist() and out[4] == np.float32(2**31)


# -- mzData ------------------------------------------------------------------

def _mzdata(path: Path, spectra, *, polarities=None, ms_levels=None, scan_modes=None) -> Path:
    """A small mzData file as MassHunter exports one, of spectra (rt, mz, intensities)."""
    def encoded(values, dtype):
        return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")

    blocks = []
    for i, (rt, mz, inten) in enumerate(spectra):
        polarity = polarities[i] if polarities else "Positive"
        ms_level = ms_levels[i] if ms_levels else 1
        scan_mode = scan_modes[i] if scan_modes else "Scan"
        blocks.append(f"""
    <spectrum id="{i + 1}">
      <spectrumDesc>
        <spectrumSettings>
          <acqSpecification spectrumType="continuous" methodOfCombination="average" count="1">
            <acquisition acqNumber="{i + 1}" />
          </acqSpecification>
          <spectrumInstrument msLevel="{ms_level}" mzRangeStart="50" mzRangeStop="1000">
            <cvParam cvLabel="psi" accession="PSI:1000036" name="ScanMode" value="{scan_mode}" />
            <cvParam cvLabel="psi" accession="PSI:1000037" name="Polarity" value="{polarity}" />
            <cvParam cvLabel="psi" accession="PSI:1000038" name="TimeInMinutes" value="{rt}" />
          </spectrumInstrument>
        </spectrumSettings>
      </spectrumDesc>
      <mzArrayBinary>
        <data precision="64" endian="little" length="{len(mz)}">{encoded(mz, "<f8")}</data>
      </mzArrayBinary>
      <intenArrayBinary>
        <data precision="32" endian="little" length="{len(inten)}">{encoded(inten, "<f4")}</data>
      </intenArrayBinary>
    </spectrum>""")
    path.write_text(f"""<?xml version="1.0" encoding="utf-8"?>
<mzData version="1.05" accessionNumber="psi-ms:100" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <description>
    <instrument>
      <instrumentName>Instrument 1</instrumentName>
      <source>
        <cvParam cvLabel="psi" accession="PSI:1000008" name="IonizationType" value="Esi" />
      </source>
      <analyzerList count="1">
        <analyzer>
          <cvParam cvLabel="psi" accession="PSI:1000010" name="AnalyzerType" value="TimeOfFlight" />
        </analyzer>
      </analyzerList>
    </instrument>
  </description>
  <spectrumList count="{len(spectra)}">{"".join(blocks)}
  </spectrumList>
</mzData>
""", encoding="utf-8")
    return path


def _spectra(n_scans=3):
    return [
        (round(0.1 * (i + 1), 3), 150.0 + 0.02 * np.arange(60 - i), np.arange(60 - i) + 0.75)
        for i in range(n_scans)
    ]


@pytest.mark.parametrize(
    "given, written",
    [("a/x.mzdata.xml", "a/x.rpd"), ("a/x.MZDATA.XML", "a/x.rpd"), ("a/x.y.mzdata.xml", "a/x.y.rpd"), ("a/x.xml", "a/x.rpd")],
)
def test_an_mzdata_file_is_converted_next_to_it(given, written):
    assert mzdata.rpd_path(given) == Path(written)


def test_an_mzdata_file_is_converted_as_rapid2_converts_it(tmp_path):
    path = _mzdata(tmp_path / "run.mzdata.xml", _spectra())
    assert mzdata.convert(path) == tmp_path / "run.rpd"
    back = read_rpd(tmp_path / "run.rpd")
    assert back.spectrum_type == "continuous" and back.RT_list.tolist() == [0.1, 0.2, 0.3]
    assert back.spectrum_settings_dict == {"ScanMode": "Scan", "Polarity": "Positive"}
    assert (back.ionization_type, back.analyzer_type) == ("Esi", "TimeOfFlight")
    assert back.point_counts.tolist() == [60, 59, 58]
    np.testing.assert_allclose(back.mz_set[2, :58], 150.0 + 0.02 * np.arange(58), rtol=0, atol=1e-9)
    # the intensities truncated to integers, as RAPID2 does
    assert back.inten_set.dtype == np.int32 and back.inten_set[2, :3].tolist() == [0, 1, 2]
    assert back.inten_set[2, 58:].tolist() == [0, 0]
    assert back.polarity_list == ["Positive"] * 3


def test_the_polarity_of_each_scan_is_kept_beside_the_first_scans_setting(tmp_path):
    path = _mzdata(tmp_path / "run.mzdata.xml", _spectra(), polarities=["Negative", "Positive", "Negative"])
    back = read_rpd(mzdata.convert(path))
    # as RAPID2 reads it: all of the first's
    assert back.spectrum_settings_dict == {"ScanMode": "Scan", "Polarity": "Negative"}
    assert back.polarity_list == ["Negative", "Positive", "Negative"]
    reader = RPDReader()
    reader.load(tmp_path / "run.rpd")
    assert reader.polarity_list.tolist() == [-1, 1, -1]
    assert reader.get_scan_type_summary() == [("Positive MS1", 1), ("Negative MS1", 2)]


def test_scans_of_more_than_one_scan_mode_or_ms2_are_not_converted(tmp_path):
    with pytest.raises(ValueError, match="one setting"):
        mzdata.read_mzdata(_mzdata(tmp_path / "a.mzdata.xml", _spectra(), scan_modes=["Scan", "ProductIon", "Scan"]))
    with pytest.raises(ValueError, match="MS2"):
        mzdata.read_mzdata(_mzdata(tmp_path / "b.mzdata.xml", _spectra(), ms_levels=[1, 2, 1]))


def test_a_conversion_stopped_writes_nothing(tmp_path):
    path = _mzdata(tmp_path / "run.mzdata.xml", _spectra(40))
    told = []

    def stop_writing(what, done, total):
        told.append(what)
        if what.startswith("Writing"):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        mzdata.convert(path, stop_writing)
    assert told[0] == "Reading run.mzdata.xml" and told[-1] == "Writing run.rpd"
    assert list(tmp_path.iterdir()) == [path]


def test_the_main_window_converts_opens_and_exports_rpd_files(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from RAPID2_Agilent.services import rpd_csv
    from tests.test_target_panel import build_app, type_into

    app, view, model, presenter = build_app(())
    shown, warned = [], []
    monkeypatch.setattr(QMessageBox, "information", lambda parent, title, text: shown.append(text))
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: warned.append(text))
    path = _mzdata(tmp_path / "run.mzdata.xml", _spectra())
    presenter.convert_files_clicked([str(path)])
    assert shown == ["Written:\n\nrun.rpd"]

    presenter.open_rpd_files_clicked([str(tmp_path / "run.rpd")])
    presenter.open_rpd_files_clicked([str(tmp_path / "run.rpd")])
    (tmp_path / "broken.rpd").write_bytes(b"broken")
    presenter.open_rpd_files_clicked([str(tmp_path / "broken.rpd")])
    assert [m.file_path for m in presenter.data_stack_presenter.models] == [tmp_path / "run.rpd"]
    assert shown[1] == "Open already:\n\nrun.rpd" and warned[0].startswith("broken.rpd could not be opened.")

    type_into(view.target_panel, "adduct", "[M+H]+")
    presenter.target_panel_presenter.add_target_from_view()
    presenter.export_targets_clicked(str(tmp_path / "out"))
    lines = (tmp_path / "out.rpd.csv").read_text(encoding="utf-8-sig").splitlines()
    row = dict(zip(rpd_csv.COLUMNS, lines[1].split("\t")))
    assert (row["file_name"], row["data_hash"], row["scan_sha256"]) == ("run.rpd", stored_hash(tmp_path / "run.rpd"), "")
