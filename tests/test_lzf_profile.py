"""The older MSProfile.bin of LZF records (e.g. of 2022), written again in the token format the decoders read."""

import struct

import numpy as np
import pytest

from RAPID2_Agilent.io import AgilentReader
from RAPID2_Agilent.io._native.build_checkpoints import build_checkpoints
from RAPID2_Agilent.io._native.decode_inten_list import decode_inten_list
from RAPID2_Agilent.io._native.lzf_profile import rewrite_lzf_records
from tests.data_paths import PEAK_EVAL_DIR, require


OLD_DATA_PATH = PEAK_EVAL_DIR / "20220801_RP_LipidStd" / "5_Std_100ng-200ul-dil20_p0.d"


def _raw(values) -> bytes:
    """A record as an LZF one decompresses to: start_t, delta_t and an int32 intensity a datapoint."""
    return struct.pack("<dd", 18464.0, 0.5) + np.asarray(values, dtype="<i4").tobytes()


def _literals(data: bytes) -> bytes:
    """LZF of data in runs as they are only."""
    return b"".join(bytes([len(data[i:i + 32]) - 1]) + data[i:i + 32] for i in range(0, len(data), 32))


def _repeats(head: bytes, unit: bytes, count: int) -> bytes:
    """LZF of head and unit count times: unit once as it is, then copies of the bytes len(unit) back, which overlap
    what they write."""
    data = head + unit * count
    out, done = bytearray(_literals(head + unit)), len(head) + len(unit)
    while done < len(data):
        length = min(len(data) - done, 264)
        if length < 3:
            out += _literals(data[done:])
            break
        back = len(unit) - 1
        out += bytes([(length - 2) << 5 | back >> 8, back & 0xFF] if length < 9 else [7 << 5 | back >> 8, length - 9, back & 0xFF])
        done += length
    return bytes(out)


def _profile(records: list[bytes]):
    """An MSProfile.bin of the records after its header: its bytes, the records' offsets and byte counts."""
    data = b"\x02\x01" + bytes(66)
    offsets = []
    for record in records:
        offsets.append(len(data))
        data += record
    return np.frombuffer(data, dtype=np.uint8), offsets, [len(record) for record in records]


def _decoded(buffer: np.ndarray, offset: int, byte_count: int):
    """start_t, delta_t and the intensities of a record of the token format, as the reader decodes it."""
    view = memoryview(buffer)
    start_t, delta_t, header, first = struct.unpack_from("<ddIi", buffer, offset)
    assert header >> 24 == 0x90
    N_datapoints = header & 0xFFFFFF
    *_, end = build_checkpoints(view, offset, byte_count, N_datapoints, -first, 2048)
    out = np.zeros(N_datapoints, dtype=np.uint32)
    decode_inten_list(view, offset + 24, -first, end, 4, out)
    return start_t, delta_t, out


def test_records_of_every_token_size_and_long_zero_runs_come_back():
    rng = np.random.default_rng(0)
    spread = np.zeros(300_000, dtype=np.int64)
    spread[5:200] = rng.integers(1, 128, 195)  # 1 byte
    # 40 zeros after 1-byte values (more than its token holds), 19740 after 2-byte ones (also)
    spread[240:260] = rng.integers(128, 32768, 20)
    spread[20_000:20_010] = [2**31 - 1, 5, 2**20, 127, 128, 32767, 32768, 1, 1, 1]
    spread[20_010:20_100] = rng.integers(1, 2**31 - 1, 90) * rng.integers(0, 2, 90)
    spread[100_000:100_050] = rng.integers(1, 128, 50)
    spread[299_000:299_500] = rng.integers(0, 40_000, 500)  # then zeros to the end
    mixed = rng.choice([0, 1, 100, 127, 128, 5000, 32767, 32768, 10**6], size=5000)
    values = [spread, mixed, np.zeros(1000, dtype=np.int64), np.r_[np.zeros(999, dtype=np.int64), 7]]
    raws = [_raw(v) for v in values]
    profile, offsets, byte_counts = _profile([_literals(raw) for raw in raws])
    buffer, new_offsets, new_byte_counts = rewrite_lzf_records(profile, offsets, byte_counts, [len(raw) for raw in raws], [1] * len(raws))
    for expected, offset, byte_count in zip(values, new_offsets, new_byte_counts):
        start_t, delta_t, decoded = _decoded(buffer, int(offset), int(byte_count))
        assert (start_t, delta_t) == (18464.0, 0.5)
        np.testing.assert_array_equal(decoded, expected)


def test_lzf_copies_overlapping_what_they_write_and_records_already_of_tokens():
    values = np.full(1000, 5)
    raw = _raw(values)
    lzf = _repeats(raw[:16], raw[16:20], 1000)
    assert len(lzf) < 0.1 * len(raw)
    profile, offsets, byte_counts = _profile([lzf])
    buffer, new_offsets, new_byte_counts = rewrite_lzf_records(profile, offsets, byte_counts, [len(raw)], [True])
    np.testing.assert_array_equal(_decoded(buffer, int(new_offsets[0]), int(new_byte_counts[0]))[2], values)
    # a record of the token format is copied as it is, beside one of LZF
    tokens = bytes(buffer[new_offsets[0]:new_offsets[0] + new_byte_counts[0]])
    profile, offsets, byte_counts = _profile([tokens, lzf])
    both, both_offsets, both_byte_counts = rewrite_lzf_records(profile, offsets, byte_counts, [0, len(raw)], [False, True])
    assert bytes(both[both_offsets[0]:both_offsets[0] + both_byte_counts[0]]) == tokens
    assert bytes(both[both_offsets[1]:both_offsets[1] + both_byte_counts[1]]) == tokens


@pytest.mark.parametrize("damage, message", [
    ("truncated", "end within a run"),
    ("longer", "end before the uncompressed size"),
    ("negative", "negative intensity"),
    ("beyond", "beyond the file"),
])
def test_damaged_records_are_errors(damage, message):
    raw = _raw([1, 2, -3 if damage == "negative" else 3])
    lzf = _literals(raw)
    if damage == "truncated":
        lzf = lzf[:-1]
    profile, offsets, byte_counts = _profile([lzf])
    if damage == "beyond":
        byte_counts[0] += 1
    with pytest.raises(ValueError, match=message):
        rewrite_lzf_records(profile, offsets, byte_counts, [len(raw) + (4 if damage == "longer" else 0)], [True])


def test_reader_reads_an_older_file_of_lzf_records():
    with AgilentReader() as reader:
        reader.load(require(OLD_DATA_PATH))
        profile = reader.ms_profile
        assert profile._buffer is not None and profile._fh.closed
        # in fewer bytes than the file's LZF
        assert profile._buffer.nbytes < 0.7 * (OLD_DATA_PATH / "AcqData" / "MSProfile.bin").stat().st_size
        assert reader.get_scan_type_summary() == [("Positive MS1", 901)]
        # the older MSScan.xsd has no MzOfInterest and MzIsolationWidth
        assert not reader.precursor_mz_list.any() and not reader.isolation_width_list.any()
        calibration = reader.ms_calib_info
        assert len(calibration.default_calib_info[1]["polynomial_flag"]) > 8  # up to t^11
        for index in (0, 450, 900):
            mz, intensity = reader.extract_spectrum(index)
            assert intensity.sum() == reader.ms_scan_info.get_scan_info(index, "TIC")
            assert intensity.max() == reader.ms_scan_info.get_spectrum_info(index, 0, "MaxY")
            offset = reader.ms_scan_info.get_spectrum_info(index, 0, "SpectrumOffset")
            start, delta, count, _ = profile.decode_header(offset)
            reference = calibration.calibrate_mz_python(index, count, 1, start, delta)
            np.testing.assert_allclose(mz, reference, rtol=1e-12)
        assert mz[0] == pytest.approx(100.0, abs=0.1) and mz[-1] == pytest.approx(1498.4, abs=0.1)
        # the lock mass HP-0921 [M+H]+, of the half height's centroid, within a few ppm
        mz, intensity = reader.extract_spectrum(450)
        top = int(np.argmax(np.where(np.abs(mz - 922.0098) < 0.05, intensity, 0)))
        half = intensity[top - 5:top + 6] > intensity[top] / 2
        centroid = np.average(mz[top - 5:top + 6][half], weights=intensity[top - 5:top + 6][half])
        assert abs(centroid - 922.0098) / 922.0098 * 1e6 < 10
        rt, xic = reader.extract_chromatogram(922.0, 922.02)
        assert len(xic) == 901 and np.median(xic) > 1000
    assert profile._buffer is None
