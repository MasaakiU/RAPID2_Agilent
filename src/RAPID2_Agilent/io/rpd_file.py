"""The .rpd files of RAPID2, read and written as it does them (versions 2.2 and 2.3); see README.md.

A file is b"\\x13RAPID", its major and minor version (a byte each) and the SHA-256 of what follows (64 hex
characters), then sections, each its byte count (little-endian, in the bytes given here) and its bytes:

- a pickled Header (2): the sizes of the byte counts that follow;
- the m/z (10) and the intensities (10) of the scans, a row a scan, as .npz; those of profile data as differences;
- in 2.3 only, the columns RAPID2 looks in for an XIC (5), as .npz; empty for centroid data;
- a pickled NoCompressionData (5): the RTs, the settings of the scans and where the m/z of each ends; in the files
  written here also the polarity of each scan (polarity_list), which RAPID2 leaves unread.

The pickles' classes are RAPID2's (Modules.process.convert_open), read here as plain records. Only they and numpy's
arrays are read: a pickle could run any code.
"""

import codecs
import hashlib
import io
import pickle
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"\x13RAPID"
VERSIONS = ((2, 2), (2, 3))
WRITTEN_VERSION = (2, 3)
HASH_SIZE = 64
SPECTRUM_TYPES = ("continuous", "discrete")  # profile, centroid
_PICKLE_MODULE = "Modules.process.convert_open"
# the byte counts' sizes, as RAPID2's Header of 2.3 has them
_HEADER = {
    "major_ver": 2,
    "minor_ver": 3,
    "mz_set_bytes_info_len": 10,
    "inten_set_bytes_info_len": 10,
    "no_compression_data_bytes_info_len": 5,
    "mz_set_info_for_chromatogram_extraction_bytes_info_len": 5,
}
_HEADER_SIZE_BYTES = 2


@dataclass
class RPDData:
    """What a .rpd file holds: of each scan a row, its points first, padded to the longest scan's."""
    spectrum_type: str  # "continuous" (profile) or "discrete" (centroid)
    mz_set: np.ndarray  # (scans, points), ascending, then NaN
    inten_set: np.ndarray  # (scans, points), integers, then 0
    point_counts: np.ndarray  # of each scan, the points of its row
    RT_list: np.ndarray  # of each scan, in RT_unit
    RT_unit: str  # "TimeInMinutes"
    spectrum_settings_dict: dict  # of the first scan, e.g. {"ScanMode": "Scan", "Polarity": "Positive"}
    ionization_type: str  # e.g. "Esi"
    analyzer_type: str  # e.g. "TimeOfFlight"
    # of each scan its "Polarity", which may differ from the first's (all else is the same): written here beside
    # RAPID2's attributes, which RAPID2 leaves unread; None in RAPID2's files, of which spectrum_settings_dict's holds
    polarity_list: list[str] | None = None
    data_hash: str = ""  # the SHA-256 the file holds, of its sections
    version: tuple[int, int] = WRITTEN_VERSION


def read_rpd(path: str | Path) -> RPDData:
    """The content of a .rpd file; ValueError if it is not one, or of a version not read."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            version = _read_version(f, path)
            data_hash = f.read(HASH_SIZE).decode("ascii")
            header = _unpickled(_section(f, _HEADER_SIZE_BYTES))
            mz_bytes = _section(f, header.mz_set_bytes_info_len)
            inten_bytes = _section(f, header.inten_set_bytes_info_len)
            if version >= (2, 3):
                # RAPID2's help to extract XICs, not needed here
                f.seek(int.from_bytes(f.read(header.mz_set_info_for_chromatogram_extraction_bytes_info_len), "little"), 1)
            settings = _unpickled(_section(f, header.no_compression_data_bytes_info_len))
        mz_set = _mz_set(mz_bytes)
        del mz_bytes
        inten_set = _inten_set(inten_bytes)
    except (EOFError, KeyError, AttributeError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
        raise ValueError(f"{path.name} is not a readable .rpd file: {error!r}") from error
    if settings.spectrum_type not in SPECTRUM_TYPES:
        raise ValueError(f"{path.name}: unknown spectrum type {settings.spectrum_type!r}")
    if mz_set.shape != inten_set.shape or len(settings.RT_list) != len(mz_set):
        raise ValueError(f"{path.name}: the m/z, intensities and RTs are not of the same scans")
    nan_start_locs = getattr(settings, "mz_set_nan_start_locs", None)
    if nan_start_locs is not None:
        point_counts = np.full(len(mz_set), mz_set.shape[1], dtype=np.int64)
        point_counts[np.asarray(nan_start_locs[0], dtype=np.int64)] = nan_start_locs[1]
    else:
        # older files of 2.2 have not kept where the scans end: there the m/z, made 0 to be saved, falls back, or is
        # NaN in the oldest, saved as it was; its differences then made NaN those in the same columns of the later
        # scans as well, which are lost (as in RAPID2)
        point_counts = np.array([_ascending_length(row) for row in mz_set], dtype=np.int64)
    for mz_row, inten_row, count in zip(mz_set, inten_set, point_counts):
        mz_row[count:] = np.nan
        inten_row[count:] = 0
    return RPDData(
        spectrum_type=settings.spectrum_type,
        mz_set=mz_set,
        inten_set=inten_set,
        point_counts=point_counts,
        RT_list=np.asarray(settings.RT_list, dtype=np.float64),
        RT_unit=settings.RT_unit,
        spectrum_settings_dict=dict(settings.spectrum_settings_dict),
        ionization_type=settings.ionization_type,
        analyzer_type=settings.analyzer_type,
        polarity_list=getattr(settings, "polarity_list", None),
        data_hash=data_hash,
        version=version,
    )


def stored_hash(path: str | Path) -> str:
    """The SHA-256 a .rpd file holds of its sections, the data_hash RAPID2 exports, without reading them."""
    path = Path(path)
    with open(path, "rb") as f:
        _read_version(f, path)
        return f.read(HASH_SIZE).decode("ascii")


def stored_spectrum_type(path: str | Path) -> str:
    """The spectrum type of a .rpd file, "continuous" or "discrete", from its settings, without reading its scans;
    ValueError if it is not a readable .rpd file."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            version = _read_version(f, path)
            f.seek(HASH_SIZE, 1)
            header = _unpickled(_section(f, _HEADER_SIZE_BYTES))
            skipped = [header.mz_set_bytes_info_len, header.inten_set_bytes_info_len]
            if version >= (2, 3):
                skipped.append(header.mz_set_info_for_chromatogram_extraction_bytes_info_len)
            for size in skipped:
                f.seek(int.from_bytes(f.read(size), "little"), 1)
            settings = _unpickled(_section(f, header.no_compression_data_bytes_info_len))
    except (EOFError, KeyError, AttributeError, pickle.UnpicklingError) as error:
        raise ValueError(f"{path.name} is not a readable .rpd file: {error!r}") from error
    if settings.spectrum_type not in SPECTRUM_TYPES:
        raise ValueError(f"{path.name}: unknown spectrum type {settings.spectrum_type!r}")
    return settings.spectrum_type


def write_rpd(path: str | Path, data: RPDData, on_progress: Callable[[str, int, int], None] | None = None) -> str:
    """Write data as a .rpd file of 2.3, as RAPID2 does, and return the SHA-256 it holds; data is left as it is.
    on_progress(what, done, total) is told of the scans written; it may raise to stop, and nothing is written then."""
    path = Path(path)
    if data.spectrum_type not in SPECTRUM_TYPES:
        raise ValueError(f"unknown spectrum type {data.spectrum_type!r}")
    report = on_progress or (lambda what, done, total: None)
    scans_written = 0

    def written_scan():
        nonlocal scans_written
        scans_written += 1
        if scans_written % 32 == 0:
            report(f"Writing {path.name}", scans_written, 2 * len(data.mz_set))

    if data.spectrum_type == "continuous":
        mz_bytes = _npz(_mz_members(data.mz_set, written_scan))
        inten_bytes = _npz(_inten_members(data.inten_set, written_scan))
        columns_bytes = _npz(_chromatogram_columns_members(data.mz_set, data.point_counts))
    else:
        mz_bytes = _npz(_plain_members("mz", data.mz_set, written_scan, nan_to_zero=True))
        inten_bytes = _npz(_plain_members("inten", data.inten_set, written_scan))
        columns_bytes = b""
    columns = data.mz_set.shape[1]
    nan_rows = np.flatnonzero(np.asarray(data.point_counts) < columns).astype(np.int64)
    settings = {
        "spectrum_type": data.spectrum_type,
        "RT_list": np.asarray(data.RT_list, dtype=np.float64),
        "RT_unit": data.RT_unit,
        "spectrum_settings_dict": dict(data.spectrum_settings_dict),
        "ionization_type": data.ionization_type,
        "analyzer_type": data.analyzer_type,
        "mz_set_nan_start_locs": (nan_rows, np.asarray(data.point_counts, dtype=np.int64)[nan_rows]),
    }
    if data.polarity_list is not None:
        settings["polarity_list"] = list(data.polarity_list)
    settings_bytes = _pickled("NoCompressionData", settings)
    header_bytes = _pickled("Header", _HEADER)
    sections = [
        (_HEADER_SIZE_BYTES, header_bytes),
        (_HEADER["mz_set_bytes_info_len"], mz_bytes),
        (_HEADER["inten_set_bytes_info_len"], inten_bytes),
        (_HEADER["mz_set_info_for_chromatogram_extraction_bytes_info_len"], columns_bytes),
        (_HEADER["no_compression_data_bytes_info_len"], settings_bytes),
    ]
    digest = hashlib.sha256()
    for size, section in sections:
        digest.update(len(section).to_bytes(size, "little"))
        digest.update(section)
    data_hash = digest.hexdigest()
    # written next to it first, so that a file stopped half way is not left in its place
    part = path.with_name(path.name + ".part")
    try:
        with open(part, "wb") as f:
            f.write(MAGIC + bytes(WRITTEN_VERSION) + data_hash.encode("ascii"))
            for size, section in sections:
                f.write(len(section).to_bytes(size, "little"))
                f.write(section)
        part.replace(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return data_hash


def _read_version(f, path: Path) -> tuple[int, int]:
    if f.read(len(MAGIC)) != MAGIC:
        raise ValueError(f"{path.name} is not a .rpd file")
    version = tuple(f.read(2))
    if version not in VERSIONS:
        raise ValueError(f"{path.name}: .rpd version {'.'.join(map(str, version))} is not read, only 2.2 and 2.3")
    return version


def _section(f, size: int) -> bytes:
    count = int.from_bytes(f.read(size), "little")
    section = f.read(count)
    if len(section) != count:
        raise EOFError("the file ends within a section")
    return section


def _ascending_length(row: np.ndarray) -> int:
    """The points of a row before its first NaN, or before its values first fall."""
    nan = np.flatnonzero(np.isnan(row))
    end = int(nan[0]) if len(nan) else len(row)
    falls = np.flatnonzero(row[1:end] < row[:end - 1])
    return int(falls[0]) + 1 if len(falls) else end


### pickles ###

class _Record:
    """A Header or NoCompressionData of RAPID2: its attributes alone."""


_RECONSTRUCT = np.empty(0).__reduce__()[0]
_SCALAR = np.float64(0).__reduce__()[0]
# what a .rpd's pickles may name: RAPID2's two classes and numpy's arrays, of numpy 1 (numpy.core) or 2 (numpy._core)
_UNPICKLED = {
    (_PICKLE_MODULE, "Header"): _Record,
    (_PICKLE_MODULE, "NoCompressionData"): _Record,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("numpy", "array"): np.array,
    ("_codecs", "encode"): codecs.encode,
    **{(module, name): value for module in ("numpy.core.multiarray", "numpy._core.multiarray")
       for name, value in (("_reconstruct", _RECONSTRUCT), ("scalar", _SCALAR))},
}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return _UNPICKLED[module, name]
        except KeyError:
            raise pickle.UnpicklingError(f"{module}.{name} is not of a .rpd file") from None


def _unpickled(section: bytes):
    return _Unpickler(io.BytesIO(section)).load()


class _Pickler(pickle.Pickler):
    """numpy's arrays pickled as numpy.array of their values: numpy 2's own pickles name numpy._core, which RAPID2's
    numpy 1 has not."""

    def reducer_override(self, obj):
        if isinstance(obj, np.ndarray):
            return np.array, (obj.tolist(), obj.dtype.str)
        return NotImplemented


def _pickled(class_name: str, state: dict) -> bytes:
    """An instance of RAPID2's class_name with the attributes of state, pickled; the class is named here by hand, as
    it cannot be imported: GLOBAL, EMPTY_TUPLE, NEWOBJ, the state, BUILD."""
    buffer = io.BytesIO()
    _Pickler(buffer, protocol=2).dump(state)
    pickled_state = buffer.getvalue()  # PROTO 2, the dict, STOP
    return (
        b"\x80\x02" + f"c{_PICKLE_MODULE}\n{class_name}\n".encode("ascii") + b")\x81"
        + pickled_state[2:-1] + b"b."
    )


### arrays ###
# m/z: of profile data, the second differences along each scan, then their differences from scan to scan, as RAPID2
# compresses them (initial_mz_array0, 1 and 2 hold the first values each difference drops); intensities: of profile
# data, their differences from scan to scan; of centroid data both as they are, the names alike.

def _npz(members: Iterator) -> bytes:
    """An .npz as np.savez_compressed writes it, of members yielded as (name, shape, dtype), then their C-order rows,
    then None, each in turn: not all held at once."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name, shape, dtype in members:
            with archive.open(name + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array_header_1_0(
                    f, {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False, "shape": shape}
                )
                while (row := next(members)) is not None:
                    f.write(np.ascontiguousarray(row, dtype=dtype).tobytes())
    return buffer.getvalue()


def _whole(name: str, array: np.ndarray) -> Iterator:
    array = np.asarray(array)
    yield name, array.shape, array.dtype
    yield array
    yield None


def _mz_members(mz_set: np.ndarray, written_scan: Callable[[], None]) -> Iterator:
    n_scans, columns = mz_set.shape
    first_two = np.nan_to_num(mz_set[:, :2], nan=0.0)
    yield from _whole("initial_mz_array0", first_two[:, 0])
    yield from _whole("initial_mz_array1", first_two[:, 1] - first_two[:, 0])
    previous = np.diff(np.nan_to_num(mz_set[0], nan=0.0), n=2)
    yield from _whole("initial_mz_array2", previous)
    written_scan()
    yield "mz_set_diff", (n_scans - 1, columns - 2), mz_set.dtype
    for row in mz_set[1:]:
        second_differences = np.diff(np.nan_to_num(row, nan=0.0), n=2)
        yield second_differences - previous
        previous = second_differences
        written_scan()
    yield None


def _inten_members(inten_set: np.ndarray, written_scan: Callable[[], None]) -> Iterator:
    yield from _whole("initial_inten_array0", inten_set[0])
    written_scan()
    yield "inten_set_diff", (len(inten_set) - 1, inten_set.shape[1]), inten_set.dtype
    for previous, row in zip(inten_set[:-1], inten_set[1:]):
        yield row - previous
        written_scan()
    yield None


def _plain_members(prefix: str, array: np.ndarray, written_scan: Callable[[], None], *, nan_to_zero: bool = False) -> Iterator:
    yield f"{prefix}_set_diff", array.shape, array.dtype
    for row in array:
        yield np.nan_to_num(row, nan=0.0) if nan_to_zero else row
        written_scan()
    yield None


def _chromatogram_columns_members(mz_set: np.ndarray, point_counts: np.ndarray) -> Iterator:
    """RAPID2's ref_row, the first scan with points in all the columns, and of each column i its first and last
    column to look in for the m/z of ref_row there (mz_set_info_for_chromatogram_extraction): after the last column
    below it in all the scans, and before the first above it in all; a column with NaN in any scan is neither."""
    columns = mz_set.shape[1]
    full = np.flatnonzero(np.asarray(point_counts) == columns)
    if not len(full):
        raise ValueError("no scan has points in all the columns")
    ref_row = int(full[0])
    reference = mz_set[ref_row]
    # both ascend along the columns, as each scan does: a column with NaN is followed by such columns only
    highest = mz_set.max(axis=0)
    lowest = mz_set.min(axis=0)
    with_nan = np.isnan(highest)
    highest[with_nan] = np.inf
    first_with_nan = int(np.argmax(with_nan)) if with_nan.any() else columns
    info = np.empty((2, columns), dtype=np.int64)
    info[0] = np.searchsorted(highest, reference, side="left")
    above = np.searchsorted(lowest[:first_with_nan], reference, side="right")
    info[1] = np.where(above < first_with_nan, above - 1, columns - 1)
    yield from _whole("ref_row", np.asarray(ref_row))
    yield from _whole("mz_set_info_for_chromatogram_extraction", info)


def _open_npy(archive: zipfile.ZipFile, name: str):
    f = archive.open(name + ".npy")
    version = np.lib.format.read_magic(f)
    read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran_order, dtype = read_header(f)
    if fortran_order:
        raise ValueError(f"{name} is in Fortran order")
    return f, shape, dtype


def _read_rows_into(f, rows: np.ndarray) -> None:
    """The rows of a .npy, read one at a time into rows (2-D), each row contiguous."""
    for row in rows:
        view = memoryview(row).cast("B")
        if f.readinto(view) != len(view):
            raise EOFError("an array ends early")


def _array(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    with archive.open(name + ".npy") as f:
        return np.lib.format.read_array(f)


def _mz_set(section: bytes) -> np.ndarray:
    """The m/z of the scans, the padding 0, as RAPID2 reads them: the differences summed again in their order."""
    with zipfile.ZipFile(io.BytesIO(section)) as archive:
        if "initial_mz_array0.npy" not in archive.namelist():
            return _array(archive, "mz_set_diff")
        f, (n_differences, n_second), dtype = _open_npy(archive, "mz_set_diff")
        mz_set = np.empty((n_differences + 1, n_second + 2), dtype=dtype)
        mz_set[0, 2:] = _array(archive, "initial_mz_array2")
        with f:
            _read_rows_into(f, mz_set[1:, 2:])
        for previous, row in zip(mz_set[:-1, 2:], mz_set[1:, 2:]):
            row += previous
        first_differences, first_values = _array(archive, "initial_mz_array1"), _array(archive, "initial_mz_array0")
    for row, first_difference, first_value in zip(mz_set, first_differences, first_values):
        row[1] = first_difference
        np.cumsum(row[1:], out=row[1:])
        row[0] = first_value
        np.cumsum(row, out=row)
    return mz_set


def _inten_set(section: bytes) -> np.ndarray:
    with zipfile.ZipFile(io.BytesIO(section)) as archive:
        if "initial_inten_array0.npy" not in archive.namelist():
            return _array(archive, "inten_set_diff")
        f, (n_differences, columns), dtype = _open_npy(archive, "inten_set_diff")
        inten_set = np.empty((n_differences + 1, columns), dtype=dtype)
        inten_set[0] = _array(archive, "initial_inten_array0")
        with f:
            _read_rows_into(f, inten_set[1:])
    for previous, row in zip(inten_set[:-1], inten_set[1:]):
        row += previous
    return inten_set
