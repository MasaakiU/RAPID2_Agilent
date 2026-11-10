"""mzData files (.mzdata.xml, as MassHunter exports them) converted to .rpd files, as RAPID2 converts them."""

import base64
from collections.abc import Callable
from pathlib import Path

import numpy as np
from lxml import etree

from .rpd_file import RPDData, write_rpd

SUFFIX = ".mzdata.xml"
RT_UNIT = "TimeInMinutes"  # the cvParam of a scan's RT, as RAPID2 reads it


def rpd_path(path: str | Path) -> Path:
    """The .rpd file an mzData file is converted to, next to it: x.mzdata.xml to x.rpd (x.xml to x.rpd)."""
    path = Path(path)
    if path.name.lower().endswith(SUFFIX):
        return path.with_name(path.name[:-len(SUFFIX)] + ".rpd")
    return path.with_suffix(".rpd")


def convert(path: str | Path, on_progress: Callable[[str, int, int], None] | None = None) -> Path:
    """Convert an mzData file to its .rpd file (rpd_path), replaced if there; return that. on_progress(what, done,
    total) is told of the KiB read and the scans written; it may raise to stop, and nothing is written then."""
    target = rpd_path(path)
    write_rpd(target, read_mzdata(path, on_progress), on_progress)
    return target


def read_mzdata(path: str | Path, on_progress: Callable[[str, int, int], None] | None = None) -> RPDData:
    """The scans of an mzData file, as RAPID2 converts them: ValueError unless they are all MS1 of one setting but for
    their polarity (as of a run of polarity segments), which a .rpd file holds once, the first scan's, and the
    polarity of each beside it. Intensities are truncated to integers, as RAPID2 does."""
    path = Path(path)
    report = on_progress or (lambda what, done, total: None)
    total = path.stat().st_size
    ionization_type = analyzer_type = spectrum_type = settings = None
    mz_lists, inten_lists, rt_list, polarity_list = [], [], [], []
    with open(path, "rb") as f:
        for _event, element in etree.iterparse(f, events=("end",), tag=("description", "spectrum"), huge_tree=True):
            if element.tag == "description":
                ionization_type = _value(element, "instrument/source/cvParam")
                analyzer_type = _value(element, "instrument/analyzerList/analyzer/cvParam")
                continue
            instrument = element.find("spectrumDesc/spectrumSettings/spectrumInstrument")
            scan_settings = {param.get("name"): param.get("value") for param in instrument.findall("cvParam")}
            if RT_UNIT not in scan_settings:
                raise ValueError(f"{path.name}: spectrum {element.get('id')} has no {RT_UNIT}")
            rt_list.append(float(scan_settings.pop(RT_UNIT)))
            if instrument.get("msLevel", "1") != "1":
                raise ValueError(f"{path.name}: spectrum {element.get('id')} is MS{instrument.get('msLevel')}; only MS1 scans are converted")
            polarity_list.append(scan_settings.get("Polarity"))
            if settings is None:
                settings = scan_settings
                spectrum_type = element.find("spectrumDesc/spectrumSettings/acqSpecification").get("spectrumType")
            elif {**scan_settings, "Polarity": None} != {**settings, "Polarity": None}:
                raise ValueError(
                    f"{path.name}: spectrum {element.get('id')} is of {scan_settings}, the first of {settings}; "
                    "a .rpd file holds the scans of one setting but for their polarity"
                )
            mz_lists.append(_decoded(element.find("mzArrayBinary/data")))
            inten_lists.append(_decoded(element.find("intenArrayBinary/data")))
            # the spectra read are let go of, so that the file is not held whole
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]
            if len(rt_list) % 32 == 0:
                # in KiB, which a progress bar's int holds of files of GBs
                report(f"Reading {path.name}", f.tell() >> 10, total >> 10)
    if not rt_list:
        raise ValueError(f"{path.name} has no spectra")
    if ionization_type is None or analyzer_type is None:
        raise ValueError(f"{path.name} has no description of its instrument")
    # the dtypes of the first spectrum's, the intensities as integers of their size
    mz_dtype = mz_lists[0].dtype.newbyteorder("=")
    inten_dtype = np.dtype(np.int32 if inten_lists[0].dtype.itemsize == 4 else np.int64)
    point_counts = np.array([len(mz_list) for mz_list in mz_lists], dtype=np.int64)
    mz_set = np.full((len(mz_lists), int(point_counts.max())), np.nan, dtype=mz_dtype)
    inten_set = np.zeros(mz_set.shape, dtype=inten_dtype)
    for i, count in enumerate(point_counts):
        if len(inten_lists[i]) != count:
            raise ValueError(f"{path.name}: spectrum {i + 1} has {count} m/z and {len(inten_lists[i])} intensities")
        mz_set[i, :count] = mz_lists[i]
        inten_set[i, :count] = inten_lists[i]
        mz_lists[i] = inten_lists[i] = None
    return RPDData(
        spectrum_type=spectrum_type,
        mz_set=mz_set,
        inten_set=inten_set,
        point_counts=point_counts,
        RT_list=np.array(rt_list),
        RT_unit=RT_UNIT,
        spectrum_settings_dict=settings,
        ionization_type=ionization_type,
        analyzer_type=analyzer_type,
        polarity_list=polarity_list,
    )


def _value(description, path: str) -> str | None:
    param = description.find(path)
    return None if param is None else param.get("value")


def _decoded(data) -> np.ndarray:
    """The values of an mzArrayBinary or intenArrayBinary's data: floats of its precision and endian."""
    endian, precision = data.get("endian"), data.get("precision")
    if endian not in ("little", "big") or precision not in ("32", "64"):
        raise ValueError(f"data of endian {endian!r} and precision {precision!r} are not read")
    dtype = np.dtype(("<" if endian == "little" else ">") + ("f4" if precision == "32" else "f8"))
    values = np.frombuffer(base64.b64decode(data.text or ""), dtype=dtype)
    if len(values) != int(data.get("length")):
        raise ValueError(f"{len(values)} values where {data.get('length')} are said")
    return values
