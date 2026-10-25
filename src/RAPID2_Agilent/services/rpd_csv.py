"""Targets and their values in runs, written as a .rpd.csv file, and targets read back from one, or from a file of
RAPID2's Export Targets; see docs/rpd_csv_format.md. No Qt.

A row for each target and run, the targets by number and the runs in the order given; with no runs, a row for each
target, the run's columns empty. Tab-separated, UTF-8 with a BOM, LF line ends, nothing quoted.
"""

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..io.ms_data import MSData
from ..io.run_hashes import data_size, is_known, run_hashes
# the terms of an adduct, to take them off a RAPID2 ion formula; parse_adduct gives only their mass
from ..models.chemistry import _ADDUCT_SYNTAX, _ADDUCT_TERM, SOLVENT_FORMULAS, parse_adduct, parse_formula
from ..models.run_peaks import RunPeaks, RunPeaksStore
from ..models.target import MSLevel, Target, ToleranceUnit, TraceType, new_target_id
from .peak_detection import find_run_peaks
from .quantification import Quantity, quantify_target

SUFFIX = ".rpd.csv"
COLUMNS = (
    "#",
    "compound_name",
    "formula",
    "adduct",
    "file_path",
    "file_name",
    "data_hash",
    "scan_sha256",
    "scan_settings",
    "trace_type",
    "MS_level",
    "precursor_m/z",
    "collision_energy",
    "isolation_width",
    "m/z",
    "m/z_range",
    "m/z_range_unit",
    "RT",
    "RT_range",
    "m/z_btm",
    "m/z_top",
    "RT_btm",
    "RT_top",
    "RT_btm (value used for calculation)",
    "RT_top (value used for calculation)",
    "N_scans (value used for calculation)",
    "area",
    "area (baseline adjusted)",
    "height",
    "baseline_height",
    "RT peaktop",
    "mz peaktop",
    "note",
)
UNIT_TEXTS = {ToleranceUnit.MZ: "abs", ToleranceUnit.PPM: "ppm"}  # as on the m/z control's mode button
_EMPTY_ROW = dict.fromkeys(COLUMNS, "")
# the columns of a target, which an import reads (§3.1); the others it skips
TARGET_COLUMNS = (
    "#", "compound_name", "formula", "adduct", "trace_type", "MS_level", "precursor_m/z", "collision_energy",
    "isolation_width", "m/z", "m/z_range", "m/z_range_unit", "RT", "RT_range", "note",
)
# what RAPID2's Load Targets read of its files (§7)
RAPID2_COLUMNS = ("compound_name", "formula", "is_TIC", "m/z", "m/z_range", "RT", "RT_range")
# the m/z of a TIC or BPC target, which its row leaves empty: the plots' initial m/z window
# (views.view_constants DEFAULT_MZ and DEFAULT_MZ_TOLERANCE), in m/z
TIC_MZ = 150.5
TIC_MZ_TOLERANCE = 0.05
# an adduct for a RAPID2 target with no formula and no adduct in its name, by the polarity of its scans (§8)
POLARITY_ADDUCTS = {"+": "[M+H]+", "-": "[M-H]-"}
DEFAULT_ADDUCT = "[M+H]+"  # the target panel's
_NUMBER = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")  # 1234.5, as §4 has numbers
_ION_CHARGE = re.compile(r"([+-])([0-9]*)$")  # of a RAPID2 ion formula, C5H14NO+ or C10H20N2+2


class Cancelled(Exception):
    """Raised by an on_progress callback to stop the export, or another long task told of its progress."""


@dataclass(frozen=True)
class Run:
    data: MSData
    path: Path  # its .d folder


def rpd_csv_path(path: str | Path) -> Path:
    """path ending in .rpd.csv: a save dialog may drop the .rpd of a name typed without it."""
    path = Path(path)
    if path.name.endswith(SUFFIX):
        return path
    return path.with_name(path.name.removesuffix(".csv") + SUFFIX)


@dataclass(frozen=True)
class Measured:
    """The values of a target in a run: a row."""
    target: Target
    run: Run
    quantity: Quantity
    run_peaks: RunPeaks | None = None  # the windows of the target in the run, if not the target's

    @property
    def note(self) -> str:
        """Why its values are empty or leave scans out; "" when they do neither."""
        if self.run_peaks is not None and self.run_peaks.rt_window is None:
            return "no peak taken for the target, so no values"
        return _note(self.quantity)


def measure(
    targets: Sequence[Target],
    runs: Sequence[Run],
    *,
    merge_ppm: float = 0.0,
    run_peaks: RunPeaksStore | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> list[Measured]:
    """The values of each target in each run, in the order of the rows: the targets by number, each in the runs in
    the order given; without run_peaks in the target's windows, with it in those it has of the target in the run,
    those it has not found (peak_detection.find_run_peaks) and kept in it. on_progress(what, done, total) is told of
    the pairs measured; it may raise Cancelled."""
    report = on_progress or (lambda what, done, total: None)
    pairs = [(target, run) for target in sorted(targets, key=lambda target: target.index) for run in runs]
    measured = []
    for done, (target, run) in enumerate(pairs, start=1):
        windows = None if run_peaks is None else run_peaks.get(target.id, run.path)
        if run_peaks is not None and windows is None:
            # the peaks not looked for in the run yet, found as Auto finds them
            windows = find_run_peaks(run.data, target, merge_ppm=merge_ppm)
            run_peaks.set(target.id, run.path, windows)
        quantity = quantify_target(run.data, target, run_peaks=windows, merge_ppm=merge_ppm)
        measured.append(Measured(target, run, quantity, windows))
        report("Calculating the areas", done, len(pairs))
    return measured


def export_rows(
    targets: Sequence[Target],
    runs: Sequence[Run],
    *,
    merge_ppm: float = 0.0,
    run_peaks: RunPeaksStore | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
    measured: Sequence[Measured] | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    """The rows, each with all the columns ("" for none), and a note on each whose values are empty or leave scans out;
    the values in the windows of run_peaks, as `measure` takes them.

    on_progress(what, done, total) is told of the bytes read for the runs' hashes, and of the target-run pairs
    measured; it may raise Cancelled. measured, if given, is `measure` of the same targets and runs, not done again.
    """
    report = on_progress or (lambda what, done, total: None)
    hashes = {}
    to_read = sum(data_size(run.path) for run in runs if not is_known(run.path))
    read = 0

    def on_chunk(size):
        nonlocal read
        read += size
        report("Reading the data to hash", read, to_read)

    for run in runs:
        hashes[run.path] = run_hashes(run.path, on_chunk)

    if not runs:
        return [_EMPTY_ROW | _target_fields(target) for target in sorted(targets, key=lambda target: target.index)], []
    if measured is None:
        measured = measure(targets, runs, merge_ppm=merge_ppm, run_peaks=run_peaks, on_progress=on_progress)
    rows, notes = [], []
    for row in measured:
        rows.append(_EMPTY_ROW | _target_fields(row.target) | _run_fields(row.run, hashes[row.run.path], row.quantity))
        if row.note:
            notes.append(f"#{row.target.index} {row.target.name}, {row.run.path.name}: {row.note}")
    return rows, notes


def write_rpd_csv(path: str | Path, rows: Sequence[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            f.write("\t".join(row[column] for column in COLUMNS) + "\n")


def _target_fields(target: Target) -> dict[str, str]:
    """The target's columns, and the windows it gives."""
    rt_btm, rt_top = target.rt_window()
    fields = {
        "#": str(target.index),
        # a Target holds no tab or line break in its name and note
        "compound_name": target.name,
        "formula": target.formula,
        "adduct": target.adduct,
        "trace_type": target.trace_type.value,
        "MS_level": target.ms_level.value,
        "precursor_m/z": _number(target.precursor_mz),
        "collision_energy": _number(target.collision_energy),
        "isolation_width": _number(target.isolation_width),
        "RT": _number(target.rt),
        "RT_range": _number(target.rt_tolerance),
        "RT_btm": _number(rt_btm),
        "RT_top": _number(rt_top),
        "note": target.note,
    }
    # the m/z of an XIC only; a ppm tolerance is a whole number
    if target.trace_type is TraceType.XIC:
        mz_btm, mz_top = target.mz_window()
        ppm = target.mz_tolerance_unit is ToleranceUnit.PPM
        fields |= {
            "m/z": _number(target.mz),
            "m/z_range": _number(int(target.mz_tolerance) if ppm else target.mz_tolerance),
            "m/z_range_unit": UNIT_TEXTS[target.mz_tolerance_unit],
            "m/z_btm": _number(mz_btm),
            "m/z_top": _number(mz_top),
        }
    return fields


def _run_fields(run: Run, hashes: tuple[str, str], quantity: Quantity) -> dict[str, str]:
    scan_sha256, data_sha256 = hashes
    return {
        "file_path": _text(str(run.path), "file_path"),
        "file_name": run.path.name,
        "data_hash": data_sha256,
        "scan_sha256": scan_sha256,
        "scan_settings": quantity.scan_settings,
        "RT_btm (value used for calculation)": _number(quantity.rt_first),
        "RT_top (value used for calculation)": _number(quantity.rt_last),
        "N_scans (value used for calculation)": str(quantity.n_scans),
        "area": _number(quantity.area),
        "area (baseline adjusted)": _number(quantity.baseline_adjusted_area),
        "height": _number(quantity.height),
        "baseline_height": _number(quantity.baseline_height),
        "RT peaktop": _number(quantity.rt_peaktop),
        "mz peaktop": _number(quantity.mz_peaktop),
    }


def _note(quantity: Quantity) -> str:
    if quantity.n_scans < 2:
        return f"{quantity.n_scans} scans to use in the RT window, so no values"
    if quantity.n_missing:
        return f"{quantity.n_missing} scans left out, as the m/z window misses their data points"
    return ""


def _number(value) -> str:
    """All the digits of a number, without an exponent; "" for none."""
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return np.format_float_positional(float(value), unique=True, trim="-")


def _text(value: str, column: str) -> str:
    if any(character in value for character in "\t\r\n"):
        raise ValueError(f"{column} cannot hold a tab or a line break: {value!r}")
    return value


class TargetFileError(ValueError):
    """The file cannot be imported; .problems lists every one found: "line 3, RT: not a number"."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


@dataclass(frozen=True)
class ImportedGroup:
    """The rows of one target of a file. Rows of one '#' should hold the same target, and do in a file written here;
    those that do not are its variants, for the importer to choose among (§5.3)."""
    number: int  # the file's '#'; of a RAPID2 file, 1, 2, ... in the order the targets first appear
    variants: tuple[Target, ...]  # the distinct targets of its rows, in the order first seen; usually one
    lines: tuple[tuple[int, ...], ...]  # the file's line numbers (1 = the header) of the rows of each variant


def read_targets_file(path: str | Path) -> list[ImportedGroup]:
    """The targets of a .rpd.csv file, or of a file of RAPID2's Export Targets, by '#'; told apart by the header (§5.1).

    TargetFileError lists every problem of the file, and then nothing is imported. The variants carry the '#' as
    their index; resolve_groups numbers them.
    """
    try:
        text = Path(path).read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError:
        raise TargetFileError(["The file is not UTF-8 text"]) from None
    lines = [line.removesuffix("\r") for line in text.split("\n")]
    rows = [(number, line.split("\t")) for number, line in enumerate(lines[1:], start=2) if line.strip()]
    header = lines[0].split("\t")
    if "adduct" in header or "trace_type" in header:
        return _read_rows(header, rows)
    if "is_TIC" in header:
        return _read_rapid2_rows(header, rows)
    raise TargetFileError(["line 1: not a file of targets: its header has no adduct, trace_type or is_TIC column"])


def resolve_groups(
    groups: Sequence[ImportedGroup], choices: dict[int, int | None] | None = None, *, first_index: int = 1,
) -> list[Target]:
    """The targets to import: of each group the variant at the position choices gives for its number, or with None,
    or not in choices, each variant as a target of its own. Numbered first_index, first_index + 1, ... by '#', the
    variants of one '#' one after another, each with a new id."""
    targets = []
    for group in sorted(groups, key=lambda group: group.number):
        choice = (choices or {}).get(group.number)
        for variant in group.variants if choice is None else (group.variants[choice],):
            targets.append(replace(variant, index=first_index + len(targets), id=new_target_id()))
    return targets


def _read_rows(header: list[str], rows: list[tuple[int, list[str]]]) -> list[ImportedGroup]:
    problems = [f"line 1, {name}: the column is there more than once" for name in sorted(_repeated(header))]
    problems += [f"line 1, {name}: not a column of the format" for name in header if name not in COLUMNS]
    problems += [f"line 1, {name}: the column is missing" for name in TARGET_COLUMNS if name not in header]
    if problems:
        raise TargetFileError(problems)
    found = []
    for line, fields in rows:
        if len(fields) != len(header):
            problems.append(f"line {line}: {len(fields)} fields, where the header has {len(header)}")
            continue
        target = _target(line, dict(zip(header, fields)), problems)
        if target is not None:
            found.append((target, line))
    if problems:
        raise TargetFileError(problems)
    return _grouped(found)


def _target(line: int, row: dict[str, str], problems: list[str]) -> Target | None:
    """The target of a row of the new format; None, and its problems added, when it holds none."""
    found = len(problems)

    def problem(column, reason):
        problems.append(f"line {line}, {column}: {reason}")

    def number(column, *, empty_allowed=False):
        text = row[column]
        if not text and empty_allowed:
            return None
        if not _NUMBER.fullmatch(text):
            problem(column, f"not a number: {text!r}" if text else "empty")
            return None
        return float(text)

    def choice(column, kinds):
        try:
            return kinds(row[column])
        except ValueError:
            problem(column, f"one of {', '.join(kind.value for kind in kinds)}, not {row[column]!r}")
            return None

    index = row["#"]
    if not (index.isdigit() and int(index) > 0):
        problem("#", f"not a positive whole number: {index!r}")
    if not row["adduct"]:
        problem("adduct", "empty")
    trace_type = choice("trace_type", TraceType)
    ms_level = choice("MS_level", MSLevel)
    ms2 = {column: number(column, empty_allowed=True) for column in ("precursor_m/z", "collision_energy", "isolation_width")}
    rt, rt_tolerance = number("RT"), number("RT_range")
    # empty for a TIC or a BPC
    mz_empty = trace_type is not TraceType.XIC and not (row["m/z"] or row["m/z_range"] or row["m/z_range_unit"])
    if mz_empty:
        mz, mz_tolerance, mz_tolerance_unit = TIC_MZ, TIC_MZ_TOLERANCE, ToleranceUnit.MZ
    else:
        mz, mz_tolerance = number("m/z"), number("m/z_range")
        mz_tolerance_unit = next((unit for unit, text in UNIT_TEXTS.items() if text == row["m/z_range_unit"]), None)
        if mz_tolerance_unit is None:
            problem("m/z_range_unit", f"abs or ppm, not {row['m/z_range_unit']!r}")
    if len(problems) > found:
        return None
    try:
        return Target(
            index=int(index),
            name=row["compound_name"],
            formula=row["formula"],
            adduct=row["adduct"],
            trace_type=trace_type,
            ms_level=ms_level,
            rt=rt,
            rt_tolerance=rt_tolerance,
            mz=mz,
            mz_tolerance=mz_tolerance,
            mz_tolerance_unit=mz_tolerance_unit,
            precursor_mz=ms2["precursor_m/z"],
            collision_energy=ms2["collision_energy"],
            isolation_width=ms2["isolation_width"],
            note=row["note"],
        )
    except ValueError as error:
        problems.append(f"line {line}: {error}")
        return None


def _read_rapid2_rows(header: list[str], rows: list[tuple[int, list[str]]]) -> list[ImportedGroup]:
    """The targets of a file of RAPID2's Export Targets, its rows with the same values of RAPID2_COLUMNS one target
    (§7). Its first column, unnamed, numbers the rows; a column may be there twice, and the first is read."""
    problems = [f"line 1, {name}: the column is missing" for name in RAPID2_COLUMNS if name not in header]
    if problems:
        raise TargetFileError(problems)
    columns = {name: header.index(name) for name in (*RAPID2_COLUMNS, "scan_settings") if name in header}
    lines: dict[tuple[str, ...], list[int]] = {}
    targets = {}
    for line, fields in rows:
        if len(fields) != len(header):
            problems.append(f"line {line}: {len(fields)} fields, where the header has {len(header)}")
            continue
        key = tuple(fields[columns[name]] for name in RAPID2_COLUMNS)
        if key not in lines:
            # read from its first row
            scan_settings = fields[columns["scan_settings"]] if "scan_settings" in columns else ""
            targets[key] = _rapid2_target(line, len(lines) + 1, dict(zip(RAPID2_COLUMNS, key)), scan_settings, problems)
        lines.setdefault(key, []).append(line)
    if problems:
        raise TargetFileError(problems)
    return [ImportedGroup(target.index, (target,), (tuple(lines[key]),)) for key, target in targets.items()]


def _rapid2_target(line: int, index: int, row: dict[str, str], scan_settings: str, problems: list[str]) -> Target | None:
    found = len(problems)

    def number(column):
        try:
            value = float(row[column])
        except ValueError:
            value = math.nan
        if not math.isfinite(value):
            problems.append(f"line {line}, {column}: not a number: {row[column]!r}")
        return value

    is_tic = row["is_TIC"]
    if is_tic not in ("True", "False"):
        problems.append(f"line {line}, is_TIC: True or False, not {is_tic!r}")
    mz, mz_tolerance, rt, rt_tolerance = (number(column) for column in ("m/z", "m/z_range", "RT", "RT_range"))
    if len(problems) > found:
        return None
    # RAPID2 wrote -1 for the m/z of a TIC
    if is_tic == "True" and mz < 0:
        mz, mz_tolerance = TIC_MZ, TIC_MZ_TOLERANCE
    try:
        name, formula, adduct = _neutral(row["compound_name"], row["formula"], scan_settings)
        return Target(
            index=index,
            name=name,
            formula=formula,
            adduct=adduct,
            trace_type=TraceType.TIC if is_tic == "True" else TraceType.XIC,
            ms_level=MSLevel.MS1,
            rt=rt,
            rt_tolerance=rt_tolerance,
            # the file's, though RAPID2 left the electrons out of it (§7)
            mz=mz,
            mz_tolerance=mz_tolerance,
            mz_tolerance_unit=ToleranceUnit.MZ,
        )
    except ValueError as error:
        problems.append(f"line {line}: {error}")
        return None


def _neutral(name: str, ion_formula: str, scan_settings: str) -> tuple[str, str, str]:
    """(name, neutral formula, adduct) of a RAPID2 target, from its name, which may end in its adduct, and its ion
    formula, which ends in its charge: C38H76NO8PCl- of PC 30:0 [M+Cl]- is C38H76NO8P. When the name's adduct does
    not make the ion of the formula, or the name has none, the formula is the ion's without the charge and the adduct
    [M]+ or [M]-, and the name stays as it is; with no formula either, the adduct is that of the polarity of the scans
    (POLARITY_ADDUCTS), or DEFAULT_ADDUCT."""
    named_adduct = None
    start = name.rfind("[")
    if start >= 0 and _ADDUCT_SYNTAX.fullmatch(name[start:]):
        named_adduct = name[start:]
    if not ion_formula:
        if named_adduct is not None:
            return name[:start].rstrip(), "", named_adduct
        # the sign after the ionization of each kind of scan: ESI+ QuadrupoleTimeOfFlight SCAN, IonMode 32- Scan
        signs = set(re.findall(r"(?:^|, )[^,]*?([+-])(?=\s|,|$)", scan_settings))
        return name, "", POLARITY_ADDUCTS[signs.pop()] if len(signs) == 1 else DEFAULT_ADDUCT
    charge = _ION_CHARGE.search(ion_formula)
    if charge is None:
        # written neutral
        return name, _hill(parse_formula(ion_formula)), named_adduct or DEFAULT_ADDUCT
    sign, size = charge.group(1), int(charge.group(2) or 1)
    counts = parse_formula(ion_formula[:charge.start()])
    if named_adduct is not None:
        neutral = _less_adduct(counts, named_adduct, size if sign == "+" else -size)
        if neutral is not None:
            return name[:start].rstrip(), _hill(neutral), named_adduct
    return name, _hill(counts), f"[M]{size if size > 1 else ''}{sign}"


def _less_adduct(counts: dict[str, int], adduct: str, charge: int) -> dict[str, int] | None:
    """The neutral molecule of the ion with these counts, made with adduct of this charge; None if it is not one."""
    if parse_adduct(adduct).charge != charge:
        return None
    multiplier, terms, _charge, _sign = _ADDUCT_SYNTAX.fullmatch(adduct).groups()
    molecules = dict(counts)
    for term_sign, count, group in _ADDUCT_TERM.findall(terms):
        for symbol, n in parse_formula(SOLVENT_FORMULAS.get(group, group)).items():
            molecules[symbol] = molecules.get(symbol, 0) - (n if term_sign == "+" else -n) * int(count or 1)
    multiplier = int(multiplier or 1)
    if any(n < 0 or n % multiplier for n in molecules.values()):
        return None
    neutral = {symbol: n // multiplier for symbol, n in molecules.items() if n}
    return neutral or None


def _hill(counts: dict[str, int]) -> str:
    """A formula in the Hill order: C, then H, then the rest alphabetically, with no C all alphabetically; an isotope,
    [13C], right after its element."""
    def element(symbol):
        return re.sub(r"[\[\]0-9]", "", symbol)

    first = ("C", "H") if any(element(symbol) == "C" for symbol in counts) else ()

    def order(symbol):
        name = element(symbol)
        return (first.index(name) if name in first else len(first), name, symbol != name, symbol)

    return "".join(f"{symbol}{n if n > 1 else ''}" for symbol, n in sorted(counts.items(), key=lambda item: order(item[0])))


def _grouped(found: list[tuple[Target, int]]) -> list[ImportedGroup]:
    """Rows of one '#' grouped, those of equal targets one variant (their ids aside), by '#'."""
    groups: dict[int, dict[Target, list[int]]] = {}
    for target, line in found:
        groups.setdefault(target.index, {}).setdefault(replace(target, id=""), []).append(line)
    return [
        ImportedGroup(
            number,
            tuple(replace(variant, id=new_target_id()) for variant in variants),
            tuple(tuple(lines) for lines in variants.values()),
        )
        for number, variants in sorted(groups.items())
    ]


def _repeated(names: Sequence[str]) -> set[str]:
    return {name for name in names if names.count(name) > 1}
