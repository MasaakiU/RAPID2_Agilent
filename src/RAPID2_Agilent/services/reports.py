"""Summaries of runs for the CLI and the MCP tools, as JSON-serializable dicts.

Each one is an operation (`operations`, or a job of `alignment.runner`) plus
the shaping a reader of text needs and a plot does not: a long trace reduced to
max_points, a spectrum reduced to its strongest peaks, numpy values made plain.
Nothing is computed here that the operations do not already compute.
"""

from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from ..alignment import AlignmentConfig, align_and_save, with_options
from ..alignment import scale_candidates as alignment_scale_candidates
from ..io import AgilentReader
from ..models.chemistry import formula_mass, ion_mz, parse_adduct
from ..models.run_peaks import RunPeaksStore
from ..models.target import MSLevel, Target, ToleranceUnit, TraceType, tolerance_window
from . import maps, operations, rpd_csv
from .runs import RunCache, run_path


# the name of a target described without a name or a formula: the target panel's
DEFAULT_TARGET_NAME = "New target"
# the polarities a report takes, as the signs operations uses: +1 positive, -1 negative
POLARITIES = {"positive": 1, "negative": -1}


def polarity_sign(polarity: str | None, adduct: str | None = None) -> int | None:
    """The sign of a polarity named, or of an adduct's charge; None for neither. ValueError if they disagree."""
    sign = None if polarity is None else POLARITIES[polarity]
    if adduct is not None:
        charge_sign = 1 if parse_adduct(adduct).charge > 0 else -1
        if sign is not None and sign != charge_sign:
            raise ValueError(f"The adduct {adduct} is {operations.polarity_name(charge_sign)}, not {polarity}")
        sign = charge_sign
    return sign


def _scans(reader, ms_level, rt_range=None, *, ms2=None, polarity=None) -> np.ndarray:
    """The scans of a report, as operations.select_scans chooses them. Of a run of both polarities one has to be
    chosen: a trace through both alternates between two different sets of ions."""
    scan_indices = operations.select_scans(reader, ms_level, rt_range, ms2=ms2, polarity=polarity)
    if polarity is None and reader.polarity_list is not None and len(np.unique(reader.polarity_list[scan_indices])) > 1:
        raise ValueError("The scans are of both polarities: give polarity (positive or negative)")
    return scan_indices


def _polarity_dict(sign: int | None) -> dict:
    return {} if sign is None else {"polarity": operations.polarity_name(sign)}


def scan_types(data) -> list[dict]:
    """The polarity and MS level of the scans, with how many of each: "Negative MS1", 118."""
    return [{"type": name, "scans": count} for name, count in data.get_scan_type_summary()]


def describe_run(reader: AgilentReader) -> dict:
    summary = {
        "name": reader.file_path.name,
        "path": str(reader.file_path),
        "n_scans": int(reader.N_scans),
        "rt_unit": "min",
        "rt_range": [float(reader.RT_list.min()), float(reader.RT_list.max())],
        "scan_types": scan_types(reader),
    }
    for ms_level in sorted({int(level) for level in reader.MS_level_list}):
        scan_indices = reader.scan_indices_for_ms_level(ms_level)
        summary[f"ms{ms_level}_profile_mz_range"] = list(reader.profile_mz_range(scan_indices))
    if (reader.MS_level_list == 2).any():
        summary["n_ms2_precursors"] = len(operations.precursors([reader]))
    return summary


def precursor_dict(precursor: operations.Precursor) -> dict:
    return {
        "precursor_mz": precursor.precursor_mz,
        "collision_energy": precursor.collision_energy,
        "isolation_width": precursor.isolation_width,
        "n_scans": precursor.n_scans,
    }


def ms2_precursors(
    reader: AgilentReader,
    *,
    merge_ppm: float = 0.0,
    mz_range: tuple[float, float] | None = None,
    max_items: int = 200,
    polarity: str | None = None,
) -> dict:
    """The precursors and collision energies of the MS2 scans, as the window lists them, from the lowest precursor;
    with a polarity, of its scans only."""
    listed = operations.precursors([reader], merge_ppm, polarity_sign(polarity))
    if mz_range is not None:
        listed = [precursor for precursor in listed if mz_range[0] <= precursor.precursor_mz <= mz_range[1]]
    return {
        "merge_ppm": merge_ppm,
        "n_precursors": len(listed),
        "truncated": len(listed) > max_items,
        "precursors": [precursor_dict(precursor) for precursor in listed[:max_items]],
    }


def ms2_selection(
    datas, precursor_mz: float, collision_energy: float | None = None, merge_ppm: float = 0.0, polarity: int | None = None,
) -> operations.MS2Selection:
    """The MS2 scans of the listed precursor nearest to precursor_mz whose isolation window holds it, at
    collision_energy; which may be left out when that precursor has scans of one CE only. Listed of the polarity
    (+1, -1) if one is given."""
    return ms2_precursor(datas, precursor_mz, collision_energy, merge_ppm, polarity)[0]


def ms2_precursor(
    datas, precursor_mz: float, collision_energy: float | None = None, merge_ppm: float = 0.0, polarity: int | None = None,
) -> tuple[operations.MS2Selection, operations.Precursor]:
    """ms2_selection, and the listed precursor it selects: its isolation width is that of a target."""
    listed = operations.precursors(datas, merge_ppm, polarity)
    nearest = operations.precursor_isolating(listed, precursor_mz)
    if nearest is None:
        raise ValueError(f"No MS2 scans isolate m/z {precursor_mz:g}; rapid2_list_precursors lists the precursors")
    # the list is by precursor and then CE
    of_nearest = [precursor for precursor in listed if precursor.precursor_mz == nearest.precursor_mz]
    energies = ", ".join(f"{precursor.collision_energy:g}" for precursor in of_nearest)
    if collision_energy is None:
        if len(of_nearest) > 1:
            raise ValueError(f"Give collision_energy: the MS2 scans of precursor {nearest.precursor_mz:.4f} have CE {energies}")
        chosen = nearest
    else:
        chosen = next(
            (
                precursor
                for precursor in of_nearest
                if round(precursor.collision_energy, operations.CE_DECIMALS) == round(collision_energy, operations.CE_DECIMALS)
            ),
            None,
        )
        if chosen is None:
            raise ValueError(
                f"No MS2 scans of precursor {nearest.precursor_mz:.4f} at CE {collision_energy:g}; they have CE {energies}"
            )
    return operations.MS2Selection(chosen.precursor_mz, chosen.collision_energy, merge_ppm), chosen


def selection_dict(selection: operations.MS2Selection) -> dict:
    return {
        "precursor_mz": selection.precursor_mz,
        "collision_energy": selection.collision_energy,
        "merge_ppm": selection.merge_ppm,
    }


def _ms2(reader, ms_level, precursor_mz, collision_energy, merge_ppm, polarity=None) -> operations.MS2Selection | None:
    """The MS2 selection of a report's arguments; None without precursor_mz."""
    if precursor_mz is None:
        if collision_energy is not None:
            raise ValueError("collision_energy selects MS2 scans of a precursor: give precursor_mz too")
        return None
    if ms_level != 2:
        raise ValueError("precursor_mz selects MS2 scans: the MS level must be 2")
    return ms2_selection([reader], precursor_mz, collision_energy, merge_ppm, polarity)


def _with_ms2(result: dict, ms2: operations.MS2Selection | None) -> dict:
    if ms2 is not None:
        result["ms2"] = selection_dict(ms2)
    return result


def calculate_ion(formula: str, adduct: str) -> dict:
    return {"formula": formula, "adduct": adduct, "neutral_mass": formula_mass(formula), "mz": ion_mz(formula, adduct)}


def xic(
    reader: AgilentReader,
    mz: float | None = None,
    tolerance: float = 10.0,
    unit: str = "ppm",
    *,
    formula: str | None = None,
    adduct: str | None = None,
    polarity: str | None = None,
    ms_level: int | None = 1,
    rt_range: tuple[float, float] | None = None,
    max_points: int | None = 300,
    precursor_mz: float | None = None,
    collision_energy: float | None = None,
    merge_ppm: float = 0.0,
) -> dict:
    """Summed profile intensity in the m/z window around mz, or the ion of formula + adduct, per scan; with
    precursor_mz, mz is a product ion. An adduct chooses the scans of its polarity, as the plots do.

    As in the plots and the .rpd.csv export, a scan whose profile data points miss the m/z window is left out, not
    taken for 0: that m/z was not measured there (n_missing, and the plots' warning).
    """
    if formula is not None:
        if mz is not None:
            raise ValueError("Give mz, or formula and adduct, not both")
        if adduct is None:
            raise ValueError("A formula needs an adduct to make its ion")
        mz = ion_mz(formula, adduct)
    elif mz is None:
        raise ValueError("Give mz, or formula and adduct")
    sign = polarity_sign(polarity, adduct)
    mz_window = tolerance_window(mz, tolerance, ToleranceUnit(unit))
    ms2 = _ms2(reader, ms_level, precursor_mz, collision_energy, merge_ppm, sign)
    scan_indices = _scans(reader, ms_level, rt_range, ms2=ms2, polarity=sign)
    misses = operations.mz_window_misses(reader, mz_window, scan_indices)
    measured = scan_indices[~misses.missing]
    rt, intensity = [], []
    if len(measured):
        rt, intensity = operations.extract_trace(reader, TraceType.XIC, mz_window=mz_window, scan_indices=measured)
    chemistry = {name: value for name, value in (("formula", formula), ("adduct", adduct)) if value is not None}
    result = {
        "trace": TraceType.XIC.value,
        "mz": float(mz),
        **chemistry,
        "mz_window": list(mz_window),
        "ms_level": ms_level,
        **_polarity_dict(sign),
        "n_missing": int(np.count_nonzero(misses.missing)),
        **_trace(rt, intensity, max_points),
    }
    warning = misses.warning()
    if warning:
        result["warning"] = warning
    return _with_ms2(result, ms2)


def trace(
    reader: AgilentReader,
    trace_type: str = "TIC",
    *,
    polarity: str | None = None,
    ms_level: int | None = None,
    rt_range: tuple[float, float] | None = None,
    max_points: int | None = 300,
    precursor_mz: float | None = None,
    collision_energy: float | None = None,
    merge_ppm: float = 0.0,
) -> dict:
    """TIC or BPC from the scan records."""
    trace_type = TraceType(trace_type.upper())
    if trace_type is TraceType.XIC:
        raise ValueError("Use xic for XIC traces: it needs an m/z and a tolerance")
    sign = polarity_sign(polarity)
    ms2 = _ms2(reader, ms_level, precursor_mz, collision_energy, merge_ppm, sign)
    scan_indices = _scans(reader, ms_level, rt_range, ms2=ms2, polarity=sign)
    rt, intensity = operations.extract_trace(reader, trace_type, scan_indices=scan_indices)
    return _with_ms2(
        {"trace": trace_type.value, "ms_level": ms_level, **_polarity_dict(sign), **_trace(rt, intensity, max_points)}, ms2,
    )


def spectrum(
    reader: AgilentReader,
    rt: float,
    rt_tolerance: float = 0.0,
    *,
    ms_level: int | None = 1,
    mode: str = "mean",
    mz_range: tuple[float, float] | None = None,
    top_n: int = 50,
    polarity: str | None = None,
    precursor_mz: float | None = None,
    collision_energy: float | None = None,
    merge_ppm: float = 0.0,
) -> dict:
    """The most intense profile peaks of the spectrum over [rt - rt_tolerance, rt + rt_tolerance)."""
    rt_window = (rt - rt_tolerance, rt + rt_tolerance)
    sign = polarity_sign(polarity)
    ms2 = _ms2(reader, ms_level, precursor_mz, collision_energy, merge_ppm, sign)
    scan_indices = _scans(reader, ms_level, ms2=ms2, polarity=sign)
    mz, intensity, representative_scan = operations.averaged_spectrum(reader, rt_window, mode=mode, scan_indices=scan_indices)
    peaks = operations.profile_peaks(mz, intensity, mz_range=mz_range, top_n=top_n)
    return _with_ms2({
        "rt_window": list(rt_window),
        "representative_scan": int(representative_scan),
        "representative_rt": float(reader.RT_list[representative_scan]),
        "n_scans": operations.scans_in_window(reader, rt_window, scan_indices),
        "mode": mode,
        "ms_level": ms_level,
        **_polarity_dict(sign),
        "peaks": [{"mz": float(mz[i]), "intensity": float(intensity[i])} for i in peaks],
    }, ms2)


def map_spots(
    reader: AgilentReader,
    *,
    top_n: int = 20,
    rt_range: tuple[float, float] | None = None,
    mz_range: tuple[float, float] | None = None,
) -> dict:
    """The most intense spots of the run's m/z x RT map of MS1 scans, as the m/z-RT map window shows it."""
    mz_rt_map = maps.built_map(reader)
    spots = maps.strongest_spots(mz_rt_map, top_n=top_n, rt_range=rt_range, mz_range=mz_range)
    return {"bin_ppm": mz_rt_map.mid.bin_ppm, "spots": [_spot_dict(spot) for spot in spots]}


def compare_runs(
    runs: RunCache,
    reference: str | Path,
    query: str | Path,
    *,
    top_n: int = 20,
    min_fold_change: float = 2.0,
    rt_range: tuple[float, float] | None = None,
    mz_range: tuple[float, float] | None = None,
) -> dict:
    """Where the query's m/z x RT map differs most from the reference's, aligned in RT: the window's B - A map. D is
    about the log fold change where both are strong, so min_fold_change is a least |D| of ln(min_fold_change)."""
    map_a = maps.built_map(runs.get(reference))
    map_b = maps.built_map(runs.get(query))
    comparison = maps.compare_maps(
        map_a, map_b, top_n=top_n, min_difference=float(np.log(min_fold_change)), rt_range=rt_range, mz_range=mz_range,
    )
    return {
        "reference": map_a.ms_data_model.file_path.name,
        "query": map_b.ms_data_model.file_path.name,
        "bin_ppm": map_a.mid.bin_ppm,
        "scale": comparison.scale,
        "normalization": comparison.normalization,
        "max_rt_shift_minutes": comparison.max_rt_shift,
        "higher_in_query": [_difference_dict(difference) for difference in comparison.higher_in_query],
        "higher_in_reference": [_difference_dict(difference) for difference in comparison.higher_in_reference],
    }


def _spot_dict(spot: maps.Spot) -> dict:
    return {"rt": spot.rt, "mz": spot.mz, "mz_bin": list(spot.mz_bin), "intensity": spot.intensity, "scan": spot.scan}


def _difference_dict(difference: maps.Difference) -> dict:
    return {
        "rt": difference.rt,
        "query_rt": difference.query_rt,
        "mz": difference.spot.mz,
        "mz_bin": list(difference.spot.mz_bin),
        "difference": difference.difference,
        "reference_intensity": difference.reference_intensity,
        "query_intensity": difference.query_intensity,
    }


def target_dict(target: Target) -> dict:
    """A target as the tools give it and take it (target_from_spec): its fields but its id, its m/z only for an
    XIC, and the m/z of its formula + adduct."""
    shown = {field.name: getattr(target, field.name) for field in fields(Target) if field.name != "id"}
    shown |= {
        "trace_type": target.trace_type.value,
        "ms_level": target.ms_level.value,
        "mz_tolerance_unit": target.mz_tolerance_unit.value,
        "calculated_mz": target.calculated_mz(),
    }
    if target.trace_type is not TraceType.XIC:
        for name in ("mz", "mz_tolerance", "mz_tolerance_unit"):
            del shown[name]
    return shown


def target_from_spec(spec: dict, index: int, datas, merge_ppm: float = 0.0) -> Target:
    """A target from the tools' description of one, numbered index; the precursor of an MS2 target is the one listed
    for datas (the runs it will be traced in) nearest to precursor_mz, at collision_energy, as a target the window
    makes has it.

    The m/z of an XIC, left out, is that of formula + adduct; for MS2 it is a product ion, so it is to be given, and
    the ion of formula + adduct chooses the precursor when precursor_mz is left out, as in the window. A TIC or BPC
    has the m/z of the plots' initial window, as one read from a .rpd.csv file.
    """
    spec = {name: value for name, value in spec.items() if value is not None}
    trace_type = TraceType(spec.get("trace_type", TraceType.XIC.value))
    ms_level = MSLevel(spec.get("ms_level", MSLevel.MS1.value))
    formula, adduct = spec.get("formula", ""), spec["adduct"]
    ion = ion_mz(formula, adduct) if formula else None
    if trace_type is not TraceType.XIC:
        mz = spec.get("mz", rpd_csv.TIC_MZ)
        mz_tolerance = spec.get("mz_tolerance", rpd_csv.TIC_MZ_TOLERANCE)
        unit = spec.get("mz_tolerance_unit", ToleranceUnit.MZ.value)
    else:
        mz = spec.get("mz", None if ms_level is MSLevel.MS2 else ion)
        if mz is None:
            raise ValueError(
                "An MS2 XIC target needs mz, a product ion" if ms_level is MSLevel.MS2 else "An XIC target needs mz, or a formula"
            )
        mz_tolerance = spec.get("mz_tolerance", 10)
        unit = spec.get("mz_tolerance_unit", ToleranceUnit.PPM.value)
    ms2 = {}
    if ms_level is not MSLevel.MS2:
        if "precursor_mz" in spec or "collision_energy" in spec:
            raise ValueError("precursor_mz and collision_energy are for MS2 targets")
    elif spec.get("precursor_mz", ion) is not None:
        _selection, precursor = ms2_precursor(
            datas, spec.get("precursor_mz", ion), spec.get("collision_energy"), merge_ppm, polarity_sign(None, adduct),
        )
        ms2 = {
            "precursor_mz": precursor.precursor_mz,
            "collision_energy": precursor.collision_energy,
            "isolation_width": precursor.isolation_width,
        }
    elif "collision_energy" in spec:
        raise ValueError("collision_energy selects the MS2 scans of a precursor: give precursor_mz, or a formula")
    return Target(
        index=index,
        name=spec.get("name", formula or DEFAULT_TARGET_NAME),
        formula=formula,
        adduct=adduct,
        trace_type=trace_type,
        ms_level=ms_level,
        rt=spec["rt"],
        rt_tolerance=spec["rt_tolerance"],
        mz=mz,
        mz_tolerance=mz_tolerance,
        mz_tolerance_unit=ToleranceUnit(unit),
        note=spec.get("note", ""),
        **ms2,
    )


def targets_of_file(path: str | Path, first_index: int = 1) -> tuple[list[Target], list[str]]:
    """The targets of a .rpd.csv file, or of RAPID2's, numbered from first_index; rows of one '#' that hold
    different targets are kept as targets of their own, and said so."""
    groups = rpd_csv.read_targets_file(Path(path).expanduser())
    notes = [
        f"#{group.number}: its rows hold {len(group.variants)} different targets, kept as targets of their own"
        for group in groups
        if len(group.variants) > 1
    ]
    return rpd_csv.resolve_groups(groups, first_index=first_index), notes


def new_rpd_csv_path(path: str | Path) -> Path:
    """Where to write a .rpd.csv file asked for: the name made to end in .rpd.csv, and nothing there yet."""
    path = rpd_csv.rpd_csv_path(Path(path).expanduser().resolve())
    if path.exists():
        raise FileExistsError(f"Choose a new file: {path} exists")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"No such directory: {path.parent}")
    return path


def quantify_targets(
    paths: Sequence[str | Path],
    *,
    targets: Sequence[Target] = (),
    specs: Sequence[dict] = (),
    merge_ppm: float = 0.0,
    run_peaks: RunPeaksStore | None = None,
    rpd_csv_file: str | Path | None = None,
    max_rows: int = 200,
) -> dict:
    """The values of the targets in the runs -- area, height, peak top and the rest, as the .rpd.csv export has
    them --, a row per target and run; written as a .rpd.csv file too if rpd_csv_file is given. targets, then
    specs (target_from_spec), numbered on from them; in the windows of run_peaks, as rpd_csv.measure takes them.

    The runs are opened for the call, all at once, not from a RunCache: each is read for every target.
    """
    if not paths:
        raise ValueError("No runs to quantify the targets in")
    output = None if rpd_csv_file is None else new_rpd_csv_path(rpd_csv_file)
    readers = []
    try:
        for path in paths:
            reader = AgilentReader()
            reader.load(run_path(path))
            readers.append(reader)
        runs = [rpd_csv.Run(reader, reader.file_path) for reader in readers]
        first = max((target.index for target in targets), default=0) + 1
        targets = [*targets, *(target_from_spec(spec, first + i, readers, merge_ppm) for i, spec in enumerate(specs))]
        if not targets:
            raise ValueError("No targets to quantify")
        measured = rpd_csv.measure(targets, runs, merge_ppm=merge_ppm, run_peaks=run_peaks)
        if output is not None:
            rows, _notes = rpd_csv.export_rows(targets, runs, merge_ppm=merge_ppm, measured=measured)
            rpd_csv.write_rpd_csv(output, rows)
    finally:
        for reader in readers:
            reader.close()
    return {
        "n_targets": len(targets),
        "runs": [run.path.name for run in runs],
        "merge_ppm": merge_ppm,
        "rpd_csv": None if output is None else str(output),
        "n_rows": len(measured),
        "truncated": len(measured) > max_rows,
        "rows": [_measured_dict(row) for row in measured[:max_rows]],
    }


def _measured_dict(row: rpd_csv.Measured) -> dict:
    return {
        "target": row.target.index,
        "name": row.target.name,
        "run": row.run.path.name,
        **asdict(row.quantity),
        "note": row.note,
    }


def scale_candidates(runs: RunCache, path: str | Path, **options) -> dict:
    """Candidate asinh scales of a run; the caller chooses one. options: `alignment.OPTIONS`."""
    found = alignment_scale_candidates(runs.get(path), with_options(AlignmentConfig(), **options))
    return {"run": found["run"]["name"], "n_scans": found["n_scans"], "scale_candidates": found["scale_candidates"]}


def align_runs(reference: str | Path, query: str | Path, output_dir: str | Path, **options) -> dict:
    """Map the query's RTs onto the reference's and save the result in output_dir. options: `alignment.OPTIONS`."""
    output_dir = Path(output_dir).expanduser().resolve()
    result, figures = align_and_save(run_path(reference), run_path(query), output_dir, with_options(AlignmentConfig(), **options))
    shift = result.mapping.transform(result.query_rt_seconds) - result.query_rt_seconds
    shift = shift[np.isfinite(shift)]
    return {
        "output_dir": str(output_dir),
        "mapping": str(output_dir / "mapping.json"),
        "figures": [str(figure) for figure in figures],
        "n_anchors": len(result.anchors),
        "rt_shift_seconds": {
            "min": float(shift.min()), "median": float(np.median(shift)), "max": float(shift.max()),
        } if len(shift) else None,
    }


def _trace(rt: np.ndarray, intensity: np.ndarray, max_points: int | None) -> dict:
    """The apex -- with no RT for a trace of 0, as a peak top has none --, and the trace reduced to max_points
    (None: every scan) by keeping each group's maximum."""
    rt, intensity = np.asarray(rt, dtype=float), np.asarray(intensity, dtype=float)
    apex = int(np.argmax(intensity)) if len(rt) else None
    summary = {
        "n_scans": len(rt),
        "apex_rt": float(rt[apex]) if apex is not None and intensity[apex] > 0 else None,
        "apex_intensity": None if apex is None else float(intensity[apex]),
        "downsampled": max_points is not None and len(rt) > max_points,
    }
    if summary["downsampled"]:
        groups = np.array_split(np.arange(len(rt)), max_points)
        keep = np.array([group[np.argmax(intensity[group])] for group in groups])
        rt, intensity = rt[keep], intensity[keep]
    return {**summary, "rt": rt.tolist(), "intensity": intensity.tolist()}
