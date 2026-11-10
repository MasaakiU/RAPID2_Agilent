# -*- coding: utf-8 -*-

"""The tool surface: what an assistant may ask of a running RAPID2.

Two kinds, kept apart because they run in different places.

*Viewer tools* read or change what the window shows the user. They run on the
Qt main thread through `run_on_main_thread`, through a context whose whole job
is to be replaceable by a fake in tests and to keep the reachable surface
small enough to audit -- an MCP tool is remote input. They take an `instance`
target, and every reply names the window that answered.

*Analysis tools* read runs (.d folders, .rpd files) by path through `services` and never touch
the window. They run on the request thread, one at a time: a minute of
alignment on the main thread would freeze the window for that minute. They
take no target, because any window gives the same answer.

Nothing here imports Qt at module level, so the registry can be built and its
schemas checked without a QApplication.
"""

from __future__ import annotations

import base64
import threading

from pathlib import Path

from .. import services
from ..alignment.config import OPTIONS as ALIGNMENT_OPTIONS
from ..services import operations, reports
from ..models.target import MSLevel, TraceType
from . import registry
from .protocol import IMAGES_KEY, Tool, ToolError, ToolRegistry
from .relay import launch as relay_launch
from .trail import traced


# Bumped by hand when the tool set changes, so that a reader of the registry
# can notice two running builds disagree without listing tool names.
TOOLSET_VERSION = 5

# What the services raise for a wrong path or option. The message is the
# answer, so it travels as a failed tool result rather than as a crash.
EXPECTED_FAILURES = (ValueError, FileNotFoundError, FileExistsError, NotImplementedError)

MAX_CAPTURE_WIDTH = 1600

# rapid2_get_status lists this many of the window's MS2 precursors; rapid2_list_precursors lists more
MAX_STATUS_PRECURSORS = 50


class ViewerContext:
    """The part of the live window the viewer tools may touch. Main thread only."""

    def __init__(self, presenter):
        self._presenter = presenter

    @property
    def _stack(self):
        return self._presenter.data_stack_presenter

    @property
    def _merge_ppm(self):
        return self._presenter.model.settings_model.ms2_merge_ppm

    @property
    def _targets(self):
        return self._presenter.model.target_list_model

    @property
    def _target_panel(self):
        return self._presenter.target_panel_presenter

    def status(self):
        stack = self._stack
        precursors = operations.precursors(stack.models, self._merge_ppm)
        files = [
            {
                "index": index,
                "name": model.file_path.name,
                "path": str(model.file_path),
                "spectra": model.spectra,
                "scan_types": reports.scan_types(model),
            }
            for index, model in enumerate(stack.models)
        ]
        return {
            "files": files,
            "rt_window": _listed(stack.current_rt_window),
            "mz_window": _listed(stack.current_mz_window),
            "trace_type": stack.trace_type.value,
            "ms_level": stack.ms_level,
            # of MS level 2 only
            "ms2": reports.selection_dict(stack.ms2_selection) if stack.ms_level == 2 and stack.ms2_selection else None,
            "ms2_merge_ppm": self._merge_ppm,
            "n_ms2_precursors": len(precursors),
            "ms2_precursors": [reports.precursor_dict(precursor) for precursor in precursors[:MAX_STATUS_PRECURSORS]],
            "targets": len(self._targets.targets),
            "selected_target": self._selected_target_number(),
        }

    def _selected_target_number(self):
        current_id = self._target_panel.current_id
        return None if current_id is None else self._targets.get(current_id).index

    def list_targets(self):
        return {
            "targets": [reports.target_dict(target) for target in self._targets.targets],
            "selected_target": self._selected_target_number(),
        }

    def add_target(self, spec):
        """Add a target, numbered on from the others, and show it, as Add does with the draft."""
        target = reports.target_from_spec(spec, self._targets.next_index(), self._stack.models, self._merge_ppm)
        self._targets.add(target)
        self._show_target(target)
        return {"added": reports.target_dict(target), "targets": len(self._targets.targets)}

    def select_target(self, number):
        target = next((target for target in self._targets.targets if target.index == number), None)
        if target is None:
            numbers = ", ".join(str(target.index) for target in self._targets.targets) or "none"
            raise ToolError(f"No target #{number}. The window has: {numbers}. rapid2_list_targets lists them.")
        self._show_target(target)
        return {"selected": reports.target_dict(target)}

    def _show_target(self, target):
        """Select a target, so that the plots show it, and pan to it."""
        self._target_panel.select_target(target.id)
        self._stack.ensure_visible(target.rt, target.mz if target.trace_type is TraceType.XIC else None)

    def import_targets(self, path, mode):
        """Read a .rpd.csv file's targets into the list, as Load Targets does, in place of the targets or after them."""
        existing = self._targets.targets
        if existing and mode is None:
            raise ToolError(f"The window has {len(existing)} targets: give mode, replace or append.")
        replace = mode == "replace"
        targets, notes = reports.targets_of_file(path, first_index=1 if replace else self._targets.next_index())
        self._targets.import_targets(targets, replace=replace)
        self._target_panel.select_target(None)
        return {"imported": len(targets), "replaced": replace and bool(existing), "targets": len(self._targets.targets), "notes": notes}

    def target_snapshot(self):
        """What an export of the window's targets reads: the targets, the runs open (.d folders, .rpd files) and the
        spectra each is read of, the MS2 merge, and the files' own RT windows of the targets, as File > Export
        Targets takes them."""
        models = self._stack.models
        run_peaks = self._presenter.run_peaks_to_export()
        readable = [model for model in models if _is_run(model.file_path)]
        return {
            "targets": self._targets.targets,
            "paths": [str(model.file_path) for model in readable],
            "spectra": [model.spectra for model in readable],
            "left_out": [model.file_path.name for model in models if not _is_run(model.file_path)],
            "merge_ppm": self._merge_ppm,
            # a copy, read while the window goes on
            "run_peaks": None if run_peaks is None else run_peaks.copy(),
        }

    def open_files(self, run_paths, spectra=None):
        """Open the runs not open yet: a .d folder of its profile or its centroid spectra (a run of centroid spectra
        only, of them), a .rpd file of what it holds. One that has not the centroid spectra asked for opens none;
        one that cannot be read stops there."""
        open_paths = {str(model.file_path) for model in self._stack.models}
        wanted = []
        for run_path in run_paths:
            try:
                path = services.run_path(run_path)
                if not _is_run(path):
                    raise ToolError(f"Not a .d folder or a .rpd file: {path}")
                if spectra == "centroid" and "centroid" not in services.stored_spectra(path):
                    raise ToolError(f"{path.name} has no centroid spectra: open it of its profile")
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
            if str(path) not in open_paths and str(path) not in wanted:
                wanted.append(str(path))
        opened = []
        for path in wanted:
            try:
                if services.is_rpd(path):
                    self._presenter.open_rpd_file(path)
                else:
                    self._presenter.open_d_folder(path, centroid=spectra == "centroid")
            except (OSError, ValueError) as exc:
                raise ToolError(f"{Path(path).name} could not be opened ({len(opened)} opened before it): {exc}") from exc
            opened.append(path)
        return {"opened": opened, **self.status()}

    def set_view(
        self, *, rt_window=None, mz_window=None, trace_type=None, ms_level=False, precursor_mz=None, collision_energy=None,
    ):
        """`ms_level` False leaves it as it is; None selects all scans. precursor_mz selects the MS2 scans of the
        window's precursor that isolates it, at collision_energy, and makes the MS level 2 if it is left out.

        Shown as a click on the m/z x RT map is: a target selected in the window is deselected, not
        changed, and the panel's draft shows the view with the plots.
        """

        stack = self._stack
        if not stack.models:
            raise ToolError("No file is open. Call rapid2_open_files first.")
        if ms_level not in (False, None, 1, 2):
            raise ToolError("The window shows MS level 1, 2, or all scans (null).")
        selection = None
        if precursor_mz is not None:
            if ms_level is False:
                ms_level = 2
            if ms_level != 2:
                raise ToolError("precursor_mz selects MS2 scans: ms_level must be 2 or left out.")
            try:
                selection = reports.ms2_selection(stack.models, precursor_mz, collision_energy, self._merge_ppm)
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
        elif collision_energy is not None:
            raise ToolError("collision_energy selects MS2 scans of a precursor: give precursor_mz too.")
        if any(part is not None for part in (rt_window, mz_window, trace_type, selection)) or ms_level is not False:
            self._presenter.target_panel_presenter.show_view(
                rt_window=rt_window,
                mz_window=mz_window,
                trace_type=None if trace_type is None else TraceType(trace_type),
                ms_level=None if ms_level is False else next(level for level in MSLevel if level.number == ms_level),
                ms2=selection,
            )
        if rt_window is not None or mz_window is not None:
            rt_center = sum(stack.current_rt_window) / 2
            mz_center = sum(mz_window) / 2 if mz_window is not None else None
            stack.ensure_visible(rt_center, mz_center)
        return self.status()

    def capture(self, part, max_width=MAX_CAPTURE_WIDTH):
        from PyQt6.QtCore import QBuffer, QIODevice, Qt

        view = self._presenter.view
        widget = {"window": view, "plots": view.data_stack_view}[part]
        pixmap = widget.grab()
        if pixmap.width() > max_width:
            pixmap = pixmap.scaledToWidth(max_width, Qt.TransformationMode.SmoothTransformation)
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, "PNG")
        data = base64.b64encode(bytes(buffer.data())).decode("ascii")
        return {
            "part": part,
            "width": pixmap.width(),
            "height": pixmap.height(),
            IMAGES_KEY: [{"data": data, "mimeType": "image/png"}],
        }


class AnalysisSession:
    """The runs the analysis tools have open, and the lock they take turns on.

    Readers and their caches are not thread-safe, and the HTTP server answers
    each request on a thread of its own.
    """

    def __init__(self):
        self.runs = services.RunCache()
        self.lock = threading.Lock()

    def close(self):
        with self.lock:
            self.runs.close()


def _listed(window):
    return None if window is None else [float(window[0]), float(window[1])]


def _is_run(path):
    """Whether the services read a file the window has open: a .d folder or a .rpd file."""
    return services.is_rpd(path) or Path(path).suffix.lower() == ".d"


def _range(arguments, low, high, default_low):
    lo, hi = arguments.get(low), arguments.get(high)
    if lo is None and hi is None:
        return None
    if lo is not None and hi is not None and lo > hi:
        raise ToolError(f"{low} must not be greater than {high}.")
    return (default_low if lo is None else lo, float("inf") if hi is None else hi)


def _window(arguments, low, high):
    """A window given by both of its edges, by neither, or an error."""

    lo, hi = arguments.get(low), arguments.get(high)
    if lo is None and hi is None:
        return None
    if lo is None or hi is None:
        raise ToolError(f"Give both {low} and {high}, or neither.")
    if not lo < hi:
        raise ToolError(f"{low} must be less than {high}.")
    return (lo, hi)


# -- schemas -------------------------------------------------------------
_NO_ARGUMENTS = {"type": "object", "properties": {}, "additionalProperties": False}

PATH_PROPERTY = {"type": "string", "description": "Path of a run: an Agilent .d folder, or a RAPID2 .rpd file."}
D_PATH_PROPERTY = {"type": "string", "description": "Path of an Agilent .d folder."}
SPECTRA_PROPERTY = {
    "type": "string",
    "enum": list(services.SPECTRA),
    "description": (
        "The spectra to read: the profile, or the centroid spectra stored at acquisition, about 1/17 of the "
        "data and much quicker to read from a network drive (rapid2_describe_run gives the sizes). Left out, "
        "the profile, or the centroid spectra of a run that holds only them; a .rpd file holds one of the two. "
        "Centroid intensities are peak heights: an XIC, height or area of them is about 0.15 to 0.4 of the "
        "profile's, so compare only values read of the same spectra. Weak ions may be missing from the centroids."
    ),
}
MS_LEVEL_PROPERTY = {
    "type": ["integer", "null"],
    "minimum": 1,
    "description": "MS level of the scans to use; null for all scans.",
}
MAX_POINTS_PROPERTY = {
    "type": ["integer", "null"],
    "minimum": 2,
    "default": 300,
    "description": "Longer traces are reduced to this many points, keeping the maxima; null keeps every scan.",
}
RT_MIN_PROPERTY = {"type": "number", "description": "Lower RT limit, in minutes."}
PRECURSOR_MZ_PROPERTY = {
    "type": "number",
    "exclusiveMinimum": 0,
    "description": (
        "For MS2: the m/z whose MS2 scans to use, matched to the nearest precursor whose isolation "
        "window holds it (see rapid2_list_precursors). Makes the MS level 2."
    ),
}
COLLISION_ENERGY_PROPERTY = {
    "type": "number",
    "description": "With precursor_mz: the collision energy of the MS2 scans; needed when the precursor has several.",
}
MERGE_PPM_PROPERTY = {
    "type": "number",
    "minimum": 0,
    "default": 0,
    "description": (
        "Precursors within this many ppm of each other count as one, for data-dependent scans that "
        "record slightly different values of one ion; 0 keeps each recorded value."
    ),
}
MS2_PROPERTIES = {
    "precursor_mz": PRECURSOR_MZ_PROPERTY,
    "collision_energy": COLLISION_ENERGY_PROPERTY,
    "merge_ppm": MERGE_PPM_PROPERTY,
}
RT_MAX_PROPERTY = {"type": "number", "description": "Upper RT limit, in minutes."}
FORMULA_PROPERTY = {"type": "string", "description": "Neutral formula, e.g. \"C6H12O6\"."}
ADDUCT_PROPERTY = {"type": "string", "description": "Adduct, e.g. \"[M-H]-\"; its charge chooses the polarity."}
POLARITY_PROPERTY = {
    "type": "string",
    "enum": ["positive", "negative"],
    "description": "The polarity of the scans to use. Required for a run with both polarities.",
}
# The alignment options both alignment tools take. Their names are keys of
# `alignment.config.OPTIONS` and are passed through unchanged.
SCALE_OPTION_PROPERTIES = {
    "ppm": {"type": "number", "exclusiveMinimum": 0},
    "polarity": POLARITY_PROPERTY,
    "max_scans": {"type": "integer", "minimum": 2},
}

TARGET_PROPERTIES = {
    "name": {"type": "string", "description": "Left out, the formula, or \"New target\"."},
    "formula": FORMULA_PROPERTY,
    "adduct": {"type": "string", "description": "Adduct, e.g. \"[M+H]+\"; its charge chooses the polarity of the scans."},
    "trace_type": {"type": "string", "enum": [kind.value for kind in TraceType], "default": "XIC"},
    "ms_level": {"type": "string", "enum": [level.value for level in MSLevel], "default": "MS1"},
    "rt": {"type": "number", "minimum": 0, "description": "RT of the window's centre, in minutes."},
    "rt_tolerance": {"type": "number", "minimum": 0, "description": "Half the RT window, in minutes."},
    "mz": {
        "type": "number",
        "exclusiveMinimum": 0,
        "description": "The m/z of an XIC; left out, that of formula + adduct. For MS2, a product ion, to be given.",
    },
    "mz_tolerance": {"type": "number", "minimum": 0, "default": 10, "description": "Half the m/z window; ppm are whole."},
    "mz_tolerance_unit": {"type": "string", "enum": ["ppm", "m/z"], "default": "ppm"},
    "precursor_mz": {
        "type": "number",
        "exclusiveMinimum": 0,
        "description": (
            "MS2: the precursor whose scans to use, the listed one nearest to this (rapid2_list_precursors); left "
            "out, the one whose isolation window holds the ion of formula + adduct."
        ),
    },
    "collision_energy": {"type": "number", "description": "MS2: needed when the precursor has scans of several CEs."},
    "note": {"type": "string"},
}
TARGET_REQUIRED = ("adduct", "rt", "rt_tolerance")
TOP_N_SPOTS_PROPERTY = {"type": "integer", "minimum": 1, "maximum": 200, "default": 20}

INSTANCE_PROPERTY = {
    "type": "string",
    "description": (
        "Which RAPID2 window to act in: its letter, as shown in the window "
        "title (for example \"A\"), or its full instance_id. Leave it out "
        "while only one window is open. Call rapid2_list_windows to see what "
        "is open. Every reply names the window that answered, so pass its "
        "`instance_id` back to stay in the same one."
    ),
}


def _takes_null(property_schema):
    declared = (property_schema or {}).get("type")
    return declared == "null" or (isinstance(declared, list) and "null" in declared)


def _without_nulls(schema, arguments):
    """The arguments less a null given for a property that does not take one, which is then absent.

    The validator lets null through for any argument that is not required, as "not given"; a handler reading
    `arguments.get(name, default)` would take it for the value, and 10 ppm or the top 50 peaks would become None.
    """

    properties = schema.get("properties") or {}
    return {name: value for name, value in arguments.items() if value is not None or _takes_null(properties.get(name))}


def _object(properties, required=()):
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _accepting_a_target(schema):
    """Every viewer tool takes a window, whatever else it takes.

    Added here rather than declared per tool: it is about the call, not about
    the operation, and a tool that had to remember to accept it could forget.
    """

    schema = dict(schema)
    properties = dict(schema.get("properties") or {})
    properties["instance"] = INSTANCE_PROPERTY
    schema["properties"] = properties
    return schema


# -- targeting -----------------------------------------------------------
def _window_row(document):
    """What the registry knows, and nothing it does not.

    No file names: what a window holds changes with every file opened, and a
    file that had to keep up with that would be a cache of Qt state rather than
    a register of addresses. Ask the window itself.
    """

    return {
        "instance": document.get("label"),
        "instance_id": document.get("instance_id"),
        "started_at": document.get("started_at"),
        "app_version": document.get("app_version"),
        "toolset_version": document.get("toolset_version"),
    }


def _open_windows(identity):
    """The roster, with ourselves in it even if our own file went missing.

    Swept first: a window that crashed leaves its address behind, and listing
    it would make a caller ask the user which of two windows to use when only
    one of them exists.
    """

    try:
        registry.sweep_stale()
        found = registry.read_instances()
    except Exception:  # noqa: BLE001 - a roster must not take the server down
        found = []
    rows = [_window_row(document) for document in found]
    if not any(row["instance_id"] == identity["instance_id"] for row in rows):
        rows.append(_window_row(identity))
        rows.sort(key=lambda row: row["instance"] or "")
    return rows


def _ambiguous(rows):
    return {
        "status": "ambiguous_instance",
        "message": (
            f"{len(rows)} RAPID2 windows are open. Pass `instance` to say which "
            "one to act in. Use rapid2_get_status with a target to see what a "
            "window has open."
        ),
        "windows": rows,
    }


def _answered_by(payload, identity):
    """Say which window this reply came from: the letter a person can say, and
    the exact id a caller passes back to stay where it is."""

    if not isinstance(payload, dict):
        return payload
    payload["instance"] = identity.get("label")
    payload["instance_id"] = identity.get("instance_id")
    return payload


def _targeted(handler, identity):
    """Refuse to act on a guess about which window was meant.

    A target that names another window is an error -- the message reached the
    wrong process. No target at all is only a problem when there is a choice to
    make, and then the answer is the roster, reported as a normal result.
    """

    def guarded(arguments):
        wanted = str(arguments.get("instance") or "").strip()
        if wanted:
            if not registry.names_instance(identity, wanted):
                raise ToolError(
                    f"This is window {identity.get('label')} "
                    f"({identity.get('instance_id')}), but the call asked for "
                    f"{wanted!r}. Call rapid2_list_windows to see what is open."
                )
        else:
            rows = _open_windows(identity)
            if len(rows) > 1:
                return _ambiguous(rows)
        return _answered_by(handler(arguments), identity)

    return guarded


# -- analysis tools --------------------------------------------------------
def _analysis_tool(analysis, name, title, description, body, input_schema):
    """One analysis tool. ``body(arguments)`` runs on the request thread."""

    def call(arguments):
        arguments = _without_nulls(input_schema, arguments)
        with analysis.lock:
            try:
                return body(arguments)
            except EXPECTED_FAILURES as exc:
                raise ToolError(str(exc)) from exc

    return Tool(name=name, title=title, description=description, input_schema=input_schema, handler=traced(call, name))


def _ms2_arguments(arguments, default_ms_level):
    """ms_level, precursor_mz, collision_energy and merge_ppm of a report; a precursor makes the default MS level 2."""

    return {
        "ms_level": arguments.get("ms_level", 2 if "precursor_mz" in arguments else default_ms_level),
        "precursor_mz": arguments.get("precursor_mz"),
        "collision_energy": arguments.get("collision_energy"),
        "merge_ppm": arguments.get("merge_ppm", 0.0),
    }


def _quantify(arguments):
    targets, notes = [], []
    if "targets_file" in arguments:
        targets, notes = reports.targets_of_file(arguments["targets_file"])
    if not targets and not arguments.get("targets"):
        raise ToolError("Give targets_file, or targets.")
    result = reports.quantify_targets(
        arguments["runs"],
        targets=targets,
        specs=arguments.get("targets", ()),
        merge_ppm=arguments.get("merge_ppm", 0.0),
        rpd_csv_file=arguments.get("rpd_csv"),
        max_rows=arguments.get("max_rows", 200),
        spectra=arguments.get("spectra"),
        saturation_height=arguments.get("saturation_height"),
    )
    return {**result, "notes": notes} if notes else result


def _alignment_options(arguments):
    """The arguments that are `alignment.OPTIONS`, under the same names."""

    return {name: arguments[name] for name in ALIGNMENT_OPTIONS if name in arguments}


def analysis_tools(analysis):
    """The analysis tools: the window registers them, and `RAPID2_Agilent.cli`
    makes each one a command. Add a tool here and it is in both."""

    runs = analysis.runs
    return [
        _analysis_tool(
            analysis,
            "rapid2_find_runs",
            "Find runs",
            (
                "List the runs under a directory, recursively: Agilent .d folders and RAPID2 .rpd files; of a "
                ".d folder also its sample name, sample type, position and acquisition time (acquired_at), as "
                "MassHunter's worklist wrote them. order: by path, or by acquisition (the injection order), "
                "runs not told of it last. Read-only."
            ),
            lambda arguments: {"runs": services.find_runs(arguments["directory"], arguments.get("order", "path"))},
            _object(
                {
                    "directory": {"type": "string"},
                    "order": {"type": "string", "enum": list(services.RUN_ORDERS), "default": "path"},
                },
                required=("directory",),
            ),
        ),
        _analysis_tool(
            analysis,
            "rapid2_describe_run",
            "Describe a run",
            (
                "The sample information of a .d folder as MassHunter wrote it (sample name, type and position, "
                "method, injection volume, dilution, acquisition time, instrument, ...), and the scan count, RT "
                "range (min), scan types (polarity and MS level), m/z range of the spectra read and the number "
                "of MS2 precursors of a run, and the spectra it holds (profile, centroid; a .rpd file one of "
                "them) with the bytes of each. Read-only."
            ),
            lambda arguments: reports.describe_run(runs.get(arguments["path"], arguments.get("spectra"))),
            _object({"path": PATH_PROPERTY, "spectra": SPECTRA_PROPERTY}, required=("path",)),
        ),
        _analysis_tool(
            analysis,
            "rapid2_list_precursors",
            "List MS2 precursors",
            (
                "The precursors and collision energies of the MS2 scans of a run, with "
                "their isolation width and scan count, by precursor m/z; one entry per "
                "precursor and CE. Pass one to the extract tools as precursor_mz and "
                "collision_energy. Read-only."
            ),
            lambda arguments: reports.ms2_precursors(
                runs.get(arguments["path"], arguments.get("spectra")),
                merge_ppm=arguments.get("merge_ppm", 0.0),
                mz_range=_range(arguments, "mz_min", "mz_max", 0.0),
                max_items=arguments.get("max_items", 200),
                polarity=arguments.get("polarity"),
            ),
            _object(
                {
                    "path": PATH_PROPERTY,
                    "spectra": SPECTRA_PROPERTY,
                    "merge_ppm": MERGE_PPM_PROPERTY,
                    "polarity": POLARITY_PROPERTY,
                    "mz_min": {"type": "number", "description": "Lowest precursor m/z to list."},
                    "mz_max": {"type": "number", "description": "Highest precursor m/z to list."},
                    "max_items": {"type": "integer", "minimum": 1, "default": 200},
                },
                required=("path",),
            ),
        ),
        _analysis_tool(
            analysis,
            "rapid2_calculate_ion_mz",
            "Calculate an ion m/z",
            (
                "Monoisotopic neutral mass and ion m/z of a formula, e.g. \"C6H12O6\", "
                "with an adduct, e.g. \"[M-H]-\"."
            ),
            lambda arguments: reports.calculate_ion(arguments["formula"], arguments["adduct"]),
            _object({"formula": {"type": "string"}, "adduct": {"type": "string"}}, required=("formula", "adduct")),
        ),
        _analysis_tool(
            analysis,
            "rapid2_calculate_isotope_pattern",
            "Calculate an isotope pattern",
            (
                "The isotope pattern of the ion of a formula and an adduct, by nominal mass, from the natural "
                "abundances (NIST): of each M+k, the mean m/z of its isotopologues, its fraction of all the ions "
                "and its abundance relative to the most abundant (relative). Atoms written as an isotope, e.g. "
                "C5H5[2H]9NO or [13C]6H12O6, are of that isotope alone. Use it to expect a compound's M+1, M+2, "
                "..., or to correct isotopologues for natural abundance."
            ),
            lambda arguments: reports.isotope_pattern(
                arguments["formula"], arguments["adduct"], min_relative=arguments.get("min_relative", 0.001),
            ),
            _object(
                {
                    "formula": FORMULA_PROPERTY,
                    "adduct": {"type": "string", "description": "Adduct, e.g. \"[M+H]+\", \"[M+2H]2+\"."},
                    "min_relative": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.001,
                        "description": "Peaks less abundant than this, relative to the most abundant, are left out.",
                    },
                },
                required=("formula", "adduct"),
            ),
        ),
        _analysis_tool(
            analysis,
            "rapid2_extract_xic",
            "Extract an XIC",
            (
                "Extracted ion chromatogram of a run: the summed intensity within "
                "mz ± tolerance per scan; give mz, or formula and adduct for the m/z of their "
                "ion. An adduct, or polarity, chooses the scans of that polarity. With "
                "precursor_mz, of the MS2 scans of that precursor, and mz is a product ion. "
                "A scan whose data points miss the m/z window is left out as not measured "
                "there (n_missing, warning), as the window's plots do. Returns the apex and "
                "the trace. Read-only; use rapid2_set_view to show it to the user."
            ),
            lambda arguments: reports.xic(
                runs.get(arguments["path"], arguments.get("spectra")),
                arguments.get("mz"),
                arguments.get("tolerance", 10.0),
                arguments.get("unit", "ppm"),
                formula=arguments.get("formula"),
                adduct=arguments.get("adduct"),
                polarity=arguments.get("polarity"),
                rt_range=_range(arguments, "rt_min", "rt_max", float("-inf")),
                max_points=arguments.get("max_points", 300),
                **_ms2_arguments(arguments, 1),
            ),
            _object(
                {
                    "path": PATH_PROPERTY,
                    "spectra": SPECTRA_PROPERTY,
                    "mz": {"type": "number", "exclusiveMinimum": 0},
                    "formula": FORMULA_PROPERTY,
                    "adduct": ADDUCT_PROPERTY,
                    "tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 10.0},
                    "unit": {"type": "string", "enum": ["ppm", "m/z"], "default": "ppm"},
                    "polarity": POLARITY_PROPERTY,
                    "ms_level": {**MS_LEVEL_PROPERTY, "default": 1},
                    **MS2_PROPERTIES,
                    "rt_min": RT_MIN_PROPERTY,
                    "rt_max": RT_MAX_PROPERTY,
                    "max_points": MAX_POINTS_PROPERTY,
                },
                required=("path",),
            ),
        ),
        _analysis_tool(
            analysis,
            "rapid2_extract_trace",
            "Extract a TIC or BPC",
            (
                "Total ion chromatogram (TIC) or base peak chromatogram (BPC) of a run, "
                "from its scan records (of the profile, whichever spectra are read); with "
                "precursor_mz, of the MS2 scans of that precursor. Read-only."
            ),
            lambda arguments: reports.trace(
                runs.get(arguments["path"], arguments.get("spectra")),
                arguments.get("trace_type", "TIC"),
                polarity=arguments.get("polarity"),
                rt_range=_range(arguments, "rt_min", "rt_max", float("-inf")),
                max_points=arguments.get("max_points", 300),
                **_ms2_arguments(arguments, None),
            ),
            _object(
                {
                    "path": PATH_PROPERTY,
                    "spectra": SPECTRA_PROPERTY,
                    "trace_type": {"type": "string", "enum": ["TIC", "BPC"], "default": "TIC"},
                    "polarity": POLARITY_PROPERTY,
                    "ms_level": MS_LEVEL_PROPERTY,
                    **MS2_PROPERTIES,
                    "rt_min": RT_MIN_PROPERTY,
                    "rt_max": RT_MAX_PROPERTY,
                    "max_points": MAX_POINTS_PROPERTY,
                },
                required=("path",),
            ),
        ),
        _analysis_tool(
            analysis,
            "rapid2_extract_spectrum",
            "Extract a spectrum",
            (
                "The most intense peaks (of a profile, its local maxima; of centroid spectra, "
                "the centroids) of the spectrum averaged "
                "(mean) or summed over the scans in rt ± rt_tolerance (min), or of the "
                "scan nearest to rt when none is in the window. With precursor_mz, the "
                "product-ion spectrum of the MS2 scans of that precursor. Read-only."
            ),
            lambda arguments: reports.spectrum(
                runs.get(arguments["path"], arguments.get("spectra")),
                arguments["rt"],
                arguments.get("rt_tolerance", 0.0),
                mode=arguments.get("mode", "mean"),
                mz_range=_range(arguments, "mz_min", "mz_max", 0.0),
                top_n=arguments.get("top_n", 50),
                polarity=arguments.get("polarity"),
                **_ms2_arguments(arguments, 1),
            ),
            _object(
                {
                    "path": PATH_PROPERTY,
                    "spectra": SPECTRA_PROPERTY,
                    "rt": {"type": "number", "description": "RT, in minutes."},
                    "rt_tolerance": {"type": "number", "minimum": 0, "default": 0.0},
                    "polarity": POLARITY_PROPERTY,
                    "ms_level": {**MS_LEVEL_PROPERTY, "default": 1},
                    **MS2_PROPERTIES,
                    "mode": {"type": "string", "enum": ["mean", "sum"], "default": "mean"},
                    "mz_min": {"type": "number"},
                    "mz_max": {"type": "number"},
                    "top_n": {"type": "integer", "minimum": 1, "default": 50},
                },
                required=("path", "rt"),
            ),
        ),
        _analysis_tool(
            analysis,
            "rapid2_quantify_targets",
            "Quantify targets in runs",
            (
                "The area, baseline-adjusted area, height, RT and m/z of the peak top and the scans "
                "used of each target in each run, as RAPID2's Export Targets computes them: a row "
                "per target and run. Targets from a .rpd.csv file (targets_file, as Export Targets "
                "writes it, or RAPID2's), or described in targets, or both. Each run's scans of the "
                "target's adduct polarity, MS level (MS2: precursor and CE) and RT window are used. "
                "rpd_csv also writes the rows as a .rpd.csv file, which must not exist yet. Use it to "
                "compare a compound across runs. Each row has warnings to look at again before trusting it "
                "(the peak top at the first or last scan of the RT window, and, with saturation_height, a "
                "height at or above it), and flagged lists the rows with any: look at "
                "their XIC (rapid2_extract_xic, or the window: rapid2_select_target and rapid2_capture)."
            ),
            lambda arguments: _quantify(arguments),
            _object(
                {
                    "runs": {"type": "array", "items": PATH_PROPERTY, "description": "Paths of runs: .d folders, .rpd files."},
                    "spectra": {
                        **SPECTRA_PROPERTY,
                        "description": "Of every run (left out, those of the first run, for all): " + SPECTRA_PROPERTY["description"],
                    },
                    "targets_file": {"type": "string", "description": "A .rpd.csv file of targets."},
                    "targets": {"type": "array", "items": _object(TARGET_PROPERTIES, required=TARGET_REQUIRED)},
                    "merge_ppm": MERGE_PPM_PROPERTY,
                    "rpd_csv": {"type": "string", "description": "Also write the rows to this new .rpd.csv file."},
                    "max_rows": {"type": "integer", "minimum": 1, "default": 200, "description": "Rows returned; the file has all."},
                    "saturation_height": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": (
                            "A height from which the detector may have saturated, to warn of; it differs by "
                            "instrument and spectra, and is not known to RAPID2. Left out, no such warning."
                        ),
                    },
                },
                required=("runs",),
            ),
        ),
        _analysis_tool(
            analysis,
            "rapid2_map_spots",
            "Find the strongest spots of a run",
            (
                "The most intense spots of the m/z x RT map of a run's MS1 scans, as RAPID2's "
                "m/z-RT Map shows it (bins of about 100 ppm): their RT, the m/z of the highest "
                "data point in the spot and its intensity, apart from one another. A first look "
                "at what a run holds; follow up with rapid2_extract_xic or rapid2_extract_spectrum. "
                "Read-only."
            ),
            lambda arguments: reports.map_spots(
                runs.get(arguments["path"], arguments.get("spectra")),
                top_n=arguments.get("top_n", 20),
                rt_range=_range(arguments, "rt_min", "rt_max", float("-inf")),
                mz_range=_range(arguments, "mz_min", "mz_max", 0.0),
            ),
            _object(
                {
                    "path": PATH_PROPERTY,
                    "spectra": SPECTRA_PROPERTY,
                    "top_n": TOP_N_SPOTS_PROPERTY,
                    "rt_min": RT_MIN_PROPERTY,
                    "rt_max": RT_MAX_PROPERTY,
                    "mz_min": {"type": "number"},
                    "mz_max": {"type": "number"},
                },
                required=("path",),
            ),
        ),
        _analysis_tool(
            analysis,
            "rapid2_compare_runs",
            "Find where two runs differ",
            (
                "Where the query run's MS1 m/z x RT map differs most from the reference's, as the "
                "B - A view of RAPID2's m/z-RT Map computes it: the query aligned onto the "
                "reference's RT, normalized to its intensity, each side compared with the other "
                "within 1 scan and 10 ppm. The spots higher in the query and those higher in the "
                "reference, the largest first, with D (about the natural log of the fold change "
                "where both are strong) and both intensities. RTs are the reference's; query_rt is "
                "the query's own. Read-only."
            ),
            lambda arguments: reports.compare_runs(
                runs,
                arguments["reference"],
                arguments["query"],
                top_n=arguments.get("top_n", 20),
                min_fold_change=arguments.get("min_fold_change", 2.0),
                rt_range=_range(arguments, "rt_min", "rt_max", float("-inf")),
                mz_range=_range(arguments, "mz_min", "mz_max", 0.0),
                spectra=arguments.get("spectra"),
            ),
            _object(
                {
                    "reference": PATH_PROPERTY,
                    "query": PATH_PROPERTY,
                    "spectra": {
                        **SPECTRA_PROPERTY,
                        "description": "Of both runs (left out, those of the reference, for both): " + SPECTRA_PROPERTY["description"],
                    },
                    "top_n": TOP_N_SPOTS_PROPERTY,
                    "min_fold_change": {
                        "type": "number",
                        "exclusiveMinimum": 1,
                        "default": 2.0,
                        "description": "Spots of a smaller difference are left out.",
                    },
                    "rt_min": RT_MIN_PROPERTY,
                    "rt_max": RT_MAX_PROPERTY,
                    "mz_min": {"type": "number"},
                    "mz_max": {"type": "number"},
                },
                required=("reference", "query"),
            ),
        ),
        _analysis_tool(
            analysis,
            "rapid2_suggest_alignment_scales",
            "Suggest asinh scales for alignment",
            (
                "Candidate asinh scales from the binned MS1 intensities of a .d folder's "
                "profile, for rapid2_align_runs; none is chosen for you. max_scans limits "
                "the scans read. Read-only."
            ),
            lambda arguments: reports.scale_candidates(runs, arguments["path"], **_alignment_options(arguments)),
            _object({"path": D_PATH_PROPERTY, **SCALE_OPTION_PROPERTIES}, required=("path",)),
        ),
        _analysis_tool(
            analysis,
            "rapid2_align_runs",
            "Align the RTs of two runs",
            (
                "Map the query run's RTs onto the reference's from the raw MS1 scans of "
                "their profile (no peak picking), and save mapping.json, metadata.json and a "
                "diagnostics figure to a new output_dir. Returns the RT shift range in "
                "seconds. Two full runs can take minutes; max_scans limits the scans."
            ),
            lambda arguments: reports.align_runs(
                arguments["reference"], arguments["query"], arguments["output_dir"], **_alignment_options(arguments),
            ),
            _object(
                {
                    "reference": D_PATH_PROPERTY,
                    "query": D_PATH_PROPERTY,
                    "output_dir": {"type": "string", "description": "A directory with no alignment result in it yet."},
                    "scale": {"type": "number", "exclusiveMinimum": 0, "description": "From rapid2_suggest_alignment_scales."},
                    **SCALE_OPTION_PROPERTIES,
                    "metric": {"type": "string", "enum": ["cosine", "pearson"]},
                    "device": {"type": "string", "enum": ["auto", "cpu", "cuda"]},
                },
                required=("reference", "query", "output_dir", "scale"),
            ),
        ),
    ]


# -- registry ------------------------------------------------------------
def build_registry(context, analysis, *, run_on_main_thread, identity=None):
    """Register the tools. ``run_on_main_thread`` wraps every viewer tool."""

    tools = ToolRegistry()
    identity = identity or {"label": None, "instance_id": None}

    def viewer(name, title, description, body, *, input_schema=None):
        """One viewer tool. ``body(arguments)`` runs on the Qt main thread."""

        schema = _accepting_a_target(input_schema or _NO_ARGUMENTS)

        def on_main_thread(arguments):
            try:
                return body(arguments)
            except EXPECTED_FAILURES as exc:
                raise ToolError(str(exc)) from exc

        def call(arguments):
            arguments = _without_nulls(schema, arguments)
            return run_on_main_thread(lambda: on_main_thread(arguments))

        tools.add(
            Tool(
                name=name,
                title=title,
                description=description,
                input_schema=schema,
                handler=traced(_targeted(call, identity), name),
            )
        )

    # Answered from the registry, so not marshalled onto the main thread: it
    # reads files, and queueing it behind a running tool call would make
    # "which windows are open?" wait on work in one of them.
    tools.add(
        Tool(
            name="rapid2_list_windows",
            title="List RAPID2 windows",
            description=(
                "List the RAPID2 windows open on this computer with AI access on: "
                "the letter each answers to, its instance_id, and when it "
                "started. Pass a letter as `instance` to a viewer tool to act in "
                "that window. Read-only."
            ),
            input_schema=_NO_ARGUMENTS,
            handler=traced(
                lambda _arguments: {"windows": _open_windows(identity)},
                "rapid2_list_windows",
            ),
        )
    )

    # Starts RAPID2. The relay answers it, being the one thing running while
    # RAPID2 is not; it is registered here so that it is in the list a window
    # gives out and in the catalog left for the relay. A window answering is
    # RAPID2 already open.
    tools.add(
        Tool(
            name=relay_launch.TOOL_NAME,
            title=relay_launch.TITLE,
            description=relay_launch.DESCRIPTION,
            input_schema=relay_launch.INPUT_SCHEMA,
            handler=traced(
                lambda _arguments: {
                    "status": "ok",
                    "started": False,
                    "message": "RAPID2 is already open and taking requests.",
                    "windows": _open_windows(identity),
                },
                relay_launch.TOOL_NAME,
            ),
        )
    )

    # -- viewer ----------------------------------------------------------
    viewer(
        "rapid2_get_status",
        "Get the RAPID2 window's state",
        (
            "What the RAPID2 window shows: the open files with the spectra "
            "read (profile or centroid) and their scan "
            "types, the selected RT window (min) and m/z window, the trace type "
            "(XIC, TIC or BPC), the MS level (null for all scans), the precursor "
            "and CE of the MS2 scans shown, the MS2 precursors of the files as the "
            "window lists them, and the number of targets. Read-only."
        ),
        lambda _arguments: context.status(),
    )
    viewer(
        "rapid2_open_files",
        "Open runs in RAPID2",
        (
            "Open Agilent .d folders or RAPID2 .rpd files in the RAPID2 window, "
            "below the files already open: a .d folder of its profile or (spectra) "
            "its centroid spectra, as the window's Open dialog asks, a .rpd file of "
            "the spectra it holds; one already open is not opened again. Returns "
            "the window's state afterwards, as rapid2_get_status, which gives each "
            "file's spectra."
        ),
        lambda arguments: context.open_files(arguments["paths"], arguments.get("spectra")),
        input_schema=_object(
            {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths of .d folders or .rpd files.",
                },
                "spectra": SPECTRA_PROPERTY,
            },
            required=("paths",),
        ),
    )
    viewer(
        "rapid2_set_view",
        "Show an RT and m/z window in RAPID2",
        (
            "Select what the RAPID2 window shows for every open file: the RT "
            "window (min) the spectra are averaged over, the m/z window the XIC "
            "is extracted from, the trace type and the MS level. precursor_mz "
            "and collision_energy show the MS2 scans of one of the window's "
            "precursors (rapid2_get_status lists them); the m/z window is then "
            "of a product ion. Leave out what should stay as it is. A target "
            "selected in the window is deselected, not changed, and the window's "
            "controls show the new view. The plots pan to the new windows, and "
            "the user can undo that with View > Undo View Range. Returns the "
            "window's state afterwards."
        ),
        lambda arguments: context.set_view(
            rt_window=_window(arguments, "rt_min", "rt_max"),
            mz_window=_window(arguments, "mz_min", "mz_max"),
            trace_type=arguments.get("trace_type"),
            ms_level=arguments["ms_level"] if "ms_level" in arguments else False,
            precursor_mz=arguments.get("precursor_mz"),
            collision_energy=arguments.get("collision_energy"),
        ),
        input_schema=_object(
            {
                "rt_min": RT_MIN_PROPERTY,
                "rt_max": RT_MAX_PROPERTY,
                "mz_min": {"type": "number", "exclusiveMinimum": 0, "description": "Lower m/z limit."},
                "mz_max": {"type": "number", "exclusiveMinimum": 0, "description": "Upper m/z limit."},
                "trace_type": {"type": "string", "enum": [kind.value for kind in TraceType]},
                "ms_level": MS_LEVEL_PROPERTY,
                "precursor_mz": PRECURSOR_MZ_PROPERTY,
                "collision_energy": COLLISION_ENERGY_PROPERTY,
            }
        ),
    )
    viewer(
        "rapid2_capture",
        "Capture the RAPID2 window",
        (
            "A PNG of what the RAPID2 window shows: the whole window, or only the "
            "chromatogram and spectrum plots. Use it to check what the user sees "
            "after rapid2_set_view. Read-only."
        ),
        lambda arguments: context.capture(arguments.get("part", "window")),
        input_schema=_object(
            {"part": {"type": "string", "enum": ["window", "plots"], "default": "window"}}
        ),
    )

    viewer(
        "rapid2_list_targets",
        "List the targets in RAPID2",
        (
            "The targets in the RAPID2 window's Targets list, by number (#), with all their "
            "fields, and the one selected. Read-only."
        ),
        lambda _arguments: context.list_targets(),
    )
    viewer(
        "rapid2_add_target",
        "Add a target in RAPID2",
        (
            "Add a target to the RAPID2 window's Targets list, numbered after the others, and "
            "select it, so that the plots show it. For MS2, the precursor is chosen among the "
            "window's (rapid2_get_status lists them). The user can undo it with Edit > Undo "
            "Target Edit. Returns the target added."
        ),
        lambda arguments: context.add_target({name: value for name, value in arguments.items() if name != "instance"}),
        input_schema=_object(TARGET_PROPERTIES, required=TARGET_REQUIRED),
    )
    viewer(
        "rapid2_select_target",
        "Show a target in RAPID2",
        (
            "Select a target of the RAPID2 window by its number (#, as rapid2_list_targets gives "
            "it), so that the plots show its trace, RT and m/z windows, and pan to it."
        ),
        lambda arguments: context.select_target(arguments["target"]),
        input_schema=_object({"target": {"type": "integer", "minimum": 1}}, required=("target",)),
    )
    viewer(
        "rapid2_import_targets",
        "Import targets into RAPID2",
        (
            "Read the targets of a .rpd.csv file (as Export Targets writes it, or RAPID2's) into "
            "the RAPID2 window's Targets list, as File > Load Targets does: in place of the "
            "targets there (replace) or after them (append); mode is needed when the list is not "
            "empty. Rows of one # that hold different targets are kept as targets of their own. "
            "The user can undo it with Edit > Undo Target Edit."
        ),
        lambda arguments: context.import_targets(arguments["path"], arguments.get("mode")),
        input_schema=_object(
            {"path": {"type": "string"}, "mode": {"type": "string", "enum": ["replace", "append"]}},
            required=("path",),
        ),
    )

    # The targets and the files are taken on the main thread, and quantified on
    # the request thread, as an analysis tool would: an export can take minutes,
    # and the window must not freeze for them.
    export_schema = _accepting_a_target(
        _object(
            {
                "path": {"type": "string", "description": "A new .rpd.csv file; .rpd.csv is added to a name without it."},
                "max_rows": {"type": "integer", "minimum": 1, "default": 200, "description": "Rows returned; the file has all."},
            },
            required=("path",),
        )
    )

    def export_targets(arguments):
        arguments = _without_nulls(export_schema, arguments)
        snapshot = run_on_main_thread(context.target_snapshot)
        if not snapshot["targets"]:
            raise ToolError("The window has no targets to export.")
        try:
            result = reports.quantify_targets(
                snapshot["paths"],
                targets=snapshot["targets"],
                merge_ppm=snapshot["merge_ppm"],
                run_peaks=snapshot["run_peaks"],
                rpd_csv_file=arguments["path"],
                max_rows=arguments.get("max_rows", 200),
                spectra=snapshot["spectra"],
            )
        except EXPECTED_FAILURES as exc:
            raise ToolError(str(exc)) from exc
        if snapshot["left_out"]:
            result["left_out"] = snapshot["left_out"]  # neither .d folders nor .rpd files, which only the window can read
        return result

    tools.add(
        Tool(
            name="rapid2_export_targets",
            title="Export the targets of RAPID2",
            description=(
                "Write the RAPID2 window's targets and their values in its open files, each read "
                "of the spectra the window has it of, to a new .rpd.csv file, as File > Export Targets "
                "does, and return the rows, as rapid2_quantify_targets does."
            ),
            input_schema=export_schema,
            handler=traced(_targeted(export_targets, identity), "rapid2_export_targets"),
        )
    )

    for tool in analysis_tools(analysis):
        tools.add(tool)
    return tools
