import math
import threading
from collections import OrderedDict
from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from RAPID2_Agilent.models.mz_rt_diff import DiffOverview, DiffParams, MzRtDiff, RTAlignment, align_maps, rasterize_diff
from RAPID2_Agilent.models.mz_rt_map import MapLayer, MapLevel, MzRtMap, rasterize
from RAPID2_Agilent.models.rpd_model import BaseMSDataModel
from RAPID2_Agilent.models.target_model import TraceType
from RAPID2_Agilent.views.data_stack import ChromatogramSpectrumStackPresenter
from RAPID2_Agilent.views.target_panel import TargetPanelPresenter

from .diagnostics import Diagnostics, DiagnosticsPlot, dtw_path, path_similarity, rt_shift, squared_difference, tic
from .mz_rt_map_window import MzRtMapWindow


class _TaskSignals(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(object)


class _Task(QRunnable):
    """Runs work(progress, is_cancelled) in the thread pool."""

    def __init__(self, work: Callable[[Callable[[int, int], None], Callable[[], bool]], object]):
        super().__init__()
        self.work = work
        self.signals = _TaskSignals()
        self._cancelled = threading.Event()

    def cancel(self):
        self._cancelled.set()

    def run(self):
        try:
            result = self.work(self.signals.progress.emit, self._cancelled.is_set)
        except Exception as error:
            self.signals.failed.emit(error)
            return
        self.signals.finished.emit(result)


class MzRtMapPresenter(QObject):
    """Heat map of file A, or of the difference of file B (aligned to A in RT) from A.

    Clicking jumps the main view there.
    """

    max_cached_maps = 2  # each keeps ~ n_scans x 35k float32 values; A and B both stay

    def __init__(
        self,
        window: MzRtMapWindow,
        data_stack_presenter: ChromatogramSpectrumStackPresenter,
        target_panel_presenter: TargetPanelPresenter,
        parent=None,
    ):
        super().__init__(parent)
        self.window = window
        self.data_stack_presenter = data_stack_presenter
        self.target_panel_presenter = target_panel_presenter
        self.current_model: BaseMSDataModel | None = None  # A
        self.compare_model: BaseMSDataModel | None = None  # B
        self._maps: OrderedDict[BaseMSDataModel, MzRtMap] = OrderedDict()  # built maps, oldest first
        self._scales: dict[BaseMSDataModel, float] = {}  # s of each file as A, also used for B - A
        self._alignments: dict[tuple[BaseMSDataModel, BaseMSDataModel], RTAlignment] = {}  # (A, B)
        self._show_whole_range = False  # once A's map is there
        self._build_task: _Task | None = None
        self._build_task_model: BaseMSDataModel | None = None
        self._align_task: _Task | None = None
        self._overview_task: _Task | None = None
        self._overview_task_params: DiffParams | None = None
        self._diff_overview: tuple[DiffParams, DiffOverview] | None = None
        self._detail_task: _Task | None = None
        self._detail_task_range: tuple[float, float] | None = None
        self._detail_layers: tuple[MapLayer, MapLayer | None] | None = None  # of A, and of B in difference mode
        # last rasterized image (intensities of A, or D) and its RT and m/z ranges, reused when only the colors change
        self._raster: tuple[np.ndarray, tuple[float, float], tuple[float, float]] | None = None

        self.window.file_selected.connect(self.on_file_selected)
        self.window.compare_file_selected.connect(self.on_compare_file_selected)
        self.window.scale_changed.connect(self.on_scale_changed)
        self.window.diff_settings_changed.connect(self.on_diff_settings_changed)
        self.window.diagnostics_changed.connect(self._update_diagnostics)
        self.window.fold_change_limit_changed.connect(lambda _limit: self._show_raster())
        self.window.view_range_changed.connect(self.render)
        self.window.clicked.connect(self.target_panel_presenter.jump_to)
        self.window.marker_range_changed.connect(self.on_marker_range_changed)
        self.data_stack_presenter.model_added.connect(lambda _model: self._update_file_list())
        self.data_stack_presenter.current_windows_changed.connect(self.update_marker)

    def show(self):
        self._update_file_list()
        self.update_marker()
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        models = self.data_stack_presenter.models
        if self.current_model is None and models:
            self.select_model(models[0])

    ### files ###
    def on_file_selected(self, index: int):
        self.select_model(self.data_stack_presenter.models[index])

    def on_compare_file_selected(self, index: int):
        self.select_compare_model(None if index < 0 else self.data_stack_presenter.models[index])

    def select_model(self, model: BaseMSDataModel):
        """Set A."""
        self.current_model = model
        self._show_whole_range = True
        self._on_selection_changed()

    def select_compare_model(self, model: BaseMSDataModel | None):
        """Set B; None shows A alone."""
        self.compare_model = model
        self._on_selection_changed()

    def _on_selection_changed(self):
        needed = (self.current_model, self.compare_model)
        if self._build_task is not None and self._build_task_model not in needed:
            self._build_task.cancel()
            self._build_task = None
        for task in (self._align_task, self._overview_task, self._detail_task):
            if task is not None:
                task.cancel()
        self._align_task = self._overview_task = self._detail_task = None
        self._diff_overview = None
        self._detail_layers = None
        self._raster = None
        for model in needed:
            if model in self._maps:
                self._maps.move_to_end(model)
        self._update_file_list()
        self.window.set_diff_mode(self.compare_model is not None)
        self.window.set_normalization_factor(None)
        self.window.set_diagnostics(None)
        self.window.clear_image()
        self._prepare()

    def _prepare(self):
        """Start what the selection still needs, or render it."""
        model_a, model_b = self.current_model, self.compare_model
        if model_a is None:
            return
        for model, role in ((model_a, "A"), (model_b, "B")):
            if model is not None and model not in self._maps:
                self._request_build(model, role)
                return
        map_a = self._maps[model_a]
        if self._show_whole_range:
            self._show_whole_range = False
            scale = self._scales.setdefault(model_a, map_a.default_scale())
            self.window.set_scale_range(*map_a.scale_range())
            self.window.set_scale(scale)
            self.window.show_range(map_a.rt_range, map_a.mz_range)
        if model_b is not None:
            alignment = self._alignments.get((model_a, model_b))
            if alignment is None:
                self._request_alignment(map_a, self._maps[model_b])
                return
            self.window.set_normalization_factor(alignment.normalization)
            self._update_diagnostics()
        self.render()

    def _request_build(self, model: BaseMSDataModel, role: str):
        if self._build_task is not None:
            if self._build_task_model is model:
                return  # already building
            self._build_task.cancel()
        mz_rt_map = MzRtMap(model)
        self.window.set_status(f"Building {role}...")
        task = _Task(lambda progress, is_cancelled: mz_rt_map if mz_rt_map.build(progress, is_cancelled) else None)
        task.signals.progress.connect(lambda done, total: self.window.set_status(f"Building {role}... {done}/{total} scans"))
        task.signals.finished.connect(lambda result, task=task: self._on_built(task, result))
        task.signals.failed.connect(self._raise)
        self._build_task, self._build_task_model = task, model
        QThreadPool.globalInstance().start(task)

    def _on_built(self, task: _Task, mz_rt_map: MzRtMap | None):
        if task is not self._build_task or mz_rt_map is None:
            return  # cancelled or superseded
        self._build_task = None
        self._maps[mz_rt_map.ms_data_model] = mz_rt_map
        needed = (self.current_model, self.compare_model)
        unneeded = [model for model in self._maps if model not in needed]  # oldest first
        for model in unneeded[:max(len(self._maps) - self.max_cached_maps, 0)]:
            del self._maps[model]
        self._prepare()

    def _request_alignment(self, map_a: MzRtMap, map_b: MzRtMap):
        if self._align_task is not None:
            return  # running for this pair; others are cancelled when the selection changes
        pair = (map_a.ms_data_model, map_b.ms_data_model)
        self.window.set_status("Aligning B to A in RT...")
        task = _Task(lambda progress, is_cancelled: align_maps(map_a, map_b, is_cancelled))
        task.signals.finished.connect(lambda result, task=task: self._on_aligned(task, pair, result))
        task.signals.failed.connect(lambda error, task=task: self._on_alignment_failed(task, error))
        self._align_task = task
        QThreadPool.globalInstance().start(task)

    def _on_aligned(self, task: _Task, pair: tuple[BaseMSDataModel, BaseMSDataModel], alignment: RTAlignment | None):
        if task is not self._align_task or alignment is None:
            return  # cancelled or superseded
        self._align_task = None
        self._alignments[pair] = alignment
        self._prepare()

    def _on_alignment_failed(self, task: _Task, error: Exception):
        if task is not self._align_task:
            return
        self._align_task = None
        if isinstance(error, ImportError):
            package = error.name or "scipy"
            message = f"RT alignment needs {package}: pip install {package}"
        else:
            message = f"RT alignment failed: {error}"
        self.window.set_status(message)
        self.window.set_diagnostics(None, message)

    def _update_file_list(self):
        models = self.data_stack_presenter.models
        names = [f"{i + 1}: {model.file_path.name}" for i, model in enumerate(models)]
        current_index = models.index(self.current_model) if self.current_model in models else -1
        compare_index = models.index(self.compare_model) if self.compare_model in models else -1
        self.window.set_files(names, [str(model.file_path) for model in models], current_index, compare_index)

    ### image ###
    def on_scale_changed(self, scale: float):
        if self.current_model is None:
            return
        self._scales[self.current_model] = scale
        if self.compare_model is None:
            self._show_raster()
        else:
            self.render()  # s enters D before the pooling into pixels
            self._update_diagnostics(if_showing=(Diagnostics.SQUARED_DIFFERENCE,))

    def on_diff_settings_changed(self):
        self.render()
        self._update_diagnostics(if_showing=(Diagnostics.TIC, Diagnostics.SQUARED_DIFFERENCE))

    def render(self):
        map_a = self._maps.get(self.current_model)
        diff = self._current_diff()
        if map_a is None or (self.compare_model is not None and diff is None):
            return  # not built or not aligned yet
        rt_range, mz_range = self.window.visible_range()
        shape = self.window.pixel_shape()
        pixel_ppm = math.expm1(math.log(mz_range[1] / mz_range[0]) / shape[1]) * 1e6
        level = map_a.level_for(pixel_ppm)
        if diff is None:
            self._render_map(map_a, level, rt_range, mz_range, shape)
        else:
            self._render_diff(diff, level, rt_range, mz_range, shape)

    def _current_diff(self) -> MzRtDiff | None:
        """B - A, None if only A is shown or B is not built or aligned yet."""
        model_a, model_b = self.current_model, self.compare_model
        alignment = self._alignments.get((model_a, model_b))
        if alignment is None or model_a not in self._maps or model_b not in self._maps:
            return None
        return MzRtDiff(self._maps[model_a], self._maps[model_b], alignment)

    def _render_map(self, map_a: MzRtMap, level: MapLevel, rt_range, mz_range, shape):
        waiting = False
        if level is MapLevel.OVERVIEW:
            layer = map_a.overview
        elif level is MapLevel.MID:
            layer = map_a.mid
        elif self._detail_covers(mz_range):
            layer = self._detail_layers[0]
        else:
            self._request_detail(map_a, None, mz_range)
            layer, waiting = map_a.mid, True  # until the detail arrives
        self._raster = (rasterize(layer, rt_range, mz_range, shape), rt_range, mz_range)
        self._show_raster()
        self.window.set_status(f"{layer.bin_ppm:.0f} ppm bins" + (" (computing detail...)" if waiting else ""))

    def _render_diff(self, diff: MzRtDiff, level: MapLevel, rt_range, mz_range, shape):
        params = self._diff_params(diff.alignment)
        waiting = False
        if level is MapLevel.OVERVIEW:
            if self._diff_overview is None or self._diff_overview[0] != params:
                self._request_diff_overview(diff, params)
                waiting = True
            layers = None if self._diff_overview is None else self._diff_overview[1].layers  # possibly of older params
        else:
            if level is MapLevel.DETAIL and self._detail_covers(mz_range):
                layer_a, layer_b = self._detail_layers
            else:
                if level is MapLevel.DETAIL:
                    self._request_detail(diff.map_a, diff.map_b, mz_range)
                    waiting = True
                layer_a, layer_b = diff.map_a.mid, diff.map_b.mid  # until the detail arrives
            layers = diff.parts(layer_a, layer_b, rt_range, mz_range, params)
        if layers is None:
            self.window.set_status("Computing difference...")
            return
        self._raster = (rasterize_diff(layers, rt_range, mz_range, shape), rt_range, mz_range)
        self._show_raster()
        shift = diff.alignment.max_shift(diff.map_b.rt)
        status = f"{layers.higher_in_b.bin_ppm:.0f} ppm bins, B shifted by up to {shift:.2f} min"
        self.window.set_status(status + (" (computing...)" if waiting else ""))

    def _diff_params(self, alignment: RTAlignment) -> DiffParams:
        normalize, rt_tolerance, mz_tolerance_ppm = self.window.diff_settings()
        return DiffParams(
            scale=self._scales[self.current_model],
            b_factor=alignment.normalization if normalize else 1.0,
            rt_tolerance=rt_tolerance,
            mz_tolerance_ppm=mz_tolerance_ppm,
        )

    def _show_raster(self):
        if self._raster is None:
            return
        image, rt_range, mz_range = self._raster
        if self.compare_model is None:
            scale = self._scales[self.current_model]
            max_intensity = self._maps[self.current_model].max_intensity()
            self.window.set_image(
                np.arcsinh(image / scale),
                rt_range,
                mz_range,
                levels=(0.0, math.asinh(max_intensity / scale)),
                color_bar_ticks=intensity_ticks(scale, max_intensity),
            )
        else:
            limit = self.window.fold_change_limit()
            self.window.set_image(
                image,
                rt_range,
                mz_range,
                levels=(-math.log(limit), math.log(limit)),
                color_bar_ticks=fold_change_ticks(limit),
            )

    def _request_diff_overview(self, diff: MzRtDiff, params: DiffParams):
        if self._overview_task is not None:
            if self._overview_task_params == params:
                return  # already computing these
            self._overview_task.cancel()
        task = _Task(lambda progress, is_cancelled: diff.overview(params, is_cancelled))
        task.signals.finished.connect(lambda overview, task=task: self._on_diff_overview(task, params, overview))
        task.signals.failed.connect(self._raise)
        self._overview_task, self._overview_task_params = task, params
        QThreadPool.globalInstance().start(task)

    def _on_diff_overview(self, task: _Task, params: DiffParams, overview: DiffOverview | None):
        if task is not self._overview_task or overview is None:
            return  # cancelled or superseded
        self._overview_task = None
        self._diff_overview = (params, overview)
        self.render()
        self._update_diagnostics(if_showing=(Diagnostics.SQUARED_DIFFERENCE,))

    ### diagnostics ###
    def _update_diagnostics(self, if_showing: tuple[Diagnostics, ...] = tuple(Diagnostics)):
        kind = self.window.diagnostics()
        if kind is Diagnostics.NONE or kind not in if_showing:
            return
        diff = self._current_diff()
        if diff is None:
            self.window.set_diagnostics(None, "Waiting for the RT alignment...")
            return
        if kind is Diagnostics.SQUARED_DIFFERENCE:
            self._show_squared_difference(diff)
            return
        params = self._diff_params(diff.alignment)
        plots = {
            Diagnostics.DTW_PATH: lambda: dtw_path(diff),
            Diagnostics.RT_SHIFT: lambda: rt_shift(diff),
            Diagnostics.PATH_SIMILARITY: lambda: path_similarity(diff),
            Diagnostics.TIC: lambda: tic(diff, params.b_factor),
        }
        self.window.set_diagnostics(plots[kind]())

    def _show_squared_difference(self, diff: MzRtDiff):
        """From the overview computation, which also yields D^2 per scan."""
        params = self._diff_params(diff.alignment)
        current = self._diff_overview is not None and self._diff_overview[0] == params
        if not current:
            self._request_diff_overview(diff, params)
        plot: DiagnosticsPlot | None = None
        if self._diff_overview is not None:
            plot = squared_difference(diff, self._diff_overview[1])  # possibly of older params
        self.window.set_diagnostics(plot, "" if current else "Computing...")

    def _detail_covers(self, mz_range: tuple[float, float]) -> bool:
        if self._detail_layers is None:
            return False
        layer = self._detail_layers[0]
        return _covers((layer.mz_edges[0], layer.mz_edges[-1]), mz_range)

    def _request_detail(self, map_a: MzRtMap, map_b: MzRtMap | None, mz_range: tuple[float, float]):
        """Detail of A, and of B if given, around mz_range."""
        if self._detail_task is not None:
            if _covers(self._detail_task_range, mz_range):
                return  # the running computation will cover this view
            self._detail_task.cancel()
        # a margin of half the visible width on each side, so that panning reuses the layer
        half_width = (mz_range[1] - mz_range[0]) / 2
        map_mz_min, map_mz_max = map_a.mz_range
        mz_min = max(mz_range[0] - half_width, map_mz_min)
        mz_max = min(mz_range[1] + half_width, map_mz_max)

        def work(progress, is_cancelled):
            layer_a = map_a.detail(mz_min, mz_max, is_cancelled)
            if layer_a is None or map_b is None:
                return None if layer_a is None else (layer_a, None)
            b_min, b_max = max(mz_min, map_b.mz_range[0]), min(mz_max, map_b.mz_range[1])
            if b_max <= b_min:
                return layer_a, None  # B has no bins here
            layer_b = map_b.detail(b_min, b_max, is_cancelled)
            return None if layer_b is None else (layer_a, layer_b)

        task = _Task(work)
        task.signals.finished.connect(lambda layers, task=task: self._on_detail(task, layers))
        task.signals.failed.connect(self._raise)
        self._detail_task = task
        self._detail_task_range = (mz_min, mz_max)
        QThreadPool.globalInstance().start(task)

    def _on_detail(self, task: _Task, layers: tuple[MapLayer, MapLayer | None] | None):
        if task is not self._detail_task or layers is None:
            return  # cancelled or superseded
        self._detail_task = None
        self._detail_layers = layers
        self.render()

    ### marker ###
    def update_marker(self):
        rt_window = self.data_stack_presenter.current_rt_window
        mz_window = self.data_stack_presenter.current_mz_window
        if rt_window is None or mz_window is None:
            self.window.set_marker(None)
            return
        self.window.set_marker(rt_window, mz_window)

    def on_marker_range_changed(self, rt_min: float, rt_max: float, mz_min: float, mz_max: float):
        if not self.data_stack_presenter.child_presenters:
            return
        # Like a map click, resizing navigates without editing a saved target.
        target_panel = self.target_panel_presenter
        target_panel.select_target(None)
        self.data_stack_presenter.set_rt_window((rt_min, rt_max))
        target_panel.on_plot_rt_window_changed(rt_min, rt_max)
        mz = None
        if self.data_stack_presenter.trace_type is TraceType.XIC:
            self.data_stack_presenter.set_mz_window((mz_min, mz_max))
            target_panel.on_plot_mz_window_changed(mz_min, mz_max)
            mz = (mz_min + mz_max) / 2
        self.data_stack_presenter.ensure_visible((rt_min + rt_max) / 2, mz)
        self.update_marker()

    @staticmethod
    def _raise(error: Exception):
        raise error


def fold_change_ticks(limit: float) -> list[tuple[float, str]]:
    """Color bar ticks (ln f, label) at fold changes f of B over A within [1 / limit, limit]:
    powers of 2, or of 10 if those would be too many."""
    base = 2 if math.log(limit, 2) < 5 else 10
    ticks = [(0.0, "×1")]
    fold = base
    while fold <= limit * (1 + 1e-9):
        ticks += [(math.log(fold), f"×{fold:g}"), (-math.log(fold), f"×1/{fold:g}")]
        fold *= base
    if len(ticks) == 1:  # limit < 2
        ticks += [(math.log(limit), f"×{limit:g}"), (-math.log(limit), f"×1/{limit:g}")]
    return sorted(ticks)


def intensity_ticks(scale: float, max_intensity: float, min_gap_fraction: float = 0.08) -> list[tuple[float, str]]:
    """Color bar ticks (asinh(x / scale), label) at round intensities x in [0, max_intensity].

    Powers of ten come first, then 2 and 5 times them, then (with more room
    required) the other multiples, so the ticks crowd neither the log-like top
    nor the linear-like bottom.
    """
    top = math.asinh(max_intensity / scale)
    min_gap = top * min_gap_fraction
    first_decade = math.floor(math.log10(scale * min_gap))  # smaller x all sit within min_gap of 0
    last_decade = math.floor(math.log10(max_intensity))
    ticks = {0.0: 0.0}  # position: intensity
    for mantissas, gap in (((1,), min_gap), ((2, 5), min_gap), ((3, 4, 6, 7, 8, 9), 1.5 * min_gap)):
        for decade in range(first_decade, last_decade + 1):
            for mantissa in mantissas:
                intensity = mantissa * 10.0**decade
                position = math.asinh(intensity / scale)
                if position <= top and all(abs(position - other) >= gap for other in ticks):
                    ticks[position] = intensity
    return [(position, _intensity_label(ticks[position])) for position in sorted(ticks)]


def _intensity_label(intensity: float) -> str:
    if intensity < 1e4:
        return f"{intensity:g}"
    mantissa, exponent = f"{intensity:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _covers(outer: tuple[float, float], inner: tuple[float, float]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]
