from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from RAPID2_Agilent.models.rpd_model import BaseMSDataModel
from RAPID2_Agilent.models.target_model import TraceType
from RAPID2_Agilent.views.view_constants import DEFAULT_MZ_WINDOW
from RAPID2_Agilent.views.data_view import ChromatogramSpectrumPresenter

from .chromatogram_spectrum_stack_view import ChromatogramSpectrumStackView


ViewRange = tuple[tuple[float, float], tuple[float, float]]  # (RT x range, m/z x range)


class ChromatogramSpectrumStackPresenter(QObject):
    # emitted when the windows are changed on the plots (not by the setters below)
    rt_window_changed = pyqtSignal(float, float)
    mz_window_changed = pyqtSignal(float, float)
    # emitted whenever current_rt_window / current_mz_window is set, by any path
    current_windows_changed = pyqtSignal()
    model_added = pyqtSignal(object)
    view_range_undo_state_changed = pyqtSignal(bool, bool)  # can_undo, can_redo
    view_range_record_delay_ms = 300

    def __init__(self, view: ChromatogramSpectrumStackView, parent=None):
        super().__init__(parent)
        self.view = view
        self.child_presenters: list[ChromatogramSpectrumPresenter] = []
        self.link_y_enabled = {"chromatogram_window": True, "spectrum_window": True}
        self.auto_y_enabled = {"chromatogram_window": True, "spectrum_window": True}
        self._current_rt_window: tuple[float, float] | None = None
        self._current_mz_window: tuple[float, float] | None = None
        self.trace_type = TraceType.XIC
        self.ms_level: int | None = None  # None: all scans
        self._syncing = False
        self._syncing_axis = False
        self._syncing_region_drag = False
        self._committed_view_range: ViewRange | None = None
        self._view_range_undo_stack: list[ViewRange] = []
        self._view_range_redo_stack: list[ViewRange] = []
        # coalesce continuous range changes (pan, zoom) into one history entry
        self._view_range_timer = QTimer(self)
        self._view_range_timer.setSingleShot(True)
        self._view_range_timer.setInterval(self.view_range_record_delay_ms)
        self._view_range_timer.timeout.connect(self.record_view_range)

    def add_model(self, model: BaseMSDataModel, *, scan_index: int = 0, mz_window: tuple[float, float] = DEFAULT_MZ_WINDOW):
        item = self.view.add_data_view()
        item.file_info_panel.set_file_info(model.file_path, model.get_scan_type_summary())
        presenter = ChromatogramSpectrumPresenter(item, model)
        presenter.trace_type = self.trace_type
        presenter.ms_level = self.ms_level
        presenter.initialize(scan_index=scan_index, mz_window=mz_window)
        presenter.selected_rt_changed.connect(
            lambda rt_min, rt_max, source=presenter: self.on_child_rt_changed(source, rt_min, rt_max)
        )
        presenter.selected_mz_window_changed.connect(
            lambda mz_min, mz_max, source=presenter: self.on_child_mz_window_changed(source, mz_min, mz_max)
        )
        item.chromatogram_window.vb().sigRangeChanged.connect(
            lambda view_box, ranges, source=presenter: self.on_child_chromatogram_range_changed(source, ranges)
        )
        item.spectrum_window.vb().sigRangeChanged.connect(
            lambda view_box, ranges, source=presenter: self.on_child_spectrum_range_changed(source, ranges)
        )
        item.chromatogram_window.region0.sigRegionChanged.connect(
            lambda region, source=presenter: self.on_child_chromatogram_region_dragged(source)
        )
        item.spectrum_window.region0.sigRegionChanged.connect(
            lambda region, source=presenter: self.on_child_spectrum_region_dragged(source)
        )
        item.chromatogram_window.vb().sigXRangeChanged.connect(lambda *_: self._view_range_timer.start())
        item.spectrum_window.vb().sigXRangeChanged.connect(lambda *_: self._view_range_timer.start())
        self.child_presenters.append(presenter)
        for window_name in self.link_y_enabled:
            window = getattr(item, window_name)
            window.set_auto_y_range_enabled(self.auto_y_enabled[window_name])
            window.y_range_bounds_provider = (
                lambda x_min, x_max, index=0, name=window_name: self._get_y_bounds(
                    presenter, name, x_min, x_max, index=index
                )
            )
            first = self.child_presenters[0]
            first_window = getattr(first.view, window_name)
            first_window.apply_y_range_policy()
            self._sync_plot_ranges(first, first_window.viewRange(), window_name)
        if self.current_rt_window is None:
            self.current_rt_window = presenter.current_rt_window
            self.rt_window_changed.emit(*self.current_rt_window)
        else:
            presenter.set_rt_window(self.current_rt_window, emit_signal=False)
        if self.current_mz_window is None:
            self.current_mz_window = presenter.current_mz_window
            self.mz_window_changed.emit(*self.current_mz_window)
        else:
            presenter.set_mz_window(self.current_mz_window, emit_signal=False)
        self.model_added.emit(model)
        return presenter

    @property
    def current_rt_window(self) -> tuple[float, float] | None:
        return self._current_rt_window

    @current_rt_window.setter
    def current_rt_window(self, rt_window: tuple[float, float] | None):
        self._current_rt_window = rt_window
        self.current_windows_changed.emit()

    @property
    def current_mz_window(self) -> tuple[float, float] | None:
        return self._current_mz_window

    @current_mz_window.setter
    def current_mz_window(self, mz_window: tuple[float, float] | None):
        self._current_mz_window = mz_window
        self.current_windows_changed.emit()

    @property
    def models(self) -> list[BaseMSDataModel]:
        return [presenter.model for presenter in self.child_presenters]

    def ensure_visible(self, rt: float, mz: float | None):
        """Pan the plots (keeping the zoom) so that rt and mz are within their x ranges."""
        first_view = self.child_presenters[0].view
        targets = [(first_view.chromatogram_window, rt)]
        if mz is not None:
            targets.append((first_view.spectrum_window, mz))
        for window, value in targets:
            x_min, x_max = window.plotItem.viewRange_x()
            if not x_min <= value <= x_max:
                half_width = (x_max - x_min) / 2
                window.plotItem.setXRange(value - half_width, value + half_width, padding=0)

    def clear(self):
        self.child_presenters.clear()
        self.current_rt_window = None
        self.current_mz_window = None
        self.view.clear_data_views()
        self._view_range_timer.stop()
        self._committed_view_range = None
        self._view_range_undo_stack.clear()
        self._view_range_redo_stack.clear()
        self._emit_view_range_undo_state()

    def set_rt_window(self, rt_window: tuple[float, float]):
        rt_window = (float(rt_window[0]), float(rt_window[1]))
        if rt_window == self.current_rt_window:
            return
        self.current_rt_window = rt_window
        for presenter in self.child_presenters:
            presenter.set_rt_window(rt_window, emit_signal=False)

    def set_mz_window(self, mz_window: tuple[float, float]):
        mz_window = (float(mz_window[0]), float(mz_window[1]))
        if mz_window == self.current_mz_window:
            return
        self.current_mz_window = mz_window
        for presenter in self.child_presenters:
            presenter.set_mz_window(mz_window, emit_signal=False)

    def set_trace_type(self, trace_type: TraceType):
        if trace_type is self.trace_type:
            return
        self.trace_type = trace_type
        for presenter in self.child_presenters:
            presenter.set_trace_type(trace_type)

    def set_ms_level(self, ms_level: int | None):
        if ms_level == self.ms_level:
            return
        self.ms_level = ms_level
        for presenter in self.child_presenters:
            presenter.set_ms_level(ms_level)

    def set_link_y_enabled(self, window_name: str, enabled: bool):
        self.link_y_enabled[window_name] = enabled
        for presenter in self.child_presenters:
            getattr(presenter.view, window_name).apply_y_range_policy()
        if enabled and self.child_presenters:
            first = self.child_presenters[0]
            self._sync_plot_ranges(first, getattr(first.view, window_name).viewRange(), window_name)

    def set_auto_y_enabled(self, window_name: str, enabled: bool):
        self.auto_y_enabled[window_name] = enabled
        for presenter in self.child_presenters:
            getattr(presenter.view, window_name).set_auto_y_range_enabled(enabled)
        for presenter in self.child_presenters:
            getattr(presenter.view, window_name).apply_y_range_policy()

    def reset_x_range(self, window_name: str):
        windows = [getattr(presenter.view, window_name) for presenter in self.child_presenters]
        bounds = [window.get_x_bounds() for window in windows if window.plot_items[0].xData is not None]
        if bounds:
            windows[0].plotItem.setMyXRange(
                min(x_min for x_min, _ in bounds), max(x_max for _, x_max in bounds)
            )

    def _get_y_bounds(self, source, window_name, x_min, x_max, *, index=0):
        presenters = self.child_presenters if self.link_y_enabled[window_name] else [source]
        bounds = [
            getattr(presenter.view, window_name).get_displayed_y_bounds(x_min, x_max, index=index)
            for presenter in presenters
        ]
        bounds = [(y_min, y_max) for y_min, y_max in bounds if y_min is not None]
        if not bounds:
            return None, None
        return min(y_min for y_min, _ in bounds), max(y_max for _, y_max in bounds)

    ### view range history ###
    def can_undo_view_range(self) -> bool:
        return bool(self._view_range_undo_stack)

    def can_redo_view_range(self) -> bool:
        return bool(self._view_range_redo_stack)

    def record_view_range(self):
        self._view_range_timer.stop()
        view_range = self._current_view_range()
        if self._committed_view_range is not None and view_range != self._committed_view_range:
            self._view_range_undo_stack.append(self._committed_view_range)
            self._view_range_redo_stack.clear()
            self._emit_view_range_undo_state()
        self._committed_view_range = view_range

    def undo_view_range(self):
        if self._view_range_timer.isActive():
            self.record_view_range()
        view_range = self._view_range_undo_stack.pop()
        self._view_range_redo_stack.append(self._committed_view_range)
        self._apply_view_range(view_range)

    def redo_view_range(self):
        if self._view_range_timer.isActive():
            self.record_view_range()
        view_range = self._view_range_redo_stack.pop()
        self._view_range_undo_stack.append(self._committed_view_range)
        self._apply_view_range(view_range)

    def _current_view_range(self) -> ViewRange:
        first_view = self.child_presenters[0].view
        rt_range = first_view.chromatogram_window.plotItem.viewRange_x()
        mz_range = first_view.spectrum_window.plotItem.viewRange_x()
        return (float(rt_range[0]), float(rt_range[1])), (float(mz_range[0]), float(mz_range[1]))

    def _apply_view_range(self, view_range: ViewRange):
        rt_range, mz_range = view_range
        first_view = self.child_presenters[0].view
        first_view.chromatogram_window.plotItem.setXRange(*rt_range, padding=0)
        first_view.spectrum_window.plotItem.setXRange(*mz_range, padding=0)
        self._view_range_timer.stop()
        self._committed_view_range = self._current_view_range()
        self._emit_view_range_undo_state()

    def _emit_view_range_undo_state(self):
        self.view_range_undo_state_changed.emit(self.can_undo_view_range(), self.can_redo_view_range())

    def on_child_rt_changed(self, source: ChromatogramSpectrumPresenter, rt_min: float, rt_max: float):
        if self._syncing:
            return
        self._syncing = True
        try:
            rt_window = (float(rt_min), float(rt_max))
            self.current_rt_window = rt_window
            for presenter in self.child_presenters:
                if presenter is not source:
                    presenter.set_rt_window(rt_window, emit_signal=False)
        finally:
            self._syncing = False
        self.rt_window_changed.emit(*rt_window)

    def on_child_mz_window_changed(self, source: ChromatogramSpectrumPresenter, mz_min: float, mz_max: float):
        if self._syncing:
            return
        self._syncing = True
        try:
            mz_window = (float(mz_min), float(mz_max))
            self.current_mz_window = mz_window
            for presenter in self.child_presenters:
                if presenter is not source:
                    presenter.set_mz_window(mz_window, emit_signal=False)
        finally:
            self._syncing = False
        self.mz_window_changed.emit(*mz_window)

    def on_child_chromatogram_range_changed(self, source: ChromatogramSpectrumPresenter, ranges):
        self._sync_plot_ranges(
            source,
            ranges,
            "chromatogram_window",
        )

    def on_child_spectrum_range_changed(self, source: ChromatogramSpectrumPresenter, ranges):
        self._sync_plot_ranges(
            source,
            ranges,
            "spectrum_window",
        )

    def on_child_chromatogram_region_dragged(self, source: ChromatogramSpectrumPresenter):
        self._sync_regions_during_drag(
            source,
            source.view.chromatogram_window.get_region(),
            lambda presenter: presenter.view.chromatogram_window,
        )

    def on_child_spectrum_region_dragged(self, source: ChromatogramSpectrumPresenter):
        self._sync_regions_during_drag(
            source,
            source.view.spectrum_window.get_region(),
            lambda presenter: presenter.view.spectrum_window,
        )

    def _sync_plot_ranges(self, source: ChromatogramSpectrumPresenter, ranges, window_name):
        if self._syncing_axis:
            return
        x_range, y_range = ranges
        self._syncing_axis = True
        try:
            for presenter in self.child_presenters:
                if presenter is source:
                    continue
                window = getattr(presenter.view, window_name)
                window.plotItem.setRange(
                    xRange=(float(x_range[0]), float(x_range[1])),
                    yRange=(float(y_range[0]), float(y_range[1])) if self.link_y_enabled[window_name] else None,
                    padding=0,
                )
        finally:
            self._syncing_axis = False

    def _sync_regions_during_drag(self, source: ChromatogramSpectrumPresenter, region, window_getter):
        if self._syncing_region_drag or self._syncing:
            return
        self._syncing_region_drag = True
        try:
            for presenter in self.child_presenters:
                if presenter is source:
                    continue
                target_region = window_getter(presenter).region0
                target_region.blockSignals(True)
                try:
                    target_region.setRegion(region)
                finally:
                    target_region.blockSignals(False)
        finally:
            self._syncing_region_drag = False
