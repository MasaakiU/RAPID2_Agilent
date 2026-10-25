from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from RAPID2_Agilent.models.rpd_model import BaseMSDataModel
from RAPID2_Agilent.models.target import TraceType
from RAPID2_Agilent.services import operations
from RAPID2_Agilent.views.view_constants import (
    BROWSING_RANGE_LABEL_COLOR,
    BROWSING_REGION_COLOR,
    DEFAULT_MZ_WINDOW,
    OWN_RT_RANGE_LABEL_COLOR,
    OWN_RT_REGION_COLOR,
    TARGET_RANGE_LABEL_COLOR,
    TARGET_REGION_COLOR,
    ZOOM_MZ_PADDING,
    ZOOM_RT_PADDING_RATIO,
)
from RAPID2_Agilent.views.data_view import ChromatogramSpectrumPresenter

from .chromatogram_spectrum_stack_view import ChromatogramSpectrumStackView


ViewRange = tuple[tuple[float, float], tuple[float, float]]  # (RT x range, m/z x range)


class ChromatogramSpectrumStackPresenter(QObject):
    # emitted when the windows are changed on the plots (not by the setters below)
    rt_window_changed = pyqtSignal(float, float)
    mz_window_changed = pyqtSignal(float, float)
    # emitted while a window is dragged on the plots, before the signals above at its end
    rt_window_dragged = pyqtSignal(float, float)
    mz_window_dragged = pyqtSignal(float, float)
    # while a target is edited, its RT windows not linked, each file's RT window is its own, the target's peak in it:
    # one changed on the plots is not passed on to the others, and is signalled with the file's model in place of the
    # RT signals above
    run_rt_window_changed = pyqtSignal(object, float, float)
    run_no_peak_requested = pyqtSignal(object)  # then, by the menu of a file's RT window, with its model
    # emitted whenever current_rt_window / current_mz_window is set, by any path
    current_windows_changed = pyqtSignal()
    model_added = pyqtSignal(object)
    models_changed = pyqtSignal()  # files reordered or closed
    view_range_undo_state_changed = pyqtSignal(bool, bool)  # can_undo, can_redo
    view_range_record_delay_ms = 300

    def __init__(self, view: ChromatogramSpectrumStackView, parent=None):
        super().__init__(parent)
        self.view = view
        self.child_presenters: list[ChromatogramSpectrumPresenter] = []
        self.editing_target = False  # whether the RT / m/z windows edit a target
        self.link_rt_windows = True  # whether a target's RT window is one for all files; see run_rt_window_changed
        self.link_y_enabled = {"chromatogram_window": True, "spectrum_window": True}
        self.auto_y_enabled = {"chromatogram_window": True, "spectrum_window": True}
        self._current_rt_window: tuple[float, float] | None = None
        self._current_mz_window: tuple[float, float] | None = None
        self.trace_type = TraceType.XIC
        self.ms_level: int | None = None  # None: all scans
        self.ms2_selection: operations.MS2Selection | None = None  # see ChromatogramSpectrumPresenter
        self.polarity: int | None = None  # see ChromatogramSpectrumPresenter
        # of several targets selected: the files whose RT windows are hidden, whether the m/z windows are, and the
        # notice of the one shown; see set_windows_hidden, set_notice
        self._rt_hidden_models: set[BaseMSDataModel] = set()
        self._mz_window_hidden = False
        self._notice = ""
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
        presenter.ms2_selection = self.ms2_selection
        presenter.polarity = self.polarity
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
        presenter.rt_window_dragged.connect(
            lambda rt_min, rt_max, source=presenter: self.on_child_chromatogram_region_dragged(source)
        )
        presenter.mz_window_dragged.connect(
            lambda mz_min, mz_max, source=presenter: self.on_child_spectrum_region_dragged(source)
        )
        item.chromatogram_window.no_peak_requested.connect(
            lambda source=presenter: self.run_no_peak_requested.emit(source.model)
        )
        item.chromatogram_window.vb().sigXRangeChanged.connect(lambda *_: self._view_range_timer.start())
        item.spectrum_window.vb().sigXRangeChanged.connect(lambda *_: self._view_range_timer.start())
        self.child_presenters.append(presenter)
        self._style_windows(item)
        presenter.set_windows_hidden(model in self._rt_hidden_models, self._mz_window_hidden)
        self._show_notice(item)
        for window_name in self.link_y_enabled:
            window = getattr(item, window_name)
            window.cursor_moved.connect(
                lambda x, y, source=window, name=window_name: self._sync_cursor(name, source, x, y)
            )
            window.cursor_left.connect(lambda name=window_name: self._clear_cursors(name))
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

    ### reordering and closing the selected files ###
    @staticmethod
    def _is_selected(presenter: ChromatogramSpectrumPresenter) -> bool:
        return presenter.view.file_info_panel.is_selected()

    def move_selected_to_top(self):
        self._reorder(sorted(self.child_presenters, key=lambda presenter: not self._is_selected(presenter)))

    def move_selected_to_bottom(self):
        self._reorder(sorted(self.child_presenters, key=self._is_selected))

    def move_selected_up(self):
        """Each selected file one row up, past an unselected one; those at the top stay, with the ones right below."""
        order = list(self.child_presenters)
        for index in range(1, len(order)):
            if self._is_selected(order[index]) and not self._is_selected(order[index - 1]):
                order[index - 1], order[index] = order[index], order[index - 1]
        self._reorder(order)

    def move_selected_down(self):
        order = list(self.child_presenters)
        for index in range(len(order) - 2, -1, -1):
            if self._is_selected(order[index]) and not self._is_selected(order[index + 1]):
                order[index], order[index + 1] = order[index + 1], order[index]
        self._reorder(order)

    def _reorder(self, order: list[ChromatogramSpectrumPresenter]):
        if order == self.child_presenters:
            return
        # the moved files pass over the others
        with self.view.animated_row_changes(on_top=self.view.selected_items):
            self.child_presenters[:] = order
            self.view.set_item_order([presenter.view for presenter in order])
        self.models_changed.emit()

    def close_selected(self) -> list[BaseMSDataModel]:
        """Close the selected files; the RT / m/z windows stay as they are, also once the last is closed.

        The file right below the closed ones is selected then, or the bottom one when none is below, so that
        closing again closes the next.
        """
        closed = [presenter for presenter in self.child_presenters if self._is_selected(presenter)]
        if not closed:
            return []
        below = self.child_presenters[self.child_presenters.index(closed[-1]) + 1:]
        with self.view.animated_row_changes():
            for presenter in closed:
                self.child_presenters.remove(presenter)
                self.view.remove_data_view(presenter.view)
            if self.child_presenters:
                self.view.select_item((below[0] if below else self.child_presenters[-1]).view)
                # the linked Y ranges were pooled over the closed files as well
                for window_name in self.link_y_enabled:
                    for presenter in self.child_presenters:
                        getattr(presenter.view, window_name).apply_y_range_policy()
        if not self.child_presenters:
            # the history is of the plots' ranges, and no plot is left to take them
            self._clear_view_range_history()
        self.models_changed.emit()
        return [presenter.model for presenter in closed]

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
        self._clear_view_range_history()

    def _clear_view_range_history(self):
        self._view_range_timer.stop()
        self._committed_view_range = None
        self._view_range_undo_stack.clear()
        self._view_range_redo_stack.clear()
        self._emit_view_range_undo_state()

    def set_rt_window(
        self,
        rt_window: tuple[float, float],
        run_windows: dict[BaseMSDataModel, tuple[float, float] | None] | None = None,
    ):
        """Set the RT window of the files, but of those in run_windows, which get their own (a target's peak in them),
        or, for None, no peak: rt_window hidden (ChromatogramSpectrumPresenter.set_no_peak)."""
        rt_window = (float(rt_window[0]), float(rt_window[1]))
        run_windows = run_windows or {}
        if rt_window != self.current_rt_window:
            self.current_rt_window = rt_window
        for presenter in self.child_presenters:
            window = run_windows.get(presenter.model, rt_window)
            presenter.set_no_peak(window is None)
            window = rt_window if window is None else window
            if presenter.current_rt_window != (float(window[0]), float(window[1])):
                presenter.set_rt_window(window, emit_signal=False)

    def set_mz_window(self, mz_window: tuple[float, float]):
        mz_window = (float(mz_window[0]), float(mz_window[1]))
        if mz_window == self.current_mz_window:
            return
        self.current_mz_window = mz_window
        for presenter in self.child_presenters:
            presenter.set_mz_window(mz_window, emit_signal=False)

    @property
    def own_rt_windows(self) -> bool:
        """Whether each file has an RT window of its own: while a target is edited, its RT windows not linked."""
        return self.editing_target and not self.link_rt_windows

    def set_editing_target(self, editing: bool):
        """Colour the RT / m/z windows by whether dragging them edits a target."""
        self.editing_target = editing
        self._apply_window_state()

    def set_link_rt_windows(self, linked: bool):
        self.link_rt_windows = linked
        self._apply_window_state()

    def _apply_window_state(self):
        """Colour the windows by what they edit; RT windows no longer the files' own are joined, their peaks dropped."""
        for presenter in self.child_presenters:
            self._style_windows(presenter.view)
        if not self.own_rt_windows:
            # the files' own RT windows give way to the one of them all, and the peaks found in them go
            if self.current_rt_window is not None:
                self.set_rt_window(self.current_rt_window)
            self.set_peak_areas({})

    def set_windows_hidden(self, rt_models=(), *, mz: bool = False):
        """Hide the RT windows of the files of rt_models, and the m/z windows if mz: of the targets selected, theirs
        differ there."""
        if (set(rt_models), mz) == (self._rt_hidden_models, self._mz_window_hidden):
            return
        self._rt_hidden_models = set(rt_models)
        self._mz_window_hidden = mz
        for presenter in self.child_presenters:
            presenter.set_windows_hidden(presenter.model in self._rt_hidden_models, mz)

    def set_notice(self, text: str):
        """A notice in red in the middle of every plot, of which data they show; "" for none."""
        if text == self._notice:
            return
        self._notice = text
        for presenter in self.child_presenters:
            self._show_notice(presenter.view)

    def _show_notice(self, item):
        item.chromatogram_window.set_notice_html(self._notice)
        item.spectrum_window.set_notice_html(self._notice)

    def set_peak_areas(self, peak_areas: dict[BaseMSDataModel, list[tuple[float, float]]]):
        """Shade the peaks (start, end) found in each file of peak_areas under its chromatogram; none in the others."""
        for presenter in self.child_presenters:
            presenter.view.chromatogram_window.set_peak_areas(peak_areas.get(presenter.model, ()))

    def _style_windows(self, item):
        """Colour the windows by what they edit; the RT window has its menu while it is the file's own."""
        item.chromatogram_window.set_rt_window_menu_enabled(self.own_rt_windows)
        color = TARGET_REGION_COLOR if self.editing_target else BROWSING_REGION_COLOR
        label_color = TARGET_RANGE_LABEL_COLOR if self.editing_target else BROWSING_RANGE_LABEL_COLOR
        # the RT window on the chromatogram, its range written on the spectrum; red while it is the file's own
        rt_color, rt_label_color = (OWN_RT_REGION_COLOR, OWN_RT_RANGE_LABEL_COLOR) if self.own_rt_windows else (
            color, label_color
        )
        item.chromatogram_window.set_region_color(rt_color)
        item.spectrum_window.set_range_label_color(rt_label_color)
        item.spectrum_window.set_region_color(color)
        item.chromatogram_window.set_range_label_color(label_color)

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

    def set_ms2_selection(self, selection: operations.MS2Selection | None):
        if selection == self.ms2_selection:
            return
        self.ms2_selection = selection
        for presenter in self.child_presenters:
            presenter.set_ms2_selection(selection)

    def set_polarity(self, polarity: int | None):
        if polarity == self.polarity:
            return
        self.polarity = polarity
        for presenter in self.child_presenters:
            presenter.set_polarity(polarity)

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
        bounds = [window.get_x_bounds() for window in windows if window.plot_items[0].xData is not None and len(window.plot_items[0].xData)]
        if bounds:
            windows[0].plotItem.setMyXRange(
                min(x_min for x_min, _ in bounds), max(x_max for _, x_max in bounds)
            )

    def zoom_to_window(self, window_name: str):
        """Show the RT / m/z window with room on each side (see ZOOM_RT_PADDING_RATIO, ZOOM_MZ_PADDING)."""
        if window_name == "chromatogram_window":
            window = self.current_rt_window
            padding = None if window is None else (window[1] - window[0]) * ZOOM_RT_PADDING_RATIO
        else:
            window = self.current_mz_window
            padding = ZOOM_MZ_PADDING
        if window is None or not self.child_presenters:
            return
        first = getattr(self.child_presenters[0].view, window_name)
        first.plotItem.setXRange(window[0] - padding, window[1] + padding, padding=0)

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
        if self.own_rt_windows:
            self.run_rt_window_changed.emit(source.model, float(rt_min), float(rt_max))
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
        rt_window = source.view.chromatogram_window.get_region()
        if self.own_rt_windows:
            source.show_rt_window_label(rt_window)  # the file's own
            return
        self._sync_regions_during_drag(
            source,
            rt_window,
            lambda presenter: presenter.view.chromatogram_window,
        )
        for presenter in self.child_presenters:
            presenter.show_rt_window_label(rt_window)
        self.rt_window_dragged.emit(*rt_window)

    def on_child_spectrum_region_dragged(self, source: ChromatogramSpectrumPresenter):
        mz_window = source.view.spectrum_window.get_region()
        self._sync_regions_during_drag(
            source,
            mz_window,
            lambda presenter: presenter.view.spectrum_window,
        )
        for presenter in self.child_presenters:
            presenter.show_mz_window_label(mz_window)
        self.mz_window_dragged.emit(*mz_window)

    def _sync_cursor(self, window_name: str, source, x_value: float, y_value: float):
        for presenter in self.child_presenters:
            window = getattr(presenter.view, window_name)
            if window is not source:
                window.set_shared_cursor(x_value, y_value)

    def _clear_cursors(self, window_name: str):
        for presenter in self.child_presenters:
            getattr(presenter.view, window_name).clear_cursor()

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
