import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from RAPID2_Agilent.models.target import TraceType
from RAPID2_Agilent.services import operations
from RAPID2_Agilent.views.view_constants import DEFAULT_MZ_TOLERANCE, DEFAULT_MZ_WINDOW, DEFAULT_RT_TOLERANCE


class ChromatogramSpectrumPresenter(QObject):
    selected_rt_changed = pyqtSignal(float, float)
    selected_mz_window_changed = pyqtSignal(float, float)
    # emitted while a window is dragged on the plots; the selected_* signals follow at its end
    rt_window_dragged = pyqtSignal(float, float)
    mz_window_dragged = pyqtSignal(float, float)

    def __init__(self, view, model, parent=None):
        super().__init__(parent)
        self.view = view
        self.model = model
        self.current_scan_index = 0
        self.current_rt_window = (0.0, 0.0)
        self.current_mz_window = DEFAULT_MZ_WINDOW
        self.trace_type = TraceType.XIC
        self.ms_level: int | None = None  # None: all scans
        # with MS level 2, the precursor and CE of the scans used; None uses all MS2 scans
        self.ms2_selection: operations.MS2Selection | None = None
        self.polarity: int | None = None  # +1, -1: of the scans used, the sign of the adduct's charge; None, all
        # the last results drawn, kept while a plot is empty for want of scans (of the adduct's polarity, say), so
        # that turning back draws them again at once: (trace type, m/z window, scans, RT, intensity, misses) and
        # (RT window, scans, m/z, intensity, scan index)
        self._last_chromatogram: tuple | None = None
        self._last_spectrum: tuple | None = None
        self._updating_region = False
        self.no_peak = False  # the file has no peak of the target: its RT window hidden; see set_no_peak
        # of the targets selected, the RT windows in this file differ, and the m/z windows: those hidden
        self.rt_window_hidden = False
        self.mz_window_hidden = False

        self.view.chromatogram_window.view_box_clicked.connect(
            self.on_chromatogram_clicked
        )
        self.view.spectrum_window.view_box_clicked.connect(
            self.on_spectrum_clicked
        )
        self.view.chromatogram_window.region0.sigRegionChangeFinished.connect(
            self.on_chromatogram_region_changed
        )
        self.view.spectrum_window.region0.sigRegionChangeFinished.connect(
            self.on_spectrum_region_changed
        )
        self.view.chromatogram_window.region0.sigRegionChanged.connect(
            self.on_chromatogram_region_dragged
        )
        self.view.spectrum_window.region0.sigRegionChanged.connect(
            self.on_spectrum_region_dragged
        )

    def initialize(self, scan_index: int = 0, mz_window: tuple[float, float] = DEFAULT_MZ_WINDOW):
        self.current_scan_index = self._clamp_scan_index(scan_index)
        rt = float(self.model.RT_list[self.current_scan_index])
        self.current_rt_window = (rt - DEFAULT_RT_TOLERANCE, rt + DEFAULT_RT_TOLERANCE)
        self.current_mz_window = mz_window
        self._update_mz_region_visibility()
        self.update_chromatogram()
        self.update_spectrum_for_rt_window(self.current_rt_window)

    def update_chromatogram(self):
        scan_indices = self.scan_indices()
        mz_window = self.current_mz_window if self.trace_type is TraceType.XIC else None
        last = self._last_chromatogram
        if len(scan_indices) == 0:
            rt_list, inten_list, misses = np.array([]), np.array([]), None
        elif last is not None and last[:2] == (self.trace_type, mz_window) and np.array_equal(last[2], scan_indices):
            rt_list, inten_list, misses = last[3:]
        else:
            rt_list, inten_list = operations.extract_trace(
                self.model,
                self.trace_type,
                mz_window=self.current_mz_window,
                scan_indices=scan_indices,
            )
            # of an XIC, the scans whose profile data points miss the m/z window, left out as not measured there
            misses = None
            if self.trace_type is TraceType.XIC:
                misses = operations.mz_window_misses(self.model, self.current_mz_window, scan_indices)
            self._last_chromatogram = (self.trace_type, mz_window, scan_indices, rt_list, inten_list, misses)
        chromatogram_window = self.view.chromatogram_window
        chromatogram_window.set_primary_plot_data(rt_list, inten_list, missing=None if misses is None else misses.missing)
        chromatogram_window.set_warning_html(self._chromatogram_warning(misses))
        self._set_window_region(chromatogram_window, self.current_rt_window)
        self.show_mz_window_label(self.current_mz_window)

    def _chromatogram_warning(self, misses: operations.MzWindowMisses | None) -> str:
        """Why the chromatogram is empty or its XIC leaves scans out, if it is or does, and that it has no peak of the
        target, if so; "" otherwise."""
        if self.polarity is not None and not np.any(self.model.polarity_list == self.polarity):
            return f"No {operations.polarity_name(self.polarity)} scans in this file, for the adduct's charge"
        warnings = ["" if misses is None else misses.warning(), "No peak in the target's RT window" if self.no_peak else ""]
        return "<br>".join(warning for warning in warnings if warning)

    def set_no_peak(self, no_peak: bool):
        """Whether the file has no peak of the target: its RT window hidden then, and that said on the chromatogram.
        The RT window stays, the spectrum's; a click on the chromatogram sets one there."""
        if no_peak == self.no_peak:
            return
        self.no_peak = no_peak
        self._update_rt_region_visibility()
        self.update_chromatogram()

    def set_windows_hidden(self, rt: bool, mz: bool):
        """Hide the RT window, if rt, and the m/z window, if mz: of the targets selected, theirs differ. A click on the
        plot sets one there, for them all."""
        self.rt_window_hidden = rt
        self.mz_window_hidden = mz
        self._update_rt_region_visibility()
        self._update_mz_region_visibility()

    def _update_rt_region_visibility(self):
        self.view.chromatogram_window.region0.setVisible(not (self.no_peak or self.rt_window_hidden))

    def update_spectrum_for_rt_window(self, rt_window: tuple[float, float]):
        scan_indices = self.scan_indices()
        last = self._last_spectrum
        if last is not None and last[0] == rt_window and np.array_equal(last[1], scan_indices):
            mz_list, inten_list, scan_index = last[2:]
        else:
            mz_list, inten_list, scan_index = self.extract_spectrum_for_rt_window(rt_window)
            if len(mz_list):
                self._last_spectrum = (rt_window, scan_indices, mz_list, inten_list, scan_index)
        self.current_scan_index = scan_index
        spectrum_window = self.view.spectrum_window
        spectrum_window.set_primary_plot_data(mz_list, inten_list)
        spectrum_window.set_warning_html(
            "RT window beyond the scanned RT range" if self._rt_window_beyond_scans(rt_window, scan_indices) else ""
        )
        self._set_window_region(spectrum_window, self.current_mz_window)
        self.show_rt_window_label(rt_window)

    def show_mz_window_label(self, mz_window: tuple[float, float]):
        """The chromatogram's label: the m/z window of an XIC, the trace type otherwise; after the precursor for MS2."""
        if self.trace_type is TraceType.XIC:
            low_mz, high_mz = mz_window
            label = f"{low_mz:.4f} - {high_mz:.4f}"
        else:
            label = self.trace_type.value
        if self.ms_level == 2 and self.ms2_selection is not None:
            selection = self.ms2_selection
            label = f"{selection.precursor_mz:.4f} / CE {selection.collision_energy:g} &gt; {label}"
        self.view.chromatogram_window.set_top_right_label_html(label)

    def show_rt_window_label(self, rt_window: tuple[float, float]):
        """The spectrum's label: the RT window it averages."""
        self.view.spectrum_window.set_top_right_label_html(
            f"{rt_window[0]:.4f} - {rt_window[1]:.4f}"
        )

    def on_chromatogram_clicked(self, rt: float, intensity: float):
        rt_window = self.view.chromatogram_window.make_region_centered_at(
            rt,
            fallback_width=2 * DEFAULT_RT_TOLERANCE,
        )
        self.set_rt_window(rt_window, emit_signal=True)

    def on_spectrum_clicked(self, mz: float, intensity: float):
        if self.trace_type is not TraceType.XIC:
            return  # the m/z window is used only for XIC
        mz_window = self.view.spectrum_window.make_region_centered_at(
            mz,
            fallback_width=2 * DEFAULT_MZ_TOLERANCE,
        )
        self.set_mz_window(mz_window, emit_signal=True)

    def on_chromatogram_region_changed(self):
        if self._updating_region:
            return
        self.set_rt_window(self.view.chromatogram_window.get_region(), emit_signal=True)

    def on_spectrum_region_changed(self):
        if self._updating_region:
            return
        self.set_mz_window(self.view.spectrum_window.get_region(), emit_signal=True)

    def on_chromatogram_region_dragged(self):
        if self._updating_region:
            return
        self.rt_window_dragged.emit(*self.view.chromatogram_window.get_region())

    def on_spectrum_region_dragged(self):
        if self._updating_region:
            return
        self.mz_window_dragged.emit(*self.view.spectrum_window.get_region())

    def set_rt_window(self, rt_window: tuple[float, float], *, emit_signal: bool = False):
        self.current_rt_window = self._normalized_window(rt_window)
        self.update_spectrum_for_rt_window(self.current_rt_window)
        self._set_window_region(self.view.chromatogram_window, self.current_rt_window)
        if emit_signal:
            self._emit_rt_window_changed()

    def set_mz_window(self, mz_window: tuple[float, float], *, emit_signal: bool = False):
        self.current_mz_window = self._normalized_window(mz_window)
        self.update_chromatogram()
        self._set_window_region(self.view.spectrum_window, self.current_mz_window)
        if emit_signal:
            self._emit_mz_window_changed()

    def set_trace_type(self, trace_type: TraceType):
        self.trace_type = trace_type
        self._update_mz_region_visibility()
        self.update_chromatogram()

    def _update_mz_region_visibility(self):
        self.view.spectrum_window.region0.setVisible(self.trace_type is TraceType.XIC and not self.mz_window_hidden)

    def set_ms_level(self, ms_level: int | None):
        self.ms_level = ms_level
        self.update_chromatogram()
        self.update_spectrum_for_rt_window(self.current_rt_window)

    def set_ms2_selection(self, selection: operations.MS2Selection | None):
        self.ms2_selection = selection
        if self.ms_level == 2:
            self.update_chromatogram()
            self.update_spectrum_for_rt_window(self.current_rt_window)

    def set_polarity(self, polarity: int | None):
        used = self.scan_indices()
        self.polarity = polarity
        # a file of one polarity uses the same scans with its own as with none: nothing to draw again
        if len(used) and np.array_equal(self.scan_indices(), used):
            return
        self.update_chromatogram()
        self.update_spectrum_for_rt_window(self.current_rt_window)

    def scan_indices(self) -> np.ndarray:
        """The scans the plots are made from; empty when the file has none of the MS level or the MS2 selection, or
        of the polarity."""
        return operations.trace_scans(self.model, self.ms_level, ms2=self.ms2_selection, polarity=self.polarity)

    def extract_spectrum_for_rt_window(self, rt_window: tuple[float, float]):
        scan_indices = self.scan_indices()
        if len(scan_indices) == 0:
            return np.array([]), np.array([]), self.current_scan_index
        if self._rt_window_beyond_scans(rt_window, scan_indices):
            return np.array([]), np.array([]), self.current_scan_index
        return operations.averaged_spectrum(
            self.model,
            rt_window,
            mode="mean",
            scan_indices=scan_indices,
        )

    def _rt_window_beyond_scans(self, rt_window: tuple[float, float], scan_indices: np.ndarray) -> bool:
        """RT window beyond the scanned range."""
        rt_list = self.model.RT_list[scan_indices]
        return len(rt_list) > 0 and (rt_window[1] < rt_list.min() or rt_list.max() < rt_window[0])

    def _clamp_scan_index(self, scan_index: int) -> int:
        return max(0, min(scan_index, self.model.N_scans - 1))

    @staticmethod
    def _normalized_window(values: tuple[float, float]) -> tuple[float, float]:
        left, right = values
        if right < left:
            left, right = right, left
        return float(left), float(right)

    def _set_window_region(self, window, values: tuple[float, float]):
        self._updating_region = True
        try:
            window.region0.setRegion(values)
        finally:
            self._updating_region = False

    def _emit_rt_window_changed(self):
        self.selected_rt_changed.emit(float(self.current_rt_window[0]), float(self.current_rt_window[1]))

    def _emit_mz_window_changed(self):
        self.selected_mz_window_changed.emit(float(self.current_mz_window[0]), float(self.current_mz_window[1]))
