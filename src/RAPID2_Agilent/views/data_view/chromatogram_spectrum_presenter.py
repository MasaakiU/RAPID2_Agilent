from PyQt6.QtCore import QObject, pyqtSignal

from RAPID2_Agilent.models.target_model import TraceType
from RAPID2_Agilent.views.view_constants import DEFAULT_MZ_TOLERANCE, DEFAULT_MZ_WINDOW, DEFAULT_RT_TOLERANCE


class ChromatogramSpectrumPresenter(QObject):
    selected_rt_changed = pyqtSignal(float, float)
    selected_mz_window_changed = pyqtSignal(float, float)

    def __init__(self, view, model, parent=None):
        super().__init__(parent)
        self.view = view
        self.model = model
        self.current_scan_index = 0
        self.current_rt_window = (0.0, 0.0)
        self.current_mz_window = DEFAULT_MZ_WINDOW
        self.trace_type = TraceType.XIC
        self.ms_level: int | None = None  # None: all scans
        self._updating_region = False

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

    def initialize(self, scan_index: int = 0, mz_window: tuple[float, float] = DEFAULT_MZ_WINDOW):
        self.current_scan_index = self._clamp_scan_index(scan_index)
        rt = float(self.model.RT_list[self.current_scan_index])
        self.current_rt_window = (rt - DEFAULT_RT_TOLERANCE, rt + DEFAULT_RT_TOLERANCE)
        self.current_mz_window = mz_window
        self._update_mz_region_visibility()
        self.update_chromatogram()
        self.update_spectrum_for_rt_window(self.current_rt_window)

    def update_chromatogram(self):
        low_mz, high_mz = self.current_mz_window
        scan_indices = self.model.scan_indices_for_ms_level(self.ms_level)
        if self.trace_type is TraceType.XIC:
            rt_list, inten_list = self.model.extract_chromatogram(low_mz, high_mz, scan_indices=scan_indices)
            label = f"{low_mz:.4f} - {high_mz:.4f}"
        elif self.trace_type is TraceType.TIC:
            rt_list, inten_list = self.model.extract_tic(scan_indices=scan_indices)
            label = "TIC"
        elif self.trace_type is TraceType.BPC:
            rt_list, inten_list = self.model.extract_bpc(scan_indices=scan_indices)
            label = "BPC"
        else:
            raise ValueError(f"Unknown trace type: {self.trace_type!r}")
        chromatogram_window = self.view.chromatogram_window
        chromatogram_window.set_primary_plot_data(rt_list, inten_list)
        self._set_window_region(chromatogram_window, self.current_rt_window)
        chromatogram_window.set_top_right_label_html(label)

    def update_spectrum_for_rt_window(self, rt_window: tuple[float, float]):
        mz_list, inten_list, scan_index = self.extract_spectrum_for_rt_window(rt_window)
        self.current_scan_index = scan_index
        spectrum_window = self.view.spectrum_window
        spectrum_window.set_primary_plot_data(mz_list, inten_list)
        self._set_window_region(spectrum_window, self.current_mz_window)
        spectrum_window.set_top_right_label_html(
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
        self.view.spectrum_window.region0.setVisible(self.trace_type is TraceType.XIC)

    def set_ms_level(self, ms_level: int | None):
        self.ms_level = ms_level
        self.update_chromatogram()
        self.update_spectrum_for_rt_window(self.current_rt_window)

    def extract_spectrum_for_rt_window(self, rt_window: tuple[float, float]):
        return self.model.extract_averaged_spectrum(
            rt_window[0],
            rt_window[1],
            mode="mean",
            scan_indices=self.model.scan_indices_for_ms_level(self.ms_level),
        )

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
