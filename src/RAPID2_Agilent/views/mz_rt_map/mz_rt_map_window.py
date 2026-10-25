import math

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from RAPID2_Agilent.views.data_view import MyViewBox
from RAPID2_Agilent.views.view_constants import BROWSING_MARKER_COLOR

from RAPID2_Agilent.views.layout_metrics import SMALL
from .diagnostics import Diagnostics, DiagnosticsPlot


class MzRtMapWindow(QWidget):
    """Floating m/z (vertical) x RT (horizontal) heat map of file A, or of B - A."""

    file_selected = pyqtSignal(int)
    compare_file_selected = pyqtSignal(int)  # index of B in the files, -1: none
    diff_settings_changed = pyqtSignal()     # normalization or tolerances
    fold_change_limit_changed = pyqtSignal(float)
    diagnostics_changed = pyqtSignal()
    scale_changed = pyqtSignal(float)
    view_range_changed = pyqtSignal()   # after panning / zooming settles
    clicked = pyqtSignal(float, float)  # RT, m/z
    marker_range_changed = pyqtSignal(float, float, float, float)  # RT min/max, m/z min/max
    view_range_settle_ms = 100
    marker_hover_color = "#ffffff"
    scale_slider_steps = 1000  # over log10(s)
    color_maps = {  # shown name: pyqtgraph name
        "inferno": "inferno",
        "magma": "magma",
        "plasma": "plasma",
        "viridis": "viridis",
        "cividis": "cividis",
        "turbo": "turbo",
        "rainbow": "CET-R4",
        "hot": "CET-L3",
        "gray": "CET-L1",
    }
    diff_color_maps = {  # B higher at the top
        "blue-white-red": "CET-D1A",
        "blue-black-red": "CET-D4",
        "green-white-violet": "CET-D2",
        "blue-yellow": "CET-CBD1",  # color-blind safe
    }
    no_compare_text = "—"
    left_axis_width = 60
    diagnostics_styles = {  # A blue and B red, as on the B - A map
        "a": {"pen": pg.mkPen("#2166ac", width=1.5)},
        "b": {"pen": pg.mkPen("#b2182b", width=1.5)},
        "b_original": {"pen": pg.mkPen("#b2182b", width=1, style=Qt.PenStyle.DashLine)},
        "main": {"pen": pg.mkPen("#222222", width=1.5)},
        "path": {"pen": pg.mkPen("#ff3030", width=1.5)},
        "anchors": {"pen": None, "symbol": "o", "symbolSize": 3, "symbolPen": None, "symbolBrush": (0, 184, 212, 160)},
        "guide": {"pen": pg.mkPen("#999999", width=1, style=Qt.PenStyle.DashLine)},
    }
    diagnostics_color_map = "CET-L1"  # gray, under the red path
    color_map_icon_size = QSize(48, 12)
    control_panel_width = 200

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.setObjectName("MzRtMapWindow")
        self.setWindowTitle("m/z-RT Map")
        self.resize(1100, 600)

        self.file_combo = _file_combo_box()
        self.file_combo.setToolTip("Reference")
        self.compare_combo = _file_combo_box()
        self.compare_combo.setToolTip("Choose a file to show its difference from A, after aligning it to A in RT")
        self.scale_spin_box = QDoubleSpinBox()
        self.scale_spin_box.setDecimals(1)
        self.scale_spin_box.setRange(0.1, 1e12)
        self.scale_spin_box.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self.scale_spin_box.setKeyboardTracking(False)
        self.scale_spin_box.setToolTip("Intensity x is shown as asinh(x / s)")
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(0, self.scale_slider_steps)
        self.scale_slider.setMinimumWidth(120)
        self.scale_slider.setToolTip("s on a log scale: left is log-like, right is linear-like")
        self._log_scale_range = (0.0, 1.0)  # log10(s) at the slider ends
        self.color_map_combo = QComboBox()
        self.color_map_combo.setIconSize(self.color_map_icon_size)
        self._diff_mode = False
        self._color_map_choices = {False: next(iter(self.color_maps)), True: next(iter(self.diff_color_maps))}
        self.fold_change_spin_box = QDoubleSpinBox()
        self.fold_change_spin_box.setPrefix("×")
        self.fold_change_spin_box.setDecimals(1)
        self.fold_change_spin_box.setRange(1.1, 1e4)
        self.fold_change_spin_box.setValue(8.0)
        self.fold_change_spin_box.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self.fold_change_spin_box.setKeyboardTracking(False)
        self.fold_change_spin_box.setToolTip("Colors saturate at this fold change of B over A (or A over B)")
        self.normalize_check_box = QCheckBox("Normalize intensity")
        self.normalize_check_box.setChecked(True)
        self.normalize_check_box.setToolTip("Multiply B by the median of A / B over the bins where both are strong")
        self.rt_tolerance_spin_box = QSpinBox()
        self.rt_tolerance_spin_box.setRange(0, 10)
        self.rt_tolerance_spin_box.setValue(1)
        self.rt_tolerance_spin_box.setSuffix(" scans")
        self.mz_tolerance_spin_box = QSpinBox()
        self.mz_tolerance_spin_box.setRange(0, 200)
        self.mz_tolerance_spin_box.setValue(10)
        self.mz_tolerance_spin_box.setSuffix(" ppm")
        self.diagnostics_combo = QComboBox()
        self.diagnostics_combo.addItems([diagnostics.value for diagnostics in Diagnostics])
        self.diagnostics_combo.setToolTip("Shown to the right of the map, with its RT range linked")
        for spin_box in (self.rt_tolerance_spin_box, self.mz_tolerance_spin_box):
            spin_box.setKeyboardTracking(False)
            spin_box.setToolTip(
                "Each file is compared with the maximum of the other within this distance (each side), "
                "so that slightly shifted peaks do not show as differences"
            )
        self.show_marker_check_box = QCheckBox("Marker")
        self.show_marker_check_box.setChecked(True)
        self.show_marker_check_box.setToolTip("Drag the rectangle to move the RT/m/z selection; drag its eight handles to resize it")
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)

        self.view_box = MyViewBox()
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box)
        self.plot_widget.setBackground("w")
        self.plot_widget.setLabel("bottom", "RT")
        self.plot_widget.setLabel("left", "m/z")
        self.plot_widget.getAxis("left").setWidth(self.left_axis_width)
        self.image_item = pg.ImageItem(axisOrder="col-major")  # image[rt_pixel, mz_pixel]
        self.color_bar = pg.ColorBarItem(values=(0.0, 1.0), width=15, colorMap=pg.colormap.get(self.color_map_name()), interactive=False)
        self.color_bar.axis.setLabel("Intensity")
        self.marker = pg.ROI(
            pos=(0, 0),
            size=(1, 1),
            rotatable=False,
            pen=pg.mkPen(BROWSING_MARKER_COLOR, width=2),
            hoverPen=pg.mkPen(self.marker_hover_color, width=2),
            handlePen=pg.mkPen(BROWSING_MARKER_COLOR),
            handleHoverPen=pg.mkPen(self.marker_hover_color),
        )
        for x, y in ((0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1), (0.5, 1), (0, 1), (0, 0.5)):
            self.marker.addScaleHandle((x, y), (1 - x, 1 - y))
        self.marker.setZValue(1)
        self.marker.setVisible(False)
        self._marker_ranges = None
        self.plot_widget.addItem(self.image_item)
        self.plot_widget.addItem(self.marker)
        self.color_bar.setImageItem(self.image_item, insert_in=self.plot_widget.getPlotItem())  # also sets its color map

        self.diagnostics_plot = pg.PlotWidget()
        self.diagnostics_plot.setBackground("w")
        self.diagnostics_plot.setLabel("bottom", "RT")
        self.diagnostics_plot.getAxis("left").setWidth(self.left_axis_width)
        self.diagnostics_plot.setXLink(self.plot_widget)
        self.diagnostics_plot.addLegend(offset=(-10, 10), brush=(255, 255, 255, 200))
        self.diagnostics_image_item = pg.ImageItem(axisOrder="col-major")
        self.diagnostics_color_bar = pg.ColorBarItem(
            values=(0.0, 1.0), width=15, colorMap=pg.colormap.get(self.diagnostics_color_map), interactive=False,
        )
        self.diagnostics_color_bar.axis.setLabel("Similarity (cosine)")
        self.diagnostics_color_bar.axis.enableAutoSIPrefix(False)
        self.diagnostics_color_bar.setImageItem(self.diagnostics_image_item, insert_in=self.diagnostics_plot.getPlotItem())
        self.diagnostics_color_bar.setVisible(False)
        self.diagnostics_color_map_combo = QComboBox()
        self.diagnostics_color_map_combo.setIconSize(self.color_map_icon_size)
        for name, color_map_name in self.color_maps.items():
            self.diagnostics_color_map_combo.addItem(_color_map_icon(pg.colormap.get(color_map_name), self.color_map_icon_size), name)
        self.diagnostics_color_map_combo.setCurrentText("gray")
        self.diagnostics_min_spin_box = QDoubleSpinBox()
        self.diagnostics_max_spin_box = QDoubleSpinBox()
        for spin_box, prefix in ((self.diagnostics_min_spin_box, "Min "), (self.diagnostics_max_spin_box, "Max ")):
            spin_box.setDecimals(6)
            spin_box.setRange(-1.0, 1.0)  # cosine similarity
            spin_box.setSingleStep(0.01)
            spin_box.setPrefix(prefix)
            spin_box.setKeyboardTracking(False)
        self.diagnostics_max_spin_box.setValue(1.0)
        self.diagnostics_auto_range_check_box = QCheckBox("Auto")
        self.diagnostics_auto_range_check_box.setChecked(True)
        self.diagnostics_auto_range_check_box.setToolTip("Use the 1st–99th percentiles of similarity")
        self._diagnostics_auto_levels = (0.0, 1.0)
        self.diagnostics_color_controls = QWidget()
        color_layout = QFormLayout(self.diagnostics_color_controls)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.addRow("Color", self.diagnostics_color_map_combo)
        levels_layout = QHBoxLayout()
        levels_layout.addWidget(self.diagnostics_min_spin_box)
        levels_layout.addWidget(self.diagnostics_max_spin_box)
        levels_layout.addWidget(self.diagnostics_auto_range_check_box)
        color_layout.addRow(levels_layout)
        self.diagnostics_color_controls.setVisible(False)
        self.diagnostics_panel = QWidget()
        diagnostics_layout = QVBoxLayout(self.diagnostics_panel)
        diagnostics_layout.setContentsMargins(0, 0, 0, 0)
        diagnostics_layout.addWidget(self.diagnostics_color_controls)
        diagnostics_layout.addWidget(self.diagnostics_plot)
        self.diagnostics_panel.setVisible(False)
        self._apply_diagnostics_levels()
        self._fill_color_map_combo()

        slider_end_labels = QHBoxLayout()
        slider_end_labels.setContentsMargins(0, 0, 0, 0)
        slider_end_labels.addWidget(_small_label("log-like"))
        slider_end_labels.addStretch(1)
        slider_end_labels.addWidget(_small_label("linear-like"))
        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form_layout.addRow("A", self.file_combo)
        form_layout.addRow("B", self.compare_combo)
        form_layout.addRow("s", self.scale_spin_box)
        form_layout.addRow(self.scale_slider)
        form_layout.addRow(slider_end_labels)
        form_layout.addRow("Color", self.color_map_combo)
        form_layout.addRow(self.show_marker_check_box)
        self.diff_group = QGroupBox("B − A")
        diff_layout = QFormLayout(self.diff_group)
        diff_layout.setContentsMargins(SMALL, SMALL, SMALL, SMALL)
        diff_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        diff_layout.addRow("Range", self.fold_change_spin_box)
        diff_layout.addRow(self.normalize_check_box)
        diff_layout.addRow("RT tol.", self.rt_tolerance_spin_box)
        diff_layout.addRow("m/z tol.", self.mz_tolerance_spin_box)
        diff_layout.addRow("Diagnostics", self.diagnostics_combo)
        self.diff_group.setVisible(False)
        self.control_panel = QWidget()
        self.control_panel.setObjectName("MzRtMapControlPanel")
        control_layout = QVBoxLayout(self.control_panel)
        control_layout.setContentsMargins(SMALL, SMALL, SMALL, SMALL)
        control_layout.addLayout(form_layout)
        control_layout.addWidget(self.diff_group)
        control_layout.addStretch(1)
        control_layout.addWidget(self.status_label)

        self.plot_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.plot_splitter.addWidget(self.plot_widget)
        self.plot_splitter.addWidget(self.diagnostics_panel)
        self.plot_splitter.setCollapsible(0, False)
        self.plot_splitter.setStretchFactor(0, 2)
        self.plot_splitter.setStretchFactor(1, 1)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.control_panel)
        self.splitter.addWidget(self.plot_splitter)
        self.splitter.setCollapsible(0, True)   # drag the handle to the left edge to hide the panel
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([self.control_panel_width, self.width() - self.control_panel_width])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)

        self._range_timer = QTimer(self)
        self._range_timer.setSingleShot(True)
        self._range_timer.setInterval(self.view_range_settle_ms)
        self._range_timer.timeout.connect(self.view_range_changed.emit)
        self.view_box.sigRangeChanged.connect(lambda *_: self._range_timer.start())
        self.view_box.sigResized.connect(lambda *_: self._range_timer.start())
        self.view_box.view_box_clicked.connect(self.clicked.emit)
        self.file_combo.currentIndexChanged.connect(self._on_file_index_changed)
        self.compare_combo.currentIndexChanged.connect(lambda index: self.compare_file_selected.emit(index - 1))
        self.scale_spin_box.valueChanged.connect(self._on_scale_spin_box_changed)
        self.scale_slider.valueChanged.connect(self._on_scale_slider_changed)
        self.color_map_combo.currentTextChanged.connect(self._on_color_map_changed)
        self.fold_change_spin_box.valueChanged.connect(self.fold_change_limit_changed.emit)
        self.normalize_check_box.toggled.connect(lambda _checked: self.diff_settings_changed.emit())
        self.rt_tolerance_spin_box.valueChanged.connect(lambda _value: self.diff_settings_changed.emit())
        self.mz_tolerance_spin_box.valueChanged.connect(lambda _value: self.diff_settings_changed.emit())
        self.diagnostics_combo.currentIndexChanged.connect(self._on_diagnostics_changed)
        self.diagnostics_color_map_combo.currentTextChanged.connect(
            lambda name: self.diagnostics_color_bar.setColorMap(pg.colormap.get(self.color_maps[name]))
        )
        self.diagnostics_auto_range_check_box.toggled.connect(self._apply_diagnostics_levels)
        self.diagnostics_min_spin_box.valueChanged.connect(self._apply_diagnostics_levels)
        self.diagnostics_max_spin_box.valueChanged.connect(self._apply_diagnostics_levels)
        self.show_marker_check_box.toggled.connect(lambda _checked: self._update_marker_visibility())
        self.marker.sigRegionChangeFinished.connect(self._on_marker_dropped)

    def _on_file_index_changed(self, index: int):
        if index >= 0:
            self.file_selected.emit(index)

    def set_files(self, names: list[str], tool_tips: list[str], current_index: int, compare_index: int):
        """compare_index: index of B, -1 for none."""
        _fill_file_combo(self.file_combo, names, tool_tips, current_index)
        _fill_file_combo(self.compare_combo, [self.no_compare_text, *names], ["", *tool_tips], compare_index + 1)

    def set_diff_mode(self, diff_mode: bool):
        """Show B - A (diverging colors, fold change on the color bar) or A alone."""
        if diff_mode == self._diff_mode:
            return
        self._diff_mode = diff_mode
        self.diff_group.setVisible(diff_mode)
        self._fill_color_map_combo()
        self.color_bar.axis.setLabel("B / A" if diff_mode else "Intensity")
        self._update_diagnostics_visibility()

    def diagnostics(self) -> Diagnostics:
        """Chosen diagnostics, NONE when only A is shown."""
        if not self._diff_mode:
            return Diagnostics.NONE
        return Diagnostics(self.diagnostics_combo.currentText())

    def set_diagnostics(self, plot: DiagnosticsPlot | None, message: str = ""):
        """Replace what the diagnostics panel shows; message while plot is None (e.g. not computed yet)."""
        plot_item = self.diagnostics_plot.getPlotItem()
        plot_item.clear()  # also empties the legend
        plot_item.setTitle(message or None, size="9pt")
        has_image = plot is not None and plot.image is not None
        self.diagnostics_color_bar.setVisible(has_image)
        self.diagnostics_color_controls.setVisible(has_image)
        if plot is None:
            return
        plot_item.setLabel("left", plot.y_label)
        largest = max((float(np.nanmax(np.abs(curve.y))) for curve in plot.curves if len(curve.y)), default=0.0)
        plot_item.getAxis("left").enableAutoSIPrefix(largest >= 1e3)  # not "x0.001" for similarities
        if plot.image is not None:
            self._diagnostics_auto_levels = plot.image.levels
            self._apply_diagnostics_levels()
            self.diagnostics_image_item.setImage(plot.image.values, levels=self.diagnostics_color_bar.levels(), autoLevels=False)
            self.diagnostics_image_item.setRect(QRectF(*plot.image.rect))
            plot_item.addItem(self.diagnostics_image_item)
        for curve in plot.curves:
            plot_item.plot(curve.x, curve.y, name=curve.name, **self.diagnostics_styles[curve.role])
        view_box = plot_item.getViewBox()
        if plot.image is None:
            view_box.setAutoVisible(y=True)  # fit y to the RT range shown
            view_box.enableAutoRange(axis=pg.ViewBox.YAxis)
        else:
            x, y, width, height = plot.image.rect
            view_box.setAutoVisible(y=False)
            view_box.setYRange(y, y + height, padding=0)

    def _on_diagnostics_changed(self):
        self._update_diagnostics_visibility()
        self.diagnostics_changed.emit()

    def _update_diagnostics_visibility(self):
        visible = self.diagnostics() is not Diagnostics.NONE
        was_hidden = self.diagnostics_panel.isHidden()
        self.diagnostics_panel.setVisible(visible)
        if visible and was_hidden:
            total = sum(self.plot_splitter.sizes()) or self.plot_splitter.width()
            self.plot_splitter.setSizes([total // 2, total - total // 2])

    def _apply_diagnostics_levels(self):
        auto = self.diagnostics_auto_range_check_box.isChecked()
        if auto:
            lower, upper = (round(value, 6) for value in self._diagnostics_auto_levels)
            if upper <= lower:
                padding = max(abs(lower) * 0.01, 0.000001)
                lower, upper = lower - padding, upper + padding
            lower, upper = max(-1.0, lower), min(1.0, upper)
        else:
            lower, upper = self.diagnostics_min_spin_box.value(), self.diagnostics_max_spin_box.value()
        for spin_box, value, bounds in (
            (self.diagnostics_min_spin_box, lower, (-1.0, upper - 0.000001)),
            (self.diagnostics_max_spin_box, upper, (lower + 0.000001, 1.0)),
        ):
            spin_box.blockSignals(True)
            try:
                spin_box.setRange(*bounds)
                spin_box.setValue(value)
                spin_box.setEnabled(not auto)
            finally:
                spin_box.blockSignals(False)
        self.diagnostics_color_bar.setLevels((lower, upper))

    def _fill_color_map_combo(self):
        color_maps = self.diff_color_maps if self._diff_mode else self.color_maps
        self.color_map_combo.blockSignals(True)
        try:
            self.color_map_combo.clear()
            for name, color_map_name in color_maps.items():
                self.color_map_combo.addItem(_color_map_icon(pg.colormap.get(color_map_name), self.color_map_icon_size), name)
            self.color_map_combo.setCurrentText(self._color_map_choices[self._diff_mode])
        finally:
            self.color_map_combo.blockSignals(False)
        self._on_color_map_changed(self.color_map_combo.currentText())

    def _on_color_map_changed(self, name: str):
        self._color_map_choices[self._diff_mode] = name
        self.color_bar.setColorMap(pg.colormap.get(self.color_map_name()))

    def color_map_name(self) -> str:
        """pyqtgraph name of the color map in use."""
        color_maps = self.diff_color_maps if self._diff_mode else self.color_maps
        return color_maps[self._color_map_choices[self._diff_mode]]

    def diff_settings(self) -> tuple[bool, int, float]:
        """normalize, RT tolerance in scans, m/z tolerance in ppm."""
        return (
            self.normalize_check_box.isChecked(),
            self.rt_tolerance_spin_box.value(),
            float(self.mz_tolerance_spin_box.value()),
        )

    def fold_change_limit(self) -> float:
        return self.fold_change_spin_box.value()

    def set_normalization_factor(self, factor: float | None):
        """factor: by which B is multiplied when normalizing, None if not known yet."""
        suffix = "" if factor is None else f" (B ×{factor:.3g})"
        self.normalize_check_box.setText(f"Normalize intensity{suffix}")

    def set_scale(self, scale: float):
        self.scale_spin_box.blockSignals(True)
        try:
            self.scale_spin_box.setValue(scale)
        finally:
            self.scale_spin_box.blockSignals(False)
        self._set_scale_slider(scale)

    def set_scale_range(self, scale_min: float, scale_max: float):
        """s at the slider ends; the spin box accepts values beyond them."""
        log_min = math.log10(max(scale_min, self.scale_spin_box.minimum()))
        log_max = max(math.log10(min(scale_max, self.scale_spin_box.maximum())), log_min + 1)
        self._log_scale_range = (log_min, log_max)
        self._set_scale_slider(self.scale_spin_box.value())

    def _set_scale_slider(self, scale: float):
        log_min, log_max = self._log_scale_range
        fraction = (math.log10(scale) - log_min) / (log_max - log_min)
        self.scale_slider.blockSignals(True)
        try:
            self.scale_slider.setValue(round(min(max(fraction, 0.0), 1.0) * self.scale_slider_steps))
        finally:
            self.scale_slider.blockSignals(False)

    def _on_scale_spin_box_changed(self, scale: float):
        self._set_scale_slider(scale)
        self.scale_changed.emit(scale)

    def _on_scale_slider_changed(self, value: int):
        log_min, log_max = self._log_scale_range
        scale = 10 ** (log_min + value / self.scale_slider_steps * (log_max - log_min))
        self.scale_spin_box.blockSignals(True)
        try:
            self.scale_spin_box.setValue(float(f"{scale:.3g}"))
        finally:
            self.scale_spin_box.blockSignals(False)
        self.scale_changed.emit(self.scale_spin_box.value())

    def show_range(self, rt_range: tuple[float, float], mz_range: tuple[float, float]):
        """Show the whole data range; panning and zooming stay within it."""
        self.view_box.setLimits(xMin=rt_range[0], xMax=rt_range[1], yMin=mz_range[0], yMax=mz_range[1])
        self.view_box.setRange(xRange=rt_range, yRange=mz_range, padding=0)

    def visible_range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        (rt_min, rt_max), (mz_min, mz_max) = self.view_box.viewRange()
        return (float(rt_min), float(rt_max)), (float(mz_min), float(mz_max))

    def pixel_shape(self) -> tuple[int, int]:
        """(RT, m/z) size of the plot area in device pixels."""
        ratio = self.devicePixelRatioF()
        rect = self.view_box.sceneBoundingRect()
        return max(1, round(rect.width() * ratio)), max(1, round(rect.height() * ratio))

    def set_image(
        self,
        image: np.ndarray,
        rt_range: tuple[float, float],
        mz_range: tuple[float, float],
        levels: tuple[float, float],
        color_bar_ticks: list[tuple[float, str]],
    ):
        """color_bar_ticks: (image value, label) pairs."""
        self.image_item.setImage(image, levels=levels, autoLevels=False)
        self.image_item.setRect(QRectF(rt_range[0], mz_range[0], rt_range[1] - rt_range[0], mz_range[1] - mz_range[0]))
        self.color_bar.setLevels(levels)
        self.color_bar.axis.setTicks([color_bar_ticks])

    def clear_image(self):
        self.image_item.clear()

    def set_marker_color(self, color: tuple[int, int, int, int]):
        self.marker.setPen(pg.mkPen(color, width=2))
        for handle in self.marker.getHandles():
            handle.pen = handle.currentPen = pg.mkPen(color)
            handle.update()

    def set_marker(self, rt_range: tuple[float, float] | None, mz_range: tuple[float, float] | None = None):
        self._marker_ranges = None if rt_range is None or mz_range is None else (rt_range, mz_range)
        if self._marker_ranges is not None:
            self.marker.blockSignals(True)
            try:
                self.marker.setPos((rt_range[0], mz_range[0]), update=False)
                self.marker.setSize((rt_range[1] - rt_range[0], mz_range[1] - mz_range[0]))
            finally:
                self.marker.blockSignals(False)
        self._update_marker_visibility()

    def _update_marker_visibility(self):
        self.marker.setVisible(self.show_marker_check_box.isChecked() and self._marker_ranges is not None)

    def _on_marker_dropped(self):
        """Keep the selection center inside the view and report its new bounds."""
        (rt_min, rt_max), (mz_min, mz_max) = self.visible_range()
        position, size = self.marker.pos(), self.marker.size()
        rt = min(max(position.x() + size.x() / 2, rt_min), rt_max)
        mz = min(max(position.y() + size.y() / 2, mz_min), mz_max)
        self.marker_range_changed.emit(rt - size.x() / 2, rt + size.x() / 2, mz - size.y() / 2, mz + size.y() / 2)

    def set_status(self, text: str):
        self.status_label.setText(text)


def _file_combo_box() -> QComboBox:
    combo = QComboBox()
    # long names do not widen the panel; the popup shows them in full
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(8)
    return combo


def _fill_file_combo(combo: QComboBox, names: list[str], tool_tips: list[str], current_index: int):
    combo.blockSignals(True)
    try:
        combo.clear()
        for name, tool_tip in zip(names, tool_tips):
            combo.addItem(name)
            combo.setItemData(combo.count() - 1, tool_tip, Qt.ItemDataRole.ToolTipRole)
        combo.setCurrentIndex(current_index)
    finally:
        combo.blockSignals(False)
    combo.view().setMinimumWidth(combo.view().sizeHintForColumn(0))


def _small_label(text: str) -> QLabel:
    label = QLabel(text)
    font = label.font()
    font.setPointSizeF(font.pointSizeF() * 0.85)
    label.setFont(font)
    label.setEnabled(False)  # drawn in the disabled (gray) color
    return label


def _color_map_icon(color_map: pg.ColorMap, size: QSize) -> QIcon:
    """The color map as a horizontal gradient, low values on the left."""
    lut = color_map.getLookupTable(nPts=size.width(), alpha=False)
    rgb = np.ascontiguousarray(np.broadcast_to(lut, (size.height(), size.width(), 3)), dtype=np.uint8)
    image = QImage(rgb.data, size.width(), size.height(), 3 * size.width(), QImage.Format.Format_RGB888)
    return QIcon(QPixmap.fromImage(image))  # fromImage copies, so rgb can go
