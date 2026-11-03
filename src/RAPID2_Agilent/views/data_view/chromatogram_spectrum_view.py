from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QMenu, QWidget

from RAPID2_Agilent.views.data_view.peak_label_item import PeakLabelItem
from RAPID2_Agilent.views.view_constants import (
    BROWSING_RANGE_LABEL_COLOR,
    BROWSING_REGION_COLOR,
    PEAK_AREA_COLOR,
    PEAK_FIT_COLOR,
    PLOT_WARNING_COLOR,
)


def make_label_html(text: str = "", size: int = 2, color: str = "#808080") -> str:
    return f"<font size='{size}' color='{color}'>{text}</font>"


class ExpAxisItem(pg.AxisItem):
    exponent_changed = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp = 0
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values, scale, spacing):
        if self.logMode:
            return self.logTickStrings(values, scale, spacing)
        if len(values) == 0:
            return []

        max_abs = max(abs(value) for value in values)
        next_exp = 0 if max_abs == 0 else int(np.floor(np.log10(max_abs)))
        if next_exp != self.exp:
            self.exp = next_exp
            self.exponent_changed.emit(self.exp)

        scale_factor = 10 ** self.exp
        return [f"{value / scale_factor:.1f}" for value in values]


class MyViewBox(pg.ViewBox):
    view_box_clicked = pyqtSignal(float, float)

    def isInViewRange(self, x_value: float, y_value: float) -> bool:
        (min_x, max_x), (min_y, max_y) = self.viewRange()
        return min_x < x_value < max_x and min_y < y_value < max_y

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            xy_position = self.mapSceneToView(ev.scenePos())
            x_value = xy_position.x()
            y_value = xy_position.y()
            if self.isInViewRange(x_value, y_value):
                ev.accept()
                self.view_box_clicked.emit(x_value, y_value)
            else:
                ev.ignore()
        elif ev.button() == Qt.MouseButton.RightButton and self.menuEnabled():
            ev.accept()
            self.raiseContextMenu(ev)
        else:
            super().mouseClickEvent(ev)


class MyPlotItem(pg.PlotItem):
    padding_x = 0.1
    padding_y = 0.1

    def __init__(
        self,
        parent=None,
        name=None,
        labels=None,
        title=None,
        viewBox=None,
        axisItems=None,
        enableMenu=True,
        **kargs,
    ):
        super().__init__(parent, name, labels, title, viewBox, axisItems, enableMenu, **kargs)
        self.autoBtn.hide()

    def updateButtons(self):
        pass

    @staticmethod
    def _expanded_range(min_value: float, max_value: float) -> tuple[float, float]:
        value_range = max_value - min_value
        if value_range == 0:
            value_range = abs(min_value) * 0.01 or 1.0
            min_value -= value_range / 2
            max_value += value_range / 2
        return min_value, max_value

    def calc_my_y_range(self, y_min: float, y_max: float) -> tuple[float, float]:
        y_min, y_max = self._expanded_range(y_min, y_max)
        padding_y = (y_max - y_min) * self.padding_y
        return y_min - padding_y, y_max + padding_y

    def setMyRange(self, x_min: float, x_max: float, y_min: float, y_max: float):
        x_min, x_max = self.setMyXRange(x_min, x_max)
        y_btm, y_top = self.calc_my_y_range(y_min, y_max)
        self.setRange(xRange=(x_min, x_max), yRange=(y_btm, y_top), padding=0)

    def setMyYRange(self, y_min: float, y_max: float):
        y_btm, y_top = self.calc_my_y_range(y_min, y_max)
        self.setYRange(y_btm, y_top, padding=0)

    def setMyXRange(self, x_min: float, x_max: float) -> tuple[float, float]:
        x_min, x_max = self._expanded_range(x_min, x_max)
        x_range = x_max - x_min
        x_min = x_min - x_range * self.padding_x
        x_max = x_max + x_range * self.padding_x
        super().setXRange(x_min, x_max, padding=0)
        return x_min, x_max

    def setXRange(self, x_min, x_max, padding=0):
        super().setXRange(x_min, x_max, padding=padding)

    def setYRange(self, y_min, y_max, padding=0):
        super().setYRange(y_min, y_max, padding=padding)

    def viewRange_x(self):
        return self.viewRange()[0]

    def viewRange_y(self):
        return self.viewRange()[1]


class WindowRegionItem(pg.LinearRegionItem):
    """The RT / m/z window, which asks for its menu when right-clicked (but while it is dragged, which that undoes)."""
    menu_requested = pyqtSignal(object)  # the QPointF on the screen

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton and not self.moving:
            ev.accept()
            self.menu_requested.emit(ev.screenPos())
            return
        super().mouseClickEvent(ev)


class MyPlotWidget(pg.PlotWidget):
    view_box_clicked = pyqtSignal(float, float)
    wheel_event_forwarded = pyqtSignal(object)
    cursor_moved = pyqtSignal(float, float)  # (x, y) of the local crosshair
    cursor_left = pyqtSignal()

    left_axis_width = 48
    axis_color = "#000000"
    cursor_label_color = "#EDA0A0"
    plot_color = "#1f77b4"

    def __init__(self, parent=None, initial_plot_count: int = 1):
        self.left_axis = ExpAxisItem(orientation="left", maxTickLength=5)
        self.left_axis.setWidth(w=self.left_axis_width)
        bottom_axis = pg.AxisItem(orientation="bottom", maxTickLength=5)
        self._set_axis_color(self.left_axis)
        self._set_axis_color(bottom_axis)
        plot_item = MyPlotItem(
            viewBox=MyViewBox(),
            axisItems={"left": self.left_axis, "bottom": bottom_axis},
        )
        super().__init__(parent=parent, plotItem=plot_item)
        plot_item.setMenuEnabled(False)  # pyqtgraph's, of no use here

        self.setBackground("w")
        self.vb().setMouseMode(pg.ViewBox.RectMode)
        self.disableAutoRange(axis="xy")
        self.auto_y_range_enabled = True
        # Linked plots can supply bounds pooled across files.
        self.y_range_bounds_provider = None
        self.x_range_initialized = False
        self._applying_y_range_policy = False

        self.plot_items: list[pg.PlotDataItem] = []
        for _ in range(initial_plot_count):
            self.add_plot_item()
        # the primary line across points not measured, dashed: as an area is integrated over them
        self.bridge_item = pg.PlotDataItem(pen=pg.mkPen(self.plot_color, width=1, style=Qt.PenStyle.DashLine))
        self.bridge_item.setZValue(2)
        self.addItem(self.bridge_item)

        self.region0 = WindowRegionItem()
        self.set_region_color(BROWSING_REGION_COLOR)
        self.addItem(self.region0)

        self.exp_label = pg.TextItem(anchor=(0, 0), html=make_label_html("e0"))
        self.exp_label.setParentItem(self.vb())
        self.range_label_color = BROWSING_RANGE_LABEL_COLOR
        self.range_label_text = ""
        self.top_right_label = pg.TextItem(anchor=(1, 0), html=make_label_html(color=self.range_label_color))
        self.top_right_label.setParentItem(self.vb())
        self.cursor_label = pg.TextItem(anchor=(1, 0), html=make_label_html(color=self.cursor_label_color))
        self.cursor_label.setParentItem(self.vb())
        # what is wrong with the data drawn, if anything, in the middle of the plot, under a notice of what it is
        self.warning_label = pg.TextItem(anchor=(0.5, 0.5), html=make_label_html(color=PLOT_WARNING_COLOR))
        self.warning_label.setParentItem(self.vb())
        self.notice_text = ""
        self.warning_text = ""
        for label in (self.exp_label, self.top_right_label, self.cursor_label, self.warning_label):
            label.textItem.document().setDocumentMargin(0)

        crosshair_pen = pg.mkPen("#FF0000", width=1)
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)
        for line in (self.crosshair_v, self.crosshair_h):
            line.setZValue(3)
            line.setVisible(False)
            self.addItem(line, ignoreBounds=True)
        self.scene().sigMouseMoved.connect(self.on_mouse_moved)

        self.vb().view_box_clicked.connect(self.view_box_clicked.emit)
        self.vb().sigResized.connect(self.vb_resized)
        self.vb().sigRangeChanged.connect(self.on_view_range_changed)
        self.left_axis.exponent_changed.connect(self.set_exponent_label)

    def vb(self):
        return self.plotItem.vb

    def set_region_color(self, color: tuple[int, int, int, int]):
        """The RT / m/z window in ``color`` (RGBA), its area and its edges.

        Under the mouse, as pyqtgraph's own defaults were for the former red window: the area twice
        as opaque, an edge in the colour, opaque.
        """
        self.region0.setBrush(pg.mkBrush(color))
        self.region0.setHoverBrush(pg.mkBrush((*color[:3], min(2 * color[3], 255))))
        for line in self.region0.lines:
            line.setPen(pg.mkPen(color, width=2))
            line.setHoverPen(pg.mkPen((*color[:3], 255), width=2))
        self.region0.update()

    @classmethod
    def _set_axis_color(cls, axis: pg.AxisItem):
        pen = pg.mkPen(cls.axis_color)
        axis.setPen(pen)
        axis.setTextPen(pen)
        axis.setTickPen(pen)

    def wheelEvent(self, ev, axis=None):
        self.wheel_event_forwarded.emit(ev)
        ev.ignore()

    @staticmethod
    def get_local_minmax(x_list, y_list, min_x, max_x):
        if x_list is None or y_list is None:
            return None, None
        local_area = (min_x <= x_list) & (x_list <= max_x)
        # points not measured are NaN
        local_y = y_list[local_area & np.isfinite(y_list)]
        if len(local_y) == 0:
            return None, None
        return np.min(local_y), np.max(local_y)

    def add_plot_item(self, *, pen=None, name=None) -> pg.PlotDataItem:
        plot_item = pg.PlotDataItem(pen=pen or pg.mkPen(self.plot_color, width=1), name=name)
        plot_item.setZValue(2)
        self.plot_items.append(plot_item)
        self.addItem(plot_item)
        return plot_item

    def set_plot_data(self, index: int, x, y, **kwargs):
        self.plot_items[index].setData(x, y, **kwargs)

    def set_primary_plot_data(self, x, y, *, missing: np.ndarray | None = None, **kwargs):
        """missing: of each point, whether it was not measured; the line leaves those out and joins the points on
        either side of them dashed."""
        if missing is not None and missing.any():
            y = np.where(missing, np.nan, np.asarray(y, dtype=float))
            kwargs["connect"] = "finite"
            measured = np.flatnonzero(~missing)
            # measured points with ones not measured between them, joined in pairs
            jumps = np.flatnonzero(np.diff(measured) > 1)
            ends = np.column_stack((measured[jumps], measured[jumps + 1])).ravel()
            self.bridge_item.setData(x[ends], y[ends], connect="pairs")
        else:
            kwargs.setdefault("connect", "auto")
            self.bridge_item.setData([], [])
        self.set_plot_data(0, x, y, **kwargs)
        # empty when a file has no scans of the choice, e.g. no MS2 scans of the precursor
        if len(x):
            self.set_initial_x_range(float(x[0]), float(x[-1]))
        self.apply_y_range_policy()

    def set_auto_y_range_enabled(self, enabled: bool):
        self.auto_y_range_enabled = enabled

    def set_initial_x_range(self, x_min: float, x_max: float, *, force: bool = False):
        if self.x_range_initialized and not force:
            return
        self.plotItem.setMyXRange(x_min, x_max)
        self.x_range_initialized = True

    def get_region(self) -> tuple[float, float]:
        left, right = self.region0.getRegion()
        if right < left:
            left, right = right, left
        return float(left), float(right)

    def get_region_width(self) -> float:
        left, right = self.get_region()
        return right - left

    def make_region_centered_at(self, center: float, fallback_width: float) -> tuple[float, float]:
        width = self.get_region_width()
        if width <= 0:
            width = fallback_width
        return center - width / 2, center + width / 2

    def clear_plot_data(self, index: int | None = None):
        if index is None:
            for plot_item in self.plot_items:
                plot_item.clear()
        else:
            self.plot_items[index].clear()

    def get_x_bounds(self, index: int = 0):
        x_data = self.plot_items[index].xData
        return np.nanmin(x_data), np.nanmax(x_data)

    def get_y_bounds(self, index: int = 0):
        y_data = self.plot_items[index].yData
        return np.nanmin(y_data), np.nanmax(y_data)

    def get_displayed_y_bounds(self, x_min, x_max, index: int = 0):
        plot_item = self.plot_items[index]
        return self.get_local_minmax(plot_item.xData, plot_item.yData, x_min, x_max)

    def apply_y_range_policy(self, index: int = 0):
        if not self.auto_y_range_enabled:
            return
        x_view_range, y_view_range = self.viewRange()
        bounds_provider = self.y_range_bounds_provider or self.get_displayed_y_bounds
        y_min, y_max = bounds_provider(*x_view_range, index=index)
        if y_min is not None:
            # 0 is always shown, and the data sit in the vertical middle: from 0 to the max plus the min
            if y_min >= 0:
                y_min, y_max = 0.0, y_max + y_min
            self._applying_y_range_policy = True
            try:
                self.plotItem.setMyYRange(y_min, y_max)
            finally:
                self._applying_y_range_policy = False

    def setMyYRange_within_x_view_range(self, index: int = 0):
        self.apply_y_range_policy(index=index)

    def on_view_range_changed(self, *args):
        if self._applying_y_range_policy:
            return
        self.apply_y_range_policy()

    def on_mouse_moved(self, scene_pos):
        vb = self.vb()
        if not vb.sceneBoundingRect().contains(scene_pos):
            self.hide_crosshair()
            return
        point = vb.mapSceneToView(scene_pos)
        x_value, y_value = point.x(), point.y()
        self.crosshair_v.setPos(x_value)
        self.crosshair_h.setPos(y_value)
        self.crosshair_v.setVisible(True)
        self.crosshair_h.setVisible(True)
        self._set_cursor_text(f"({x_value:.4f}, {y_value:.3g})")
        self.cursor_moved.emit(float(x_value), float(y_value))

    def set_shared_cursor(self, x_value: float, y_value: float):
        """Show the crosshair position of a linked plot."""
        self.crosshair_v.setPos(x_value)
        self.crosshair_h.setPos(y_value)
        self.crosshair_v.setVisible(True)
        self.crosshair_h.setVisible(True)
        self._set_cursor_text(f"({x_value:.4f}, {y_value:.3g})")

    def _set_cursor_text(self, text: str):
        self.cursor_label.setHtml(make_label_html(text, color=self.cursor_label_color))
        self._place_top_right_labels()

    def clear_cursor(self):
        self.crosshair_v.setVisible(False)
        self.crosshair_h.setVisible(False)
        self.cursor_label.setHtml(make_label_html(color=self.cursor_label_color))

    def hide_crosshair(self):
        was_visible = self.crosshair_v.isVisible()
        self.clear_cursor()
        if was_visible:
            self.cursor_left.emit()

    def leaveEvent(self, ev):
        self.hide_crosshair()
        super().leaveEvent(ev)

    def _place_top_right_labels(self):
        width = self.vb().boundingRect().width()
        self.top_right_label.setPos(width, 0)
        self.cursor_label.setPos(width, self.top_right_label.boundingRect().height())

    def vb_resized(self, vb=None):
        self.exp_label.setPos(0, 0)
        rect = self.vb().boundingRect()
        self.warning_label.setPos(rect.width() / 2, rect.height() / 2)
        # wrapped, so that a long one is read whole in a narrow plot
        self.warning_label.setTextWidth(max(1.0, rect.width()))
        self._place_top_right_labels()

    def set_warning_html(self, text: str):
        """A warning about the data drawn, in red in the middle; "" for none."""
        self.warning_text = text
        self._show_warning()

    def set_notice_html(self, text: str):
        """Which data are drawn, in red in the middle above the warning: of the targets selected, the one shown;
        "" for none."""
        self.notice_text = text
        self._show_warning()

    def _show_warning(self):
        text = "<br>".join(part for part in (self.notice_text, self.warning_text) if part)
        self.warning_label.setHtml(f"<div align='center'>{make_label_html(text, color=PLOT_WARNING_COLOR)}</div>")

    def set_exponent_label(self, exponent: int):
        self.exp_label.setHtml(make_label_html(f"e{exponent}"))

    def set_top_right_label_html(self, text: str):
        self.range_label_text = text
        self.top_right_label.setHtml(make_label_html(text, color=self.range_label_color))
        self._place_top_right_labels()

    def set_range_label_color(self, color: str):
        self.range_label_color = color
        self.set_top_right_label_html(self.range_label_text)


class ChromatogramWindow(MyPlotWidget):
    no_peak_requested = pyqtSignal()  # by the RT window's menu

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("ChromatogramWindow")
        # self.setLabel("left", "Intensity")
        self.peak_areas: list[tuple[float, float]] = []  # (start, end) in RT, of the peaks found
        self.peak_area_items: list[pg.PlotDataItem] = []
        self.peak_fit_items: list = []  # the lines and labels of set_peak_fits
        # the RT window's menu, while it is a target's peak in the file (set_rt_window_menu_enabled)
        self.rt_window_menu = QMenu(self)
        self.no_peak_action = self.rt_window_menu.addAction("No peak in this file")
        self.no_peak_action.triggered.connect(self.no_peak_requested.emit)
        self.rt_window_menu_enabled = False
        self.region0.menu_requested.connect(self._show_rt_window_menu)

    def set_rt_window_menu_enabled(self, enabled: bool):
        self.rt_window_menu_enabled = enabled

    def _show_rt_window_menu(self, screen_pos):
        if self.rt_window_menu_enabled:
            self.rt_window_menu.popup(screen_pos.toPoint())

    def set_primary_plot_data(self, x, y, **kwargs):
        super().set_primary_plot_data(x, y, **kwargs)
        self._draw_peak_areas()

    def set_peak_areas(self, peak_areas: Sequence[tuple[float, float]]):
        """Shade the area of each peak (start, end) under the trace, as it is integrated: over the points measured
        from its start to its end, down to 0."""
        self.peak_areas = list(peak_areas)
        self._draw_peak_areas()

    def _draw_peak_areas(self):
        for item in self.peak_area_items:
            self.removeItem(item)
        self.peak_area_items = []
        x, y = self.plot_items[0].xData, self.plot_items[0].yData
        if x is None:
            return
        measured = np.isfinite(y)  # points not measured are NaN
        for start, end in self.peak_areas:
            inside = measured & (start <= x) & (x <= end)
            if np.count_nonzero(inside) < 2:
                continue
            item = pg.PlotDataItem(x[inside], y[inside], pen=None, fillLevel=0, fillBrush=pg.mkBrush(PEAK_AREA_COLOR))
            item.setZValue(1)  # over the RT window, under the trace
            self.addItem(item)
            self.peak_area_items.append(item)


    def set_peak_fits(self, fits: Sequence[tuple[tuple, tuple, str]]):
        """Draw the fit of each peak (RTs, intensities), with its label at its top; for now, as the peak detector
        that finds the peaks' shapes has them."""
        for item in self.peak_fit_items:
            self.removeItem(item)
        self.peak_fit_items = []
        for rt, intensity, label in fits:
            if len(rt) < 2:
                continue
            line = pg.PlotDataItem(rt, intensity, pen=pg.mkPen(PEAK_FIT_COLOR, width=1))
            line.setZValue(3)  # over the trace
            top = int(np.argmax(intensity))
            text = pg.TextItem(label, color=PEAK_FIT_COLOR, anchor=(0.5, 1))
            text.setPos(rt[top], intensity[top])
            for item in (line, text):
                self.addItem(item)
                self.peak_fit_items.append(item)


class SpectrumWindow(MyPlotWidget):
    plot_color = "#000000"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("SpectrumWindow")
        # self.setLabel("left", "Intensity")
        # the m/z of the peak tops, clear of the plot's labels other than the cursor's, which moves with the mouse
        self.peak_labels = PeakLabelItem(self.vb(), obstacles=(self.exp_label, self.top_right_label, self.warning_label))

    def set_primary_plot_data(self, x, y, **kwargs):
        super().set_primary_plot_data(x, y, **kwargs)
        self.peak_labels.set_spectrum(x, y)

    def _show_warning(self):
        super()._show_warning()
        self.peak_labels.invalidate()

    def set_top_right_label_html(self, text: str):
        super().set_top_right_label_html(text)
        self.peak_labels.invalidate()


class ChromatogramSpectrumView(QWidget):
    def __init__(
        self,
        chromatogram_window: QWidget | None = None,
        spectrum_window: QWidget | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("ChromatogramSpectrumView")

        self.chromatogram_window = chromatogram_window or ChromatogramWindow()
        self.spectrum_window = spectrum_window or SpectrumWindow()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.chromatogram_window, 1)
        layout.addWidget(self.spectrum_window, 1)
