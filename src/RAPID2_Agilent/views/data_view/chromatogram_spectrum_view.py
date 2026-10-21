import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QWidget


def make_label_html(text: str = "", size: int = 2) -> str:
    return f"<font size='{size}' color='#808080'>{text}</font>"


def main_r_pen():
    return pg.mkPen("#FF000032", width=2)


def main_r_brush():
    return pg.mkBrush("#FF000032")


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


class MyPlotWidget(pg.PlotWidget):
    view_box_clicked = pyqtSignal(float, float)
    wheel_event_forwarded = pyqtSignal(object)

    left_axis_width = 48
    axis_color = "#000000"
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

        self.region0 = pg.LinearRegionItem(pen=main_r_pen(), brush=main_r_brush())
        self.addItem(self.region0)

        self.exp_label = pg.TextItem(anchor=(0, 0), html=make_label_html("e0"))
        self.exp_label.setParentItem(self.vb())
        self.top_right_label = pg.TextItem(anchor=(1, 0), html=make_label_html())
        self.top_right_label.setParentItem(self.vb())

        self.vb().view_box_clicked.connect(self.view_box_clicked.emit)
        self.vb().sigResized.connect(self.vb_resized)
        self.vb().sigRangeChanged.connect(self.on_view_range_changed)
        self.left_axis.exponent_changed.connect(self.set_exponent_label)

    def vb(self):
        return self.plotItem.vb

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
        local_y = y_list[local_area]
        if len(local_y) == 0:
            return None, None
        return np.nanmin(local_y), np.nanmax(local_y)

    def add_plot_item(self, *, pen=None, name=None) -> pg.PlotDataItem:
        plot_item = pg.PlotDataItem(pen=pen or pg.mkPen(self.plot_color, width=1), name=name)
        plot_item.setZValue(2)
        self.plot_items.append(plot_item)
        self.addItem(plot_item)
        return plot_item

    def set_plot_data(self, index: int, x, y, **kwargs):
        self.plot_items[index].setData(x, y, **kwargs)

    def set_primary_plot_data(self, x, y, **kwargs):
        self.set_plot_data(0, x, y, **kwargs)
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

    def vb_resized(self, vb=None):
        vb = vb or self.vb()
        geometry = vb.screenGeometry()
        self.exp_label.setPos(0, 0)
        self.top_right_label.setPos(geometry.width(), 0)

    def set_exponent_label(self, exponent: int):
        self.exp_label.setHtml(make_label_html(f"e{exponent}"))

    def set_top_right_label_html(self, text: str):
        self.top_right_label.setHtml(text)


class ChromatogramWindow(MyPlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("ChromatogramWindow")
        # self.setLabel("left", "Intensity")


class SpectrumWindow(MyPlotWidget):
    plot_color = "#000000"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("SpectrumWindow")
        # self.setLabel("left", "Intensity")


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
