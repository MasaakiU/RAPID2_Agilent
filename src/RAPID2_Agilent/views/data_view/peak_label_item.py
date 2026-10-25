import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetricsF

from RAPID2_Agilent.services import operations


class PeakLabelItem(pg.GraphicsObject):
    """The m/z of the peak tops of a profile spectrum, small above them to the right.

    The most intense peaks first, each only where its label lies clear of those already placed, of the
    spectrum's line and of ``obstacles`` (labels of the plot): zoomed in, the peaks lie farther apart and more
    are labelled. Laid out in the ViewBox's pixels when it is next painted, after the view changed.
    """

    color = "#808080"
    font_scale = 0.8  # as the plot's other labels, of <font size='2'>
    gap = 2.0  # px between a peak top and its label
    margin = 2.0  # px kept free around a label
    # peaks lower than this fraction of the view's top are not labelled, the noise of the baseline
    min_relative_intensity = 0.02
    max_candidates = 300  # most intense peaks in view considered
    max_labels = 100

    def __init__(self, view_box: pg.ViewBox, obstacles=()):
        super().__init__()
        self.setParentItem(view_box)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.obstacles = tuple(obstacles)
        self.label_font = QFont()
        self.label_font.setPointSizeF(self.label_font.pointSizeF() * self.font_scale)
        self.font_metrics = QFontMetricsF(self.label_font)
        # a label's width, the sum of its characters': far faster than measuring each label
        self.char_widths = {char: self.font_metrics.horizontalAdvance(char) for char in "-.0123456789"}
        self.mz = np.array([])
        self.intensity = np.array([])
        # with one point past the last, where the reduction of a pixel column without points begins
        self.padded_intensity = np.array([-np.inf])
        # of the peak tops, most intense first
        self.apex_mz = np.array([])
        self.apex_intensity = np.array([])
        self._labels: list[tuple[QRectF, str]] | None = None
        view_box.sigRangeChanged.connect(self.invalidate)
        view_box.sigResized.connect(self.on_resized)

    def set_spectrum(self, mz, intensity):
        """mz ascending."""
        self.mz = np.asarray(mz, dtype=float)
        self.intensity = np.asarray(intensity, dtype=float)
        self.padded_intensity = np.append(self.intensity, -np.inf)
        apexes = operations.profile_peaks(self.mz, self.intensity) if len(self.mz) > 2 else np.array([], dtype=np.intp)
        self.apex_mz = self.mz[apexes]
        self.apex_intensity = self.intensity[apexes]
        self.invalidate()

    def invalidate(self, *args):
        self._labels = None
        self.update()

    def on_resized(self, *args):
        self.prepareGeometryChange()
        self.invalidate()

    def boundingRect(self):
        view_box = self.parentItem()
        return QRectF() if view_box is None else view_box.rect()

    def paint(self, painter, *args):
        labels = self.labels()
        if not labels:
            return
        painter.setFont(self.label_font)
        painter.setPen(QColor(self.color))
        for rect, text in labels:
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def labels(self) -> list[tuple[QRectF, str]]:
        """(rect in the ViewBox's pixels, text) of each label shown."""
        if self._labels is None:
            self._labels = self._layout()
        return self._labels

    def _layout(self) -> list[tuple[QRectF, str]]:
        view_box = self.parentItem()
        view_rect = view_box.rect()
        width, height = view_rect.width(), view_rect.height()
        (x_min, x_max), (y_min, y_max) = view_box.viewRange()
        if len(self.apex_mz) == 0 or width < 1 or height < 1 or x_max <= x_min or y_max <= y_min:
            return []
        mz, intensity = self.mz, self.intensity
        x_scale = width / (x_max - x_min)
        y_scale = height / (y_max - y_min)

        # the peak tops from the view's top down to the lowest labelled, in view; most intense first
        start, stop = np.searchsorted(-self.apex_intensity, (-y_max, -self.min_relative_intensity * y_max))
        candidate_mz = self.apex_mz[start:stop]
        candidates = start + np.flatnonzero((x_min <= candidate_mz) & (candidate_mz <= x_max))[: self.max_candidates]
        if len(candidates) == 0:
            return []
        candidate_mz = self.apex_mz[candidates]
        texts = [f"{value:.4f}" for value in candidate_mz]
        text_widths = np.array([sum(self.char_widths[char] for char in text) for text in texts])
        text_height = self.font_metrics.height()
        # its bottom left at the peak top, or shifted inside the view at its right edge
        apex_x = (candidate_mz - x_min) * x_scale
        left = np.minimum(apex_x, width - text_widths)
        right = left + text_widths
        bottom = (y_max - self.apex_intensity[candidates]) * y_scale - self.gap
        top = bottom - text_height
        inside = (left >= 0) & (top >= 0)

        # the highest the line is drawn in each pixel column: at the column's edges or at a point within it
        n_columns = int(np.ceil(width))
        edges = x_min + np.arange(n_columns + 1) / x_scale
        edge_intensity = np.interp(edges, mz, intensity, left=-np.inf, right=-np.inf)
        column_intensity = np.maximum(edge_intensity[:-1], edge_intensity[1:])
        # the points of column c: [ends[c], ends[c + 1])
        ends = np.searchsorted(mz, edges)
        has_points = ends[:-1] < ends[1:]
        column_max = np.maximum.reduceat(self.padded_intensity, ends)[:-1]
        column_intensity[has_points] = np.maximum(column_intensity[has_points], column_max[has_points])
        # the line's top in pixels, one past the last column for the reduction below
        line_top = np.append((y_max - column_intensity) * y_scale, np.inf)
        # of each label, the line's highest under its columns, below its bottom; reduceat over [first, end) pairs
        first_columns = np.clip(np.floor(left), 0, n_columns - 1).astype(np.intp)
        end_columns = np.clip(np.ceil(right), first_columns + 1, n_columns).astype(np.intp)
        bounds_per_label = np.column_stack((first_columns, end_columns)).ravel()
        clear_of_line = np.minimum.reduceat(line_top, bounds_per_label)[::2] >= bottom

        taken = [self._obstacle_rect(item) for item in self.obstacles]
        taken = [rect for rect in taken if rect is not None]
        labels = []
        for index in np.flatnonzero(inside & clear_of_line):
            rect = QRectF(left[index], top[index], text_widths[index], text_height)
            free = rect.adjusted(-self.margin, -self.margin, self.margin, self.margin)
            if any(free.intersects(other) for other in taken):
                continue
            taken.append(rect)
            labels.append((rect, texts[index]))
            if len(labels) >= self.max_labels:
                break
        return labels

    def _obstacle_rect(self, item) -> QRectF | None:
        """The rect in the ViewBox of a label of the plot that shows text, None otherwise."""
        if not item.isVisible() or not item.textItem.toPlainText():
            return None
        return self.parentItem().mapRectFromScene(item.sceneBoundingRect())
