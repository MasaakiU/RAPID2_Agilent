from collections.abc import Callable, Hashable, Iterable, Mapping

from PyQt6.QtCore import QEasingCurve, QPointF, QRect, Qt, QVariantAnimation
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QWidget


ROW_CHANGE_MS = 200

RowSpan = tuple[int, int]  # a row's top in the viewport, and its height


class RowChangeOverlay(QWidget):
    """A change to a list's rows, played over its viewport once the change is made.

    The list shows its new state at once -- what a click hits, the selection, the data -- and this paints over it
    for a moment, from pictures of the rows: each one sliding from where it was to where it is, and each removed
    one fading where it was, so that what happened can be seen. It covers the viewport from the first row that
    changed down, takes no mouse events, and is gone at the end.
    """

    # held here while they play. Their parent's Python wrapper would hold them otherwise, and the wrapper of a
    # viewport is often a passing one: once it went, so did the overlay's Python side, leaving a blank widget
    _playing: set["RowChangeOverlay"] = set()

    def __init__(
        self,
        viewport: QWidget,
        top: int,
        removed: list[tuple[QPixmap, int]],
        moved: list[tuple[QPixmap, int, int]],
        background: QColor,
    ):
        super().__init__(viewport)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        # it paints all it covers, so the list under it is not painted again for every step
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setGeometry(0, top, viewport.width(), viewport.height() - top)
        self._top = top
        self._removed = removed  # (picture, top)
        self._moved = moved  # (picture, top before, top after), drawn in order
        self._background = background
        self._progress = 0.0
        self._animation = QVariantAnimation(self)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.setDuration(ROW_CHANGE_MS)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.valueChanged.connect(self._set_progress)
        self._animation.finished.connect(self._finish)
        # gone with its viewport, say, before the end
        self.destroyed.connect(lambda _object=None, overlay=self: RowChangeOverlay._playing.discard(overlay))

    @classmethod
    def play(
        cls,
        viewport: QWidget,
        before: Mapping[Hashable, RowSpan],
        after: Mapping[Hashable, RowSpan],
        removed_pictures: Mapping[Hashable, QPixmap],
        picture_of: Callable[[Hashable], QPixmap],
        background: QColor,
        on_top: Iterable[Hashable] = (),
    ) -> "RowChangeOverlay | None":
        """Play the rows going from ``before`` to ``after``; None when none moved or went.

        ``removed_pictures``: the removed rows in view, as they were. ``picture_of``: a row there now, as it is;
        asked only for the rows the overlay reaches. The rows in ``on_top`` pass over the others.
        """
        changed_tops = [before[key][0] for key in removed_pictures] + [
            min(before[key][0], span[0]) for key, span in after.items() if key in before and before[key] != span
        ]
        if not changed_tops:
            return None
        top = max(0, min(changed_tops))
        height = viewport.height()

        def reaches(span: RowSpan) -> bool:
            return span[0] + span[1] > top and span[0] < height

        removed = [(picture, before[key][0]) for key, picture in removed_pictures.items()]
        on_top = set(on_top)
        moved = [
            (picture_of(key), before.get(key, span)[0], span[0])
            for key, span in sorted(after.items(), key=lambda key_span: key_span[0] in on_top)
            if reaches(before.get(key, span)) or reaches(span)
        ]
        overlay = cls(viewport, top, removed, moved, background)
        cls._playing.add(overlay)
        overlay.show()
        overlay.raise_()
        overlay._animation.start()
        return overlay

    @classmethod
    def finish_all(cls, viewport: QWidget):
        """End the changes still playing over ``viewport``, so that the list shows as it is."""
        for overlay in [overlay for overlay in cls._playing if overlay.parentWidget() is viewport]:
            overlay._finish()

    def _finish(self):
        self._animation.stop()
        self.hide()
        RowChangeOverlay._playing.discard(self)
        self.deleteLater()

    @staticmethod
    def picture_of_rows(viewport: QWidget) -> QImage:
        """A picture of ``viewport`` to cut rows from (see cut_row)."""
        # with transparency, so that the part of a row out of view is cut clear rather than black
        return viewport.grab().toImage().convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)

    @staticmethod
    def cut_row(picture: QImage, span: RowSpan) -> QPixmap:
        """The row at ``span`` cut from a picture of the whole viewport."""
        # in the picture's pixels, which a high-DPI screen has more of
        ratio = picture.devicePixelRatio()
        top, height = span
        row = picture.copy(QRect(0, round(top * ratio), picture.width(), round(height * ratio)))
        row.setDevicePixelRatio(ratio)
        return QPixmap.fromImage(row)

    def _set_progress(self, progress: float):
        self._progress = progress
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._background)
        painter.translate(0, -self._top)
        painter.setOpacity(1.0 - self._progress)
        for picture, top in self._removed:
            painter.drawPixmap(QPointF(0, top), picture)
        painter.setOpacity(1.0)
        for picture, old_top, new_top in self._moved:
            painter.drawPixmap(QPointF(0, round(old_top + (new_top - old_top) * self._progress)), picture)
