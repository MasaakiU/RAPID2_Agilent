from PyQt6.QtCore import QEvent, QObject, QPoint
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication, QToolTip, QWidget


TOOLTIP_OFFSET = QPoint(10, -18)


class TooltipPositionFilter(QObject):
    """Show every tooltip next to the mouse cursor, shifted by TOOLTIP_OFFSET."""

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.ToolTip:
            return False
        tooltip_text = obj.toolTip() if isinstance(obj, QWidget) else ""
        QToolTip.hideText()
        if tooltip_text:
            if hasattr(event, "pos"):
                position = obj.mapToGlobal(event.pos()) + TOOLTIP_OFFSET
            else:
                position = QCursor.pos() + TOOLTIP_OFFSET
            QToolTip.showText(position, tooltip_text, obj, obj.rect())
        return True


def install_tooltip_position(app: QApplication) -> TooltipPositionFilter:
    tooltip_filter = TooltipPositionFilter(app)
    app.installEventFilter(tooltip_filter)
    return tooltip_filter
