from PyQt6.QtCore import QPoint, QSize, Qt
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QApplication, QFrame, QLabel, QToolTip, QVBoxLayout, QWidget

from RAPID2_Agilent.views.widgets.my_push_button import MyPushButton


class InfoPopup(QFrame):
    """Rich text shown under the widget it explains, in the tooltip's colours, until a click elsewhere (as in mywb)."""

    MARGIN = 10

    def __init__(self, html: str, parent: QWidget, width: int):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        # framed in the text colour, as a tooltip is; a palette, so that it is the mode's own
        tooltip = QToolTip.palette()
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, tooltip.color(QPalette.ColorRole.ToolTipBase))
        palette.setColor(QPalette.ColorRole.WindowText, tooltip.color(QPalette.ColorRole.ToolTipText))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.setFrameShape(QFrame.Shape.Box)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self.label = QLabel(html)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(self.MARGIN, self.MARGIN, self.MARGIN, self.MARGIN)
        layout.addWidget(self.label)
        # the text wrapped to the width, and as high as that makes it: a wrapped label's own height is a guess
        label_width = width - 2 * (self.MARGIN + self.frameWidth())
        self.label.setFixedSize(label_width, self.label.heightForWidth(label_width))
        self.setFixedWidth(width)

    def show_near(self, widget: QWidget):
        """Under ``widget``, or above it where the screen ends below; within the screen either way."""
        self.adjustSize()
        screen = widget.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry()
        point = widget.mapToGlobal(widget.rect().bottomLeft() + QPoint(0, 2))
        if point.x() + self.width() > available.right():
            point.setX(max(available.left(), available.right() - self.width()))
        if point.y() + self.height() > available.bottom():
            point = widget.mapToGlobal(widget.rect().topLeft() - QPoint(0, self.height() + 2))
            point.setX(max(available.left(), min(point.x(), available.right() - self.width())))
        point.setY(max(point.y(), available.top()))
        self.move(point)
        self.show()


class InfoIconButton(MyPushButton):
    """An info icon, flat, that shows ``html`` in an InfoPopup when clicked, and closes it when clicked again (as in mywb)."""

    ICON_SIZE = 15
    BUTTON_SIZE = ICON_SIZE + 6
    POPUP_WIDTH = 320

    def __init__(self, html: str, parent=None, *, popup_width: int = POPUP_WIDTH, accessible_name: str = "Show information"):
        super().__init__(parent)
        self._html = html
        self._popup_width = popup_width
        self._popup: InfoPopup | None = None
        self.set_icon_name("circle-info-solid")
        self.setIconSize(QSize(self.ICON_SIZE, self.ICON_SIZE))
        # flat rather than a transparent style sheet, which would keep it from hearing of Light/Dark changes
        self.setFlat(True)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.setAccessibleName(accessible_name)
        self.clicked.connect(self.show_info)

    def adjusted_size_hint(self) -> QSize:
        return QSize(self.BUTTON_SIZE, self.BUTTON_SIZE)

    def minimumSizeHint(self) -> QSize:
        return self.adjusted_size_hint()

    def show_info(self):
        if self._popup is not None and self._popup.isVisible():
            self._popup.close()
            return
        self._popup = InfoPopup(self._html, self, self._popup_width)
        self._popup.show_near(self)
