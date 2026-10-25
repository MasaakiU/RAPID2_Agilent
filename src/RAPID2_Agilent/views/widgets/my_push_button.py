from PyQt6.QtCore import QEvent, QRect, QSize, Qt
from PyQt6.QtGui import QIcon, QIconEngine, QPainter, QPalette, QPixmap
from PyQt6.QtWidgets import QPushButton, QStyle, QStyleOptionButton, QStylePainter

from RAPID2_Agilent.utils.path import AppPath


IconLike = QIcon | str | None


class _DisabledOpacityIconEngine(QIconEngine):
    """An icon drawn translucent when disabled, in place of the style's disabled pixmap.

    Qt makes a disabled icon by greying it towards the window colour: a black icon on a light window comes out a
    middle grey, and the light icon of Dark mode stays all but as light, so that a disabled button there looked like
    an enabled one. The icon itself, translucent, fades the same in both modes (as in mywb).
    """

    def __init__(self, source_icon: QIcon, disabled_opacity: float):
        super().__init__()
        self.source_icon = QIcon(source_icon)
        self.disabled_opacity = disabled_opacity

    def clone(self):
        return _DisabledOpacityIconEngine(self.source_icon, self.disabled_opacity)

    def paint(self, painter, rect, mode, state):
        painter.save()
        if mode == QIcon.Mode.Disabled:
            painter.setOpacity(painter.opacity() * self.disabled_opacity)
            mode = QIcon.Mode.Normal
        self.source_icon.paint(painter, rect, Qt.AlignmentFlag.AlignCenter, mode, state)
        painter.restore()

    def pixmap(self, size, mode, state):
        pixmap = QPixmap(size)
        if pixmap.isNull():
            return pixmap
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        self.paint(painter, pixmap.rect(), mode, state)
        painter.end()
        return pixmap


class MyPushButton(QPushButton):
    DISABLED_ICON_OPACITY = 0.38
    HEIGHT_ADJUSTMENT = -3
    # the style leaves this much more room right of a menu's arrow than there is left of the icon; taken off the width
    MENU_ARROW_RIGHT_TRIM = 6
    default_icon_size = 16

    def __init__(
        self,
        parent=None,
        *,
        light_icon: IconLike = None,
        dark_icon: IconLike = None,
    ):
        super().__init__(parent)
        self.setObjectName("MyPushButton")
        self._light_icon = QIcon()
        self._dark_icon = QIcon()
        self._menu_arrow_shown = True
        self.setIconSize(QSize(self.default_icon_size, self.default_icon_size))

        self.set_icons(light_icon, dark_icon)
        self._apply_size()

    def set_menu_arrow_shown(self, shown: bool) -> None:
        """Whether a button with a menu shows the arrow; one whose icon already tells of a menu does without."""
        self._menu_arrow_shown = shown
        self._apply_size()
        self.update()

    def adjusted_size_hint(self) -> QSize:
        # square, a little lower than a text button; one with a menu wider by the arrow the style draws for it
        height = max(1, self._text_button_height() + self.HEIGHT_ADJUSTMENT)
        width = height
        if self.menu() is not None and self._menu_arrow_shown:
            width += self.style().pixelMetric(QStyle.PixelMetric.PM_MenuButtonIndicator, None, self) - self.MENU_ARROW_RIGHT_TRIM
        return QSize(width, height)

    def _text_button_height(self) -> int:
        """The height of a text button in this font and style, as QPushButton.sizeHint works it out.

        Not the button's own: with its icon, a little taller than one without, it was the one without while the
        size was set before the icon, and the taller after any later font or style change.
        """
        option = QStyleOptionButton()
        self.initStyleOption(option)
        option.icon = QIcon()
        text_height = self.fontMetrics().size(Qt.TextFlag.TextShowMnemonic, "XXXX").height()
        return self.style().sizeFromContents(QStyle.ContentsType.CT_PushButton, option, QSize(0, text_height), self).height()

    def sizeHint(self) -> QSize:
        return self.adjusted_size_hint()

    def minimumSizeHint(self) -> QSize:
        size = QSize(super().minimumSizeHint())
        size.setHeight(max(1, size.height() + self.HEIGHT_ADJUSTMENT))
        return size

    def paintEvent(self, ev):
        if self.menu() is None:
            super().paintEvent(ev)
            return
        painter = QStylePainter(self)
        option = QStyleOptionButton()
        self.initStyleOption(option)
        option.features &= ~QStyleOptionButton.ButtonFeature.HasMenu
        if not self._menu_arrow_shown:
            painter.drawControl(QStyle.ControlElement.CE_PushButton, option)
            return
        # as QPushButton draws it, the icon and the arrow where they are on a button MENU_ARROW_RIGHT_TRIM wider
        indicator = self.style().pixelMetric(QStyle.PixelMetric.PM_MenuButtonIndicator, option, self)
        painter.drawControl(QStyle.ControlElement.CE_PushButtonBevel, option)
        label = QStyleOptionButton(option)
        contents = self.style().subElementRect(QStyle.SubElement.SE_PushButtonContents, option, self)
        label.rect = contents.adjusted(0, 0, self.MENU_ARROW_RIGHT_TRIM - indicator, 0)
        painter.drawControl(QStyle.ControlElement.CE_PushButtonLabel, label)
        # the rect QCommonStyle gives the arrow
        arrow = QStyleOptionButton(option)
        rect = option.rect
        arrow.rect = QRect(
            rect.right() - indicator - 2 + self.MENU_ARROW_RIGHT_TRIM,
            rect.height() // 2 - indicator // 2 + 3,
            indicator - 6,
            indicator - 6,
        )
        painter.drawPrimitive(QStyle.PrimitiveElement.PE_IndicatorArrowDown, arrow)

    def _apply_size(self) -> None:
        self.setFixedSize(self.adjusted_size_hint())

    def setMenu(self, menu) -> None:
        super().setMenu(menu)
        self._apply_size()

    def changeEvent(self, ev):
        super().changeEvent(ev)
        if ev.type() in (
            QEvent.Type.ApplicationFontChange,
            QEvent.Type.FontChange,
            QEvent.Type.StyleChange,
        ):
            self._apply_size()
        elif ev.type() == QEvent.Type.PaletteChange:
            self._refresh_icon()

    @classmethod
    def from_icon_name(cls, icon_name: str, parent=None) -> "MyPushButton":
        button = cls(parent)
        button.set_icon_name(icon_name)
        return button

    def set_icons(self, light_icon: IconLike, dark_icon: IconLike) -> None:
        """The icon for Light mode (black) and for Dark mode (gray20); one given serves both."""
        self._light_icon = self._to_icon(light_icon)
        self._dark_icon = self._to_icon(dark_icon) if dark_icon is not None else self._light_icon
        self._refresh_icon()

    def set_icon_name(self, icon_name: str) -> None:
        base_name = icon_name.removesuffix(".svg")
        light_icon_path = AppPath.icon_path.joinpath(f"{base_name}.svg")
        dark_icon_path = AppPath.icon_path.joinpath(f"{base_name}_gray20.svg")
        for icon_path in (light_icon_path, dark_icon_path):
            if not icon_path.is_file():
                raise FileNotFoundError(f"Icon file not found: {icon_path}")
        self.set_icons(str(light_icon_path), str(dark_icon_path))

    @staticmethod
    def _to_icon(icon: IconLike) -> QIcon:
        if isinstance(icon, QIcon):
            return icon
        if isinstance(icon, str):
            return QIcon(icon)
        return QIcon()

    def _refresh_icon(self) -> None:
        is_dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
        icon = self._dark_icon if is_dark else self._light_icon
        super().setIcon(QIcon(_DisabledOpacityIconEngine(icon, self.DISABLED_ICON_OPACITY)) if not icon.isNull() else icon)

    def setIcon(self, icon: IconLike) -> None:
        self._light_icon = self._dark_icon = self._to_icon(icon)
        self._refresh_icon()


def make_icon_button(icon_name: str, tooltip: str) -> MyPushButton:
    button = MyPushButton.from_icon_name(icon_name)
    button.setToolTip(tooltip)
    return button
