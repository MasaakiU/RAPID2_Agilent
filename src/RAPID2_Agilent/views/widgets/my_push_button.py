from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QPushButton

from RAPID2_Agilent.utils.path import AppPath


IconLike = QIcon | str | None


class MyPushButton(QPushButton):
    default_width = 27
    default_icon_size = 16

    def __init__(
        self,
        parent=None,
        *,
        normal_icon: IconLike = None,
        pressed_icon: IconLike = None,
    ):
        super().__init__(parent)
        self.setObjectName("MyPushButton")
        self._normal_icon = QIcon()
        self._pressed_icon = QIcon()
        self._is_pressed = False
        self._is_inside = False
        self.setMouseTracking(True)
        self.setFixedWidth(self.default_width)
        self.setIconSize(QSize(self.default_icon_size, self.default_icon_size))

        self.toggled.connect(self._refresh_icon)
        self.pressed.connect(self._on_pressed)
        self.released.connect(self._on_released)

        self.set_icons(normal_icon, pressed_icon)

    @classmethod
    def from_icon_name(cls, icon_name: str, parent=None) -> "MyPushButton":
        button = cls(parent)
        button.set_icon_name(icon_name)
        return button

    def set_icons(self, normal_icon: IconLike, pressed_icon: IconLike) -> None:
        self._normal_icon = self._to_icon(normal_icon)
        self._pressed_icon = self._to_icon(pressed_icon) if pressed_icon is not None else self._normal_icon
        self._refresh_icon()

    def setIcons(self, icon0: IconLike, icon1: IconLike) -> None:
        self.set_icons(icon0, icon1)

    def set_icon_name(self, icon_name: str) -> None:
        base_name = icon_name.removesuffix(".svg")
        normal_icon_path = AppPath.icon_path.joinpath(f"{base_name}.svg")
        pressed_icon_path = AppPath.icon_path.joinpath(f"{base_name}_gray20.svg")
        for icon_path in (normal_icon_path, pressed_icon_path):
            if not icon_path.is_file():
                raise FileNotFoundError(f"Icon file not found: {icon_path}")
        self.set_icons(str(normal_icon_path), str(pressed_icon_path))

    @staticmethod
    def _to_icon(icon: IconLike) -> QIcon:
        if isinstance(icon, QIcon):
            return icon
        if isinstance(icon, str):
            return QIcon(icon)
        return QIcon()

    def _on_pressed(self) -> None:
        self._is_pressed = True
        self._is_inside = True
        self._refresh_icon()

    def _on_released(self) -> None:
        self._is_pressed = False
        self._is_inside = False
        self._refresh_icon()

    def _refresh_icon(self, *_args) -> None:
        use_pressed_icon = self.isEnabled() and (
            self.isChecked() or (self._is_pressed and self._is_inside)
        )
        super().setIcon(self._pressed_icon if use_pressed_icon else self._normal_icon)

    def mouseMoveEvent(self, ev):
        if self._is_pressed and not self.isChecked():
            self._is_inside = self._event_pos_is_inside(ev)
            self._refresh_icon()
        super().mouseMoveEvent(ev)

    def enterEvent(self, ev):
        self._is_inside = True
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._is_inside = False
        if self._is_pressed and not self.isChecked():
            self._refresh_icon()
        super().leaveEvent(ev)

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self._refresh_icon()

    def setIcon(self, icon: IconLike) -> None:
        self._normal_icon = self._to_icon(icon)
        self._refresh_icon()

    def _event_pos_is_inside(self, ev) -> bool:
        position = ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
        return self.hitButton(position)


def make_icon_button(icon_name: str, tooltip: str) -> MyPushButton:
    button = MyPushButton.from_icon_name(icon_name)
    button.setToolTip(tooltip)
    return button
