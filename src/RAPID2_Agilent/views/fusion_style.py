# -*- Coding: utf-8 -*-

from contextlib import contextmanager

from PyQt6.QtCore import QRect
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QAbstractButton, QPushButton, QProxyStyle, QStackedWidget, QStyle, QTabWidget

# Fusion lights a button under the mouse by drawing it from its colour rather
# than from that colour 4% darker, a step too slight to notice. Where no
# colour for a hovered button has been given (see AppFusionStyle), it is
# handed a lighter one in one of two ways.
#
# Fusion draws a button's top about 20% lighter than its colour, so from this
# value (HSV, 0-255) up the top is white already, and a lift can only bring
# the bottom up towards it.
_WHITE_TOPPED_VALUE = 212

# Such a light button is lifted all the way to this value: the most that
# leaves it any shading, its bottom about 5 levels below its top. Past it the
# face is flat white.
LIGHT_HOVER_VALUE = 242

# A darker one, whose shading has room to move, is handed a colour this much
# lighter, which about doubles Fusion's own step.
HOVER_LIGHTER = 104

# Lent palettes kept for reuse, at most this many; see _palette_with_button.
_LENT_PALETTES_KEPT = 64

_State = QStyle.StateFlag


def hover_button_color(color):
    """The colour Fusion is handed for a button under the mouse."""

    if color.value() < _WHITE_TOPPED_VALUE:
        return color.lighter(HOVER_LIGHTER)
    lifted = QColor(color)
    # A colour already past it keeps Fusion's own step, and no more.
    lifted.setHsv(
        color.hsvHue(),
        color.hsvSaturation(),
        max(color.value(), LIGHT_HOVER_VALUE),
        color.alpha(),
    )
    return lifted


def _lit_by_fusion(option):
    """Whether Fusion draws this face in its hovered colour."""

    state = option.state
    return bool(
        state & _State.State_Enabled
        and state & _State.State_MouseOver
        # Pressed or checked, Fusion draws it flat and darker instead.
        and not state & (_State.State_Sunken | _State.State_On)
    )


def on_tab_page(widget):
    """Whether ``widget`` sits on one of a QTabWidget's pages."""

    while widget is not None:
        parent = widget.parentWidget()
        # The pages are children of the tab widget's own stack.
        if isinstance(parent, QStackedWidget) and isinstance(
            parent.parentWidget(), QTabWidget
        ):
            return True
        widget = parent
    return False


class AppFusionStyle(QProxyStyle):
    """Fusion, with a button under the mouse lit more clearly.

    Push buttons, tool buttons and combo boxes all draw their face through
    ``PE_PanelButtonCommand``, so the one change reaches all of them. Sizes
    are untouched: they come from ``sizeFromContents`` and ``pixelMetric``,
    which this leaves to Fusion.

    Tabs, and the page under them, are drawn from the window colour. Fusion
    paints them from the button colour, lightened, and once buttons had a
    colour of their own (see ``palette_fix``) a page painted from it
    took the buttons on it back into itself. The window colour is what the
    page was drawn from before.

    Two colours a palette cannot carry can be given for the current Light or
    Dark mode, as ``palette_fix`` does: ``tab_page_button_color`` for a
    button on a tab's page, and ``hovered_button_color`` for one under the
    mouse, on a page or off it. Left as None, a button on a page is like any
    other, and a hovered one is lifted by ``hover_button_color``.

    A table's heading under the mouse is lit as a button is. Fusion draws one
    the same with the mouse over it or not, and a heading is told the mouse
    is there only where a view asks for it -- one a click on does something
   . Selected or not: a click on a heading
    of a chosen lane still chooses.
    """

    def __init__(self):
        super().__init__("Fusion")
        # Code that asks whether Fusion draws a widget reads its style's name,
        # and a proxy has none of its own.
        self.setObjectName(self.baseStyle().objectName())
        self.tab_page_button_color = None
        self.hovered_button_color = None
        self._lent_palettes = {}

    def _palette_with_button(self, palette, color):
        """``palette`` with ``color`` as its Button, the same one each time.

        Fusion keeps a drawn combo box in a cache keyed by its palette. A
        palette made afresh for every drawing would never be found there
        again, and fill it with drawings nobody asks for.
        """

        key = (palette.cacheKey(), palette.currentColorGroup(), color.rgba())
        lent = self._lent_palettes.get(key)
        if lent is None:
            if len(self._lent_palettes) >= _LENT_PALETTES_KEPT:
                self._lent_palettes.clear()
            lent = QPalette(palette)
            lent.setColor(QPalette.ColorRole.Button, color)
            self._lent_palettes[key] = lent
        return lent

    @contextmanager
    def _button_color_lent(self, option, color):
        """Hand Fusion ``color`` as the Button colour for one drawing.

        The caller's own option, lent and given back: a copy made here would
        be a bare QStyleOption, and a tool button's is not one.
        """

        original = QPalette(option.palette)
        option.palette = self._palette_with_button(original, color)
        try:
            yield
        finally:
            option.palette = original

    def _resting_button_color(self, widget):
        """The colour for ``widget``'s face where the palette's will not do."""

        if self.tab_page_button_color is not None and on_tab_page(widget):
            return self.tab_page_button_color
        return None

    def _button_face_color(self, option, widget):
        """The colour for a button's face, or None to leave the palette's."""

        if isinstance(widget, QAbstractButton) and option.state & _State.State_Enabled:
            # pressed, or a checked toggle: the icon colour is the theme's alone
            if option.state & _State.State_Sunken or (
                widget.isCheckable() and option.state & _State.State_On
            ):
                return option.palette.color(QPalette.ColorRole.Highlight)
        resting = self._resting_button_color(widget)
        if not _lit_by_fusion(option):
            return resting
        if self.hovered_button_color is not None:
            return self.hovered_button_color
        return hover_button_color(
            resting or option.palette.color(QPalette.ColorRole.Button)
        )

    def drawPrimitive(self, element, option, painter, widget=None):
        if element == QStyle.PrimitiveElement.PE_IndicatorArrowDown and isinstance(widget, QPushButton):
            # A menu button's arrow, as large as a combo box's. Fusion draws
            # an arrow no larger than it may be, and smaller to fit a smaller
            # rect: the combo box's room for it is larger than that, but the
            # square QCommonStyle gives a menu button is not. Three times as
            # wide about the same centre, it is.
            rect = QRect(option.rect)
            option.rect = rect.adjusted(-rect.width(), -rect.height(), rect.width(), rect.height())
            try:
                super().drawPrimitive(element, option, painter, widget)
            finally:
                option.rect = rect
            return
        if element == QStyle.PrimitiveElement.PE_PanelButtonCommand:
            button = self._button_face_color(option, widget)
        elif element == QStyle.PrimitiveElement.PE_FrameTabWidget:
            button = option.palette.color(QPalette.ColorRole.Window)
        else:
            button = None
        if button is None:
            super().drawPrimitive(element, option, painter, widget)
            return
        with self._button_color_lent(option, button):
            super().drawPrimitive(element, option, painter, widget)

    def drawComplexControl(self, control, option, painter, widget=None):
        # Fusion keeps a drawn combo box in a cache keyed by its palette, so a
        # page's colour has to be in that palette before the key is taken:
        # lent only once the face is drawn, a combo box on a page and one off
        # it, alike otherwise, would share one drawing.
        resting = (
            self._resting_button_color(widget)
            if control == QStyle.ComplexControl.CC_ComboBox
            else None
        )
        if resting is None:
            super().drawComplexControl(control, option, painter, widget)
            return
        with self._button_color_lent(option, resting):
            super().drawComplexControl(control, option, painter, widget)

    def _heading_face_color(self, option):
        """The colour for a heading under the mouse, or None to leave the palette's."""

        state = option.state
        if not (state & _State.State_Enabled and state & _State.State_MouseOver):
            return None
        if self.hovered_button_color is not None:
            return self.hovered_button_color
        return hover_button_color(option.palette.color(QPalette.ColorRole.Button))

    def drawControl(self, element, option, painter, widget=None):
        if element == QStyle.ControlElement.CE_HeaderSection:
            lit = self._heading_face_color(option)
            if lit is None:
                super().drawControl(element, option, painter, widget)
                return
            with self._button_color_lent(option, lit):
                super().drawControl(element, option, painter, widget)
            return
        if element != QStyle.ControlElement.CE_TabBarTabShape:
            super().drawControl(element, option, painter, widget)
            return
        with self._button_color_lent(
            option, option.palette.color(QPalette.ColorRole.Window)
        ):
            super().drawControl(element, option, painter, widget)
