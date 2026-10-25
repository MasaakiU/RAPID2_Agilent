# -*- Coding: utf-8 -*-

import sys

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QColor, QPalette

# The colours below are kept per platform (sys.platform) and, within one, per
# Light or Dark mode. A platform with no buttons of its own is left as Qt
# hands it over, and so is a mode with none: Windows's High Contrast themes,
# which Qt reports as neither Light nor Dark.

# Nuitka/PyInstaller macOS app bundles resolve NSColor.windowBackgroundColor
# to its flat catalog value instead of the vibrancy-composited one a real
# window shows, so Qt's system palette ends up near-white/near-black instead
# of the actual Aqua gray. A plain source run (or any non-bundled process)
# is unaffected. Windows's window grey is right as Qt hands it over.
_WINDOW_BACKGROUND = {
    "darwin": {
        Qt.ColorScheme.Light: QColor(236, 236, 236),
        Qt.ColorScheme.Dark: QColor(50, 50, 50),
    },
}

# Buttons are given a colour of their own as well. macOS hands the Button
# role the same grey as the Window one -- its own style draws a button
# without either -- but Fusion paints a button's face from Button and the
# panel round it from Window, so a button stood out by its outline alone.
# Both modes keep macOS's own window grey (236 and 50). Light mode takes the
# button below it, so that the window grey falls between the top of its
# shading and the middle; Dark mode lifts the button a shade above it.
#
# Windows's Light mode hands both roles one grey too, 240, and the button is
# taken as far below it as on macOS. Its Dark mode hands a button of its own
# already, 60 on a window of 30; it is taken two steps of 5 darker, so that
# one under the mouse (see _HOVERED_BUTTON) stands out from it.
_BUTTON = {
    "darwin": {
        Qt.ColorScheme.Light: QColor(211, 211, 211),
        Qt.ColorScheme.Dark: QColor(65, 65, 65),
    },
    "win32": {
        Qt.ColorScheme.Light: QColor(215, 215, 215),
        Qt.ColorScheme.Dark: QColor(50, 50, 50),
    },
}

# Two more colours, which the palette cannot carry and the app's style is handed
# instead (see fusion_style.AppFusionStyle).
#
# A tab's page is lighter than the window round it -- Fusion paints it a step
# up from the window grey -- so a button on one is lighter as well: in Light
# mode the grey whose middle is the window grey. On Windows, as many steps up
# from a button off a page as on macOS.
_TAB_PAGE_BUTTON = {
    "darwin": {
        Qt.ColorScheme.Light: QColor(222, 222, 222),
        Qt.ColorScheme.Dark: QColor(70, 70, 70),
    },
    "win32": {
        Qt.ColorScheme.Light: QColor(226, 226, 226),
        Qt.ColorScheme.Dark: QColor(55, 55, 55),
    },
}
# A button under the mouse, on a page or off it: in Light mode all but white,
# with a trace of shading left (254 at the top, 249 at the foot); in Dark
# mode a grey Fusion draws with 110 in the middle of its shading (119 at the
# top, 101 at the foot). The same on Windows, where the lift the style gives
# otherwise (fusion_style.hover_button_color) left a button looking as it
# was: its middle 245 to 252 in Light mode, 77 to 83 in Dark.
_HOVERED_BUTTON = {
    "darwin": {
        Qt.ColorScheme.Light: QColor(242, 242, 242),
        Qt.ColorScheme.Dark: QColor(85, 85, 85),
    },
    "win32": {
        Qt.ColorScheme.Light: QColor(242, 242, 242),
        Qt.ColorScheme.Dark: QColor(85, 85, 85),
    },
}


# The selection in an item view, focused and not (Active and Inactive). Qt's macOS
# theme hands a pale accent blue and, once the view has lost focus, a grey that
# all but vanishes: Light's 212 on a white row, Dark's 54 on a row of 23. Both
# modes take a blue that stands clear of the rows, and a greyer blue of it for a
# view without focus, so a selection still reads as one there. The text on them
# stays the theme's: black in Light mode, white in Dark.
_HIGHLIGHT = {
    "darwin": {
        Qt.ColorScheme.Light: (QColor(120, 170, 240), QColor(185, 205, 232)),
        Qt.ColorScheme.Dark: (QColor(40, 92, 170), QColor(58, 74, 98)),
    },
}


def _this_modes(table, app):
    """This platform's colour in ``table`` for the current mode, or None."""

    return table.get(sys.platform, {}).get(app.styleHints().colorScheme())


# Qt's macOS theme hands the Inactive group a black ButtonText in Dark mode
# while every other text role stays light, so a button's words vanish into
# its dark face whenever another app is in front. The active colour is the
# right one either way: macOS does not dim a button's text for an inactive
# window, only the accent.
_INACTIVE_TEXT_ROLES = {
    "darwin": (QPalette.ColorRole.ButtonText,),
}


def _corrected_palette(app):
    """The application palette as it should be, worked out from what it is.

    None where this platform has no colours for the current mode.

    Only ever from the current palette, and only after Qt has put the new
    mode's colours into it. A colour handed to ``setPalette`` stays set
    through every later switch -- Qt lays the theme's new palette under the
    roles an application set, not over them -- so a correction made in one
    mode is still there in the next unless it is made again. Worked out
    before the switch had reached the palette, it was made from the old
    mode's colours: Dark mode's white, copied into the Inactive group, stayed
    white in Light mode and left the words on a light button unreadable
    whenever another app was in front.
    """

    button = _this_modes(_BUTTON, app)
    if button is None:
        return None
    palette = QPalette(app.palette())
    background = _this_modes(_WINDOW_BACKGROUND, app)
    if background is not None:
        palette.setColor(QPalette.ColorRole.Window, background)
        palette.setColor(QPalette.ColorRole.Light, background.lighter(110))
        palette.setColor(QPalette.ColorRole.Dark, background.darker(160))
        palette.setColor(QPalette.ColorRole.Mid, background.darker(140))
    palette.setColor(QPalette.ColorRole.Button, button)
    highlight = _this_modes(_HIGHLIGHT, app)
    if highlight is not None:
        active, inactive = highlight
        palette.setColor(QPalette.ColorGroup.Active, QPalette.ColorRole.Highlight, active)
        palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Highlight, inactive)
    for role in _INACTIVE_TEXT_ROLES.get(sys.platform, ()):
        palette.setColor(
            QPalette.ColorGroup.Inactive,
            role,
            palette.color(QPalette.ColorGroup.Active, role),
        )
    return palette


def _hand_style_its_colours(app):
    """Give the app's style this mode's colours that a palette cannot carry."""

    style = app.style()
    if not hasattr(style, "tab_page_button_color"):
        return
    style.tab_page_button_color = _this_modes(_TAB_PAGE_BUTTON, app)
    style.hovered_button_color = _this_modes(_HOVERED_BUTTON, app)


def _any_role_set(palette):
    """Whether the application has set any of ``palette``'s roles.

    PyQt6 has no ``QPalette.resolveMask``; asking role by role is the same.
    """

    return any(
        palette.isBrushSet(group, role)
        for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive, QPalette.ColorGroup.Disabled)
        for role in QPalette.ColorRole
        if role != QPalette.ColorRole.NColorRoles
    )


def _apply_corrected_palette(app):
    # Before the palette: setting that repaints, and the repaint wants these.
    _hand_style_its_colours(app)
    palette = _corrected_palette(app)
    if palette is None:
        # The theme's own colours, in place of any an earlier mode set: they
        # would stay set (see _corrected_palette), and a High Contrast
        # theme's words would go onto the app's grey. A palette with no role
        # set takes every one from the theme.
        if _any_role_set(app.palette()):
            app.setPalette(QPalette())
        return
    # Only when something is wrong: setting it is itself a palette change,
    # and this runs on every one, so an unconditional set would never stop.
    if palette != app.palette():
        app.setPalette(palette)


class _PaletteChangeWatcher(QObject):
    """Corrects the palette each time the application palette changes.

    A palette change rather than ``colorSchemeChanged``, because the scheme
    is announced before the theme's new colours are in the palette, and a
    correction has to be made from those. Filtering ``QStyleHints`` hears
    every ``ApplicationPaletteChange`` without seeing every other event of
    every object, as a filter on the application would.
    """

    def __init__(self, app):
        super().__init__(app.styleHints())
        self._app = app

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.ApplicationPaletteChange:
            _apply_corrected_palette(self._app)
        return False


def install_palette_fix(app):
    """Give the palette the app's colours and keep them through Dark/Light changes.

    On macOS and Windows: the Button role, see ``_BUTTON``, and the style's
    colours for a button on a tab's page and under the mouse, see
    ``_TAB_PAGE_BUTTON`` and ``_HOVERED_BUTTON``. On macOS also the Window
    role, see ``_WINDOW_BACKGROUND``, and the Inactive button text, see
    ``_INACTIVE_TEXT_ROLES``. Set the style first.
    """
    if sys.platform not in _BUTTON:
        return
    _apply_corrected_palette(app)
    style_hints = app.styleHints()
    # Held on the app: PyQt6, unlike PySide6, drops the watcher's Python side
    # with the short-lived ``styleHints()`` wrapper its parent is, leaving a
    # bare QObject that never filters.
    app._palette_change_watcher = _PaletteChangeWatcher(app)
    style_hints.installEventFilter(app._palette_change_watcher)
