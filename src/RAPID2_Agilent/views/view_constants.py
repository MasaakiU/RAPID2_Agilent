from dataclasses import dataclass


@dataclass(frozen=True)
class DoubleRangeControlSpec:
    decimals: int
    single_step: float
    center_minimum: float
    center_maximum: float
    tolerance_minimum: float
    tolerance_maximum: float


@dataclass(frozen=True)
class MzRangeControlSpec(DoubleRangeControlSpec):
    ppm_minimum: int
    ppm_maximum: int
    ppm_single_step: int


RT_RANGE_CONTROL = DoubleRangeControlSpec(
    decimals =          4,
    single_step =       0.1,
    center_minimum =    0.0,
    center_maximum =    1_000.0 - 10**-4,
    tolerance_minimum = 0.0,
    tolerance_maximum = 10_00.0 - 10**-4,
)

MZ_RANGE_CONTROL = MzRangeControlSpec(
    decimals =          6,
    single_step =       1.0,
    center_minimum =    0.0,
    center_maximum =    10_000.0 - 10**-6,
    tolerance_minimum = 0.0,
    tolerance_maximum = 10_000.0 - 10**-6,
    ppm_minimum =       1,
    ppm_maximum =       999,
    ppm_single_step =   1,
)

# target m/z differs from the one calculated by formula + adduct, and a field's text cannot be read; translucent
# (#AARRGGBB), so that the same red reads over Light and Dark rows and over the selection
MZ_MISMATCH_COLOR = "#73e62828"
# a warning written in a plot, white in Light and Dark mode alike
PLOT_WARNING_COLOR = "#d00000"
# grey while browsing, the selection blue (palette_fix) while a target is edited; RGBA. The RT / m/z windows lie on
# the plots, white in Light and Dark mode alike; the marker of the m/z-RT map on a mostly dark heatmap, so it is light
BROWSING_REGION_COLOR = (128, 128, 128, 80)
TARGET_REGION_COLOR = (120, 170, 240, 96)
BROWSING_MARKER_COLOR = (200, 200, 200, 255)
TARGET_MARKER_COLOR = (156, 195, 255, 255)
# the RT / m/z range text at the top right of a plot follows the same state, as hex for the label HTML
BROWSING_RANGE_LABEL_COLOR = "#C0C0C0"
TARGET_RANGE_LABEL_COLOR = "#9CC3FF"
# red, a target's RT window that is each file's own (the target's peak in it), with its range text; the areas of the
# peaks found under a chromatogram in the same red, so that the target's, under its window too, reads darker
OWN_RT_REGION_COLOR = (255, 0, 0, 50)
OWN_RT_RANGE_LABEL_COLOR = "#FF8080"
PEAK_AREA_COLOR = (255, 0, 0, 50)
# the fit of each peak the peak detector found (its shape on the baseline), and its area +- sigma: for now
PEAK_FIT_COLOR = (160, 0, 0, 255)
# the former style, the same in both states: uncomment to restore
# BROWSING_REGION_COLOR = TARGET_REGION_COLOR = (255, 0, 0, 50)  # red, "#FF000032"
# BROWSING_MARKER_COLOR = TARGET_MARKER_COLOR = (0, 229, 255, 255)  # cyan, "#00e5ff"

# initial view windows (center ± tolerance), also used when a click needs a window width
DEFAULT_RT_TOLERANCE = 0.5
DEFAULT_MZ = 150.5
DEFAULT_MZ_TOLERANCE = 0.05
DEFAULT_MZ_WINDOW = (DEFAULT_MZ - DEFAULT_MZ_TOLERANCE, DEFAULT_MZ + DEFAULT_MZ_TOLERANCE)

# zooming to the RT / m/z window leaves room on each side: as wide as the window for RT; a fixed width for m/z,
# which shows the isotope peaks up to M+2 (+2.007) clear of the edge, and down to M-2 alike
ZOOM_RT_PADDING_RATIO = 1.0
ZOOM_MZ_PADDING = 2.5
