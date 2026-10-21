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

MZ_MISMATCH_COLOR = "#ffd6d6"   # target m/z differs from the one calculated by formula + adduct
FIELD_ERROR_COLOR = "#d00000"

# initial view windows (center ± tolerance), also used when a click needs a window width
DEFAULT_RT_TOLERANCE = 0.5
DEFAULT_MZ = 150.5
DEFAULT_MZ_TOLERANCE = 0.05
DEFAULT_MZ_WINDOW = (DEFAULT_MZ - DEFAULT_MZ_TOLERANCE, DEFAULT_MZ + DEFAULT_MZ_TOLERANCE)
