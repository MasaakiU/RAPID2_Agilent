"""Targets to trace (XIC, TIC or BPC); no Qt dependency."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from .chemistry import ion_mz, parse_adduct, parse_formula


class TraceType(StrEnum):
    XIC = "XIC"
    TIC = "TIC"
    BPC = "BPC"


class MSLevel(StrEnum):
    ALL = "All"
    MS1 = "MS1"
    MS2 = "MS2"

    @property
    def number(self) -> int | None:
        """MSLevel of the scans to use; None for all scans."""
        return {MSLevel.ALL: None, MSLevel.MS1: 1, MSLevel.MS2: 2}[self]


class ToleranceUnit(StrEnum):
    MZ = "m/z"
    PPM = "ppm"


def new_target_id() -> str:
    return uuid.uuid4().hex


def tolerance_window(center: float, tolerance: float, unit: ToleranceUnit) -> tuple[float, float]:
    half_width = center * tolerance * 1e-6 if unit is ToleranceUnit.PPM else tolerance
    return center - half_width, center + half_width


@dataclass(frozen=True)
class Target:
    index: int              # unique in the list; 1, 2, ... in the order of addition until reindexed
    name: str
    formula: str            # neutral molecule; "" when unknown
    adduct: str             # carries charge and polarity, e.g. "[M+H]+"
    trace_type: TraceType
    ms_level: MSLevel
    rt: float
    rt_tolerance: float
    mz: float               # the m/z traced: the ion, or the product ion for MS2 targets
    mz_tolerance: float
    mz_tolerance_unit: ToleranceUnit
    # MS2 only: the MS2 scans used, by their precursor and collision energy, and their isolation width
    precursor_mz: float | None = None
    collision_energy: float | None = None
    isolation_width: float | None = None
    note: str = ""
    id: str = field(default_factory=new_target_id)

    def __post_init__(self):
        if self.index < 1:
            raise ValueError(f"index must be >= 1, got {self.index}")
        # as the .rpd.csv export writes them, tab-separated, a line a row; the panel's fields make them spaces
        for field_name in ("name", "note"):
            if any(character in getattr(self, field_name) for character in "\t\r\n"):
                raise ValueError(f"The {field_name} cannot hold a tab or a line break")
        parse_adduct(self.adduct)
        if self.formula:
            parse_formula(self.formula)
        if not isinstance(self.trace_type, TraceType):
            raise TypeError(f"trace_type must be TraceType, got {self.trace_type!r}")
        if not isinstance(self.ms_level, MSLevel):
            raise TypeError(f"ms_level must be MSLevel, got {self.ms_level!r}")
        if not isinstance(self.mz_tolerance_unit, ToleranceUnit):
            raise TypeError(f"mz_tolerance_unit must be ToleranceUnit, got {self.mz_tolerance_unit!r}")
        if self.rt_tolerance < 0 or self.mz_tolerance < 0:
            raise ValueError("Tolerances must be non-negative")
        # the ppm control is an integer spin box
        if self.mz_tolerance_unit is ToleranceUnit.PPM and not float(self.mz_tolerance).is_integer():
            raise ValueError(f"ppm tolerance must be an integer, got {self.mz_tolerance}")
        if self.ms_level is not MSLevel.MS2 and any(
            value is not None for value in (self.precursor_mz, self.collision_energy, self.isolation_width)
        ):
            raise ValueError("precursor_mz, collision_energy and isolation_width are only for MS2 targets")

    def calculated_mz(self) -> float | None:
        if not self.formula:
            return None
        return ion_mz(self.formula, self.adduct)

    def mz_deviates_from_formula(self, decimals: int) -> bool:
        # an MS2 target traces a product ion: its precursor is the one formula + adduct give
        if self.ms_level is MSLevel.MS2:
            return False
        calculated_mz = self.calculated_mz()
        return calculated_mz is not None and round(calculated_mz, decimals) != round(self.mz, decimals)

    def precursor_deviates_from_formula(self) -> bool:
        """Whether the ion of formula + adduct lies outside the isolation window of an MS2 target's precursor."""
        calculated_mz = self.calculated_mz()
        return (
            calculated_mz is not None
            and self.precursor_mz is not None
            and self.isolation_width is not None
            and abs(calculated_mz - self.precursor_mz) > self.isolation_width / 2
        )

    def rt_window(self) -> tuple[float, float]:
        return self.rt - self.rt_tolerance, self.rt + self.rt_tolerance

    def mz_window(self) -> tuple[float, float]:
        return tolerance_window(self.mz, self.mz_tolerance, self.mz_tolerance_unit)
