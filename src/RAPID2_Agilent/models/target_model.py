from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from enum import StrEnum

from PyQt6.QtCore import QObject, pyqtSignal

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
    mz: float               # precursor m/z for MS2 targets
    mz_tolerance: float
    mz_tolerance_unit: ToleranceUnit
    product_ion: float | None = None        # MS2 only
    collision_energy: float | None = None   # MS2 only
    note: str = ""
    id: str = field(default_factory=new_target_id)

    def __post_init__(self):
        if self.index < 1:
            raise ValueError(f"index must be >= 1, got {self.index}")
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
        if self.ms_level is not MSLevel.MS2 and (self.product_ion is not None or self.collision_energy is not None):
            raise ValueError("product_ion and collision_energy are only for MS2 targets")

    def calculated_mz(self) -> float | None:
        if not self.formula:
            return None
        return ion_mz(self.formula, self.adduct)

    def mz_deviates_from_formula(self, decimals: int) -> bool:
        calculated_mz = self.calculated_mz()
        return calculated_mz is not None and round(calculated_mz, decimals) != round(self.mz, decimals)

    def rt_window(self) -> tuple[float, float]:
        return self.rt - self.rt_tolerance, self.rt + self.rt_tolerance

    def mz_window(self) -> tuple[float, float]:
        return tolerance_window(self.mz, self.mz_tolerance, self.mz_tolerance_unit)


class TargetListModel(QObject):
    """Ordered target list with snapshot-based undo/redo of every edit."""

    targets_reset = pyqtSignal()
    target_changed = pyqtSignal(str)
    undo_state_changed = pyqtSignal(bool, bool)  # can_undo, can_redo

    def __init__(self):
        super().__init__()
        self._targets: list[Target] = []
        self._undo_stack: list[tuple[Target, ...]] = []
        self._redo_stack: list[tuple[Target, ...]] = []

    @property
    def targets(self) -> tuple[Target, ...]:
        return tuple(self._targets)

    def get(self, target_id: str) -> Target:
        return self._targets[self.index_of(target_id)]

    def index_of(self, target_id: str) -> int:
        for index, target in enumerate(self._targets):
            if target.id == target_id:
                return index
        raise KeyError(target_id)

    def contains(self, target_id: str) -> bool:
        return any(target.id == target_id for target in self._targets)

    def next_index(self) -> int:
        return max((target.index for target in self._targets), default=0) + 1

    def add(self, target: Target) -> None:
        if self.contains(target.id):
            raise ValueError(f"Duplicate target id: {target.id}")
        if any(existing.index == target.index for existing in self._targets):
            raise ValueError(f"Duplicate target index: {target.index}")
        self._push_undo()
        self._targets.append(target)
        self.targets_reset.emit()

    def update(self, target_id: str, **changes) -> Target:
        index = self.index_of(target_id)
        old = self._targets[index]
        if "index" in changes:
            raise ValueError("Use reindex() to change indices")
        new = replace(old, **changes)
        if new == old:
            return old
        self._push_undo()
        self._targets[index] = new
        self.target_changed.emit(target_id)
        return new

    def remove(self, target_ids: list[str]) -> None:
        indices = {self.index_of(target_id) for target_id in target_ids}
        if not indices:
            return
        self._push_undo()
        self._targets = [target for index, target in enumerate(self._targets) if index not in indices]
        self.targets_reset.emit()

    def reindex(self, target_ids: list[str]) -> None:
        """Renumber all targets 1, 2, ... in the given order."""
        if sorted(target_ids) != sorted(target.id for target in self._targets):
            raise ValueError("reindex() needs every target id exactly once")
        new_index = {target_id: i + 1 for i, target_id in enumerate(target_ids)}
        targets = [replace(target, index=new_index[target.id]) for target in self._targets]
        if targets == self._targets:
            return
        self._push_undo()
        self._targets = targets
        self.targets_reset.emit()

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> None:
        previous = self._undo_stack.pop()
        self._redo_stack.append(tuple(self._targets))
        self._targets = list(previous)
        self._emit_undo_state()
        self.targets_reset.emit()

    def redo(self) -> None:
        following = self._redo_stack.pop()
        self._undo_stack.append(tuple(self._targets))
        self._targets = list(following)
        self._emit_undo_state()
        self.targets_reset.emit()

    def _push_undo(self) -> None:
        self._undo_stack.append(tuple(self._targets))
        self._redo_stack.clear()
        self._emit_undo_state()

    def _emit_undo_state(self) -> None:
        self.undo_state_changed.emit(self.can_undo(), self.can_redo())
