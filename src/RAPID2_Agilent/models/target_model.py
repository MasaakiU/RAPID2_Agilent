"""Qt list model of the targets; the targets themselves are in target.py."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from .run_peaks import RunPeaks, RunPeaksStore
from .target import Target


class TargetListModel(QObject):
    """Ordered target list with snapshot-based undo/redo of every edit; of the targets' RT windows in the files
    (run_peaks) with them, which an edit of a target may change, as RunPeaksStore.follow does."""

    targets_reset = pyqtSignal()
    target_changed = pyqtSignal(str)
    undo_state_changed = pyqtSignal(bool, bool)  # can_undo, can_redo

    def __init__(self):
        super().__init__()
        self._targets: list[Target] = []
        self.run_peaks = RunPeaksStore()
        # (targets, run_peaks.snapshot())
        self._undo_stack: list[tuple[tuple[Target, ...], dict]] = []
        self._redo_stack: list[tuple[tuple[Target, ...], dict]] = []

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

    def update_many(self, changes: dict[str, dict]) -> None:
        """Change several targets, {target id: changes}, as one edit to undo; none of them if one cannot be."""
        targets = list(self._targets)
        changed_ids = []
        for target_id, target_changes in changes.items():
            if "index" in target_changes:
                raise ValueError("Use reindex() to change indices")
            index = self.index_of(target_id)
            new = replace(targets[index], **target_changes)
            if new != targets[index]:
                targets[index] = new
                changed_ids.append(target_id)
        if not changed_ids:
            return
        self._push_undo()
        self._targets = targets
        for target_id in changed_ids:
            self.target_changed.emit(target_id)

    def import_targets(self, targets: list[Target], *, replace: bool) -> None:
        """Put targets in place of the list, or after it, as one edit to undo."""
        kept = [] if replace else self._targets
        imported = kept + list(targets)
        if len({target.id for target in imported}) < len(imported):
            raise ValueError("Duplicate target id")
        if len({target.index for target in imported}) < len(imported):
            raise ValueError("Duplicate target index")
        if imported == self._targets:
            return
        self._push_undo()
        self._targets = imported
        self.targets_reset.emit()

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

    def set_run_peaks(self, changes: Sequence[tuple[str, str | Path, RunPeaks]]) -> None:
        """Set the RunPeaks of targets in files, (target id, file, RunPeaks) each, as one edit to undo."""
        changes = [change for change in changes if self.run_peaks.get(change[0], change[1]) != change[2]]
        if not changes:
            return
        self._push_undo()
        for target_id, run, run_peaks in changes:
            self.run_peaks.set(target_id, run, run_peaks)

    def set_found_run_peaks(self, found: Sequence[tuple[str, str | Path, RunPeaks]], *, keep_manual: bool) -> None:
        """Set the peaks found of targets in files, (target id, file, RunPeaks) each, as one edit to undo; see
        RunPeaksStore.set_found."""
        self._push_undo()
        for target_id, run, run_peaks in found:
            self.run_peaks.set_found(target_id, run, run_peaks, keep_manual=keep_manual)

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> None:
        previous = self._undo_stack.pop()
        self._redo_stack.append(self._snapshot())
        self._restore(previous)
        self._emit_undo_state()
        self.targets_reset.emit()

    def redo(self) -> None:
        following = self._redo_stack.pop()
        self._undo_stack.append(self._snapshot())
        self._restore(following)
        self._emit_undo_state()
        self.targets_reset.emit()

    def _push_undo(self) -> None:
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()
        self._emit_undo_state()

    def _snapshot(self) -> tuple[tuple[Target, ...], dict]:
        return tuple(self._targets), self.run_peaks.snapshot()

    def _restore(self, snapshot: tuple[tuple[Target, ...], dict]) -> None:
        targets, run_peaks = snapshot
        self._targets = list(targets)
        self.run_peaks.restore(run_peaks)

    def _emit_undo_state(self) -> None:
        self.undo_state_changed.emit(self.can_undo(), self.can_redo())
