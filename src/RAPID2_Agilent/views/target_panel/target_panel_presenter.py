from dataclasses import replace

from PyQt6.QtCore import QObject, pyqtSignal

from RAPID2_Agilent.models.chemistry import formula_mass, ion_mz, parse_adduct, parse_formula
from RAPID2_Agilent.models.rpd_model import BaseMSDataModel
from RAPID2_Agilent.models.run_peaks import run_peaks_for
from RAPID2_Agilent.models.target import (
    MSLevel,
    Target,
    ToleranceUnit,
    TraceType,
    new_target_id,
    tolerance_window,
)
from RAPID2_Agilent.models.settings_model import SettingsModel
from RAPID2_Agilent.models.target_model import TargetListModel
from RAPID2_Agilent.services import operations
from RAPID2_Agilent.views.data_stack import ChromatogramSpectrumStackPresenter
from RAPID2_Agilent.views.view_constants import MZ_RANGE_CONTROL

from .target_panel import TargetPanel
from .target_table_model import TargetTableModel


class TargetPanelPresenter(QObject):
    """Connects the target panel, the target list and the plots.

    With exactly one target selected, the panel edits that target and the plots
    follow it. Otherwise the panel holds a draft: its view parameters only move the
    plots, its formula and adduct set the m/z, and Add makes it a target. Edits in
    the list change their target directly. A window dragged on the plots shows in
    the panel and in the list as it moves, and changes the target where it is dropped;
    but for the RT window of a target while the files' are not linked, which is each
    file's own, the target's peak in it, kept in the target list's run_peaks and
    undone with its edits: the target's RT and tolerance, the window the peak is
    looked for in, are the panel's then.
    What is typed in the panel shows in the list as it is typed, and what is typed in
    the list in the panel; it changes the target once the field or the cell is left.

    With several targets selected, the panel edits them all at once: a field changed
    changes that field of each, as one edit to undo, the others staying as they
    are, and Add copies them all. The plots show one of them, the one last clicked,
    which they say in red; their RT and m/z windows are hidden where those of the
    targets differ, and a click on a plot sets one there, for them all.

    For MS2 the m/z is the product ion, and the scans are those of a precursor and
    collision energy chosen from the ones in the files; formula and adduct choose
    the precursor whose isolation window holds their ion, keeping the CE if it can.
    """

    editing_target_changed = pyqtSignal(bool)  # whether the panel and the plots edit a target

    default_name = "New target"
    default_adduct = "[M+H]+"
    mz_decimals = MZ_RANGE_CONTROL.decimals

    def __init__(
        self,
        panel: TargetPanel,
        target_list_model: TargetListModel,
        data_stack_presenter: ChromatogramSpectrumStackPresenter,
        settings_model: SettingsModel,
        parent=None,
    ):
        super().__init__(parent)
        self.panel = panel
        self.target_list_model = target_list_model
        self.data_stack_presenter = data_stack_presenter
        self.settings_model = settings_model
        self.run_peaks = target_list_model.run_peaks
        self.current_id: str | None = None
        # with several targets selected, those, edited all at once, and of them the one the plots show
        self.batch_ids: list[str] = []
        self.batch_shown_id: str | None = None
        self._updating_batch = False
        self.editing_target = False
        self._updating_from_plot = False
        self._precursors: list[operations.Precursor] = []  # of the files' MS2 scans
        # the draft's precursor_mz, collision_energy and isolation_width; it goes on from those of a target shown
        self._draft_ms2: dict | None = None
        self._polarity: int | None = None  # of the scans used, see _apply_polarity

        self.table_model = TargetTableModel(self.target_list_model, mz_decimals=self.mz_decimals, parent=self)
        self.panel.set_table_model(self.table_model)
        shown_fields = self.settings_model.target_table_shown_fields
        self.panel.set_column_settings(
            TargetTableModel.DEFAULT_SHOWN_FIELDS if shown_fields is None else shown_fields,
            self.settings_model.target_table_partial_fields,
        )
        self.panel.set_merge_ppm(self.settings_model.ms2_merge_ppm)
        self.data_stack_presenter.set_link_rt_windows(self.settings_model.link_rt_windows)
        self.panel.show_target(None)
        self._refresh_precursors()

        self.panel.selection_changed.connect(self.on_selection_changed)
        self.panel.rt_edited.connect(self.on_rt_edited)
        self.panel.mz_edited.connect(self.on_mz_edited)
        self.panel.trace_type_edited.connect(self.on_trace_type_edited)
        self.panel.ms_level_edited.connect(self.on_ms_level_edited)
        self.panel.field_edited.connect(self.on_field_edited)
        self.panel.field_typed.connect(self.on_field_typed)
        self.panel.typing_finished.connect(self.table_model.clear_preview)
        self.panel.cell_typed.connect(self.on_cell_typed)
        self.panel.cell_editing_finished.connect(self.on_cell_editing_finished)
        self.panel.chemistry_text_changed.connect(self.update_chemistry_preview)
        self.panel.precursor_chosen.connect(self.on_precursor_chosen)
        self.panel.merge_ppm_edited.connect(self.on_merge_ppm_edited)
        self.table_model.edit_requested.connect(self.on_table_edited)
        self.panel.add_requested.connect(self.add_target_from_view)
        self.panel.remove_requested.connect(self.remove_selected_targets)
        self.panel.reset_index_requested.connect(self.reset_index)
        self.panel.columns_changed.connect(self.on_columns_changed)
        self.target_list_model.target_changed.connect(self.on_target_changed)
        self.target_list_model.targets_reset.connect(self.on_targets_reset)
        self.data_stack_presenter.rt_window_changed.connect(self.on_plot_rt_window_changed)
        self.data_stack_presenter.mz_window_changed.connect(self.on_plot_mz_window_changed)
        self.data_stack_presenter.rt_window_dragged.connect(self.on_plot_rt_window_dragged)
        self.data_stack_presenter.mz_window_dragged.connect(self.on_plot_mz_window_dragged)
        self.data_stack_presenter.run_rt_window_changed.connect(self.on_plot_run_rt_window_changed)
        self.data_stack_presenter.run_no_peak_requested.connect(self.on_plot_run_no_peak)
        self.data_stack_presenter.model_added.connect(self.on_model_added)
        self.data_stack_presenter.models_changed.connect(self._refresh_precursors)

    ### operations ###
    @property
    def shown_id(self) -> str | None:
        """The target the plots show: the one selected, or of several the one last clicked."""
        return self.current_id if self.current_id is not None else self.batch_shown_id

    def select_target(self, target_id: str | None):
        self.panel.select_target(target_id)
        self.current_id = target_id
        self._set_batch([], None)
        self._show_current_target()

    def select_targets(self, target_ids: list[str], shown_id: str | None = None):
        """Select several targets, shown_id the one the plots show; one or none as select_target()."""
        if len(target_ids) <= 1:
            self.select_target(target_ids[0] if target_ids else None)
            return
        self.panel.select_targets(target_ids, shown_id)
        self.current_id = None
        self._set_batch(target_ids, shown_id)
        self._show_current_target()

    def jump_to(self, rt: float, mz: float):
        """Move the view to (rt, mz) keeping the tolerances.

        A selected target is deselected, not edited. m/z is kept for traces other than XIC.
        """
        self.select_target(None)
        _rt, rt_tolerance = self.panel.rt_values()
        self.panel.set_rt_values(rt, rt_tolerance)
        self.data_stack_presenter.set_rt_window((rt - rt_tolerance, rt + rt_tolerance))
        if self.panel.trace_type() is not TraceType.XIC:
            self.data_stack_presenter.ensure_visible(rt, None)
            return
        _mz, mz_tolerance, mz_tolerance_unit = self.panel.mz_values()
        self.panel.set_mz_values(mz, mz_tolerance, mz_tolerance_unit)
        self.data_stack_presenter.set_mz_window(tolerance_window(mz, mz_tolerance, mz_tolerance_unit))
        self.data_stack_presenter.ensure_visible(rt, mz)

    def show_view(
        self,
        *,
        rt_window: tuple[float, float] | None = None,
        mz_window: tuple[float, float] | None = None,
        trace_type: TraceType | None = None,
        ms_level: MSLevel | None = None,
        ms2: operations.MS2Selection | None = None,
    ):
        """Show a view asked for from outside the panel (the AI assistant); None leaves a part as it is.

        As with jump_to, a selected target is deselected, not edited, and the draft takes the view.
        ms2, for MS2, is one of the listed precursors and CEs; without it MS2 goes on with the
        draft's, or the default one. An m/z window in ppm snaps to a whole ppm.
        """
        ms2_fields = None
        if ms2 is not None:
            index = self._listed_index(ms2.precursor_mz, ms2.collision_energy)
            if index is None:
                raise ValueError(f"Precursor {ms2.precursor_mz:.4f} at CE {ms2.collision_energy:g} is not listed")
            ms2_fields = self._ms2_fields(self._precursors[index])
        self.select_target(None)
        if trace_type is not None:
            self.panel.set_trace_type(trace_type)
            self.data_stack_presenter.set_trace_type(trace_type)
        if ms2_fields is not None:
            self._draft_ms2 = ms2_fields
        if ms_level is not None:
            self.panel.set_ms_level(ms_level)
        if ms_level is not None or ms2_fields is not None:
            self.on_ms_level_edited()
        if rt_window is not None:
            self._show_plot_rt_window(*rt_window)
            self.data_stack_presenter.set_rt_window(rt_window)
        if mz_window is not None:
            changes = self._show_plot_mz_window(*mz_window)
            _mz, _mz_tolerance, mz_tolerance_unit = self.panel.mz_values()
            self.data_stack_presenter.set_mz_window(tolerance_window(changes["mz"], changes["mz_tolerance"], mz_tolerance_unit))

    def add_target_from_view(self) -> Target | None:
        """A new target: with one selected, a copy of it; without, the draft of the current view. With several
        selected, copies of them all (duplicate_selected_targets), the one of the target shown returned.

        The draft brings its name, chemistry and note, and the view the precursor
        and CE of MS2; None, and the field marked, when one of them does not parse.
        """
        # a field still being typed counts, as if it had been left
        self.panel.commit_pending_fields()
        if self.batch_ids:
            shown_index = self.batch_ids.index(self.batch_shown_id)
            return self.duplicate_selected_targets()[shown_index]
        if self.current_id is not None:
            return self.duplicate_current_target()
        identity = self._draft_identity()
        if identity is None:
            return None
        rt, rt_tolerance = self.panel.rt_values()
        mz, mz_tolerance, mz_tolerance_unit = self.panel.mz_values()
        ms_level = self.panel.ms_level()
        target = Target(
            index=self.target_list_model.next_index(),
            **identity,
            trace_type=self.panel.trace_type(),
            ms_level=ms_level,
            rt=rt,
            rt_tolerance=rt_tolerance,
            mz=mz,
            mz_tolerance=mz_tolerance,
            mz_tolerance_unit=mz_tolerance_unit,
            **((self._current_ms2_fields() or {}) if ms_level is MSLevel.MS2 else {}),
        )
        self._add_and_select(target)
        return target

    def duplicate_current_target(self) -> Target:
        target = replace(
            self.target_list_model.get(self.current_id),
            id=new_target_id(),
            index=self.target_list_model.next_index(),
        )
        self._add_and_select(target)
        return target

    def duplicate_selected_targets(self) -> list[Target]:
        """Copies of the targets selected, numbered on in the list's order, added as one edit to undo and selected in
        their place; the plots show the copy of the one they showed."""
        next_index = self.target_list_model.next_index()
        copies = [
            replace(self.target_list_model.get(target_id), id=new_target_id(), index=next_index + offset)
            for offset, target_id in enumerate(self.batch_ids)
        ]
        shown_copy = copies[self.batch_ids.index(self.batch_shown_id)]
        self.target_list_model.import_targets(copies, replace=False)
        self.select_targets([copy.id for copy in copies], shown_copy.id)
        return copies

    def remove_selected_targets(self):
        removed_ids = self.panel.selected_target_ids()
        next_id = self.panel.target_id_after_removal(removed_ids)
        self.target_list_model.remove(removed_ids)
        self.select_target(next_id)

    def remove_all_targets(self):
        self.target_list_model.remove([target.id for target in self.target_list_model.targets])

    def reset_index(self):
        self.target_list_model.reindex(self.panel.displayed_target_ids())

    def _add_and_select(self, target: Target):
        self.target_list_model.add(target)
        self.select_target(target.id)

    ### panel -> model / plots ###
    def on_selection_changed(self):
        selected_ids = self.panel.selected_target_ids()
        self.current_id = selected_ids[0] if len(selected_ids) == 1 else None
        self._set_batch(selected_ids if len(selected_ids) > 1 else [], self.panel.current_target_id())
        self._show_current_target()

    def _set_batch(self, target_ids: list[str], shown_id: str | None):
        """Several targets selected, edited all at once, shown_id the one the plots show (the first if it is not one of
        them); none for one or none selected. A draft after them goes on from the view of the one shown."""
        shown_before = self.batch_shown_id
        self.batch_ids = list(target_ids)
        self.batch_shown_id = (shown_id if shown_id in target_ids else target_ids[0]) if target_ids else None
        if (
            shown_before is not None and not target_ids and self.current_id is None
            and self.target_list_model.contains(shown_before)
        ):
            self.panel.show_target(self.target_list_model.get(shown_before))

    def _batch_targets(self) -> list[Target]:
        return [self.target_list_model.get(target_id) for target_id in self.batch_ids]

    def _update_batch(self, changes_of):
        """Change each target selected by changes_of(target), as one edit to undo, and show them; none is changed when
        changes_of raises (ValueError)."""
        changes = {target.id: changes_of(target) for target in self._batch_targets()}
        self._updating_batch = True
        try:
            self.target_list_model.update_many(changes)
        finally:
            self._updating_batch = False
        self._show_current_target(apply_to_plots=not self._updating_from_plot)

    def on_rt_edited(self, field_name: str = "rt"):
        rt, rt_tolerance = self.panel.rt_values()
        if self.batch_ids:
            value = rt if field_name == "rt" else rt_tolerance
            self._update_batch(lambda _target: {field_name: value})
        elif self.current_id is None:
            self.data_stack_presenter.set_rt_window((rt - rt_tolerance, rt + rt_tolerance))
        else:
            self.target_list_model.update(self.current_id, rt=rt, rt_tolerance=rt_tolerance)

    def on_mz_edited(self, field_name: str = "mz"):
        mz, mz_tolerance, mz_tolerance_unit = self.panel.mz_values()
        if self.batch_ids:
            self._on_batch_mz_edited(field_name, mz, mz_tolerance, mz_tolerance_unit)
        elif self.current_id is None:
            self.data_stack_presenter.set_mz_window(tolerance_window(mz, mz_tolerance, mz_tolerance_unit))
        else:
            self.target_list_model.update(
                self.current_id,
                mz=mz,
                mz_tolerance=mz_tolerance,
                mz_tolerance_unit=mz_tolerance_unit,
            )

    def _on_batch_mz_edited(self, field_name: str, mz: float, mz_tolerance: float, mz_tolerance_unit: ToleranceUnit):
        if field_name == "mz":
            self._update_batch(lambda _target: {"mz": mz})
        elif field_name == "mz_tolerance":
            # with the unit shown, which they may not all have had
            self._update_batch(lambda _target: {"mz_tolerance": mz_tolerance, "mz_tolerance_unit": mz_tolerance_unit})
        else:
            # each keeps its window, as in the list
            try:
                self._update_batch(lambda target: self._tolerance_unit_changes(target, mz_tolerance_unit))
            except ValueError:
                self._show_current_target(apply_to_plots=False)  # the unit as it was

    def on_trace_type_edited(self):
        trace_type = self.panel.trace_type()
        if self.batch_ids:
            self._update_batch(lambda _target: {"trace_type": trace_type})
        elif self.current_id is None:
            self.data_stack_presenter.set_trace_type(trace_type)
        else:
            self.target_list_model.update(self.current_id, trace_type=trace_type)

    def on_ms_level_edited(self):
        ms_level = self.panel.ms_level()
        if self.batch_ids:
            self._update_batch(lambda target: self._ms_level_changes(target, ms_level))
            return
        if self.current_id is None:
            if ms_level is MSLevel.MS2 and self._draft_ms2 is None:
                self._draft_ms2 = self._ms2_fields(self._default_precursor(self._draft_ion_mz()))
            self.data_stack_presenter.set_ms2_selection(self._ms2_selection(self._draft_ms2))
            self.data_stack_presenter.set_ms_level(ms_level.number)
            self._show_precursor()
            return
        target = self.target_list_model.get(self.current_id)
        self.target_list_model.update(self.current_id, **self._ms_level_changes(target, ms_level))

    def on_precursor_chosen(self, index: int):
        fields = self._ms2_fields(self._precursors[index])
        if self.batch_ids:
            self._update_batch(lambda _target: fields)
        elif self.current_id is None:
            self._draft_ms2 = fields
            self.data_stack_presenter.set_ms2_selection(self._ms2_selection(fields))
            self._show_precursor()
        else:
            self.target_list_model.update(self.current_id, **fields)

    def on_merge_ppm_edited(self, ppm: int):
        self.settings_model.ms2_merge_ppm = ppm
        self._refresh_precursors()

    def on_field_edited(self, field_name: str, text: str):
        if self.batch_ids:
            try:
                self._update_batch(lambda target: self._parse_field(target, field_name, text.strip()))
            except ValueError as error:
                self.panel.set_field_error(field_name, str(error))
            return
        if self.current_id is None:
            self._on_draft_field_edited(field_name, text.strip())
            return
        target = self.target_list_model.get(self.current_id)
        try:
            self.target_list_model.update(target.id, **self._parse_field(target, field_name, text.strip()))
        except ValueError as error:
            self.panel.set_field_error(field_name, str(error))
            return
        self.panel.set_field_error(field_name, None)

    def on_field_typed(self, field_name: str, value):
        if self.batch_ids:
            changes = {field_name: value.strip() if isinstance(value, str) else value}
            if field_name == "mz_tolerance":
                changes["mz_tolerance_unit"] = self.panel.mz_values()[2]
            try:
                self.table_model.set_previews([replace(target, **changes) for target in self._batch_targets()])
            except ValueError:
                pass  # not yet one of their values
        elif self.current_id is not None:
            typed = self._typed_target(self.target_list_model.get(self.current_id), field_name, value)
            if typed is not None:
                self.table_model.set_preview(typed)

    def on_cell_typed(self, target_id: str, field_name: str, value):
        if target_id == self.current_id:
            typed = self._typed_target(self.target_list_model.get(target_id), field_name, value)
            if typed is not None:
                self.panel.show_target(typed)

    def on_cell_editing_finished(self):
        # committed, or dropped: the panel shows the target as it is
        if self.current_id is not None:
            self.panel.show_target(self.target_list_model.get(self.current_id))

    @staticmethod
    def _typed_target(target: Target, field_name: str, value) -> Target | None:
        """``target`` with what is typed in a field; None while it cannot be one of its values."""
        try:
            return replace(target, **{field_name: value.strip() if isinstance(value, str) else value})
        except ValueError:
            return None

    def update_chemistry_preview(self):
        """Masses of the formula / adduct being typed (not yet committed)."""
        formula, adduct = self.panel.chemistry_texts()
        neutral_mass = mz = None
        if formula:
            try:
                neutral_mass = formula_mass(formula)
                # none of several targets whose adducts differ
                if "adduct" not in self.panel.mixed_fields:
                    mz = ion_mz(formula, adduct or self.default_adduct)
            except ValueError:
                pass  # incomplete input while typing
        self.panel.set_chemistry_preview(neutral_mass, mz, decimals=self.mz_decimals)
        # the adduct marked as it is typed, unfinished too; an empty one is the default
        try:
            if adduct:
                parse_adduct(adduct)
        except ValueError as error:
            self.panel.set_field_error("adduct", str(error))
        else:
            self.panel.set_field_error("adduct", None)
        self._apply_polarity()

    ### the draft ###
    def _on_draft_field_edited(self, field_name: str, text: str):
        try:
            self._parse_draft_field(field_name, text)
        except ValueError as error:
            self.panel.set_field_error(field_name, str(error))
            return
        self.panel.set_field_error(field_name, None)
        if field_name == "adduct":
            self._apply_polarity()
        if field_name in ("formula", "adduct"):
            self._apply_draft_chemistry()

    def _apply_draft_chemistry(self):
        """The m/z of the draft's ion, in the view; kept when there is no formula or it does not parse.

        For MS2 the ion chooses the precursor instead: the one whose isolation window holds it, if any.
        """
        mz = self._draft_ion_mz()
        if mz is None:
            return
        if self.panel.ms_level() is MSLevel.MS2:
            precursor = self._precursor_isolating(mz, (self._draft_ms2 or {}).get("collision_energy"))
            if precursor is not None:
                self._draft_ms2 = self._ms2_fields(precursor)
                self.data_stack_presenter.set_ms2_selection(self._ms2_selection(self._draft_ms2))
                self._show_precursor()
            return
        _mz, mz_tolerance, mz_tolerance_unit = self.panel.mz_values()
        self.panel.set_mz_values(mz, mz_tolerance, mz_tolerance_unit)
        self.data_stack_presenter.set_mz_window(tolerance_window(mz, mz_tolerance, mz_tolerance_unit))

    def _draft_ion_mz(self) -> float | None:
        """The m/z of the draft's formula + adduct; None without a formula or when they do not parse."""
        formula, adduct = self.panel.chemistry_texts()
        if not formula:
            return None
        try:
            return ion_mz(formula, adduct or self.default_adduct)
        except ValueError:
            return None  # the field at fault is marked

    def _parse_draft_field(self, field_name: str, text: str):
        if field_name == "formula" and text:
            parse_formula(text)
        elif field_name == "adduct":
            parse_adduct(text or self.default_adduct)
        return text

    def _draft_identity(self) -> dict | None:
        """The draft's name, chemistry and note; None, with the field marked, if one does not parse."""
        texts = self.panel.draft_texts()
        identity = {}
        for field_name, text in texts.items():
            try:
                value = self._parse_draft_field(field_name, text)
            except ValueError as error:
                self.panel.set_field_error(field_name, str(error))
                return None
            identity[field_name] = value
        identity["name"] = identity["name"] or self.default_name
        identity["adduct"] = identity["adduct"] or self.default_adduct
        return identity

    def on_columns_changed(self):
        self.settings_model.target_table_shown_fields = self.panel.column_settings()
        self.settings_model.target_table_partial_fields = self.panel.partial_column_settings()

    def on_table_edited(self, target_id: str, field_name: str, value):
        target = self.target_list_model.get(target_id)
        if isinstance(value, str):
            value = value.strip()
        try:
            self.target_list_model.update(target_id, **self._parse_field(target, field_name, value))
        except ValueError as error:
            self.panel.show_table_edit_error(target_id, field_name, str(error))

    def _ms_level_changes(self, target: Target, ms_level: MSLevel) -> dict:
        changes = {"ms_level": ms_level}
        if ms_level is not MSLevel.MS2:
            # MS/MS settings exist only for MS2 targets
            changes.update(precursor_mz=None, collision_energy=None, isolation_width=None)
        elif target.ms_level is not MSLevel.MS2:
            changes.update(self._ms2_fields(self._default_precursor(target.calculated_mz())) or {})
        return changes

    @staticmethod
    def _tolerance_unit_changes(target: Target, unit: ToleranceUnit) -> dict:
        """The tolerance in ``unit`` that keeps the m/z window; a ppm one is a whole number of the ppm range."""
        if unit is target.mz_tolerance_unit:
            return {}
        mz_min, mz_max = target.mz_window()
        half_width = (mz_max - mz_min) / 2
        if unit is ToleranceUnit.PPM:
            if target.mz <= 0:
                raise ValueError("A ppm tolerance needs a positive m/z")
            spec = MZ_RANGE_CONTROL
            tolerance = min(max(round(half_width / target.mz * 1e6), spec.ppm_minimum), spec.ppm_maximum)
        else:
            tolerance = half_width
        return {"mz_tolerance": tolerance, "mz_tolerance_unit": unit}

    def _parse_field(self, target: Target, field_name: str, text) -> dict:
        """Changes for an edited field; ``text`` is the stripped text, a number for RT, m/z and their tolerances, and
        the index of the listed precursor chosen for the precursor."""
        if field_name in ("name", "note"):
            return {field_name: text}
        if field_name in ("rt", "rt_tolerance", "mz", "mz_tolerance"):
            return {field_name: text}
        if field_name == "trace_type":
            return {"trace_type": TraceType(text)}
        if field_name == "mz_tolerance_unit":
            return TargetPanelPresenter._tolerance_unit_changes(target, ToleranceUnit(text))
        if field_name == "ms_level":
            return self._ms_level_changes(target, MSLevel(text))
        if field_name == "precursor_mz":
            return self._ms2_fields(self._precursors[text])
        # m/z follows formula / adduct only when they are actually changed
        if field_name == "formula":
            changes = {"formula": text}
            if text and text != target.formula:
                changes.update(self._chemistry_changes(target, ion_mz(text, target.adduct)))
            return changes
        if field_name == "adduct":
            changes = {"adduct": text}
            if target.formula and text != target.adduct:
                changes.update(self._chemistry_changes(target, ion_mz(target.formula, text)))
            return changes
        raise KeyError(field_name)

    def _chemistry_changes(self, target: Target, mz: float) -> dict:
        """What the ion of a new formula / adduct changes: the m/z, or for MS2 the precursor that isolates it, if any."""
        if target.ms_level is not MSLevel.MS2:
            return {"mz": mz}
        return self._ms2_fields(self._precursor_isolating(mz, target.collision_energy)) or {}

    ### the precursors of MS2 ###
    def _refresh_precursors(self):
        """List the precursors of the files, merged as set, and show and use the current choice with them."""
        self._precursors = operations.precursors(
            self.data_stack_presenter.models, self.settings_model.ms2_merge_ppm, self._polarity
        )
        self.panel.set_precursors(self._precursors)
        self.table_model.set_precursors(self._precursors, self._listed_index)
        if self.shown_id is None and self.panel.ms_level() is MSLevel.MS2 and self._draft_ms2 is None:
            self._draft_ms2 = self._ms2_fields(self._default_precursor(self._draft_ion_mz()))
        self.data_stack_presenter.set_ms2_selection(self._ms2_selection(self._current_ms2_fields()))
        self._show_precursor()

    def _apply_polarity(self):
        """Use the scans of the polarity of the adduct, and list their precursors: of the one typed, as it is typed
        (a target's too, ahead of its commit), or else the target's; all scans while the draft's is empty, as nothing
        is asked for yet."""
        _formula, adduct = self.panel.chemistry_texts()
        if not adduct and self.shown_id is not None:
            adduct = self.target_list_model.get(self.shown_id).adduct
        try:
            polarity = None if not adduct else (1 if parse_adduct(adduct).charge > 0 else -1)
        except ValueError:
            return  # marked in the field; the scans stay as they were
        if polarity == self._polarity:
            return
        self._polarity = polarity
        self.data_stack_presenter.set_polarity(polarity)
        self._refresh_precursors()

    def _show_precursor(self):
        if self.batch_ids:
            chosen = {(target.precursor_mz, target.collision_energy) for target in self._batch_targets()}
            if len(chosen) > 1:
                self.panel.set_chosen_precursor(None, None, None, mixed=True)
                return
            precursor_mz, collision_energy = chosen.pop()
        else:
            fields = self._current_ms2_fields() or {}
            precursor_mz, collision_energy = fields.get("precursor_mz"), fields.get("collision_energy")
        index = None if precursor_mz is None else self._listed_index(precursor_mz, collision_energy)
        self.panel.set_chosen_precursor(index, precursor_mz, collision_energy)

    def _listed_index(self, precursor_mz: float, collision_energy: float) -> int | None:
        """The index of the listed precursor whose scans an MS2Selection of these selects; None if none."""
        merge_ppm = self.settings_model.ms2_merge_ppm
        return next(
            (
                index
                for index, precursor in enumerate(self._precursors)
                if round(precursor.collision_energy, operations.CE_DECIMALS) == round(collision_energy, operations.CE_DECIMALS)
                and abs(precursor.precursor_mz - precursor_mz) <= precursor_mz * merge_ppm * 1e-6 + operations.PRECURSOR_EPSILON
            ),
            None,
        )

    def _current_ms2_fields(self) -> dict | None:
        """precursor_mz, collision_energy and isolation_width of the target shown or the draft; None if not chosen."""
        if self.shown_id is None:
            return self._draft_ms2
        return self._target_ms2_fields(self.target_list_model.get(self.shown_id))

    @staticmethod
    def _target_ms2_fields(target: Target) -> dict | None:
        if target.precursor_mz is None:
            return None
        return {
            "precursor_mz": target.precursor_mz,
            "collision_energy": target.collision_energy,
            "isolation_width": target.isolation_width,
        }

    @staticmethod
    def _ms2_fields(precursor: operations.Precursor | None) -> dict | None:
        if precursor is None:
            return None
        return {
            "precursor_mz": precursor.precursor_mz,
            "collision_energy": precursor.collision_energy,
            "isolation_width": precursor.isolation_width,
        }

    def _ms2_selection(self, fields: dict | None) -> operations.MS2Selection | None:
        """The MS2 scans of the fields; None (all MS2 scans) when no precursor is chosen."""
        if fields is None or fields["precursor_mz"] is None:
            return None
        return operations.MS2Selection(fields["precursor_mz"], fields["collision_energy"], self.settings_model.ms2_merge_ppm)

    def _default_precursor(self, ion_mz: float | None) -> operations.Precursor | None:
        """The precursor that isolates ion_mz, if any, or else the first one; None when the files have none."""
        precursor = None if ion_mz is None else self._precursor_isolating(ion_mz)
        if precursor is None and self._precursors:
            precursor = self._precursors[0]
        return precursor

    def _precursor_isolating(self, mz: float, collision_energy: float | None = None) -> operations.Precursor | None:
        return operations.precursor_isolating(self._precursors, mz, collision_energy)

    ### plots -> panel / model ###
    def on_plot_rt_window_dragged(self, rt_min: float, rt_max: float):
        self._preview_current(**self._show_plot_rt_window(rt_min, rt_max))

    def on_plot_mz_window_dragged(self, mz_min: float, mz_max: float):
        self._preview_current(**self._show_plot_mz_window(mz_min, mz_max))

    def on_plot_rt_window_changed(self, rt_min: float, rt_max: float):
        self.table_model.clear_preview()
        changes = self._show_plot_rt_window(rt_min, rt_max)
        if self.shown_id is not None:
            self._update_current_from_plot(**changes)

    def on_plot_mz_window_changed(self, mz_min: float, mz_max: float):
        self.table_model.clear_preview()
        changes = self._show_plot_mz_window(mz_min, mz_max)
        _mz, _mz_tolerance, mz_tolerance_unit = self.panel.mz_values()
        if mz_tolerance_unit is ToleranceUnit.PPM:
            # the ppm control holds integers: snap the plotted window to it
            self.data_stack_presenter.set_mz_window(
                tolerance_window(changes["mz"], changes["mz_tolerance"], mz_tolerance_unit)
            )
        if self.shown_id is not None:
            self._update_current_from_plot(**changes)

    def set_link_rt_windows(self, linked: bool):
        """One RT window of a target for all files, its own, or each file's, the target's peak in it."""
        self.settings_model.link_rt_windows = linked
        self.data_stack_presenter.set_link_rt_windows(linked)
        if self.shown_id is not None:
            self._show_run_peaks(self.target_list_model.get(self.shown_id))
        self._show_batch_windows()

    def on_plot_run_rt_window_changed(self, model: BaseMSDataModel, rt_min: float, rt_max: float):
        """The target's peak in a file, set by hand there; in one of no peak, too, by a click on its chromatogram. Of
        several targets selected, theirs all."""
        self._set_run_window_by_hand(model, (rt_min, rt_max))

    def on_plot_run_no_peak(self, model: BaseMSDataModel):
        """No peak of the target in a file, by the menu of its RT window; of several targets selected, of them all."""
        self._set_run_window_by_hand(model, None)

    def _set_run_window_by_hand(self, model: BaseMSDataModel, rt_window: tuple[float, float] | None):
        changes = []
        for target_id in self.batch_ids or [self.current_id]:
            target = self.target_list_model.get(target_id)
            kept = self.run_peaks.get(target.id, model.file_path)
            peaks = () if kept is None else kept.peaks
            changes.append((target.id, model.file_path, run_peaks_for(target, rt_window, peaks, manual=True)))
        self.target_list_model.set_run_peaks(changes)  # one edit to undo
        self._show_run_peaks(self.target_list_model.get(self.shown_id))
        self._show_batch_windows()

    def refresh_run_peaks(self):
        """Show the current target's peaks in the files again, once they are found."""
        if self.shown_id is not None:
            self._show_run_peaks(self.target_list_model.get(self.shown_id))
        self._show_batch_windows()

    def on_model_added(self, model: BaseMSDataModel):
        self._refresh_precursors()
        if self.shown_id is not None:
            # a file opened again shows the target's peak in it
            self._show_run_peaks(self.target_list_model.get(self.shown_id))
        self._show_batch_windows()

    def _show_plot_rt_window(self, rt_min: float, rt_max: float) -> dict:
        """Show an RT window of the plots in the panel; the changes it makes to a target."""
        rt, rt_tolerance = (rt_min + rt_max) / 2, (rt_max - rt_min) / 2
        self.panel.set_rt_values(rt, rt_tolerance)
        return {"rt": rt, "rt_tolerance": rt_tolerance}

    def _show_plot_mz_window(self, mz_min: float, mz_max: float) -> dict:
        """Show an m/z window of the plots in the panel, in its tolerance unit; the changes it makes to a target."""
        mz, half_width = (mz_min + mz_max) / 2, (mz_max - mz_min) / 2
        _mz, _mz_tolerance, mz_tolerance_unit = self.panel.mz_values()
        if mz_tolerance_unit is ToleranceUnit.PPM:
            # the ppm control holds integers
            self.panel.set_mz_values(mz, half_width / mz * 1e6, mz_tolerance_unit)
            _mz, mz_tolerance, _unit = self.panel.mz_values()
        else:
            mz_tolerance = half_width
            self.panel.set_mz_values(mz, mz_tolerance, mz_tolerance_unit)
        # with the unit shown, which several targets selected may not all have had
        return {"mz": mz, "mz_tolerance": mz_tolerance, "mz_tolerance_unit": mz_tolerance_unit}

    def _preview_current(self, **changes):
        """Show the current target, or the targets selected, with ``changes`` in the list, without changing them."""
        if self.batch_ids:
            self.table_model.set_previews([replace(target, **changes) for target in self._batch_targets()])
        elif self.current_id is not None:
            self.table_model.set_preview(replace(self.target_list_model.get(self.current_id), **changes))

    def _update_current_from_plot(self, **changes):
        # the plots already show the new values
        self._updating_from_plot = True
        try:
            if self.batch_ids:
                self._update_batch(lambda _target: changes)
            else:
                self.target_list_model.update(self.current_id, **changes)
        finally:
            self._updating_from_plot = False

    ### model -> panel / plots ###
    def on_target_changed(self, target_id: str):
        self.run_peaks.follow(self.target_list_model.get(target_id))
        if self._updating_batch:
            return  # shown once they all are changed
        if target_id == self.current_id or target_id in self.batch_ids:
            self._show_current_target(apply_to_plots=not self._updating_from_plot)

    def on_targets_reset(self):
        for target in self.target_list_model.targets:
            self.run_peaks.follow(target)
        if self.batch_ids:
            self.select_targets(
                [target_id for target_id in self.batch_ids if self.target_list_model.contains(target_id)],
                self.batch_shown_id,
            )
            return
        if self.current_id is not None and not self.target_list_model.contains(self.current_id):
            self.current_id = None
        self.select_target(self.current_id)

    def _show_current_target(self, *, apply_to_plots: bool = True):
        """The target selected, or the several, in the panel, and the one the plots show, there."""
        targets = self._batch_targets()
        target = None if self.shown_id is None else self.target_list_model.get(self.shown_id)
        if targets:
            self.panel.show_targets(targets, target)
        else:
            self.panel.show_target(target)
        edited = targets or ([] if target is None else [target])
        self.panel.set_mz_mismatch(any(edited_target.mz_deviates_from_formula(self.mz_decimals) for edited_target in edited))
        self.panel.set_precursor_mismatch(any(edited_target.precursor_deviates_from_formula() for edited_target in edited))
        if target is not None and target.ms_level is MSLevel.MS2:
            self._draft_ms2 = self._current_ms2_fields()
        self._apply_polarity()
        self._show_precursor()
        self.update_chemistry_preview()
        if self.editing_target != (target is not None):
            self.editing_target = target is not None
            self.data_stack_presenter.set_editing_target(self.editing_target)
            self.editing_target_changed.emit(self.editing_target)
        if target is not None and apply_to_plots:
            self._apply_target_to_plots(target)
        elif target is not None and not self.data_stack_presenter.link_rt_windows:
            # the plots show the target already, but for what was found in a trace it changed, which is gone
            self._show_run_peaks(target)
        self._show_batch_windows()

    def _show_batch_windows(self):
        """Of several targets selected, hide the RT and m/z windows where theirs differ, and say on the plots which
        one they show."""
        stack = self.data_stack_presenter
        if not self.batch_ids:
            stack.set_windows_hidden()
            stack.set_notice("")
            return
        targets = self._batch_targets()
        if stack.own_rt_windows:
            rt_models = [
                model for model in stack.models if len({self._run_rt_window(target, model) for target in targets}) > 1
            ]
        else:
            rt_models = stack.models if len({target.rt_window() for target in targets}) > 1 else []
        stack.set_windows_hidden(rt_models, mz=len({target.mz_window() for target in targets}) > 1)
        shown = self.target_list_model.get(self.batch_shown_id)
        stack.set_notice(f"{len(targets)} targets selected: showing #{shown.index}")

    def _run_rt_window(self, target: Target, model: BaseMSDataModel) -> tuple[float, float] | None:
        """The target's RT window in a file while the files' are not linked: of its peak there (None: no peak), or its
        own; as _run_peaks_shown has it."""
        run_peaks = self.run_peaks.get(target.id, model.file_path)
        return target.rt_window() if run_peaks is None else run_peaks.rt_window

    def _apply_target_to_plots(self, target: Target):
        self.data_stack_presenter.set_trace_type(target.trace_type)
        self.data_stack_presenter.set_ms2_selection(self._ms2_selection(self._target_ms2_fields(target)))
        self.data_stack_presenter.set_ms_level(target.ms_level.number)
        self.data_stack_presenter.set_mz_window(target.mz_window())
        self._show_run_peaks(target)

    def _show_run_peaks(self, target: Target):
        """The target's RT window in the plots, each file's own while they are not linked, and the peaks found."""
        windows, peak_areas = self._run_peaks_shown(target)
        self.data_stack_presenter.set_rt_window(target.rt_window(), windows)
        self.data_stack_presenter.set_peak_areas(peak_areas)

    def _run_peaks_shown(self, target: Target) -> tuple[dict, dict]:
        """Of the files with a peak of the target in them, its RT window (None: no peak), and of those with peaks
        found, their (start, end); while the files' RT windows are linked none, all files taking the target's window."""
        windows, peak_areas = {}, {}
        if self.data_stack_presenter.link_rt_windows:
            return windows, peak_areas
        for model in self.data_stack_presenter.models:
            run_peaks = self.run_peaks.get(target.id, model.file_path)
            if run_peaks is None:
                continue
            windows[model] = run_peaks.rt_window
            if run_peaks.peaks:
                peak_areas[model] = [(peak.start, peak.end) for peak in run_peaks.peaks]
        return windows, peak_areas
