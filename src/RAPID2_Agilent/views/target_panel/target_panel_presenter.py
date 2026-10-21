from dataclasses import replace

from PyQt6.QtCore import QObject

from RAPID2_Agilent.models.chemistry import formula_mass, ion_mz
from RAPID2_Agilent.models.target_model import (
    MSLevel,
    Target,
    TargetListModel,
    ToleranceUnit,
    TraceType,
    new_target_id,
    tolerance_window,
)
from RAPID2_Agilent.views.data_stack import ChromatogramSpectrumStackPresenter
from RAPID2_Agilent.views.view_constants import MZ_RANGE_CONTROL

from .target_panel import TargetPanel
from .target_table_model import TargetTableModel


class TargetPanelPresenter(QObject):
    """Connects the target panel, the target list and the plots.

    With exactly one target selected, the view parameters (trace type, MS
    level, RT, m/z) edit that target and the plots follow it. Otherwise the panel acts
    as a plain viewer control.
    """

    default_name = "New target"
    default_adduct = "[M+H]+"
    mz_decimals = MZ_RANGE_CONTROL.decimals

    def __init__(
        self,
        panel: TargetPanel,
        target_list_model: TargetListModel,
        data_stack_presenter: ChromatogramSpectrumStackPresenter,
        parent=None,
    ):
        super().__init__(parent)
        self.panel = panel
        self.target_list_model = target_list_model
        self.data_stack_presenter = data_stack_presenter
        self.current_id: str | None = None
        self._updating_from_plot = False

        self.table_model = TargetTableModel(self.target_list_model, mz_decimals=self.mz_decimals, parent=self)
        self.panel.set_table_model(self.table_model)
        self.panel.show_target(None, mz_decimals=self.mz_decimals)

        self.panel.selection_changed.connect(self.on_selection_changed)
        self.panel.rt_edited.connect(self.on_rt_edited)
        self.panel.mz_edited.connect(self.on_mz_edited)
        self.panel.trace_type_edited.connect(self.on_trace_type_edited)
        self.panel.ms_level_edited.connect(self.on_ms_level_edited)
        self.panel.field_edited.connect(self.on_field_edited)
        self.panel.chemistry_text_changed.connect(self.update_chemistry_preview)
        self.panel.add_requested.connect(self.add_target_from_view)
        self.panel.duplicate_requested.connect(self.duplicate_current_target)
        self.panel.remove_requested.connect(self.remove_selected_targets)
        self.panel.reset_index_requested.connect(self.reset_index)
        self.target_list_model.target_changed.connect(self.on_target_changed)
        self.target_list_model.targets_reset.connect(self.on_targets_reset)
        self.data_stack_presenter.rt_window_changed.connect(self.on_plot_rt_window_changed)
        self.data_stack_presenter.mz_window_changed.connect(self.on_plot_mz_window_changed)

    ### operations ###
    def select_target(self, target_id: str | None):
        self.panel.select_target(target_id)
        self.current_id = target_id
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

    def add_target_from_view(self) -> Target:
        rt, rt_tolerance = self.panel.rt_values()
        mz, mz_tolerance, mz_tolerance_unit = self.panel.mz_values()
        target = Target(
            index=self.target_list_model.next_index(),
            name=self.default_name,
            formula="",
            adduct=self.default_adduct,
            trace_type=self.panel.trace_type(),
            ms_level=self.panel.ms_level(),
            rt=rt,
            rt_tolerance=rt_tolerance,
            mz=mz,
            mz_tolerance=mz_tolerance,
            mz_tolerance_unit=mz_tolerance_unit,
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

    def remove_selected_targets(self):
        self.target_list_model.remove(self.panel.selected_target_ids())

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
        self._show_current_target()

    def on_rt_edited(self):
        rt, rt_tolerance = self.panel.rt_values()
        if self.current_id is None:
            self.data_stack_presenter.set_rt_window((rt - rt_tolerance, rt + rt_tolerance))
        else:
            self.target_list_model.update(self.current_id, rt=rt, rt_tolerance=rt_tolerance)

    def on_mz_edited(self):
        mz, mz_tolerance, mz_tolerance_unit = self.panel.mz_values()
        if self.current_id is None:
            self.data_stack_presenter.set_mz_window(tolerance_window(mz, mz_tolerance, mz_tolerance_unit))
        else:
            self.target_list_model.update(
                self.current_id,
                mz=mz,
                mz_tolerance=mz_tolerance,
                mz_tolerance_unit=mz_tolerance_unit,
            )

    def on_trace_type_edited(self):
        trace_type = self.panel.trace_type()
        if self.current_id is None:
            self.data_stack_presenter.set_trace_type(trace_type)
        else:
            self.target_list_model.update(self.current_id, trace_type=trace_type)

    def on_ms_level_edited(self):
        ms_level = self.panel.ms_level()
        if self.current_id is None:
            self.data_stack_presenter.set_ms_level(ms_level.number)
            return
        changes = {"ms_level": ms_level}
        if ms_level is not MSLevel.MS2:
            # MS/MS settings exist only for MS2 targets
            changes.update(product_ion=None, collision_energy=None)
        self.target_list_model.update(self.current_id, **changes)

    def on_field_edited(self, field_name: str, text: str):
        target = self.target_list_model.get(self.current_id)
        try:
            self.target_list_model.update(target.id, **self._parse_field(target, field_name, text.strip()))
        except ValueError as error:
            self.panel.set_field_error(field_name, str(error))
            return
        self.panel.set_field_error(field_name, None)

    def update_chemistry_preview(self):
        """Masses of the formula / adduct being typed (not yet committed)."""
        formula, adduct = self.panel.chemistry_texts()
        neutral_mass = mz = None
        if formula:
            try:
                neutral_mass = formula_mass(formula)
                mz = ion_mz(formula, adduct)
            except ValueError:
                pass  # incomplete input while typing
        self.panel.set_chemistry_preview(neutral_mass, mz, decimals=self.mz_decimals)

    @staticmethod
    def _parse_field(target: Target, field_name: str, text: str) -> dict:
        if field_name in ("name", "note"):
            return {field_name: text}
        # m/z follows formula / adduct only when they are actually changed
        if field_name == "formula":
            changes = {"formula": text}
            if text and text != target.formula:
                changes["mz"] = ion_mz(text, target.adduct)
            return changes
        if field_name == "adduct":
            changes = {"adduct": text}
            if target.formula and text != target.adduct:
                changes["mz"] = ion_mz(target.formula, text)
            return changes
        if field_name in ("product_ion", "collision_energy"):
            return {field_name: float(text) if text else None}
        raise KeyError(field_name)

    ### plots -> panel / model ###
    def on_plot_rt_window_changed(self, rt_min: float, rt_max: float):
        rt, rt_tolerance = (rt_min + rt_max) / 2, (rt_max - rt_min) / 2
        self.panel.set_rt_values(rt, rt_tolerance)
        if self.current_id is not None:
            self._update_current_from_plot(rt=rt, rt_tolerance=rt_tolerance)

    def on_plot_mz_window_changed(self, mz_min: float, mz_max: float):
        mz, half_width = (mz_min + mz_max) / 2, (mz_max - mz_min) / 2
        _mz, _mz_tolerance, mz_tolerance_unit = self.panel.mz_values()
        if mz_tolerance_unit is ToleranceUnit.PPM:
            # the ppm control holds integers: snap the plotted window to it
            self.panel.set_mz_values(mz, half_width / mz * 1e6, mz_tolerance_unit)
            _mz, mz_tolerance, _unit = self.panel.mz_values()
            self.data_stack_presenter.set_mz_window(tolerance_window(mz, mz_tolerance, mz_tolerance_unit))
        else:
            mz_tolerance = half_width
            self.panel.set_mz_values(mz, mz_tolerance, mz_tolerance_unit)
        if self.current_id is not None:
            self._update_current_from_plot(mz=mz, mz_tolerance=mz_tolerance)

    def _update_current_from_plot(self, **changes):
        # the plots already show the new values
        self._updating_from_plot = True
        try:
            self.target_list_model.update(self.current_id, **changes)
        finally:
            self._updating_from_plot = False

    ### model -> panel / plots ###
    def on_target_changed(self, target_id: str):
        if target_id == self.current_id:
            self._show_current_target(apply_to_plots=not self._updating_from_plot)

    def on_targets_reset(self):
        if self.current_id is not None and not self.target_list_model.contains(self.current_id):
            self.current_id = None
        self.select_target(self.current_id)

    def _show_current_target(self, *, apply_to_plots: bool = True):
        target = None if self.current_id is None else self.target_list_model.get(self.current_id)
        self.panel.show_target(target, mz_decimals=self.mz_decimals)
        self.update_chemistry_preview()
        if target is not None and apply_to_plots:
            self._apply_target_to_plots(target)

    def _apply_target_to_plots(self, target: Target):
        if target.ms_level is MSLevel.MS2:
            raise NotImplementedError("Displaying MS2 targets (precursor / product ion / CE) is not implemented yet")
        self.data_stack_presenter.set_trace_type(target.trace_type)
        self.data_stack_presenter.set_ms_level(target.ms_level.number)
        self.data_stack_presenter.set_rt_window(target.rt_window())
        self.data_stack_presenter.set_mz_window(target.mz_window())
