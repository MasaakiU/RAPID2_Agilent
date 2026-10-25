"""Several targets selected, edited all at once in the target panel (see TargetPanelPresenter)."""

import numpy as np
import pytest

from RAPID2_Agilent.models.chemistry import ion_mz
from RAPID2_Agilent.models.target import ToleranceUnit
from tests.test_target_panel import DATA_PATH, build_app, edit_cell, type_into


def select_two(presenter, view):
    """Two targets of other RTs and notes, the same name, both selected; the second the one the plots show."""
    target_presenter = presenter.target_panel_presenter
    panel = view.target_panel
    first = target_presenter.add_target_from_view()
    second = target_presenter.add_target_from_view()
    for target, rt, note in ((first, 0.5, "first"), (second, 1.0, "second")):
        edit_cell(panel, target.id, "rt", rt)
        edit_cell(panel, target.id, "note", note)
    target_presenter.select_targets([first.id, second.id], second.id)
    return first.id, second.id


def test_panel_shows_what_they_have_in_common_blank_where_they_differ():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    first_id, second_id = select_two(presenter, view)
    first = model.target_list_model.get(first_id)

    assert panel.state_label.text() == "2 targets selected"
    assert panel.name_edit.text() == first.name
    assert (panel.note_edit.text(), panel.note_edit.placeholderText()) == ("", "")
    assert panel.rt_range_control.center_spin_box.is_blank()
    assert panel.rt_range_control.tolerance() == pytest.approx(first.rt_tolerance)
    assert panel.mixed_fields == {"note", "rt"}


def test_a_field_typed_changes_that_field_of_all_as_one_edit_to_undo():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    targets = model.target_list_model
    first_id, second_id = select_two(presenter, view)
    before = {target_id: targets.get(target_id) for target_id in (first_id, second_id)}

    type_into(panel, "note", "both")
    panel.rt_range_control.tolerance_spin_box.setValue(0.3)

    for target_id, old in before.items():
        new = targets.get(target_id)
        assert (new.note, new.rt_tolerance) == ("both", pytest.approx(0.3))
        assert new.rt == old.rt  # each keeps its own
    assert panel.note_edit.text() == "both"
    view.undo_target_edit_action.trigger()
    view.undo_target_edit_action.trigger()
    assert {target_id: targets.get(target_id) for target_id in before} == before
    # still both selected
    assert presenter.target_panel_presenter.batch_ids == [first_id, second_id]


def test_leaving_fields_untouched_changes_nothing():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    targets = model.target_list_model
    select_two(presenter, view)
    before = targets.targets

    # as Tab goes through the fields: the blank ones are not filled in
    panel.note_edit.setModified(True)
    panel.note_edit.editingFinished.emit()
    panel.commit_pending_fields()
    for spin_box in (panel.rt_range_control.center_spin_box, panel.rt_range_control.tolerance_spin_box):
        spin_box.interpretText()

    assert targets.targets == before
    assert panel.rt_range_control.center_spin_box.is_blank()


def test_formula_typed_sets_each_m_z_by_its_own_adduct():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    targets = model.target_list_model
    first_id, second_id = select_two(presenter, view)
    edit_cell(panel, second_id, "adduct", "[M+Na]+")
    assert "adduct" in panel.mixed_fields

    type_into(panel, "formula", "C6H12O6")

    assert targets.get(first_id).mz == pytest.approx(ion_mz("C6H12O6", "[M+H]+"))
    assert targets.get(second_id).mz == pytest.approx(ion_mz("C6H12O6", "[M+Na]+"))
    # no one m/z of their ions to show
    assert panel.calculated_mz_label.text() == ""


def test_m_z_tolerance_is_set_with_its_unit():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    targets = model.target_list_model
    first_id, second_id = select_two(presenter, view)
    edit_cell(panel, first_id, "mz_tolerance_unit", "ppm")
    assert "mz_tolerance" in panel.mixed_fields
    # the unit of the one the plots show
    assert panel.mz_values()[2] is ToleranceUnit.MZ

    panel.mz_range_control.tolerance_spin_box.setValue(0.02)

    for target_id in (first_id, second_id):
        target = targets.get(target_id)
        assert (target.mz_tolerance, target.mz_tolerance_unit) == (pytest.approx(0.02), ToleranceUnit.MZ)


def test_plots_show_the_last_clicked_hide_differing_windows_and_a_click_sets_them_all():
    app, view, model, presenter = build_app()
    stack = presenter.data_stack_presenter
    targets = model.target_list_model
    first_id, second_id = select_two(presenter, view)
    shown = targets.get(second_id)

    for child in stack.child_presenters:
        chromatogram, spectrum = child.view.chromatogram_window, child.view.spectrum_window
        assert f"2 targets selected: showing #{shown.index}" in chromatogram.warning_label.textItem.toPlainText()
        np.testing.assert_allclose(child.current_rt_window, shown.rt_window())
        # the RTs differ, the m/z windows do not
        assert not chromatogram.region0.isVisible()
        assert spectrum.region0.isVisible()

    stack.child_presenters[0].on_chromatogram_clicked(0.8, 0.0)

    for target_id in (first_id, second_id):
        target = targets.get(target_id)
        assert (target.rt, target.rt_tolerance) == (pytest.approx(0.8), pytest.approx(shown.rt_tolerance))
    assert all(child.view.chromatogram_window.region0.isVisible() for child in stack.child_presenters)

    # one selected: no notice
    presenter.target_panel_presenter.select_target(first_id)
    assert "selected" not in stack.child_presenters[0].view.chromatogram_window.warning_label.textItem.toPlainText()


def test_a_click_sets_the_files_own_rt_window_of_them_all_while_not_linked():
    app, view, model, presenter = build_app()
    stack = presenter.data_stack_presenter
    first_id, second_id = select_two(presenter, view)
    view.link_rt_windows_check_box.setChecked(False)
    assert not stack.child_presenters[0].view.chromatogram_window.region0.isVisible()

    stack.child_presenters[0].on_chromatogram_clicked(0.8, 0.0)

    for target_id in (first_id, second_id):
        run_peaks = model.run_peaks.get(target_id, DATA_PATH)
        assert run_peaks.manual and np.mean(run_peaks.rt_window) == pytest.approx(0.8)
    # set there for both, hidden in the other file still
    assert stack.child_presenters[0].view.chromatogram_window.region0.isVisible()
    assert not stack.child_presenters[1].view.chromatogram_window.region0.isVisible()


def test_add_copies_them_all_as_one_edit_to_undo():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    targets = model.target_list_model
    target_presenter = presenter.target_panel_presenter
    first_id, second_id = select_two(presenter, view)
    assert panel.add_button.toolTip() == panel.DUPLICATE_ALL_TOOLTIP

    panel.add_button.click()

    assert len(targets.targets) == 4
    copies = targets.targets[2:]
    assert [copy.note for copy in copies] == ["first", "second"]
    assert target_presenter.batch_ids == [copy.id for copy in copies]
    assert target_presenter.batch_shown_id == copies[1].id
    view.undo_target_edit_action.trigger()
    assert [target.id for target in targets.targets] == [first_id, second_id]


def test_leaving_them_starts_a_draft_from_the_view_of_the_one_shown():
    app, view, model, presenter = build_app()
    panel = view.target_panel
    target_presenter = presenter.target_panel_presenter
    first_id, second_id = select_two(presenter, view)
    shown = model.target_list_model.get(second_id)

    target_presenter.select_target(None)

    assert panel.state_label.text() == "New target (not added)"
    assert panel.rt_values() == pytest.approx((shown.rt, shown.rt_tolerance))
    assert (panel.name_edit.text(), panel.note_edit.text()) == ("", "")
    assert panel.name_edit.placeholderText() == "New target"
