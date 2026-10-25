import pytest

from RAPID2_Agilent.models.run_peaks import Peak, run_peaks_for
from RAPID2_Agilent.models.target import MSLevel, Target, ToleranceUnit, TraceType
from RAPID2_Agilent.models.target_model import TargetListModel


def make_target(**changes) -> Target:
    values = dict(
        index=1,
        name="glucose",
        formula="C6H12O6",
        adduct="[M-H]-",
        trace_type=TraceType.XIC,
        ms_level=MSLevel.MS1,
        rt=1.0,
        rt_tolerance=0.1,
        mz=179.056112,
        mz_tolerance=0.01,
        mz_tolerance_unit=ToleranceUnit.MZ,
    )
    values.update(changes)
    return Target(**values)


@pytest.mark.parametrize("field", ["name", "note"])
@pytest.mark.parametrize("character", ["\t", "\n", "\r"])
def test_name_and_note_hold_no_tab_or_line_break(field, character):
    with pytest.raises(ValueError, match=field):
        make_target(**{field: f"a{character}b"})


def test_target_windows():
    target = make_target(mz=200.0, mz_tolerance=5, mz_tolerance_unit=ToleranceUnit.PPM)

    assert target.rt_window() == pytest.approx((0.9, 1.1))
    assert target.mz_window() == pytest.approx((199.999, 200.001))


def test_target_mz_deviation_uses_display_decimals():
    assert not make_target().mz_deviates_from_formula(6)
    assert make_target(mz=179.0562).mz_deviates_from_formula(6)
    assert not make_target(formula="", mz=100.0).mz_deviates_from_formula(6)


def test_ms2_target_compares_its_precursor_with_the_isolation_window():
    # C6H12O6 [M-H]- is 179.056112; the m/z traced is a product ion
    def ms2_target(precursor_mz):
        return make_target(ms_level=MSLevel.MS2, mz=89.0, precursor_mz=precursor_mz, collision_energy=20.0, isolation_width=1.3)

    assert not ms2_target(179.0561).mz_deviates_from_formula(6)
    assert not ms2_target(179.6).precursor_deviates_from_formula()
    assert ms2_target(179.8).precursor_deviates_from_formula()
    assert not make_target(ms_level=MSLevel.MS2, mz=89.0).precursor_deviates_from_formula()


@pytest.mark.parametrize(
    "changes",
    [
        {"adduct": ""},
        {"formula": "C6H12O6X"},
        {"rt_tolerance": -1.0},
        {"mz_tolerance": 2.5, "mz_tolerance_unit": ToleranceUnit.PPM},
        {"collision_energy": 20.0},
        {"precursor_mz": 179.06, "ms_level": MSLevel.ALL},
        {"isolation_width": 1.3},
    ],
)
def test_invalid_target_raises(changes):
    with pytest.raises(ValueError):
        make_target(**changes)


def test_msms_settings_for_ms2_target():
    target = make_target(ms_level=MSLevel.MS2, mz=89.0, precursor_mz=179.06, collision_energy=20.0, isolation_width=1.3)

    assert (target.precursor_mz, target.collision_energy, target.isolation_width) == (179.06, 20.0, 1.3)
    assert MSLevel.ALL.number is None
    assert target.ms_level.number == 2


def test_update_undo_redo():
    model = TargetListModel()
    target = make_target()
    model.add(target)

    model.update(target.id, rt=2.0)
    assert model.get(target.id).rt == 2.0

    model.undo()
    assert model.get(target.id).rt == 1.0
    model.undo()
    assert model.targets == ()
    assert not model.can_undo()

    model.redo()
    model.redo()
    assert model.get(target.id).rt == 2.0
    assert not model.can_redo()


def test_update_without_change_is_not_recorded():
    model = TargetListModel()
    target = make_target()
    model.add(target)

    model.update(target.id, rt=target.rt)
    model.undo()

    assert model.targets == ()


def test_invalid_update_keeps_state():
    model = TargetListModel()
    target = make_target()
    model.add(target)

    with pytest.raises(ValueError):
        model.update(target.id, adduct="M+H")

    assert model.get(target.id) == target
    model.undo()
    assert model.targets == ()


def test_remove_targets():
    model = TargetListModel()
    targets = [make_target(index=i + 1, name=f"t{i}") for i in range(3)]
    for target in targets:
        model.add(target)

    model.remove([targets[0].id, targets[2].id])

    assert model.targets == (targets[1],)


def test_add_rejects_duplicate_index():
    model = TargetListModel()
    model.add(make_target(index=1))

    with pytest.raises(ValueError):
        model.add(make_target(index=1))
    assert model.next_index() == 2


def test_reindex_renumbers_in_given_order():
    model = TargetListModel()
    targets = [make_target(index=index) for index in (1, 4, 7)]
    for target in targets:
        model.add(target)

    model.reindex([targets[2].id, targets[0].id, targets[1].id])

    assert [model.get(target.id).index for target in targets] == [2, 3, 1]
    with pytest.raises(ValueError):
        model.update(targets[0].id, index=10)
    with pytest.raises(ValueError):
        model.reindex([targets[0].id])

    model.undo()
    assert [model.get(target.id).index for target in targets] == [1, 4, 7]


def test_import_replaces_or_appends_as_one_edit():
    model = TargetListModel()
    first = make_target(index=1)
    model.add(first)
    resets = []
    model.targets_reset.connect(lambda: resets.append(True))
    imported = [make_target(index=2, name="a"), make_target(index=3, name="b")]

    model.import_targets(imported, replace=False)
    assert model.targets == (first, *imported) and resets == [True]
    model.undo()
    assert model.targets == (first,)

    model.import_targets([make_target(index=1, name="c")], replace=True)
    assert [target.name for target in model.targets] == ["c"]
    model.undo()
    assert model.targets == (first,)

    with pytest.raises(ValueError):
        model.import_targets([make_target(index=1)], replace=False)
    assert model.targets == (first,) and model.can_redo()


def test_the_targets_windows_in_the_files_are_undone_with_their_edits():
    model = TargetListModel()
    target = make_target()
    model.add(target)
    peak = Peak(0.9, 1.05, 1.0)
    by_hand = run_peaks_for(target, (0.95, 1.1), manual=True)

    # set by hand, in two files at once: one edit
    model.set_run_peaks([(target.id, "a.d", by_hand), (target.id, "b.d", by_hand)])
    model.set_found_run_peaks([(target.id, "a.d", run_peaks_for(target, (0.9, 1.05), (peak,)))], keep_manual=False)
    assert model.run_peaks.get(target.id, "a.d").rt_window == (0.9, 1.05)
    model.undo()
    assert model.run_peaks.get(target.id, "a.d") == model.run_peaks.get(target.id, "b.d") == by_hand
    model.undo()
    assert model.run_peaks.get(target.id, "a.d") is None and model.run_peaks.get(target.id, "b.d") is None
    model.redo()
    assert model.run_peaks.get(target.id, "b.d") == by_hand
    # nothing changed, nothing to undo
    model.set_run_peaks([(target.id, "b.d", by_hand)])
    model.undo()
    assert model.run_peaks.get(target.id, "b.d") is None

    # what an edit of the target drops (RunPeaksStore.follow, by whoever shows it), undone with it
    model.set_found_run_peaks([(target.id, "a.d", run_peaks_for(target, (0.9, 1.05), (peak,)))], keep_manual=False)
    changed = model.update(target.id, mz=200.0)
    model.run_peaks.follow(changed)
    assert model.run_peaks.get(target.id, "a.d") is None
    model.undo()
    assert model.run_peaks.get(target.id, "a.d").rt_window == (0.9, 1.05)
