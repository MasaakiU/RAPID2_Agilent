import pytest

from RAPID2_Agilent.models.target_model import MSLevel, Target, TargetListModel, ToleranceUnit, TraceType


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


def test_target_windows():
    target = make_target(mz=200.0, mz_tolerance=5, mz_tolerance_unit=ToleranceUnit.PPM)

    assert target.rt_window() == pytest.approx((0.9, 1.1))
    assert target.mz_window() == pytest.approx((199.999, 200.001))


def test_target_mz_deviation_uses_display_decimals():
    assert not make_target().mz_deviates_from_formula(6)
    assert make_target(mz=179.0562).mz_deviates_from_formula(6)
    assert not make_target(formula="", mz=100.0).mz_deviates_from_formula(6)


@pytest.mark.parametrize(
    "changes",
    [
        {"adduct": ""},
        {"formula": "C6H12O6X"},
        {"rt_tolerance": -1.0},
        {"mz_tolerance": 2.5, "mz_tolerance_unit": ToleranceUnit.PPM},
        {"collision_energy": 20.0},
        {"product_ion": 89.0, "ms_level": MSLevel.ALL},
    ],
)
def test_invalid_target_raises(changes):
    with pytest.raises(ValueError):
        make_target(**changes)


def test_msms_settings_for_ms2_target():
    target = make_target(ms_level=MSLevel.MS2, product_ion=89.0, collision_energy=20.0)

    assert (target.product_ion, target.collision_energy) == (89.0, 20.0)
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
