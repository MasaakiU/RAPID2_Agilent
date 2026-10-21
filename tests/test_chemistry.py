import pytest

from RAPID2_Agilent.models.chemistry import formula_mass, ion_mz, parse_adduct, parse_formula


def test_formula_mass_of_glucose():
    assert formula_mass("C6H12O6") == pytest.approx(180.063388, abs=1e-6)


@pytest.mark.parametrize(
    ("adduct", "expected_mz"),
    [
        ("[M+H]+", 181.070665),
        ("[M-H]-", 179.056112),
        ("[M+Na]+", 203.052609),
        ("[M+2H]2+", 91.038971),
        ("[2M-H]-", 359.119500),
        ("[M-H2O+H]+", 163.060100),
    ],
)
def test_ion_mz_of_glucose(adduct, expected_mz):
    assert ion_mz("C6H12O6", adduct) == pytest.approx(expected_mz, abs=1e-6)


def test_parse_adduct_charge():
    assert parse_adduct("[M+2H]2+").charge == 2
    assert parse_adduct("[M-H]-").charge == -1


@pytest.mark.parametrize("formula", ["", "c6", "C6H12O6X", "C6 H12"])
def test_invalid_formula_raises(formula):
    with pytest.raises(ValueError):
        parse_formula(formula)


@pytest.mark.parametrize("adduct", ["", "M+H", "[M+H]", "[M+Xx]+"])
def test_invalid_adduct_raises(adduct):
    with pytest.raises(ValueError):
        parse_adduct(adduct)
