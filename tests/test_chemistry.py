import pytest

from RAPID2_Agilent.models.chemistry import formula_mass, ion_mz, parse_adduct, parse_formula


def test_formula_mass_of_glucose():
    assert formula_mass("C6H12O6") == pytest.approx(180.063388, abs=1e-6)


@pytest.mark.parametrize(
    ("formula", "expected_mass"),
    [
        ("[13C]6H12O6", 186.083517),
        ("C5[13C]H12O6", 181.066743),
        ("C6H11[2H]O6", 181.069665),
        ("C6H11DO6", 181.069665),
    ],
)
def test_formula_mass_with_isotopes(formula, expected_mass):
    assert formula_mass(formula) == pytest.approx(expected_mass, abs=1e-6)


@pytest.mark.parametrize(
    ("formula", "expected_counts"),
    [
        ("C6H3(CH3)3", {"C": 9, "H": 12}),
        ("Ca(OH)2", {"Ca": 1, "O": 2, "H": 2}),
        ("((CH3)3Si)2O", {"C": 6, "H": 18, "Si": 2, "O": 1}),
        ("C6H5(CH2)2([13C]H3)", {"C": 8, "H": 12, "[13C]": 1}),
    ],
)
def test_parse_formula_with_groups(formula, expected_counts):
    assert parse_formula(formula) == expected_counts


@pytest.mark.parametrize(
    ("adduct", "expected_mz"),
    [
        ("[M+H]+", 181.070665),
        ("[M-H]-", 179.056112),
        ("[M+Na]+", 203.052609),
        ("[M+2H]2+", 91.038971),
        ("[2M-H]-", 359.119500),
        ("[M-H2O+H]+", 163.060100),
        ("[M+FA-H]-", 225.061591),
        ("[M+Hac-H]-", 239.077241),
    ],
)
def test_ion_mz_of_glucose(adduct, expected_mz):
    assert ion_mz("C6H12O6", adduct) == pytest.approx(expected_mz, abs=1e-6)


@pytest.mark.parametrize(
    ("solvent", "formula"),
    [("ACN", "C2H3N"), ("IsoProp", "C3H8O"), ("DMSO", "C2H6OS"), ("FA", "CH2O2"), ("Hac", "C2H4O2"), ("TFA", "C2HF3O2")],
)
def test_solvent_in_adduct_is_its_formula(solvent, formula):
    assert ion_mz("C6H12O6", f"[M+2{solvent}+H]+") == pytest.approx(ion_mz("C6H12O6", f"[M+2{formula}+H]+"), abs=1e-9)


def test_parse_adduct_charge():
    assert parse_adduct("[M+2H]2+").charge == 2
    assert parse_adduct("[M-H]-").charge == -1


@pytest.mark.parametrize(
    "formula",
    ["", "c6", "C6H12O6X", "C6 H12", "13C6H12O6", "[13C", "[99C]6", "[C]6",
     "(CH3", "CH3)2", "()", "C()2", "(2CH3)", "((CH3)2"],
)
def test_invalid_formula_raises(formula):
    with pytest.raises(ValueError):
        parse_formula(formula)


# a charge, multiplier or count of 0 too: no ion, no molecule, or nothing added
@pytest.mark.parametrize(
    "adduct", ["", "M+H", "[M+H]", "[M+Xx]+", "[M+2H]0+", "[0M+H]+", "[M+0Na+H]+", "[M+02H]2+", "[M+H]+2", "[M + H]+"],
)
def test_invalid_adduct_raises(adduct):
    with pytest.raises(ValueError):
        parse_adduct(adduct)


def test_the_isotope_pattern_of_glucose_is_of_the_natural_abundances():
    from RAPID2_Agilent.models.chemistry import isotope_pattern

    peaks = isotope_pattern("C6H12O6", "[M-H]-")
    assert [peak.shift for peak in peaks[:3]] == [0, 1, 2]
    assert peaks[0].mz == pytest.approx(ion_mz("C6H12O6", "[M-H]-"), abs=1e-9)
    # M+1 / M+0 of C6 H11 O6, by the binomial of each element
    expected = 6 * 0.0107 / 0.9893 + 11 * 0.000115 / 0.999885 + 6 * 0.00038 / 0.99757
    assert peaks[1].fraction / peaks[0].fraction == pytest.approx(expected, rel=1e-3)
    assert sum(peak.fraction for peak in peaks) == pytest.approx(1, abs=1e-9)


def test_atoms_written_as_an_isotope_are_of_it_alone_and_the_charge_divides_the_shift():
    from RAPID2_Agilent.models.chemistry import ion_composition, isotope_pattern

    labelled = isotope_pattern("C5H5[2H]9NO", "[M]+")
    assert labelled[0].mz == pytest.approx(ion_mz("C5H5[2H]9NO", "[M]+"), abs=1e-9)
    # of C5 H5 N O alone, the [2H] adding none
    assert labelled[1].fraction / labelled[0].fraction == pytest.approx(
        5 * 0.0107 / 0.9893 + 5 * 0.000115 / 0.999885 + 0.00364 / 0.99636 + 0.00038 / 0.99757, rel=1e-3,
    )
    doubly = isotope_pattern("C6H12O6", "[M+2H]2+")
    assert doubly[1].mz - doubly[0].mz == pytest.approx(1.00335 / 2, abs=1e-3)
    # an element whose most abundant isotope is not its lightest: 10B below 11B
    assert isotope_pattern("C2H5BO2", "[M+H]+")[0].shift == -1
    assert ion_composition("C6H12O6", "[M-H2O+H]+") == ({"C": 6, "H": 11, "O": 5}, 1)
    with pytest.raises(ValueError, match="takes away more"):
        ion_composition("CH4", "[M-H2O+H]+")
