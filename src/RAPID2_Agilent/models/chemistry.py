import re
from dataclasses import dataclass
from functools import cache

from RAPID2_Agilent.utils.path import AppPath


ELECTRON_MASS = 0.000548579909065

# an isotope is written in brackets, e.g. C5[13C]H12O6, [13C]6H12O6, [15N]2;
# a group in parentheses, which may be nested, e.g. C6H3(CH3)3, ((CH3)3Si)2O
_FORMULA_TOKEN = re.compile(r"(\(|\)|[A-Z][a-z]?|\[[0-9]+[A-Z][a-z]?\])([0-9]*)")
# e.g. [M+H]+, [M-H]-, [M+2H]2+, [2M+Na]+, [M-H2O+H]+, [M+FA-H]-; a multiplier, count or charge written is 1 or more
_ADDUCT_SYNTAX = re.compile(r"\[([1-9][0-9]*)?M((?:[+-](?:[1-9][0-9]*)?[A-Z][A-Za-z0-9]*)*)\]([1-9][0-9]*)?([+-])")
_ADDUCT_TERM = re.compile(r"([+-])([1-9][0-9]*)?([A-Z][A-Za-z0-9]*)")
# solvents an adduct names, as in the Fiehn lab's adduct table, Skyline and MS-DIAL; looked up before a group is
# read as a formula, so that Hac is not H + Ac
SOLVENT_FORMULAS = {
    "ACN": "C2H3N",  # acetonitrile
    "IsoProp": "C3H8O",  # isopropanol
    "DMSO": "C2H6OS",  # dimethyl sulfoxide
    "FA": "CH2O2",  # formic acid
    "Hac": "C2H4O2",  # acetic acid
    "TFA": "C2HF3O2",  # trifluoroacetic acid
}


@cache
def _isotope_table() -> list[dict[str, str]]:
    table = []
    for block in AppPath.atomic_weights_path.read_text().split("\n\n"):
        fields = {}
        for line in block.strip().splitlines():
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
        if fields:
            table.append(fields)
    return table


@cache
def monoisotopic_masses() -> dict[str, float]:
    """Mass of the most abundant isotope of each atomic symbol (NIST table)."""
    best: dict[str, tuple[float, float]] = {}
    for fields in _isotope_table():
        if fields["Isotopic Composition"] == "":
            continue
        symbol = fields["Atomic Symbol"]
        mass = float(fields["Relative Atomic Mass"].split("(")[0])
        composition = float(fields["Isotopic Composition"].split("(")[0])
        if symbol not in best or best[symbol][1] < composition:
            best[symbol] = (mass, composition)
    return {symbol: mass for symbol, (mass, _composition) in best.items()}


@cache
def isotope_masses() -> dict[str, float]:
    """Mass of each isotope keyed by its bracket notation, e.g. "[13C]" (NIST table)."""
    masses = {}
    for fields in _isotope_table():
        symbol = fields["Atomic Symbol"]
        # NIST lists 2H / 3H under their own symbols D / T
        if symbol in ("D", "T"):
            symbol = "H"
        masses[f"[{fields['Mass Number']}{symbol}]"] = float(fields["Relative Atomic Mass"].split("(")[0])
    return masses


def _atom_masses() -> dict[str, float]:
    return monoisotopic_masses() | isotope_masses()


def parse_formula(formula: str) -> dict[str, int]:
    masses = _atom_masses()
    # counts of the outermost formula, then of each open parenthesis group
    stack: list[dict[str, int]] = [{}]
    position = 0
    for m in _FORMULA_TOKEN.finditer(formula):
        token, count_text = m.groups()
        if m.start() != position or (token == "(" and count_text):
            break
        # ")" without a matching "(", or closing an empty group
        if token == ")" and (len(stack) == 1 or not stack[-1]):
            break
        position = m.end()
        if token == "(":
            stack.append({})
            continue
        count = int(count_text or 1)
        if token == ")":
            for symbol, n in stack.pop().items():
                stack[-1][symbol] = stack[-1].get(symbol, 0) + n * count
            continue
        if token not in masses:
            raise ValueError(f"Unknown element or isotope {token!r} in formula {formula!r}")
        stack[-1][token] = stack[-1].get(token, 0) + count
    if position != len(formula) or len(stack) != 1 or not stack[0]:
        raise ValueError(f"Invalid formula: {formula!r}")
    return stack[0]


def formula_mass(formula: str) -> float:
    masses = _atom_masses()
    return sum(masses[symbol] * count for symbol, count in parse_formula(formula).items())


@dataclass(frozen=True)
class Adduct:
    multiplier: int     # n of [nM...]
    mass_shift: float   # net mass of added/removed groups (electrons excluded)
    charge: int         # signed


def parse_adduct(adduct: str) -> Adduct:
    m = _ADDUCT_SYNTAX.fullmatch(adduct)
    if m is None:
        raise ValueError(f"Invalid adduct: {adduct!r} (expected e.g. [M+H]+, [M-H]-, [M+2H]2+)")
    multiplier_text, terms, charge_text, sign = m.groups()
    mass_shift = 0.0
    for term_sign, count, group in _ADDUCT_TERM.findall(terms):
        group_mass = formula_mass(SOLVENT_FORMULAS.get(group, group)) * int(count or 1)
        mass_shift += group_mass if term_sign == "+" else -group_mass
    charge = int(charge_text or 1) * (1 if sign == "+" else -1)
    return Adduct(multiplier=int(multiplier_text or 1), mass_shift=mass_shift, charge=charge)


def ion_mz(formula: str, adduct: str) -> float:
    """m/z of the ion made from a neutral formula and an adduct."""
    a = parse_adduct(adduct)
    ion_mass = a.multiplier * formula_mass(formula) + a.mass_shift - a.charge * ELECTRON_MASS
    return ion_mass / abs(a.charge)
