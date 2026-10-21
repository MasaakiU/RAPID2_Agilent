import re
from dataclasses import dataclass
from functools import cache

from RAPID2_Agilent.utils.path import AppPath


ELECTRON_MASS = 0.000548579909065

_FORMULA_SYNTAX = re.compile(r"(?:[A-Z][a-z]?[0-9]*)+")
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*)")
# e.g. [M+H]+, [M-H]-, [M+2H]2+, [2M+Na]+, [M-H2O+H]+
_ADDUCT_SYNTAX = re.compile(r"\[([0-9]*)M((?:[+-][0-9]*[A-Z][A-Za-z0-9]*)*)\]([0-9]*)([+-])")
_ADDUCT_TERM = re.compile(r"([+-])([0-9]*)([A-Z][A-Za-z0-9]*)")


@cache
def monoisotopic_masses() -> dict[str, float]:
    """Mass of the most abundant isotope of each atomic symbol (NIST table)."""
    best: dict[str, tuple[float, float]] = {}
    for block in AppPath.atomic_weights_path.read_text().split("\n\n"):
        fields = {}
        for line in block.strip().splitlines():
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
        if not fields or fields["Isotopic Composition"] == "":
            continue
        symbol = fields["Atomic Symbol"]
        mass = float(fields["Relative Atomic Mass"].split("(")[0])
        composition = float(fields["Isotopic Composition"].split("(")[0])
        if symbol not in best or best[symbol][1] < composition:
            best[symbol] = (mass, composition)
    return {symbol: mass for symbol, (mass, _composition) in best.items()}


def parse_formula(formula: str) -> dict[str, int]:
    if _FORMULA_SYNTAX.fullmatch(formula) is None:
        raise ValueError(f"Invalid formula: {formula!r}")
    masses = monoisotopic_masses()
    counts: dict[str, int] = {}
    for symbol, count in _FORMULA_TOKEN.findall(formula):
        if symbol not in masses:
            raise ValueError(f"Unknown element {symbol!r} in formula {formula!r}")
        counts[symbol] = counts.get(symbol, 0) + int(count or 1)
    return counts


def formula_mass(formula: str) -> float:
    masses = monoisotopic_masses()
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
        group_mass = formula_mass(group) * int(count or 1)
        mass_shift += group_mass if term_sign == "+" else -group_mass
    charge = int(charge_text or 1) * (1 if sign == "+" else -1)
    return Adduct(multiplier=int(multiplier_text or 1), mass_shift=mass_shift, charge=charge)


def ion_mz(formula: str, adduct: str) -> float:
    """m/z of the ion made from a neutral formula and an adduct."""
    a = parse_adduct(adduct)
    ion_mass = a.multiplier * formula_mass(formula) + a.mass_shift - a.charge * ELECTRON_MASS
    return ion_mass / abs(a.charge)
