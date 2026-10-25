import pytest

from RAPID2_Agilent.models.target_search import SearchError, parse_search


class Row:
    """A target as the search sees it: fields as shown, and numbers as kept."""

    def __init__(self, name, *, mz, rt, formula="", adduct="[M+H]+", note="", precursor=None):
        self.shown = {
            "name": name,
            "formula": formula,
            "adduct": adduct,
            "mz": f"{mz:.4f}",
            "rt": f"{rt:.2f}",
            "precursor_mz": "" if precursor is None else f"{precursor:.4f}",
            "note": note,
        }
        self.kept = {"mz": mz, "rt": rt, "precursor_mz": precursor}

    def texts(self):
        return self.shown.values()

    def text(self, attribute):
        return self.shown[attribute]

    def number(self, attribute):
        return self.kept[attribute]


ROWS = {
    "glucose": Row("glucose", formula="C6H12O6", mz=181.0707, rt=1.239),
    "citrate": Row("citrate", formula="C6H8O7", adduct="[M-H]-", mz=191.0197, rt=2.5, note='5" tube'),
    "PC": Row("PC(16:0/18:1)", mz=760.585, rt=8.0, note="check"),
    "deoxy": Row("2'-deoxyadenosine", mz=252.1091, rt=3.0, note='say "hi"', precursor=268.1),
}


def found(text):
    matches = parse_search(text)
    return [key for key, row in ROWS.items() if matches(row)]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", list(ROWS)),
        ("glucose", ["glucose"]),
        ("GLU", ["glucose"]),
        # as typed, with what a query would take for its marks
        ("PC(16:0/18:1)", ["PC"]),
        ("16:0/18:1", ["PC"]),
        ("[M+H]+", ["glucose", "PC", "deoxy"]),
        ("2'-deoxy", ["deoxy"]),
        ('5" tube', ["citrate"]),
        ("name", []),
        ("rt 2", []),
    ],
)
def test_plain_text_is_looked_for_as_typed(text, expected):
    assert found(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ('"glucose"', ["glucose"]),
        ("'glucose'", ["glucose"]),
        ('name:"glu"', ["glucose"]),
        ('NAME:"GLU"', ["glucose"]),
        ('name="glucose"', ["glucose"]),
        ('name="glu"', []),
        ('name~"^c"', ["citrate"]),
        ('name~"^PC\\(16:0/18:1\\)$"', ["PC"]),
        ('~"^c"', ["glucose", "citrate", "PC"]),  # citrate's name, glucose's formula, PC's note
        ('"PC(16:0/18:1)" AND mz > 700', ["PC"]),
        ("name:\"2'-deoxy\"", ["deoxy"]),
        ('note:"5\\" tube"', ["citrate"]),
        ("note:'say \"hi\"'", ["deoxy"]),
        ('(name~"^glu" OR formula:"C6H8") AND NOT note:"check"', ["glucose", "citrate"]),
        ('name:"c" name:"t"', ["citrate"]),
        ('NOT "glucose"', ["citrate", "PC", "deoxy"]),
    ],
)
def test_text_conditions(text, expected):
    assert found(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("200 < mz < 300", ["deoxy"]),
        ("300 > mz > 200", ["deoxy"]),
        ("mz > 700", ["PC"]),
        ("m/z >= 760.585", ["PC"]),
        ("RT<2", ["glucose"]),
        # : and = go by the value as shown: the RT of 1.239 is shown 1.24
        ("rt:1.24", ["glucose"]),
        ("rt:1.2", ["glucose"]),
        ("rt:1.23", []),
        ("rt=1.24", ["glucose"]),
        ("rt=1.239", []),
        ("mz:181", ["glucose"]),
        ("mz:181.07", ["glucose"]),
        ("mz=181.0707", ["glucose"]),
        ("mz=181.07", []),
        # none kept, none shown
        ("precursor > 1", ["deoxy"]),
        ("precursor:268", ["deoxy"]),
    ],
)
def test_number_conditions(text, expected):
    assert found(text) == expected


@pytest.mark.parametrize(
    "text, message",
    [
        ('glucose AND mz<300', 'Text in a query is quoted: "glucose"'),
        ("name=glucose", 'Text in a query is quoted: name="glucose"'),
        ('mz:"150"', "mz is a number, not quoted: mz:150"),
        ('mz~"^1"', "mz is a number"),
        ("name<3", "name is text"),
        ("mass<300", '"mass" is no field'),
        ("name:", "Text in a query is quoted"),
        ('(name:"a"', "The bracket at 1 is not closed"),
        ('name:"a")', "The bracket at 9 closes nothing"),
        ('"abc', "has no closing"),
        ('name~"["', "The expression at 6 cannot be read"),
        ("mz > 200 < 300", "Not expected at 10: <"),
        ("200 < name", "wants a comparison with a field"),
    ],
)
def test_a_query_that_cannot_be_read_says_why(text, message):
    with pytest.raises(SearchError, match=message.replace("(", r"\(").replace(")", r"\)")):
        parse_search(text)
