"""The search of the target list: plain text, or a query of conditions on the targets' fields.

Plain text -- anything with none of a query's marks -- is looked for as typed, in all fields as shown. A query is
anything with one: a condition on a field (``name:"glu"``, ``mz < 300``), a quoted text at the start of a word, or
AND / OR / NOT (in capitals). In a query, text is always quoted, Python-like, in '...' or "..." (\\\\, \\' and \\"
for the characters themselves); numbers never are.

    "glucose"                     some field has glucose
    name:"glu"   name="glucose"   the name has glu / is glucose
    name~"^glu"   ~"^glu"         the name / some field has a match of the regular expression
    mz:150.5                      the m/z as shown starts so: 150.5 <= m/z < 150.6
    mz=150.5                      the m/z as shown is 150.5 (150.5000)
    200 < mz < 300   rt >= 1.5    the value as kept
    (name~"^glu" OR formula:"C6") AND NOT note:"check"

Upper and lower case are the same, in text and expressions alike, and field names may be in either. Conditions
next to each other are both to hold (AND); NOT binds closest, then AND, then OR.

A number field's : and = go by the value as shown, so that what is seen is what is found: an RT of 1.239, shown
1.24, is rt:1.24 and rt=1.24. Its comparisons go by the value as kept.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
import operator
import re
from typing import Protocol


class Kind(Enum):
    TEXT = "text"
    NUMBER = "number"


@dataclass(frozen=True)
class SearchField:
    attribute: str  # of the Target
    kind: Kind


# the names a query calls the fields by
FIELDS = {
    "index": SearchField("index", Kind.NUMBER),
    "#": SearchField("index", Kind.NUMBER),
    "name": SearchField("name", Kind.TEXT),
    "formula": SearchField("formula", Kind.TEXT),
    "adduct": SearchField("adduct", Kind.TEXT),
    "trace": SearchField("trace_type", Kind.TEXT),
    "mz": SearchField("mz", Kind.NUMBER),
    "m/z": SearchField("mz", Kind.NUMBER),
    "mztol": SearchField("mz_tolerance", Kind.NUMBER),
    "unit": SearchField("mz_tolerance_unit", Kind.TEXT),
    "rt": SearchField("rt", Kind.NUMBER),
    "rttol": SearchField("rt_tolerance", Kind.NUMBER),
    "ms": SearchField("ms_level", Kind.TEXT),
    "precursor": SearchField("precursor_mz", Kind.NUMBER),
    "ce": SearchField("collision_energy", Kind.NUMBER),
    "note": SearchField("note", Kind.TEXT),
}
FIELD_NAMES = ", ".join(name for name in FIELDS if name not in ("#", "m/z"))


class SearchRow(Protocol):
    """A target as the search sees it."""

    def texts(self) -> Iterable[str]:
        """All its fields, as shown."""

    def text(self, attribute: str) -> str:
        """One field as shown; "" where it has none."""

    def number(self, attribute: str) -> float | None:
        """A number field as kept; None where it has none (the precursor of an MS1 target, say)."""


Matcher = Callable[[SearchRow], bool]


class SearchError(ValueError):
    """A query that cannot be read; the message says what is wrong and where."""


def parse_search(text: str) -> Matcher:
    """What a row is to have to be shown; SearchError for a query that cannot be read."""
    if not _is_query(text):
        return _on_any_field(_contains(text))
    return _Parser(text).parse()


### telling a query from plain text ###
_FIELD_ALTERNATIVES = "|".join(re.escape(name) for name in sorted(FIELDS, key=len, reverse=True))
_START = r"(?:^|(?<=[\s(]))"
_QUERY_MARKS = re.compile(
    "|".join(
        (
            # a condition on a field, the field first or the number
            rf"(?i:{_START}(?:{_FIELD_ALTERNATIVES})\s*(?:<=|>=|<|>|=|:|~))",
            rf"(?i:(?:<=|>=|<|>)\s*(?:{_FIELD_ALTERNATIVES})(?=$|[\s()<>=:~]))",
            # a condition on a word that is no field, so that it is told so rather than looked for; not a word, a
            # colon and a number, which lipids are written with (PC 34:1)
            rf"{_START}[^\s()<>=:~'\"]+\s*(?:[:=~]\s*['\"]|(?:<=|>=|<|>)\s*[-+.\d])",
            # quoted text, at the start of a word, and a regular expression on any field: a quote inside a word
            # (2'-deoxyadenosine) is plain text
            rf"{_START}~?\s*['\"]",
            rf"{_START}(?:AND|OR|NOT)(?=$|[\s()'\"])",
        )
    )
)


def _is_query(text: str) -> bool:
    return _QUERY_MARKS.search(text) is not None


### reading a query ###
@dataclass(frozen=True)
class _Token:
    kind: str  # "(", ")", "AND", "OR", "NOT", "field", "op", "string", "number", "word", "end"
    value: str | None
    position: int


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?=$|[\s()<>=:~])")
_FIELD = re.compile(rf"(?i:{_FIELD_ALTERNATIVES})(?=$|[\s()<>=:~])")
_KEYWORD = re.compile(r"(?:AND|OR|NOT)(?=$|[\s()'\"])")
_OPERATOR = re.compile(r"<=|>=|<|>|=|:|~")
_WORD = re.compile(r"[^\s()<>=:~'\"]+")
_QUOTE_ESCAPES = {"\\": "\\", "'": "'", '"': '"'}


def _tokens(text: str) -> list[_Token]:
    tokens = []
    position = 0
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            tokens.append(_Token("end", None, position))
            return tokens
        character = text[position]
        if character in "()":
            tokens.append(_Token(character, None, position))
            position += 1
        elif character in "'\"":
            value, end = _read_quoted(text, position)
            tokens.append(_Token("string", value, position))
            position = end
        elif match := _OPERATOR.match(text, position):
            tokens.append(_Token("op", match.group(), position))
            position = match.end()
        elif match := _KEYWORD.match(text, position):
            tokens.append(_Token(match.group(), None, position))
            position = match.end()
        elif match := _NUMBER.match(text, position):
            tokens.append(_Token("number", match.group(), position))
            position = match.end()
        elif match := _FIELD.match(text, position):
            tokens.append(_Token("field", match.group().lower(), position))
            position = match.end()
        else:
            match = _WORD.match(text, position)
            tokens.append(_Token("word", match.group(), position))
            position = match.end()


def _read_quoted(text: str, start: int) -> tuple[str, int]:
    quote = text[start]
    characters = []
    position = start + 1
    while position < len(text):
        character = text[position]
        if character == "\\" and position + 1 < len(text):
            following = text[position + 1]
            # as in Python: the escapes of the quotes and the backslash; any other is kept as it is, \( say
            characters.append(_QUOTE_ESCAPES.get(following, character + following))
            position += 2
        elif character == quote:
            return "".join(characters), position + 1
        else:
            characters.append(character)
            position += 1
    raise SearchError(f"The text quoted at {start + 1} has no closing {quote}")


def _expression(token: _Token) -> re.Pattern:
    try:
        return re.compile(token.value, re.IGNORECASE)
    except re.error as error:
        raise SearchError(f"The expression at {token.position + 1} cannot be read: {error}") from None


_COMPARISONS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
# the comparison a number makes on the left, as the field makes it on the right: 200 < mz is mz > 200
_TURNED = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}


def _contains(needle: str) -> Callable[[str], bool]:
    needle = needle.casefold()
    return lambda shown: needle in shown.casefold()


def _equals(value: str) -> Callable[[str], bool]:
    value = value.casefold()
    return lambda shown: shown.casefold() == value


def _shown_number(shown: str) -> Decimal | None:
    try:
        return Decimal(shown)
    except InvalidOperation:
        return None  # none shown


def _starts_as(typed: str) -> Callable[[str], bool]:
    """The number as shown begins with the digits typed: 150.5 is 150.5 <= x < 150.6, -1.2 is -1.3 < x <= -1.2."""
    number = Decimal(typed)
    step = Decimal(1).scaleb(number.as_tuple().exponent)

    def starts(shown: str) -> bool:
        value = _shown_number(shown)
        if value is None:
            return False
        if number.is_signed():
            return number - step < value <= number
        return number <= value < number + step

    return starts


def _shown_equals(typed: str) -> Callable[[str], bool]:
    number = Decimal(typed)
    return lambda shown: _shown_number(shown) == number


def _on_any_field(test: Callable[[str], bool]) -> Matcher:
    return lambda row: any(test(shown) for shown in row.texts())


def _on_shown(field: SearchField, test: Callable[[str], bool]) -> Matcher:
    return lambda row: test(row.text(field.attribute))


class _Parser:
    def __init__(self, text: str):
        self._tokens = _tokens(text)
        self._index = 0

    def parse(self) -> Matcher:
        matcher = self._or()
        token = self._peek()
        if token.kind != "end":
            raise self._unexpected(token)
        return matcher

    def _peek(self) -> _Token:
        return self._tokens[self._index]

    def _next(self) -> _Token:
        token = self._tokens[self._index]
        self._index += 1
        return token

    def _or(self) -> Matcher:
        matchers = [self._and()]
        while self._peek().kind == "OR":
            self._next()
            matchers.append(self._and())
        return matchers[0] if len(matchers) == 1 else (lambda row: any(matcher(row) for matcher in matchers))

    def _and(self) -> Matcher:
        matchers = [self._not()]
        while True:
            token = self._peek()
            if token.kind == "AND":
                self._next()
            # next to each other is AND too
            elif not (token.kind in ("(", "NOT", "field", "string", "number", "word") or token.value == "~"):
                break
            matchers.append(self._not())
        return matchers[0] if len(matchers) == 1 else (lambda row: all(matcher(row) for matcher in matchers))

    def _not(self) -> Matcher:
        if self._peek().kind == "NOT":
            self._next()
            matcher = self._not()
            return lambda row: not matcher(row)
        return self._primary()

    def _primary(self) -> Matcher:
        token = self._next()
        if token.kind == "(":
            matcher = self._or()
            if self._next().kind != ")":
                raise SearchError(f"The bracket at {token.position + 1} is not closed")
            return matcher
        if token.kind == "string":
            return _on_any_field(_contains(token.value))
        if token.kind == "op" and token.value == "~":
            expression = self._next()
            if expression.kind != "string":
                raise SearchError(f'~ at {token.position + 1} wants a quoted expression, e.g. ~"^glu"')
            return _on_any_field(lambda shown, pattern=_expression(expression): pattern.search(shown) is not None)
        if token.kind == "field":
            return self._condition(token)
        if token.kind == "number":
            return self._range_from_number(token)
        if token.kind == "word":
            if self._peek().kind == "op":
                raise SearchError(f'"{token.value}" is no field; the fields are {FIELD_NAMES}')
            raise SearchError(f'Text in a query is quoted: "{token.value}"')
        raise self._unexpected(token)

    def _condition(self, field_token: _Token) -> Matcher:
        name = field_token.value
        field = FIELDS[name]
        operator_token = self._next()
        if operator_token.kind != "op":
            example = f'{name}:"..."' if field.kind is Kind.TEXT else f"{name}:150.5"
            raise SearchError(f"{name} at {field_token.position + 1} wants a condition, e.g. {example}")
        symbol, value = operator_token.value, self._next()
        if field.kind is Kind.TEXT:
            if symbol in _COMPARISONS:
                raise SearchError(f"{name} is text: it takes :, = or ~, not {symbol}")
            if value.kind != "string":
                shown = value.value if value.kind in ("word", "number") else "..."
                raise SearchError(f'Text in a query is quoted: {name}{symbol}"{shown}"')
            if symbol == "~":
                pattern = _expression(value)
                return _on_shown(field, lambda shown: pattern.search(shown) is not None)
            return _on_shown(field, _contains(value.value) if symbol == ":" else _equals(value.value))
        if symbol == "~":
            raise SearchError(f"{name} is a number: it takes :, = or a comparison, not ~")
        if value.kind != "number":
            shown = value.value if value.kind in ("string", "word") else "150.5"
            raise SearchError(f"{name} is a number, not quoted: {name}{symbol}{shown}")
        if symbol in _COMPARISONS:
            # a range is written from its lower end, 200 < mz < 300 (see _range_from_number)
            return self._compare(field, symbol, value)
        return _on_shown(field, _starts_as(value.value) if symbol == ":" else _shown_equals(value.value))

    @staticmethod
    def _compare(field: SearchField, symbol: str, number_token: _Token) -> Matcher:
        compare, number = _COMPARISONS[symbol], float(number_token.value)

        def matches(row: SearchRow) -> bool:
            kept = row.number(field.attribute)
            return kept is not None and compare(kept, number)

        return matches

    def _range_from_number(self, number_token: _Token) -> Matcher:
        operator_token, field_token = self._next(), self._next()
        if (
            operator_token.value not in _COMPARISONS
            or field_token.kind != "field"
            or FIELDS[field_token.value].kind is not Kind.NUMBER
        ):
            raise SearchError(f"The number at {number_token.position + 1} wants a comparison with a field, e.g. 200 < mz")
        field = FIELDS[field_token.value]
        matchers = [self._compare(field, _TURNED[operator_token.value], number_token)]
        following = self._peek()
        if following.kind == "op" and following.value in _COMPARISONS:
            self._next()
            upper = self._next()
            if upper.kind != "number":
                raise SearchError(f"{field_token.value} {following.value} at {following.position + 1} wants a number")
            matchers.append(self._compare(field, following.value, upper))
        return lambda row: all(matcher(row) for matcher in matchers)

    @staticmethod
    def _unexpected(token: _Token) -> SearchError:
        if token.kind == "end":
            return SearchError("The query ends too soon")
        if token.kind == ")":
            return SearchError(f"The bracket at {token.position + 1} closes nothing")
        shown = token.value if token.value is not None else token.kind
        return SearchError(f"Not expected at {token.position + 1}: {shown}")
