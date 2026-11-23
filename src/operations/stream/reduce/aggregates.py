"""The entries of "aggregate" in a reduce (DESIGN.md section 5).

- named entry:  output field -> "criteria(field)", e.g. "peak_kw": "max(kw)"; "readings": "count()" counts rows
- field entry:  field name or pattern -> "criteria", e.g. "kw": "mean", "charge*": "sum", "*": "sum";
                the output keeps each field's name

A field entry with a pattern is resolved on the rows it gets: "*" means every field that holds numbers in all
the rows, except the keys, ts, text and True/False.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

from src.exceptions import ChainError
from src.models.rows import Row
from src.operations.stream.reduce.criteria import CRITERIA

_NAMED = re.compile(r"\s*(?P<criteria>\w+)\s*\(\s*(?P<field>[^()]*?)\s*\)\s*")
_PATTERN_CHARACTERS = "*?["


@dataclass(frozen=True)
class Computation:
    """One output field of a reduce: `criteria` over the values of `field` (None: count the rows)."""

    output: str
    criteria: str
    field: str | None

    def compute(self, group: list[Row]) -> Any:
        if self.field is None:
            return len(group)
        values = []
        for row in group:
            if self.field not in row:
                raise ChainError(f"unknown field {self.field}")
            if row[self.field] is not None:
                values.append(row[self.field])
        try:
            return CRITERIA[self.criteria](values)
        except TypeError as error:
            raise ChainError(f"{self.criteria}({self.field}) cannot be computed: {error}") from None


class Aggregate(ABC):
    """One entry of "aggregate", as written in the chain."""

    criteria: str

    @abstractmethod
    def resolve(self, items: list[Row], excluded: set[str]) -> list[Computation]:
        """The computations this entry stands for, given the rows of the reduce and the fields to leave out."""


@dataclass(frozen=True)
class NamedAggregate(Aggregate):
    output: str
    criteria: str
    field: str | None  # None only for count()

    def resolve(self, items: list[Row], excluded: set[str]) -> list[Computation]:
        return [Computation(self.output, self.criteria, self.field)]


@dataclass(frozen=True)
class FieldAggregate(Aggregate):
    pattern: str  # a field name, or a pattern with * ? [
    criteria: str

    def resolve(self, items: list[Row], excluded: set[str]) -> list[Computation]:
        if not any(character in self.pattern for character in _PATTERN_CHARACTERS):
            return [Computation(self.pattern, self.criteria, self.pattern)]
        names = [name for name in numeric_fields(items) if name not in excluded and fnmatchcase(name, self.pattern)]
        if items and not names:
            raise ChainError(f"{self.pattern!r} matches no field that holds numbers")
        return [Computation(name, self.criteria, name) for name in names]


def parse_aggregate(key: str, text: str) -> Aggregate:
    """One entry of "aggregate": a named entry if the text has parentheses, else a field entry."""
    named = _NAMED.fullmatch(text)
    if named is None:
        criteria = text.strip()
        _check_criteria(criteria, key)
        return FieldAggregate(pattern=key, criteria=criteria)
    criteria, field = named["criteria"], named["field"] or None
    _check_criteria(criteria, key)
    if field is None and criteria != "count":
        raise ChainError(f"{key}: {criteria}() needs a field, for example {criteria}(kw)")
    return NamedAggregate(output=key, criteria=criteria, field=field)


def resolve_all(aggregates: tuple[Aggregate, ...], items: list[Row], key_names: tuple[str, ...]) -> list[Computation]:
    """The computations of all entries. Two outputs with the same name, or an output named like a key, is an error."""
    excluded = {*key_names, "ts"}
    computations = [computation for entry in aggregates for computation in entry.resolve(items, excluded)]
    outputs = [computation.output for computation in computations]
    for output in outputs:
        if outputs.count(output) > 1:
            raise ChainError(f"aggregate gives the field {output} twice")
        if output in key_names:
            raise ChainError(f"aggregate gives the field {output}, which is already a key")
    return computations


def numeric_fields(items: list[Row]) -> list[str]:
    """The fields that every row has, holding a number or null, with at least one number; in the order of the
    first row."""
    if not items:
        return []
    return [name for name in items[0] if _holds_numbers(items, name)]


def _holds_numbers(items: list[Row], name: str) -> bool:
    values = [row.get(name, _MISSING) for row in items]
    if any(value is _MISSING or not (value is None or _is_number(value)) for value in values):
        return False
    return any(value is not None for value in values)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


_MISSING = object()


def _check_criteria(criteria: str, key: str) -> None:
    if criteria not in CRITERIA:
        raise ChainError(f"{key}: unknown criteria {criteria!r}; the criteria are {', '.join(CRITERIA)}")
