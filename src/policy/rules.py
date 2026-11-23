"""The rule of each operator in policy.json: what it may do on raw rows (DESIGN.md section 8).

  "map":    {"onRaw": false}                                    refused on raw rows
  "top":    {"onRaw": true}                                     allowed on raw rows; they stay raw
  "filter": {"onRaw": "time fields only"}                       may read only the time fields
  "reduce": {"onRaw": {"minWindowMinutes": 15, "criteria": [...]}}  needs a window; ends the raw rows

A rule checks one step and says whether the rows are still raw after it. To give an operator its own kind of
rule (the ML operators will have algorithms and parameter ranges), write an OperatorRule and add its parser to
_RULE_PARSERS.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.exceptions import PolicyFileError, PolicyViolation
from src.operations import FilterOperation, Operation, ReduceOperation


class OperatorRule(ABC):
    @abstractmethod
    def check(self, operation: Operation, raw: bool) -> bool:
        """Raises PolicyViolation if the step breaks the rule; returns whether the rows are raw after it."""


class NotOnRaw(OperatorRule):
    """{"onRaw": false}"""

    def check(self, operation: Operation, raw: bool) -> bool:
        if raw:
            raise PolicyViolation("not allowed on raw rows; reduce them with a window first")
        return False


class AnyOnRaw(OperatorRule):
    """{"onRaw": true}: allowed on raw rows, which stay raw."""

    def check(self, operation: Operation, raw: bool) -> bool:
        return raw


@dataclass(frozen=True)
class FilterOnTimeFields(OperatorRule):
    """{"onRaw": "time fields only"}: on raw rows, the expression may read only the time fields."""

    time_fields: frozenset[str]

    def check(self, operation: Operation, raw: bool) -> bool:
        if raw and isinstance(operation, FilterOperation):
            others = sorted(operation.where.fields - self.time_fields)
            if others:
                raise PolicyViolation(f"on raw rows a filter may only read {sorted(self.time_fields)}, not {others}")
        return raw


@dataclass(frozen=True)
class ReduceWithWindow(OperatorRule):
    """{"onRaw": {"minWindowMinutes": 15, "criteria": ["mean", "sum", "count"]}}: the first reduce of raw rows
    must use a window of at least minWindowMinutes, and only the listed criteria. After it, the rows are not raw."""

    min_window_minutes: int
    criteria: frozenset[str]

    def check(self, operation: Operation, raw: bool) -> bool:
        if not raw or not isinstance(operation, ReduceOperation):
            return False
        window = operation.window
        if window is None:
            raise PolicyViolation("on raw rows the first reduce must use a window")
        if window.minutes < self.min_window_minutes:
            raise PolicyViolation(f"the window {window.text} is shorter than {self.min_window_minutes} minutes")
        refused = sorted(operation.criteria - self.criteria)
        if refused:
            raise PolicyViolation(
                f"{', '.join(refused)} not allowed on raw rows; the allowed criteria are {sorted(self.criteria)}"
            )
        return False


def _filter_rule(on_raw: Any, time_fields: frozenset[str]) -> OperatorRule:
    if on_raw != "time fields only":
        raise PolicyFileError('filter: onRaw must be false, true or "time fields only"')
    return FilterOnTimeFields(time_fields)


def _reduce_rule(on_raw: Any, time_fields: frozenset[str]) -> OperatorRule:
    if not isinstance(on_raw, dict) or set(on_raw) != {"minWindowMinutes", "criteria"}:
        raise PolicyFileError('reduce: onRaw must be false, true or {"minWindowMinutes": ..., "criteria": [...]}')
    return ReduceWithWindow(int(on_raw["minWindowMinutes"]), frozenset(on_raw["criteria"]))


_RULE_PARSERS: dict[str, Callable[[Any, frozenset[str]], OperatorRule]] = {
    "filter": _filter_rule,
    "reduce": _reduce_rule,
}


def rule_from_json(operator: str, on_raw: Any, time_fields: frozenset[str]) -> OperatorRule:
    """The rule of one operator, from the value of its "onRaw"."""
    if on_raw is False:
        return NotOnRaw()
    if on_raw is True:
        return AnyOnRaw()
    parser = _RULE_PARSERS.get(operator)
    if parser is None:
        raise PolicyFileError(f"{operator}: onRaw must be true or false")
    return parser(on_raw, time_fields)
