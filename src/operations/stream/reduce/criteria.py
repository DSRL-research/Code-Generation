"""The criteria of reduce: what is computed over the values of a group.

Each one gets the values that are not null. With no values, every criterion gives None, except count (0), as in
SQL. To add a criterion, add a function here; the policy then decides whether it is allowed on raw rows.
"""

from collections.abc import Callable
from typing import Any


def _mean(values: list[Any]) -> Any:
    return sum(values) / len(values) if values else None


def _sum(values: list[Any]) -> Any:
    return sum(values) if values else None


def _min(values: list[Any]) -> Any:
    return min(values) if values else None


def _max(values: list[Any]) -> Any:
    return max(values) if values else None


CRITERIA: dict[str, Callable[[list[Any]], Any]] = {
    "mean": _mean,
    "sum": _sum,
    "count": len,
    "min": _min,
    "max": _max,
}
