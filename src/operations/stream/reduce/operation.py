from dataclasses import dataclass
from typing import Any, ClassVar

from src.exceptions import ChainError
from src.models.rows import Row, Rows
from src.operations.base import Operation, RunContext
from src.operations.stream.reduce.aggregates import Aggregate, resolve_all
from src.operations.stream.reduce.grouping import Grouping, GroupKey, WindowGrouping
from src.operations.stream.reduce.window import Window


@dataclass(frozen=True)
class ReduceOperation(Operation):
    """Groups the rows and computes aggregates per group (ReduceOperation.java).

    JSON: {"op": "reduce", "window": "15m", "aggregate": {"kw": "mean(kw)"}}
          {"op": "reduce", "by": {"slot": "ts.slot()"}, "aggregate": {"baseline_kw": "mean(kw)"}}
          {"op": "reduce", "aggregate": {"charge*": "sum"}}
    An output row holds the key fields and the aggregates. The output is sorted by the key.
    """

    name: ClassVar[str] = "reduce"
    grouping: Grouping
    aggregates: tuple[Aggregate, ...]

    @property
    def window(self) -> Window | None:
        """The window, or None if the rows are grouped otherwise. The policy uses it."""
        return self.grouping.window if isinstance(self.grouping, WindowGrouping) else None

    @property
    def criteria(self) -> frozenset[str]:
        """Every criteria the aggregates use. The policy uses it."""
        return frozenset(aggregate.criteria for aggregate in self.aggregates)

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        computations = resolve_all(self.aggregates, rows.items, self.grouping.key_names)
        output = []
        for key, group in _sorted_groups(self._groups(rows.items)):
            if self.grouping.keeps(group, rows.interval):
                keys = dict(zip(self.grouping.key_names, key, strict=True))
                output.append({**keys, **{c.output: c.compute(group) for c in computations}})
        return Rows(output, self.grouping.interval_after())

    def _groups(self, items: list[Row]) -> dict[GroupKey, list[Row]]:
        groups: dict[GroupKey, list[Row]] = {}
        for row in items:
            groups.setdefault(self.grouping.key_of(row), []).append(row)
        return groups


def _sorted_groups(groups: dict[GroupKey, list[Row]]) -> list[tuple[GroupKey, list[Row]]]:
    """The groups in key order; None sorts last."""
    try:
        return sorted(groups.items(), key=lambda item: _sortable(item[0]))
    except TypeError as error:
        raise ChainError(f"the keys cannot be ordered, they hold values of different types: {error}") from None


def _sortable(key: GroupKey) -> tuple[tuple[bool, Any], ...]:
    return tuple((value is None, value) for value in key)
