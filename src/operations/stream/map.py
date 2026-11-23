from dataclasses import dataclass
from typing import ClassVar

from src.expressions import Expression
from src.models.rows import Row, Rows
from src.operations.base import Operation, RunContext


@dataclass(frozen=True)
class MapOperation(Operation):
    """Sets fields from expressions, in order, so a later expression reads a field set earlier (MapOperation.java).

    JSON: {"op": "map", "set": {"diff_kw": "load - baseline_kw", "flex_kw": "max(0, diff_kw)"}}
    A value can be of any type: a number, text, True/False, None or a time.
    """

    name: ClassVar[str] = "map"
    assignments: dict[str, Expression]  # field -> expression, in the order to apply them

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        return rows.with_items([self._map(row) for row in rows.items])

    def _map(self, row: Row) -> Row:
        mapped = dict(row)
        for field, expression in self.assignments.items():
            mapped[field] = expression.evaluate(mapped)
        return mapped
