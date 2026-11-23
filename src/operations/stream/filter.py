from dataclasses import dataclass
from typing import ClassVar

from src.exceptions import ChainError
from src.expressions import Expression
from src.models.rows import Row, Rows
from src.operations.base import Operation, RunContext


@dataclass(frozen=True)
class FilterOperation(Operation):
    """Keeps the rows where `where` is True (FilterOperation.java).

    JSON: {"op": "filter", "where": "17 <= ts.hour < 21"}
    """

    name: ClassVar[str] = "filter"
    where: Expression

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        return rows.with_items([row for row in rows.items if self._keeps(row)])

    def _keeps(self, row: Row) -> bool:
        value = self.where.evaluate(row)
        if not isinstance(value, bool):
            raise ChainError(f"where must give True or False, not {value!r}, in {self.where.text!r}")
        return value
