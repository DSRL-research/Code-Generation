from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from src.exceptions import ChainError
from src.models.rows import Row, Rows
from src.operations.base import Operation, RunContext


@dataclass(frozen=True)
class TopOperation(Operation):
    """Keeps the n rows with the highest (desc) or lowest (asc) value of a field (TopOperation.java).

    JSON: {"op": "top", "n": 3, "by": "kw", "order": "desc"}
    Rows with the same value keep their input order. Rows where the field is None come last.
    """

    name: ClassVar[str] = "top"
    n: int
    by: str
    order: Literal["asc", "desc"] = "desc"

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        with_value = [row for row in rows.items if self._value(row) is not None]
        without_value = [row for row in rows.items if self._value(row) is None]
        try:
            ordered = sorted(with_value, key=self._value, reverse=self.order == "desc")  # sorted() is stable
        except TypeError as error:
            raise ChainError(f"cannot order the values of {self.by}: {error}") from None
        return rows.with_items((ordered + without_value)[: self.n])

    def _value(self, row: Row) -> Any:
        if self.by not in row:
            raise ChainError(f"unknown field {self.by}")
        return row[self.by]
