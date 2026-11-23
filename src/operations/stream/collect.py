from dataclasses import dataclass
from typing import Any, ClassVar

from src.exceptions import ChainError
from src.models.rows import Row, Rows
from src.operations.base import Operation, RunContext


@dataclass(frozen=True)
class CollectOperation(Operation):
    """Keeps the listed fields, in that order, and rounds decimal numbers (CollectOperation.java).

    JSON: {"op": "collect", "fields": ["ts", "kw"], "decimals": 1}; without "fields", every field is kept.
    It can only be the last step (ChainParser checks this).
    """

    name: ClassVar[str] = "collect"
    fields: tuple[str, ...] | None
    decimals: int = 2

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        return rows.with_items([self._collect(row) for row in rows.items])

    def _collect(self, row: Row) -> Row:
        names = self.fields if self.fields is not None else tuple(row)
        collected = {}
        for name in names:
            if name not in row:
                raise ChainError(f"unknown field {name}")
            collected[name] = round_numbers(row[name], self.decimals)
        return collected


def round_numbers(value: Any, decimals: int) -> Any:
    """Rounds decimal numbers with Python's round(), also inside lists and objects. Other values stay as they are."""
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, list | tuple):
        return [round_numbers(item, decimals) for item in value]
    if isinstance(value, dict):
        return {key: round_numbers(item, decimals) for key, item in value.items()}
    return value
