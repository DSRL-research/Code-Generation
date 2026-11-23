from dataclasses import dataclass
from typing import ClassVar

from src.exceptions import ChainError
from src.ml.metrics import errors
from src.models.rows import Row, Rows
from src.operations.base import Operation, RunContext


@dataclass(frozen=True)
class EvaluateOperation(Operation):
    """Compares predictions with actual values and returns one row: {n, mae, rmse, mape}.

    JSON: {"op": "evaluate", "actual": "kw", "predicted": "kw_pred"}
    Rows where either value is null are skipped.
    """

    name: ClassVar[str] = "evaluate"
    actual: str
    predicted: str

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        pairs = [(self._value(row, self.actual), self._value(row, self.predicted)) for row in rows.items]
        pairs = [(actual, predicted) for actual, predicted in pairs if actual is not None and predicted is not None]
        return Rows([errors([actual for actual, _ in pairs], [predicted for _, predicted in pairs])])

    @staticmethod
    def _value(row: Row, field: str) -> float | None:
        if field not in row:
            raise ChainError(f"unknown field {field}")
        value = row[field]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int | float)):
            raise ChainError(f"{field} must hold numbers, not {value!r}")
        return value
