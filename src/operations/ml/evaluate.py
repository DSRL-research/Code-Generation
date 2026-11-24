from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar

from src.exceptions import ChainError
from src.ml.metrics import errors
from src.models.rows import Row, Rows
from src.operations.base import Operation, RunContext
from src.operations.ml.common import store_of


@dataclass(frozen=True)
class EvaluateOperation(Operation):
    """Compares predictions with actual values and returns one row: {n, mae, rmse, mape}.

    JSON: {"op": "evaluate", "actual": "kw", "predicted": "kw_pred", "model": "mg2_demand"}
    Rows where either value is null are skipped. With `model`, the scores are also written on that model's card
    (mae, rmse, mape, and which rows: evaluated_rows, evaluated_from, evaluated_to), and the row names the model:
    train does not test, so this is how a model's card gets its quality.
    """

    name: ClassVar[str] = "evaluate"
    actual: str
    predicted: str
    model: str | None = None

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        compared = [row for row in rows.items if self._compared(row)]
        scores = errors([row[self.actual] for row in compared], [row[self.predicted] for row in compared])
        if self.model is None:
            return Rows([scores])
        store_of(ctx).update_card(self.model, self._card_changes(scores, compared))
        return Rows([{"model": self.model, **scores}])

    def _compared(self, row: Row) -> bool:
        return self._value(row, self.actual) is not None and self._value(row, self.predicted) is not None

    @staticmethod
    def _card_changes(scores: dict[str, Any], compared: list[Row]) -> dict[str, Any]:
        times = [row["ts"] for row in compared if isinstance(row.get("ts"), datetime)]
        return {
            "mae": scores["mae"],
            "rmse": scores["rmse"],
            "mape": scores["mape"],
            "evaluated_rows": scores["n"],
            "evaluated_from": min(times).isoformat() if times else None,
            "evaluated_to": max(times).isoformat() if times else None,
        }

    @staticmethod
    def _value(row: Row, field: str) -> float | None:
        if field not in row:
            raise ChainError(f"unknown field {field}")
        value = row[field]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int | float)):
            raise ChainError(f"{field} must hold numbers, not {value!r}")
        return value
