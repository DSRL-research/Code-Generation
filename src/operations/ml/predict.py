from dataclasses import dataclass
from typing import ClassVar

from src.ml.features import history
from src.models.rows import Rows
from src.operations.base import Operation, RunContext
from src.operations.ml.common import interval_of, store_of


@dataclass(frozen=True)
class PredictOperation(Operation):
    """Adds the model's prediction to each row, computed from the actual values before it (a backtest).

    JSON: {"op": "predict", "model": "mg1_demand", "as": "kw_pred"}
    The rows need the model's target and time step. Rows without every lag (the first ones) are dropped.
    """

    name: ClassVar[str] = "predict"
    model: str
    output: str = "prediction"

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        model = store_of(ctx).load(self.model)
        interval_of(rows, model.spec.interval)
        known = history(rows.items, model.spec.target)
        kept, features = [], []
        for row in rows.items:
            row_features = model.spec.features(row["ts"], known)
            if row_features is not None:
                kept.append(row)
                features.append(row_features)
        predictions = model.predict(features)
        return rows.with_items([{**row, self.output: value} for row, value in zip(kept, predictions, strict=True)])
