from dataclasses import dataclass
from typing import ClassVar

from src.exceptions import ChainError
from src.ml.features import row_inputs
from src.models.rows import Rows
from src.operations.base import Operation, RunContext
from src.operations.ml.common import check_same_interval, store_of


@dataclass(frozen=True)
class PredictOperation(Operation):
    """Adds the model's prediction to each row, from the model's feature columns.

    JSON: {"op": "predict", "model": "mg2_demand", "as": "kw_pred"}
    Make the feature columns with the same steps as for training, or with prepare. A row without a number in every
    feature column gets None (evaluate skips it). The rows need the model's time step, when both are known.
    """

    name: ClassVar[str] = "predict"
    model: str
    output: str = "prediction"

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        model = store_of(ctx).load(self.model)
        spec = model.spec
        check_same_interval(rows, spec.interval)
        for name in spec.features:
            if rows.items and not any(name in row for row in rows.items):
                raise ChainError(
                    f'the rows have no field {name}, an input of {model.name}; add {{"op": "prepare", "model": '
                    f'"{model.name}"}} before predict, or the steps that made it for training'
                )
        inputs = [row_inputs(row, spec.features) for row in rows.items]
        predictions = iter(model.predict([values for values in inputs if values is not None]))
        return rows.with_items(
            [{**row, self.output: next(predictions) if values is not None else None}
             for row, values in zip(rows.items, inputs, strict=True)]
        )  # fmt: skip
