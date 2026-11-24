from dataclasses import dataclass, field
from typing import Any, ClassVar

from src.exceptions import ChainError
from src.expressions import Expression
from src.ml.algorithms import check_parameters, make_estimator
from src.ml.features import ColumnSpec, column_table
from src.ml.trained import TrainedModel, make_card
from src.models.rows import Row, Rows
from src.operations.base import RecordsRecipe, RunContext
from src.operations.ml.common import MIN_TRAINING_ROWS, store_of, where_holds


@dataclass(frozen=True)
class TrainOperation(RecordsRecipe):
    """Trains a model on columns of the rows and saves it; the rows pass through unchanged.

    JSON: {"op": "train", "model": "mg2_demand", "target": "kw", "features": ["kw_lag_1", "kw_lag_24", "hour_sin"],
           "where": "set == 'train'", "algo": "gbr", "params": {"n_estimators": 200}}
    The model learns `target` from the `features` columns, made by the steps before (lag, map), on the rows where
    `where` is True (all rows without it). Rows whose target or a feature is missing or not a number are skipped.
    The model is saved with its recipe, the lag and map steps that made its inputs; its card is at
    GET /models/{model}. To test it: predict, filter the test rows, evaluate with "model" (the scores go on its card).
    """

    name: ClassVar[str] = "train"
    model: str
    target: str
    features: tuple[str, ...]
    where: Expression | None = None
    algo: str = "linear"
    params: dict[str, Any] = field(default_factory=dict)
    recipe: tuple[dict[str, Any], ...] = ()

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        chosen = [row for row in rows.items if where_holds(self.where, row)]
        self._check_fields(chosen)
        table, targets, times, skipped = column_table(chosen, self.features, self.target)
        if len(targets) < MIN_TRAINING_ROWS:
            raise ChainError(
                f"{len(targets)} rows can be trained on ({skipped} skipped: a missing or non-number target or "
                f"feature); training needs at least {MIN_TRAINING_ROWS}"
            )
        estimator = make_estimator(self.algo, self.params).fit(table, targets)
        spec = ColumnSpec(self.target, self.features, rows.interval)
        params = check_parameters(self.algo, self.params)
        card = make_card(self.model, self.algo, params, spec, self.recipe, times, len(targets), skipped)
        store_of(ctx).save(TrainedModel(self.model, estimator, spec, card))
        return rows

    def _check_fields(self, items: list[Row]) -> None:
        """A name that no row has is a typo, not a row to skip."""
        for name in (*self.features, self.target):
            if items and not any(name in row for row in items):
                raise ChainError(f"the rows have no field {name}")
