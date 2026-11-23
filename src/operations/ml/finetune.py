from dataclasses import dataclass
from typing import ClassVar

from sklearn.linear_model import Ridge

from src.exceptions import ChainError
from src.expressions import Expression
from src.ml.features import column_table
from src.ml.trained import AdaptedEstimator, TrainedModel, make_card
from src.models.rows import Rows
from src.operations.base import Operation, RunContext
from src.operations.ml.common import MIN_TRAINING_ROWS, check_same_interval, store_of, where_holds


@dataclass(frozen=True)
class FinetuneOperation(Operation):
    """Adapts a model of another node to the local rows and saves it under a new name; the rows pass through (UC3).

    JSON: {"op": "finetune", "model": "mg12_demand", "as": "mg11_demand", "where": "set == 'train'", "alpha": 100}
    The new model is the source model plus a correction (ridge regression) learned on the local rows where `where`
    is True; a larger alpha keeps it closer to the source. The rows need the source's feature columns (prepare makes
    them). The new model keeps the source's features and recipe; its card names the source model.
    """

    name: ClassVar[str] = "finetune"
    model: str
    output: str
    alpha: float = 100.0
    where: Expression | None = None

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        store = store_of(ctx)
        source = store.load(self.model)
        spec = source.spec
        check_same_interval(rows, spec.interval)
        chosen = [row for row in rows.items if where_holds(self.where, row)]
        for name in (*spec.features, spec.target):
            if chosen and not any(name in row for row in chosen):
                raise ChainError(
                    f'the rows have no field {name}, an input of {source.name}; add {{"op": "prepare", "model": '
                    f'"{source.name}"}} before finetune'
                )
        table, targets, times, skipped = column_table(chosen, spec.features, spec.target)
        if len(targets) < MIN_TRAINING_ROWS:
            raise ChainError(
                f"{len(targets)} rows can be learned from ({skipped} skipped); finetune needs at least "
                f"{MIN_TRAINING_ROWS}"
            )
        source_predictions = source.predict(table)
        correction = Ridge(alpha=self.alpha).fit(
            table, [y - p for y, p in zip(targets, source_predictions, strict=True)]
        )
        params = {**source.card["params"], "finetune_alpha": self.alpha}
        recipe = tuple(source.card.get("recipe", []))
        card = make_card(self.output, source.card["algo"], params, spec, recipe, times, len(targets), skipped,
                         kind="adapted", source_model=source.name)  # fmt: skip
        store.save(TrainedModel(self.output, AdaptedEstimator(source.estimator, correction), spec, card))
        return rows
