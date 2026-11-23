from dataclasses import dataclass
from typing import ClassVar

from sklearn.linear_model import Ridge

from src.ml.features import training_set
from src.ml.metrics import errors
from src.ml.trained import AdaptedEstimator, TrainedModel, make_card, summary_row
from src.models.rows import Rows
from src.operations.base import Operation, RunContext
from src.operations.ml.common import interval_of, split_holdout, store_of


@dataclass(frozen=True)
class FinetuneOperation(Operation):
    """Adapts a model of another node to the local rows, saves it under a new name, and returns its card (UC3).

    JSON: {"op": "finetune", "model": "mg12_demand", "as": "mg11_demand", "alpha": 100, "holdout": 24}
    The new model is the source model plus a correction (ridge regression) learned on the local rows; a larger
    alpha keeps it closer to the source. It uses the source's features, so the rows need the source's target and
    time step. Its card names the source model.
    """

    name: ClassVar[str] = "finetune"
    model: str
    output: str
    alpha: float = 100.0
    holdout: int = 0

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        store = store_of(ctx)
        source = store.load(self.model)
        interval_of(rows, source.spec.interval)
        features, targets, times = training_set(source.spec, rows.items)
        train_x, train_y, test_x, test_y = split_holdout(features, targets, self.holdout)
        source_predictions = source.predict(train_x)
        residual = Ridge(alpha=self.alpha).fit(
            train_x, [y - p for y, p in zip(train_y, source_predictions, strict=True)]
        )
        params = {**source.card["params"], "finetune_alpha": self.alpha}
        card = make_card(self.output, "adapted", source.card["algo"], params, source.spec, times[: len(train_y)],
                         len(test_y), None, source_model=self.model)  # fmt: skip
        model = TrainedModel(self.output, AdaptedEstimator(source.estimator, residual), source.spec, card)
        if test_y:
            card.update({key: value for key, value in errors(test_y, model.predict(test_x)).items() if key != "n"})
        store.save(model)
        return Rows([summary_row(card)])
