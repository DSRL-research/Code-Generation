from dataclasses import dataclass, field
from typing import Any, ClassVar

from src.ml.algorithms import check_parameters, make_estimator
from src.ml.features import FeatureSpec, training_set
from src.ml.metrics import errors
from src.ml.trained import TrainedModel, make_card, summary_row
from src.models.rows import Rows
from src.operations.base import Operation, RunContext
from src.operations.ml.common import interval_of, split_holdout, store_of


@dataclass(frozen=True)
class FitOperation(Operation):
    """Trains a forecasting model on the rows, saves it in the model store, and returns one row: its card.

    JSON: {"op": "fit", "model": "mg1_demand", "target": "kw", "lags": [1, 2, 24], "calendar": ["hour", "weekday"],
           "algo": "linear", "params": {"alpha": 1.0}, "holdout": 24}
    The model predicts the target from its lags and the calendar of ts. The last `holdout` rows are not used for
    training: the error on them (mae, rmse) is the model's validation error.
    """

    name: ClassVar[str] = "fit"
    model: str
    target: str
    lags: tuple[int, ...]
    calendar: tuple[str, ...]
    algo: str = "linear"
    params: dict[str, Any] = field(default_factory=dict)
    holdout: int = 0

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        spec = FeatureSpec(self.target, self.lags, self.calendar, interval_of(rows))
        features, targets, times = training_set(spec, rows.items)
        train_x, train_y, test_x, test_y = split_holdout(features, targets, self.holdout)
        estimator = make_estimator(self.algo, self.params).fit(train_x, train_y)
        params = check_parameters(self.algo, self.params)
        card = make_card(self.model, "trained", self.algo, params, spec, times[: len(train_y)], len(test_y), None)
        model = TrainedModel(self.model, estimator, spec, card)
        if test_y:
            card.update({key: value for key, value in errors(test_y, model.predict(test_x)).items() if key != "n"})
        store_of(ctx).save(model)
        return Rows([summary_row(card)])
