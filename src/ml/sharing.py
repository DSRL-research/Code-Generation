"""Sharing models between sandboxes (DESIGN.md section 11, "Sharing models").

A model leaves its sandbox as a ModelRecord of plain numbers (GET /models/{name}/export), and enters another one
through the load operator. No file of another sandbox is ever unpickled.

Linear models can be shared: those trained with algo "linear", and those adapted from them (finetune adds a linear
correction, so source + correction is again one linear model). gbr models cannot be shared yet.
"""

from datetime import timedelta
from typing import Any

import numpy as np
from pydantic import ValidationError
from sklearn.linear_model import Ridge

from src.exceptions import ChainError
from src.ml.features import FeatureSpec
from src.ml.trained import AdaptedEstimator, TrainedModel, make_card
from src.models.model_record import FORMAT, ModelRecord
from src.models.rows import Row
from src.store.model_store import check_name

MAX_ERRORS = 3


class LinearWeights:
    """A linear model given by its numbers: what a loaded model predicts with."""

    def __init__(self, coef: list[float], intercept: float) -> None:
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = float(intercept)

    def predict(self, features: Any) -> np.ndarray:
        return np.asarray(features, dtype=float) @ self.coef + self.intercept


def linear_parts(estimator: Any) -> tuple[list[float], float] | None:
    """(coef, intercept) of a linear estimator, or None if it is not linear."""
    if isinstance(estimator, LinearWeights):
        return [float(value) for value in estimator.coef], estimator.intercept
    if isinstance(estimator, Ridge):
        return [float(value) for value in estimator.coef_], float(estimator.intercept_)
    if isinstance(estimator, AdaptedEstimator):
        source, residual = linear_parts(estimator.source), linear_parts(estimator.residual)
        if source is None or residual is None:
            return None
        coef = [a + b for a, b in zip(source[0], residual[0], strict=True)]
        return coef, source[1] + residual[1]
    return None


def shareable(card: dict[str, Any]) -> bool:
    """Whether a model can be exported, from its card. An adapted model keeps the algo of its source, and a loaded
    model is linear, so the algo says it."""
    return card.get("algo") == "linear"


def export_record(model: TrainedModel) -> dict[str, Any]:
    """The model as one record: its card, and the numbers to predict with."""
    parts = linear_parts(model.estimator)
    if parts is None:
        raise ChainError(
            f"{model.name} is a {model.card.get('algo')} model; only linear models can be exported for now"
        )
    coef, intercept = parts
    return {"format": FORMAT, **model.card, "coef": coef, "intercept": intercept}


def model_from_record(record: Row, name: str | None = None) -> TrainedModel:
    """The model a record describes, saved as `name` (default: the name in the record), with kind "loaded".

    Its card keeps where it comes from: source_model is the name in the record; rows, first_ts, last_ts, mae and
    rmse describe its training in the sandbox that exported it.
    """
    try:
        checked = ModelRecord.model_validate(record)
    except ValidationError as error:
        raise ChainError(f"the record is not a model that can be loaded: {_describe(error)}") from None
    output = check_name(name or checked.name)
    spec = FeatureSpec(
        checked.target, tuple(checked.lags), tuple(checked.calendar), timedelta(minutes=checked.interval_minutes)
    )
    card = make_card(output, "loaded", "linear", {}, spec, [], 0, None, source_model=checked.name)
    card.update(
        rows=checked.rows,
        holdout_rows=checked.holdout_rows,
        first_ts=checked.first_ts,
        last_ts=checked.last_ts,
        mae=checked.mae,
        rmse=checked.rmse,
    )
    return TrainedModel(output, LinearWeights(checked.coef, checked.intercept), spec, card)


def _describe(error: ValidationError) -> str:
    """The first problems, each as "<field>: <problem>"."""
    problems = []
    for item in error.errors()[:MAX_ERRORS]:
        place = ".".join(str(part) for part in item["loc"])
        problem = str(item["msg"]).removeprefix("Value error, ")
        problems.append(f"{place}: {problem}" if place else problem)
    return "; ".join(problems)
