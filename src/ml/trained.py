"""Trained models: what the model store keeps, and the card that describes each model."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.ml.features import ColumnSpec


@dataclass
class TrainedModel:
    """An estimator, the inputs it needs, and its card.

    The card is the model as an asset: what it predicts, from which columns, made by which steps (its recipe),
    trained on what, how good it was (once evaluate has scored it), and where it came from (`source_model` for an
    adapted or loaded model).
    """

    name: str
    estimator: Any
    spec: ColumnSpec
    card: dict[str, Any]

    def predict(self, features: list[list[float]]) -> list[float]:
        if not features:
            return []
        return [float(value) for value in self.estimator.predict(features)]


class AdaptedEstimator:
    """A model from another node, plus a correction learned on local rows: prediction = source + residual.

    This is the transfer learning of UC3 in its simplest form. The residual model is a ridge regression: with a
    large alpha it stays close to zero, so the source model is kept unless the local rows show otherwise.
    """

    def __init__(self, source: Any, residual: Any) -> None:
        self.source = source
        self.residual = residual

    def predict(self, features: Any) -> Any:
        return self.source.predict(features) + self.residual.predict(features)


def make_card(
    name: str,
    algo: str,
    params: dict[str, Any],
    spec: ColumnSpec,
    recipe: tuple[dict[str, Any], ...],
    train_times: list[datetime],
    rows: int,
    skipped: int,
    kind: str = "trained",
    source_model: str | None = None,
) -> dict[str, Any]:
    """The card of a model: kind "trained" (train), "adapted" (finetune) or "loaded" (load). `recipe` holds the lag
    and map steps that made its inputs. mae, rmse and mape stay None until evaluate scores the model with
    "model": its name."""
    return {
        "name": name,
        "kind": kind,
        "algo": algo,
        "params": params,
        **spec.to_json(),
        "recipe": list(recipe),
        "rows": rows,
        "skipped_rows": skipped,
        "first_ts": min(train_times).isoformat() if train_times else None,
        "last_ts": max(train_times).isoformat() if train_times else None,
        "mae": None,
        "rmse": None,
        "mape": None,
        "source_model": source_model,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
