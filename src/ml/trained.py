"""Trained models: what the model store keeps, and the card that describes each model."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.ml.features import FeatureSpec


@dataclass
class TrainedModel:
    """An estimator, the features it needs, and its card.

    The card is the model as an asset: what it predicts, from which features, trained on what, how good it was on
    its holdout rows, and where it came from (`source_model` for an adapted model).
    """

    name: str
    estimator: Any
    spec: FeatureSpec
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
    kind: str,
    algo: str,
    params: dict[str, Any],
    spec: FeatureSpec,
    train_times: list[datetime],
    holdout_rows: int,
    errors: dict[str, Any] | None,
    source_model: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,  # "trained" (fit), "adapted" (finetune) or "loaded" (load)
        "algo": algo,
        "params": params,
        "features": spec.names,
        **spec.to_json(),
        "rows": len(train_times),
        "holdout_rows": holdout_rows,
        "first_ts": train_times[0].isoformat() if train_times else None,
        "last_ts": train_times[-1].isoformat() if train_times else None,
        "mae": errors["mae"] if errors else None,
        "rmse": errors["rmse"] if errors else None,
        "source_model": source_model,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


SUMMARY_FIELDS = ("name", "kind", "algo", "rows", "holdout_rows", "mae", "rmse", "source_model")


def summary_row(card: dict[str, Any]) -> dict[str, Any]:
    """The row that fit and finetune return: the main fields of the card."""
    return {field: card[field] for field in SUMMARY_FIELDS}
