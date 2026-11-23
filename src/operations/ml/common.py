"""Checks shared by the ML operators."""

from datetime import timedelta
from typing import Any

from src.exceptions import ChainError
from src.models.rows import Rows
from src.operations.base import RunContext
from src.store import ModelStore

MIN_TRAINING_ROWS = 10


def store_of(ctx: RunContext) -> ModelStore:
    if ctx.models is None:
        raise ChainError("this run has no model store")
    return ctx.models


def interval_of(rows: Rows, expected: timedelta | None = None) -> timedelta:
    """The time step of the rows, which lag features need; it must be the model's if the model is given."""
    if rows.interval is None:
        raise ChainError("the time step of the rows is unknown; reduce them with a window first, e.g. 1h")
    if expected is not None and rows.interval != expected:
        raise ChainError(
            f"the model works on {_minutes(expected)} steps, the rows have {_minutes(rows.interval)} steps; "
            f"reduce them with a {_minutes(expected)} window first"
        )
    return rows.interval


def split_holdout(features: list[Any], targets: list[float], holdout: int) -> tuple[list, list, list, list]:
    """Training rows and the last `holdout` rows, which are kept for validation."""
    training = len(targets) - holdout
    if training < MIN_TRAINING_ROWS:
        raise ChainError(
            f"{len(targets)} rows have every lag; training needs at least {MIN_TRAINING_ROWS} besides the "
            f"{holdout} holdout rows"
        )
    return features[:training], targets[:training], features[training:], targets[training:]


def _minutes(interval: timedelta) -> str:
    return f"{interval.total_seconds() / 60:g}-minute"
