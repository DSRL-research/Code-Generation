"""Checks and steps shared by the ML operators."""

from collections.abc import Sequence
from datetime import timedelta

from src.exceptions import ChainError, SandboxError
from src.expressions import Expression
from src.ml.trained import TrainedModel
from src.models.chain import Step
from src.models.rows import Row, Rows
from src.operations.base import RunContext
from src.store import ModelStore

MIN_TRAINING_ROWS = 10


def store_of(ctx: RunContext) -> ModelStore:
    if ctx.models is None:
        raise ChainError("this run has no model store")
    return ctx.models


def replay_recipe(model: TrainedModel, rows: Rows, ctx: RunContext) -> Rows:
    """The rows after the steps of the model's recipe: its inputs, made as when it was trained."""
    return run_recipe(model, recipe_steps_of(model, ctx), rows, ctx)


def recipe_steps_of(model: TrainedModel, ctx: RunContext) -> Sequence[Step]:
    """The steps of the model's recipe, parsed (and checked) once: forecast runs them at every step."""
    if ctx.parse_steps is None:
        raise ChainError("this run cannot run the recipe of a model")
    try:
        return ctx.parse_steps(model.card.get("recipe", []))
    except SandboxError as error:
        raise ChainError(f"the recipe of {model.name}, {error}") from None


def run_recipe(model: TrainedModel, steps: Sequence[Step], rows: Rows, ctx: RunContext) -> Rows:
    """The rows after `steps`; an error names the model and the recipe step."""
    try:
        for step in steps:
            try:
                rows = step.operation.apply(rows, ctx)
            except SandboxError as error:
                raise error.at(step.where) from None
    except SandboxError as error:
        raise ChainError(f"the recipe of {model.name}, {error}") from None
    return rows


def check_same_interval(rows: Rows, expected: timedelta | None) -> None:
    """When the time steps of the model and of the rows are both known, they must be the same."""
    if expected is not None and rows.interval is not None and rows.interval != expected:
        raise ChainError(_other_step(expected, rows.interval))


def interval_of(rows: Rows, expected: timedelta) -> timedelta:
    """The time step of the rows, which must be known and be the model's: forecast steps forward with it."""
    if rows.interval is None:
        raise ChainError("the time step of the rows is unknown; reduce them with a window first, e.g. 1h")
    if rows.interval != expected:
        raise ChainError(_other_step(expected, rows.interval))
    return rows.interval


def where_holds(where: Expression | None, row: Row) -> bool:
    """The value of a `where` expression (train, finetune) for a row: True or False, else an error; True if none."""
    if where is None:
        return True
    value = where.evaluate(row)
    if not isinstance(value, bool):
        raise ChainError(f"where must give True or False, not {value!r}, in {where.text!r}")
    return value


def _other_step(expected: timedelta, actual: timedelta) -> str:
    return (
        f"the model works on {_minutes(expected)} steps, the rows have {_minutes(actual)} steps; "
        f"reduce them with a {_minutes(expected)} window first"
    )


def _minutes(interval: timedelta) -> str:
    return f"{interval.total_seconds() / 60:g}-minute"
