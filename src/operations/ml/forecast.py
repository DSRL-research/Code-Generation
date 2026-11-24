from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar

from src.exceptions import ChainError
from src.ml.features import ColumnSpec, row_inputs
from src.models.chain import Step
from src.models.rows import Row, Rows
from src.models.time import Timestamp
from src.operations.base import Operation, RunContext
from src.operations.ml.common import interval_of, recipe_steps_of, run_recipe, store_of
from src.operations.stream.lag import LagOperation


@dataclass(frozen=True)
class ForecastOperation(Operation):
    """Forecasts the next `steps` time steps after the rows, and returns only those new rows: {ts, <as>}.

    JSON: {"op": "forecast", "model": "mg2_demand", "steps": 24, "as": "load"}
    Give it the rows of one series as they come before the model's recipe (e.g. after the reduce). Each step adds
    the next row, makes its inputs with the recipe, predicts, and fills in the value, which the next steps' lags
    read (a recursive forecast). So the inputs may use ts and earlier values, not fields future rows do not have.
    """

    name: ClassVar[str] = "forecast"
    model: str
    steps: int
    output: str = "forecast"

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        model = store_of(ctx).load(self.model)
        spec = model.spec
        if spec.interval is None:
            raise ChainError(f"{model.name} was trained on rows without a known time step, so it cannot step forward")
        interval = interval_of(rows, spec.interval)
        recipe = recipe_steps_of(model, ctx)
        series = _series(recipe, rows.items)
        history = [row for row in rows.items if isinstance(row.get("ts"), datetime)]
        if not history:
            raise ChainError("forecast needs past rows with a time (ts)")
        reach = _reach(recipe) * interval
        time = max(row["ts"] for row in history)
        forecast = []
        for _ in range(self.steps):
            time = Timestamp.of(time + interval)
            tail = [row for row in history if row["ts"] >= time - reach]
            new_rows = Rows([*tail, {**series, "ts": time, spec.target: None}], interval)
            prepared = run_recipe(model, recipe, new_rows, ctx)
            [value] = model.predict([_inputs_at(time, prepared.items, spec, model.name)])
            history.append({**series, "ts": time, spec.target: value})
            forecast.append({"ts": time, self.output: value})
        return Rows(forecast, interval)


def _inputs_at(time: datetime, items: list[Row], spec: ColumnSpec, model: str) -> list[float]:
    row = next((row for row in items if row.get("ts") == time), None)
    if row is None:
        raise ChainError(f"not enough history before {time.isoformat()} for the lags of the recipe of {model}")
    inputs = row_inputs(row, spec.features)
    if inputs is None:
        missing = [name for name in spec.features if row_inputs(row, (name,)) is None]
        raise ChainError(
            f"the inputs {missing} of {time.isoformat()} are not numbers; forecast can only use inputs made from "
            "ts and earlier values"
        )
    return inputs


def _reach(recipe: Sequence[Step]) -> int:
    """How many time steps back the recipe reads: the largest lag of each lag step, added up (a lag of a lag)."""
    return sum(max(step.operation.lags) for step in recipe if isinstance(step.operation, LagOperation))


def _series(recipe: Sequence[Step], items: list[Row]) -> dict[str, Any]:
    """The `by` fields of the recipe's lags, with the one series' values: the new rows belong to that series."""
    names = list(dict.fromkeys(name for step in recipe if isinstance(step.operation, LagOperation)
                               for name in step.operation.by))  # fmt: skip
    if not names:
        return {}
    values = {tuple(row.get(name) for name in names) for row in items}
    if len(values) != 1:
        raise ChainError(
            f"forecast continues one series, and the rows hold {len(values)} (by {names}); filter them to one first"
        )
    return dict(zip(names, values.pop(), strict=True))
