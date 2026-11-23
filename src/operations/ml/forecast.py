from dataclasses import dataclass
from typing import ClassVar

from src.exceptions import ChainError
from src.ml.features import history
from src.models.rows import Rows
from src.models.time import Timestamp
from src.operations.base import Operation, RunContext
from src.operations.ml.common import interval_of, store_of


@dataclass(frozen=True)
class ForecastOperation(Operation):
    """Forecasts the next `steps` time steps after the rows, and returns only those new rows: {ts, <as>}.

    JSON: {"op": "forecast", "model": "mg1_demand", "steps": 4, "as": "load"}
    Each step is predicted from the values before it: actual ones, then the forecast ones (recursive forecast).
    """

    name: ClassVar[str] = "forecast"
    model: str
    steps: int
    output: str = "forecast"

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        model = store_of(ctx).load(self.model)
        interval = interval_of(rows, model.spec.interval)
        known = history(rows.items, model.spec.target)
        if not known:
            raise ChainError(f"forecast needs past values of {model.spec.target}")
        time = max(known)
        forecast = []
        for _ in range(self.steps):
            time = Timestamp.of(time + interval)
            features = model.spec.features(time, known)
            if features is None:
                raise ChainError(f"not enough history before {time.isoformat()} for lags {list(model.spec.lags)}")
            [value] = model.predict([features])
            known[time] = value
            forecast.append({"ts": time, self.output: value})
        return Rows(forecast, interval)
