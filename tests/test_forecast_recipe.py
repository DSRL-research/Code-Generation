"""forecast with a model made by train: each new row's inputs are made by the model's recipe (DESIGN.md section 17,
task 5)."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.exceptions import ChainError
from src.store import ModelStore
from tests.conftest import SourceServer
from tests.utils import mg_rows, rows, run, t

LINK = "http://test/mg"
LAG = {"op": "lag", "field": "kw", "lags": [1, 24]}
INPUTS = {"op": "map", "set": {
    "hour_sin": "sin(2 * pi * ts.hour / 24)",
    "hour_cos": "cos(2 * pi * ts.hour / 24)",
    "weekend": "1 if ts.weekday() >= 5 else 0"}}  # fmt: skip
FEATURES = ["kw_lag_1", "kw_lag_24", "hour_sin", "hour_cos", "weekend"]
TRAIN = {"op": "train", "model": "mg1_demand", "target": "kw", "features": FEATURES}


@pytest.fixture
def store(tmp_path) -> ModelStore:
    store = ModelStore(tmp_path / "models")
    run(chain(LAG, INPUTS, TRAIN), {LINK: mg_rows(1, "2026-07-01", "2026-07-17")}, models=store)
    return store


def chain(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"source": LINK, "steps": list(steps)}


def forecast(store: ModelStore, source: Any, steps: int = 4, model: str = "mg1_demand") -> list[dict[str, Any]]:
    return run(chain({"op": "forecast", "model": model, "steps": steps, "as": "load"}), {LINK: source}, models=store)


def test_forecast_gives_the_next_hours_after_the_rows(store: ModelStore) -> None:
    result = forecast(store, mg_rows(1, "2026-07-17", "2026-07-18"))
    assert [row["ts"] for row in result] == [t(f"2026-07-19T0{hour}:00") for hour in range(4)]
    assert all(200 < row["load"] < 900 for row in result)


def test_the_first_step_is_the_prediction_from_the_actual_rows(store: ModelStore) -> None:
    """One step ahead, forecast and prepare + predict read the same inputs, so they give the same value."""
    [first] = forecast(store, mg_rows(1, "2026-07-17", "2026-07-18"), steps=1)
    predicted = run(chain({"op": "prepare", "model": "mg1_demand"}, {"op": "predict", "model": "mg1_demand"}),
                    {LINK: mg_rows(1, "2026-07-17", "2026-07-19")}, models=store)  # fmt: skip
    [same_hour] = [row for row in predicted if row["ts"] == first["ts"]]
    assert first["load"] == pytest.approx(same_hour["prediction"])


def test_each_step_reads_the_value_forecast_for_the_step_before(tmp_path) -> None:
    """A model that learned 'one more than an hour before' counts on: 20, 21, 22 after 19."""
    store = ModelStore(tmp_path / "models")
    counting = rows(*[{"ts": t(f"2026-07-01T{hour:02d}:00"), "kw": float(hour)} for hour in range(20)],
                    interval_minutes=60)  # fmt: skip
    train = {"op": "train", "model": "count", "target": "kw", "features": ["kw_lag_1"], "params": {"alpha": 0}}
    run(chain({"op": "lag", "field": "kw", "lags": [1]}, train), {LINK: counting}, models=store)
    result = forecast(store, counting, steps=3, model="count")
    assert [row["load"] for row in result] == pytest.approx([20.0, 21.0, 22.0])
    assert result[-1]["ts"] == t("2026-07-01T22:00")


def test_one_day_of_history_is_enough_for_lag_24(store: ModelStore) -> None:
    assert len(forecast(store, mg_rows(1, "2026-07-18", "2026-07-18"), steps=24)) == 24


def test_not_enough_history_for_the_lags(store: ModelStore) -> None:
    half_a_day = rows(*mg_rows(1, "2026-07-18", "2026-07-18").items[12:], interval_minutes=60)
    with pytest.raises(ChainError, match="not enough history before 2026-07-19T00:00:00 for the lags of the recipe"):
        forecast(store, half_a_day)


def test_an_input_that_future_rows_do_not_have_is_named(tmp_path) -> None:
    store = ModelStore(tmp_path / "models")
    weather = rows(*[{"ts": t(f"2026-07-01T{hour:02d}:00"), "kw": 2.0 * hour, "temp_c": 20.0 + hour}
                     for hour in range(20)], interval_minutes=60)  # fmt: skip
    train = {"op": "train", "model": "warm", "target": "kw", "features": ["warm"]}
    run(chain({"op": "map", "set": {"warm": "temp_c - 20"}}, train), {LINK: weather}, models=store)
    with pytest.raises(ChainError) as error:
        forecast(store, weather, model="warm")
    assert str(error.value) == (
        "steps[0] forecast: the recipe of warm, steps[0] map: unknown field temp_c in 'temp_c - 20'"
    )


def test_by_continues_the_one_series_given(tmp_path) -> None:
    """A model trained on two microgrids at once (lag by mg) forecasts the series it is given."""
    store = ModelStore(tmp_path / "models")
    both = rows(*[{**row, "mg": 1} for row in mg_rows(1, "2026-07-10", "2026-07-17").items],
                *[{**row, "mg": 2} for row in mg_rows(2, "2026-07-10", "2026-07-17").items],
                interval_minutes=60)  # fmt: skip
    lag_by_mg = {**LAG, "by": "mg"}
    run(chain(lag_by_mg, INPUTS, TRAIN), {LINK: both}, models=store)
    mg2 = rows(*[{**row, "mg": 2} for row in mg_rows(2, "2026-07-18", "2026-07-18").items], interval_minutes=60)
    assert len(forecast(store, mg2)) == 4
    with pytest.raises(ChainError, match=r"forecast continues one series, and the rows hold 2 \(by \['mg'\]\)"):
        forecast(store, both)


def test_the_rows_need_the_model_time_step(store: ModelStore) -> None:
    quarter_hours = rows({"ts": t("2026-07-18T00:00"), "kw": 1.0}, {"ts": t("2026-07-18T00:15"), "kw": 1.0},
                         interval_minutes=15)  # fmt: skip
    with pytest.raises(ChainError, match="the model works on 60-minute steps, the rows have 15-minute steps"):
        forecast(store, quarter_hours)


def test_a_model_trained_without_a_time_step_cannot_step_forward(tmp_path) -> None:
    store = ModelStore(tmp_path / "models")
    no_step = rows(*[{"ts": t(f"2026-07-01T{hour:02d}:00"), "x": float(hour), "kw": 2.0 * hour} for hour in range(20)])
    run(chain({"op": "train", "model": "flat", "target": "kw", "features": ["x"]}), {LINK: no_step}, models=store)
    with pytest.raises(ChainError, match="flat was trained on rows without a known time step"):
        forecast(store, rows(*no_step.items, interval_minutes=60), model="flat")


def test_over_http_train_then_forecast_the_next_day(sandbox: TestClient, source_server: SourceServer) -> None:
    def source(start: str, end: str) -> str:
        return source_server.url(f"/microgrids/1/readings?start={start}&end={end}")

    trained = sandbox.post("/chains/run", json={"source": source("2026-07-01", "2026-07-17"),
                                                "steps": [LAG, INPUTS, TRAIN]})  # fmt: skip
    assert trained.status_code == 200, trained.text
    step = {"op": "forecast", "model": "mg1_demand", "steps": 24, "as": "load"}
    reply = sandbox.post("/chains/run", json={"source": source("2026-07-18", "2026-07-18"), "steps": [step]})
    assert reply.status_code == 200, reply.text
    assert [row["ts"] for row in reply.json()][::12] == ["2026-07-19T00:00:00", "2026-07-19T12:00:00"]
