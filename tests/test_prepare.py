"""predict on the feature columns of a model made by train, and prepare, which makes them again from the model's
recipe (DESIGN.md section 17, task 4)."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.exceptions import ChainError
from src.services.chain_parser import ChainParser
from src.store import ModelStore
from tests.conftest import SourceServer
from tests.utils import mg_rows, rows, run, t

LINK = "http://test/mg"
LAG = {"op": "lag", "field": "kw", "lags": [1, 24]}
INPUTS = {"op": "map", "set": {
    "hour_sin": "sin(2 * pi * ts.hour / 24)",
    "hour_cos": "cos(2 * pi * ts.hour / 24)",
    "weekend": "1 if ts.weekday() >= 5 else 0",
    "set": "'test' if ts >= '2026-07-18' else 'train'"}}  # fmt: skip
FEATURES = ["kw_lag_1", "kw_lag_24", "hour_sin", "hour_cos", "weekend"]
TRAIN = {"op": "train", "model": "mg1_demand", "target": "kw", "features": FEATURES, "where": "set == 'train'"}
PREDICT = {"op": "predict", "model": "mg1_demand", "as": "kw_pred"}
EVALUATE = {"op": "evaluate", "actual": "kw", "predicted": "kw_pred"}
TRAINING_DAYS = mg_rows(1, "2026-07-01", "2026-07-18")
LATER_DAYS = mg_rows(1, "2026-07-18", "2026-07-20")  # the 18th only gives the lags of the 19th


@pytest.fixture
def store(tmp_path) -> ModelStore:
    return ModelStore(tmp_path / "models")


@pytest.fixture
def trained(store: ModelStore) -> ModelStore:
    run(chain(LAG, INPUTS, TRAIN), {LINK: TRAINING_DAYS}, models=store)
    return store


def chain(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"source": LINK, "steps": list(steps)}


def test_the_whole_training_workflow_step_by_step(store: ModelStore) -> None:
    """Inputs, split, train, predict and test, in one chain: the workflow of DESIGN.md section 17."""
    [scores] = run(chain(LAG, INPUTS, TRAIN, PREDICT, {"op": "filter", "where": "set == 'test'"}, EVALUATE),
                   {LINK: TRAINING_DAYS}, models=store)  # fmt: skip
    assert scores["n"] == 24  # the test day
    assert 0 < scores["mae"] < 40 and scores["mape"] < 10  # kW, on a load of about 500 kW


def test_prepare_makes_the_inputs_as_the_steps_written_by_hand(trained: ModelStore) -> None:
    by_hand = run(chain(LAG, INPUTS), {LINK: LATER_DAYS}, models=trained)
    prepared = run(chain({"op": "prepare", "model": "mg1_demand"}), {LINK: LATER_DAYS}, models=trained)
    assert prepared == by_hand and len(prepared) == 48


def test_prepare_then_predict_on_days_the_model_has_not_seen(trained: ModelStore) -> None:
    [scores] = run(chain({"op": "prepare", "model": "mg1_demand"}, PREDICT, EVALUATE), {LINK: LATER_DAYS},
                   models=trained)  # fmt: skip
    assert scores["n"] == 48 and scores["mae"] < 40


def test_predict_without_the_inputs_says_how_to_make_them(trained: ModelStore) -> None:
    with pytest.raises(ChainError) as error:
        run(chain(PREDICT), {LINK: LATER_DAYS}, models=trained)
    assert str(error.value) == (
        'steps[0] predict: the rows have no field kw_lag_1, an input of mg1_demand; add {"op": "prepare", '
        '"model": "mg1_demand"} before predict, or the steps that made it for training'
    )


def test_a_row_without_every_input_gets_no_prediction(trained: ModelStore) -> None:
    """evaluate then skips it, as it skips any null."""
    holes = {"op": "map", "set": {"weekend": "None if ts.hour == 0 else weekend"}}
    result = run(chain({"op": "prepare", "model": "mg1_demand"}, holes, PREDICT), {LINK: LATER_DAYS},
                 models=trained)  # fmt: skip
    assert [row["kw_pred"] is None for row in result] == [row["ts"].hour == 0 for row in result]
    [scores] = run(chain({"op": "prepare", "model": "mg1_demand"}, holes, PREDICT, EVALUATE), {LINK: LATER_DAYS},
                   models=trained)  # fmt: skip
    assert scores["n"] == 48 - 2


def test_a_model_works_only_on_rows_with_its_time_step(trained: ModelStore) -> None:
    inputs = dict.fromkeys(FEATURES, 1.0)
    quarter_hours = rows({"ts": t("2026-07-19T00:00"), **inputs}, {"ts": t("2026-07-19T00:15"), **inputs},
                         interval_minutes=15)  # fmt: skip
    with pytest.raises(ChainError, match="the model works on 60-minute steps, the rows have 15-minute steps"):
        run(chain(PREDICT), {LINK: quarter_hours}, models=trained)


def test_a_model_trained_on_columns_alone_has_an_empty_recipe(store: ModelStore) -> None:
    line = [{"ts": t(f"2026-07-01T{hour:02d}:00"), "x": float(hour), "y": 2.0 * hour} for hour in range(20)]
    source = rows(*line, interval_minutes=60)
    run(chain({"op": "train", "model": "line", "target": "y", "features": ["x"]}), {LINK: source}, models=store)
    assert run(chain({"op": "prepare", "model": "line"}), {LINK: source}, models=store) == line


def test_prepare_names_the_recipe_step_that_fails(store: ModelStore) -> None:
    uses_weather = {"op": "map", "set": {"warm": "temp_c > 25"}}
    weather = rows(*[{"ts": t(f"2026-07-01T{hour:02d}:00"), "kw": float(hour), "temp_c": 20.0 + hour}
                     for hour in range(20)], interval_minutes=60)  # fmt: skip
    train = {"op": "train", "model": "warm", "target": "kw", "features": ["warm"]}
    run(chain(uses_weather, train), {LINK: weather}, models=store)
    with pytest.raises(ChainError) as error:
        run(chain({"op": "prepare", "model": "warm"}), {LINK: LATER_DAYS}, models=store)
    assert (
        str(error.value) == "steps[0] prepare: the recipe of warm, steps[0] map: unknown field temp_c in 'temp_c > 25'"
    )


def test_a_model_saved_by_fit_in_an_older_version_is_refused_with_a_reason(tmp_path) -> None:
    """Its card has no recipe; its file is never unpickled (the classes it needs are gone)."""
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "old.json").write_text('{"name": "old", "kind": "trained", "algo": "linear", "lags": [1, 24]}')
    (folder / "old.joblib").write_bytes(b"not read")
    store = ModelStore(folder)
    with pytest.raises(ChainError, match="old was made by fit, which this version no longer has; train it again"):
        run(chain({"op": "prepare", "model": "old"}), {LINK: LATER_DAYS}, models=store)
    assert store.card("old") is not None  # still listed by GET /models, as it is


@pytest.mark.parametrize(
    ("recipe", "message"),
    [
        ([{"op": "filter", "where": "kw > 0"}], "a recipe may only hold lag and map steps, not 'filter'"),
        ([{"op": "load", "from": {"source": "http://x/"}}], "not 'load'"),
        (["lag"], "not None"),
        ({"op": "map"}, "a recipe must be a list of steps"),
        ([{"op": "map", "set": {"x": "__import__('os')"}}], "is not allowed"),
    ],
)
def test_a_recipe_to_run_holds_only_lag_and_map_steps(recipe: Any, message: str) -> None:
    """Recipes will also come from other sandboxes (load, task 7): they are checked like chains, and more."""
    with pytest.raises(ChainError, match=message):
        ChainParser().recipe_steps(recipe)


def test_over_http_train_in_one_request_predict_in_another(sandbox: TestClient, source_server: SourceServer) -> None:
    def source(start: str, end: str) -> str:
        return source_server.url(f"/microgrids/1/readings?start={start}&end={end}")

    trained = sandbox.post("/chains/run", json={"source": source("2026-07-01", "2026-07-18"),
                                                "steps": [LAG, INPUTS, TRAIN]})  # fmt: skip
    assert trained.status_code == 200, trained.text
    use = [{"op": "prepare", "model": "mg1_demand"}, PREDICT, EVALUATE]
    reply = sandbox.post("/chains/run", json={"source": source("2026-07-18", "2026-07-20"), "steps": use})
    assert reply.status_code == 200, reply.text
    [scores] = reply.json()
    assert scores["n"] == 48 and scores["mae"] < 40
