"""The train operator: a model trained on columns made by the steps before (DESIGN.md section 17)."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.exceptions import ChainError
from src.ml.sharing import export_record
from src.services.chain_parser import ChainParser
from src.store import ModelStore
from tests.conftest import SourceServer
from tests.utils import mg_rows, rows, run, t

LINK = "http://test/mg"
LAG = {"op": "lag", "field": "kw", "lags": [1, 24]}
INPUTS = {"op": "map", "set": {
    "hour_sin": "sin(2 * pi * ts.hour / 24)",
    "hour_cos": "cos(2 * pi * ts.hour / 24)",
    "set": "'test' if ts >= '2026-07-18' else 'train'"}}  # fmt: skip
FEATURES = ["kw_lag_1", "kw_lag_24", "hour_sin", "hour_cos"]
TRAIN = {"op": "train", "model": "mg1_demand", "target": "kw", "features": FEATURES, "where": "set == 'train'"}
DAYS = mg_rows(1, "2026-07-01", "2026-07-18")  # 18 days of MG1


@pytest.fixture
def store(tmp_path) -> ModelStore:
    return ModelStore(tmp_path / "models")


def chain(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"source": LINK, "steps": list(steps)}


def test_train_saves_the_model_and_passes_the_rows_through(store: ModelStore) -> None:
    result = run(chain(LAG, INPUTS, TRAIN), {LINK: DAYS}, models=store)
    assert len(result) == 17 * 24  # lag dropped the first day; train kept every row
    assert {"kw_lag_24", "hour_sin", "set"} <= set(result[0])

    card = store.card("mg1_demand")
    assert card["kind"] == "trained" and card["algo"] == "linear" and card["params"] == {"alpha": 1.0}
    assert card["target"] == "kw" and card["features"] == FEATURES and card["interval_minutes"] == 60
    assert card["rows"] == 16 * 24 and card["skipped_rows"] == 0  # days 2-17; day 18 is the test set
    assert (card["first_ts"], card["last_ts"]) == ("2026-07-02T00:00:00", "2026-07-17T23:00:00")
    assert card["mae"] is None  # train does not test: predict and evaluate do
    assert card["recipe"] == [LAG, INPUTS]  # the steps that made its inputs, as written


def test_the_model_learns_the_target_from_the_named_columns(store: ModelStore) -> None:
    line = [{"ts": t(f"2026-07-01T{hour:02d}:00"), "x": float(hour), "other": 5.0, "y": 2.0 * hour + 1}
            for hour in range(20)]  # fmt: skip
    train = {"op": "train", "model": "line", "target": "y", "features": ["x"], "params": {"alpha": 0}}
    run(chain(train), {LINK: rows(*line, interval_minutes=60)}, models=store)
    assert store.load("line").predict([[30.0]]) == pytest.approx([61.0])


def test_rows_without_a_number_for_every_column_are_skipped(store: ModelStore) -> None:
    line = [{"ts": t(f"2026-07-01T{hour:02d}:00"), "x": float(hour), "y": 2.0 * hour} for hour in range(20)]
    with_gaps = {"op": "map", "set": {"x2": "None if ts.hour < 3 else ('text' if ts.hour == 3 else x)"}}
    train = {"op": "train", "model": "line", "target": "y", "features": ["x2"]}
    run(chain(with_gaps, train), {LINK: rows(*line, interval_minutes=60)}, models=store)
    card = store.card("line")
    assert card["rows"] == 16 and card["skipped_rows"] == 4
    assert card["first_ts"] == "2026-07-01T04:00:00"


@pytest.mark.parametrize(("algo", "params"), [("gbr", {"n_estimators": 20}), ("mlp", {"hidden": 16})])
def test_train_with_each_algorithm(store: ModelStore, algo: str, params: dict[str, Any]) -> None:
    run(chain(LAG, INPUTS, {**TRAIN, "algo": algo, "params": params}), {LINK: DAYS}, models=store)
    card = store.card("mg1_demand")
    assert card["algo"] == algo and params.items() <= card["params"].items()


def test_the_recipe_is_the_lag_and_map_steps_just_before_train() -> None:
    steps = [
        {"op": "map", "set": {"kw": "kw * 1000"}},  # before the reduce: how the rows were obtained
        {"op": "reduce", "window": "1h", "aggregate": {"kw": "mean"}},
        {"op": "lag", "field": "kw", "lags": [1]},
        {"op": "filter", "where": "ts.hour >= 6"},  # chooses rows, makes no inputs: passed over
        {"op": "map", "set": {"h": "ts.hour"}},
        {"op": "train", "model": "m", "target": "kw", "features": ["kw_lag_1", "h"]},
    ]
    parsed = ChainParser().parse(chain(*steps))
    assert parsed.steps[-1].operation.recipe == (steps[2], steps[4])


def test_train_names_a_column_that_no_row_has(store: ModelStore) -> None:
    with pytest.raises(ChainError) as error:
        run(chain(LAG, INPUTS, {**TRAIN, "features": ["kw_lag_2"]}), {LINK: DAYS}, models=store)
    assert str(error.value) == "steps[2] train: the rows have no field kw_lag_2"


def test_train_needs_enough_rows(store: ModelStore) -> None:
    with pytest.raises(ChainError, match="0 rows can be trained on .* training needs at least 10"):
        run(chain(LAG, INPUTS, {**TRAIN, "where": "set == 'nothing'"}), {LINK: DAYS}, models=store)


def test_where_must_be_true_or_false(store: ModelStore) -> None:
    with pytest.raises(ChainError, match="where must give True or False, not 'train'"):
        run(chain(LAG, INPUTS, {**TRAIN, "where": "set"}), {LINK: DAYS}, models=store)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"features": ["kw", "kw_lag_1"]}, "the target kw cannot also be a feature"),
        ({"features": ["kw_lag_1", "kw_lag_1"]}, "features must be different field names"),
        ({"features": []}, r"steps\[0\]\.features"),
        ({"algo": "svm"}, "unknown algo 'svm'"),
        ({"algo": "mlp", "params": {"layers": 4}}, "mlp: layers must be a number from 1 to 3"),
        ({"model": "../x"}, "a model name may hold only"),
        ({"where": "set = 'train'"}, "where: cannot read the expression"),
    ],
)
def test_train_is_checked_when_the_chain_is_parsed(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ChainError, match=message):
        ChainParser().parse(chain({**TRAIN, **change}))


def test_the_export_of_a_trained_model_carries_its_features_and_recipe(store: ModelStore) -> None:
    run(chain(LAG, INPUTS, TRAIN), {LINK: DAYS}, models=store)
    record = export_record(store.load("mg1_demand"))
    assert record["format"] == "linear" and record["features"] == FEATURES and record["recipe"] == [LAG, INPUTS]
    assert len(record["coef"]) == len(FEATURES) and "lags" not in record


def test_over_http_the_card_shows_the_recipe(sandbox: TestClient, source_server: SourceServer) -> None:
    source = source_server.url("/microgrids/1/readings?start=2026-07-01&end=2026-07-18")
    reply = sandbox.post("/chains/run", json={"source": source, "steps": [LAG, INPUTS, TRAIN]})
    assert reply.status_code == 200, reply.text
    assert len(reply.json()) == 17 * 24
    card = sandbox.get("/models/mg1_demand").json()
    assert card["recipe"] == [LAG, INPUTS]
    assert card["export"] == "http://testserver/models/mg1_demand/export"  # a linear model: it can be shared
