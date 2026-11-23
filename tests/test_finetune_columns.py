"""finetune with a model made by train: learn a correction on the source's feature columns, choose the rows with
`where`, and let the rows pass through (DESIGN.md section 17, task 6). The data of UC3: MG1's model, 4 days of MG11.
"""

from typing import Any

import pytest

from src.exceptions import ChainError
from src.store import ModelStore
from tests.utils import mg_rows, run

LINK = "http://test/mg"
LAG = {"op": "lag", "field": "kw", "lags": [1, 2, 24]}
INPUTS = {"op": "map", "set": {
    "hour_sin": "sin(2 * pi * ts.hour / 24)", "hour_cos": "cos(2 * pi * ts.hour / 24)",
    "day_sin": "sin(2 * pi * ts.weekday() / 7)", "day_cos": "cos(2 * pi * ts.weekday() / 7)"}}  # fmt: skip
FEATURES = ["kw_lag_1", "kw_lag_2", "kw_lag_24", "hour_sin", "hour_cos", "day_sin", "day_cos"]
MG11_FOUR_DAYS = mg_rows(11, "2026-07-15", "2026-07-18")
PREPARE = {"op": "prepare", "model": "mg1_demand"}
SPLIT = {"op": "map", "set": {"set": "'test' if ts >= '2026-07-18' else 'train'"}}
FINETUNE = {"op": "finetune", "model": "mg1_demand", "as": "mg11_from_mg1", "where": "set == 'train'"}


@pytest.fixture
def store(tmp_path) -> ModelStore:
    """MG1's demand model, made by train on 48 days."""
    store = ModelStore(tmp_path / "models")
    train = {"op": "train", "model": "mg1_demand", "target": "kw", "features": FEATURES}
    run(chain(LAG, INPUTS, train), {LINK: mg_rows(1, end="2026-07-18")}, models=store)
    return store


def chain(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"source": LINK, "steps": list(steps)}


def mae_on_the_last_day(store: ModelStore, model: str) -> float:
    [scores] = run(chain(PREPARE, SPLIT, {"op": "predict", "model": model, "as": "p"},
                         {"op": "filter", "where": "set == 'test'"},
                         {"op": "evaluate", "actual": "kw", "predicted": "p"}),
                   {LINK: MG11_FOUR_DAYS}, models=store)  # fmt: skip
    assert scores["n"] == 24
    return scores["mae"]


def test_finetune_adapts_the_model_and_lets_the_rows_pass(store: ModelStore) -> None:
    result = run(chain(PREPARE, SPLIT, FINETUNE), {LINK: MG11_FOUR_DAYS}, models=store)
    assert len(result) == 3 * 24 and "kw_lag_24" in result[0]  # the rows go on

    card, source = store.card("mg11_from_mg1"), store.card("mg1_demand")
    assert card["kind"] == "adapted" and card["source_model"] == "mg1_demand"
    assert card["algo"] == "linear" and card["params"] == {"alpha": 1.0, "finetune_alpha": 100.0}
    assert card["features"] == FEATURES and card["recipe"] == source["recipe"] == [LAG, INPUTS]
    assert card["rows"] == 2 * 24 and card["last_ts"] == "2026-07-17T23:00:00"  # the test day is not learned


def test_on_the_day_it_did_not_learn_from_the_adapted_model_beats_the_source(store: ModelStore) -> None:
    run(chain(PREPARE, SPLIT, FINETUNE), {LINK: MG11_FOUR_DAYS}, models=store)
    assert mae_on_the_last_day(store, "mg11_from_mg1") < mae_on_the_last_day(store, "mg1_demand")


def test_adapt_predict_both_and_compare_in_one_straight_chain(store: ModelStore) -> None:
    """What needed a join before: the rows pass through finetune, so both models predict the same rows."""
    result = run(chain(
        PREPARE, SPLIT, FINETUNE,
        {"op": "predict", "model": "mg1_demand", "as": "kw_base"},
        {"op": "predict", "model": "mg11_from_mg1", "as": "kw_adapted"},
        {"op": "filter", "where": "set == 'test'"},
        {"op": "map", "set": {"better": "1 if abs(kw - kw_adapted) < abs(kw - kw_base) else 0"}},
        {"op": "reduce", "aggregate": {"hours": "count(kw)", "better_hours": "sum(better)"}},
    ), {LINK: MG11_FOUR_DAYS}, models=store)  # fmt: skip
    assert result[0]["hours"] == 24 and result[0]["better_hours"] > 12


def test_the_adapted_model_can_be_prepared_for_and_forecast(store: ModelStore) -> None:
    run(chain(PREPARE, SPLIT, FINETUNE), {LINK: MG11_FOUR_DAYS}, models=store)
    forecast = run(chain({"op": "forecast", "model": "mg11_from_mg1", "steps": 6, "as": "load"}),
                   {LINK: MG11_FOUR_DAYS}, models=store)  # fmt: skip
    assert len(forecast) == 6 and all(isinstance(row["load"], float) for row in forecast)


def test_finetune_needs_the_inputs_of_the_source(store: ModelStore) -> None:
    with pytest.raises(ChainError) as error:
        run(chain(SPLIT, FINETUNE), {LINK: MG11_FOUR_DAYS}, models=store)
    assert str(error.value) == (
        'steps[1] finetune: the rows have no field kw_lag_1, an input of mg1_demand; add {"op": "prepare", '
        '"model": "mg1_demand"} before finetune'
    )


def test_finetune_needs_enough_rows(store: ModelStore) -> None:
    with pytest.raises(ChainError, match="0 rows can be learned from"):
        run(chain(PREPARE, SPLIT, {**FINETUNE, "where": "set == 'nothing'"}), {LINK: MG11_FOUR_DAYS}, models=store)


def test_without_where_finetune_learns_from_every_row(store: ModelStore) -> None:
    every_row = {key: value for key, value in FINETUNE.items() if key != "where"}
    run(chain(PREPARE, every_row), {LINK: MG11_FOUR_DAYS}, models=store)
    assert store.card("mg11_from_mg1")["rows"] == 3 * 24


def test_rows_to_keep_for_testing_are_chosen_with_where_not_holdout(store: ModelStore) -> None:
    with pytest.raises(ChainError, match=r"steps\[1\]\.holdout: unknown key"):
        run(chain(PREPARE, {**FINETUNE, "holdout": 24}), {LINK: MG11_FOUR_DAYS}, models=store)
