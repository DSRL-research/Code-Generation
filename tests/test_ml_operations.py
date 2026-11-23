"""The ML operators, in chains, on the hourly readings of real microgrids (tests/resources/microgrids/).

Day 0 is 2026-06-01. The models are trained on days 0-47 (to 2026-07-18) and tested on the days after.
"""

import math
from datetime import timedelta
from typing import Any

import pytest

from src.exceptions import ChainError
from src.models.rows import Rows
from src.services.chain_parser import ChainParser
from src.store import ModelStore
from tests.utils import mg_rows, rows, run, t

LINK = "http://test/mg"
TRAIN_MG1 = mg_rows(1, end="2026-07-18")  # 48 days
FIT = {"op": "fit", "model": "mg1_demand", "target": "kw", "lags": [1, 2, 24], "calendar": ["hour", "weekday"],
       "holdout": 24}  # fmt: skip


@pytest.fixture
def store(tmp_path) -> ModelStore:
    return ModelStore(tmp_path / "models")


def chain(*steps: dict[str, Any], source: str = LINK) -> dict[str, Any]:
    return {"source": source, "steps": list(steps)}


def train_mg1(store: ModelStore) -> dict[str, Any]:
    [card] = run(chain(FIT), {LINK: TRAIN_MG1}, models=store)
    return card


# ---------- fit ----------


def test_fit_trains_saves_the_model_and_returns_its_card(store: ModelStore) -> None:
    card = train_mg1(store)
    assert card["name"] == "mg1_demand" and card["kind"] == "trained" and card["algo"] == "linear"
    assert card["rows"] == 48 * 24 - 24 - 24  # rows that have lag_24, without the 24 holdout rows
    assert card["holdout_rows"] == 24
    assert 0 < card["mae"] < 30  # kW, on a load of about 500 kW
    saved = store.card("mg1_demand")
    assert saved["features"] == ["lag_1", "lag_2", "lag_24", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
    assert saved["interval_minutes"] == 60 and saved["first_ts"] == "2026-06-02T00:00:00"
    assert store.names() == ["mg1_demand"]


def test_fit_with_gradient_boosting(store: ModelStore) -> None:
    step = {**FIT, "algo": "gbr", "params": {"n_estimators": 50, "max_depth": 2}}
    [card] = run(chain(step), {LINK: TRAIN_MG1}, models=store)
    assert card["algo"] == "gbr"
    assert store.card("mg1_demand")["params"] == {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.1}


def test_fit_with_a_neural_network_then_use_and_adapt_it(store: ModelStore) -> None:
    step = {**FIT, "algo": "mlp", "params": {"hidden": 16, "layers": 1}}
    [card] = run(chain(step), {LINK: TRAIN_MG1}, models=store)
    assert card["algo"] == "mlp" and 0 < card["mae"] < 40
    assert store.card("mg1_demand")["params"] == {
        "hidden": 16, "layers": 1, "alpha": 0.0001, "learning_rate": 0.001, "max_iter": 500}  # fmt: skip

    forecast = run(chain({"op": "forecast", "model": "mg1_demand", "steps": 4, "as": "load"}),
                   {LINK: mg_rows(1, "2026-07-15", "2026-07-18")}, models=store)  # fmt: skip
    assert all(200 < row["load"] < 900 for row in forecast)  # in kW: the target scaling is undone

    finetune = {"op": "finetune", "model": "mg1_demand", "as": "mg11_from_mg1", "holdout": 24}
    [adapted] = run(chain(finetune), {LINK: mg_rows(11, "2026-07-15", "2026-07-18")}, models=store)
    assert adapted["kind"] == "adapted" and adapted["algo"] == "mlp" and adapted["mae"] is not None


def test_fit_needs_the_time_step_of_the_rows(store: ModelStore) -> None:
    with pytest.raises(ChainError, match="the time step of the rows is unknown"):
        run(chain(FIT), {LINK: Rows(TRAIN_MG1.items, None)}, models=store)


def test_fit_needs_enough_rows(store: ModelStore) -> None:
    with pytest.raises(ChainError, match="training needs at least 10"):
        run(chain(FIT), {LINK: mg_rows(1, "2026-06-01", "2026-06-02")}, models=store)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"algo": "svm"}, "unknown algo 'svm'"),
        ({"params": {"depth": 3}}, r"linear has no parameters \['depth'\]"),
        ({"algo": "gbr", "params": {"n_estimators": 5000}}, "n_estimators must be a number from 1 to 1000"),
        ({"algo": "mlp", "params": {"layers": 4}}, "mlp: layers must be a number from 1 to 3"),
        ({"model": "../other"}, "a model name may hold only"),
        ({"lags": [0, 24]}, "lags must be different whole numbers from 1 to 168"),
        ({"calendar": ["minute"]}, r"steps\[0\]\.calendar"),
    ],
)
def test_fit_is_checked_when_the_chain_is_parsed(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ChainError, match=message):
        ChainParser().parse(chain({**FIT, **change}))


# ---------- predict and evaluate ----------


def test_predict_then_evaluate_scores_a_model_on_days_it_has_not_seen(store: ModelStore) -> None:
    train_mg1(store)
    days_48_49 = mg_rows(1, "2026-07-18", "2026-07-20")  # day 47 only gives the lags of the first hours
    [scores] = run(chain(
        {"op": "predict", "model": "mg1_demand", "as": "kw_pred"},
        {"op": "filter", "where": "ts >= '2026-07-19'"},
        {"op": "evaluate", "actual": "kw", "predicted": "kw_pred"},
    ), {LINK: days_48_49}, models=store)  # fmt: skip
    assert scores["n"] == 48
    assert scores["mae"] < 40 and scores["mape"] < 10


def test_predict_drops_the_rows_without_every_lag(store: ModelStore) -> None:
    train_mg1(store)
    result = run(chain({"op": "predict", "model": "mg1_demand"}), {LINK: mg_rows(1, "2026-07-18", "2026-07-20")},
                 models=store)  # fmt: skip
    assert len(result) == 72 - 24
    assert result[0]["ts"] == t("2026-07-19T00:00") and isinstance(result[0]["prediction"], float)


def test_a_model_needs_rows_with_its_time_step(store: ModelStore) -> None:
    train_mg1(store)
    quarter_hours = Rows(TRAIN_MG1.items, timedelta(minutes=15))
    with pytest.raises(ChainError, match="the model works on 60-minute steps, the rows have 15-minute steps"):
        run(chain({"op": "predict", "model": "mg1_demand"}), {LINK: quarter_hours}, models=store)


def test_an_unknown_model(store: ModelStore) -> None:
    with pytest.raises(ChainError, match=r"steps\[0\] predict: no model named nope"):
        run(chain({"op": "predict", "model": "nope"}), {LINK: TRAIN_MG1}, models=store)


# ---------- forecast ----------


def test_forecast_gives_the_next_hours_after_the_rows(store: ModelStore) -> None:
    train_mg1(store)
    result = run(chain({"op": "forecast", "model": "mg1_demand", "steps": 4, "as": "load"}),
                 {LINK: mg_rows(1, "2026-07-15", "2026-07-18")}, models=store)  # fmt: skip
    assert [row["ts"] for row in result] == [t(f"2026-07-19T0{hour}:00") for hour in range(4)]
    assert all(200 < row["load"] < 900 for row in result)


def test_forecast_needs_the_lags_of_the_first_step(store: ModelStore) -> None:
    train_mg1(store)
    half_a_day = Rows(mg_rows(1, "2026-07-18", "2026-07-18").items[12:], timedelta(hours=1))
    with pytest.raises(ChainError, match="not enough history before 2026-07-19T00:00:00"):
        run(chain({"op": "forecast", "model": "mg1_demand", "steps": 4}), {LINK: half_a_day}, models=store)


# ---------- finetune ----------


def test_finetune_adapts_a_model_of_another_microgrid(store: ModelStore) -> None:
    train_mg1(store)
    mg11_four_days = mg_rows(11, "2026-07-15", "2026-07-18")
    [card] = run(chain({"op": "finetune", "model": "mg1_demand", "as": "mg11_from_mg1", "holdout": 24}),
                 {LINK: mg11_four_days}, models=store)  # fmt: skip
    assert card["kind"] == "adapted" and card["source_model"] == "mg1_demand"
    assert card["rows"] == 48 and card["holdout_rows"] == 24
    on_the_holdout_day = chain(
        {"op": "predict", "model": "mg1_demand", "as": "p"},
        {"op": "filter", "where": "ts >= '2026-07-18'"},
        {"op": "evaluate", "actual": "kw", "predicted": "p"},
    )
    [as_is] = run(on_the_holdout_day, {LINK: mg11_four_days}, models=store)
    assert card["mae"] < as_is["mae"]  # on the same holdout day, the adapted model beats MG1's model as it is
    assert store.card("mg11_from_mg1")["features"] == store.card("mg1_demand")["features"]


def test_finetune_needs_the_source_model(store: ModelStore) -> None:
    with pytest.raises(ChainError, match="no model named mg9_demand"):
        run(chain({"op": "finetune", "model": "mg9_demand", "as": "x"}), {LINK: TRAIN_MG1}, models=store)


# ---------- evaluate ----------


def test_evaluate_skips_nulls_and_gives_mae_rmse_mape() -> None:
    data = rows({"a": 10, "p": 12}, {"a": 20, "p": 18}, {"a": None, "p": 5}, {"a": 0, "p": 1})
    [scores] = run(chain({"op": "evaluate", "actual": "a", "predicted": "p"}), {LINK: data})
    assert scores == pytest.approx({"n": 3, "mae": 5 / 3, "rmse": math.sqrt(3), "mape": 15.0})


# ---------- embed and similarity ----------


def hour_rows(days: int = 2, **extra: Any) -> list[dict[str, Any]]:
    return [{"ts": t("2026-07-01") + timedelta(hours=h), "kw": float(h % 24), **extra} for h in range(24 * days)]


def test_embed_gives_the_hour_of_day_profile() -> None:
    [row] = run(chain({"op": "embed", "field": "kw"}), {LINK: rows(*hour_rows())})
    assert row == {"embedding": [float(hour) for hour in range(24)], "days": 2}


def test_a_normalized_embedding_keeps_only_the_shape() -> None:
    [row] = run(chain({"op": "embed", "field": "kw", "normalize": True, "as": "shape"}), {LINK: rows(*hour_rows())})
    assert sum(row["shape"]) == pytest.approx(0) and math.dist(row["shape"], [0] * 24) == pytest.approx(1)


def test_embed_by_a_field_gives_one_row_per_group() -> None:
    data = rows(*hour_rows(mg=1), *hour_rows(mg=2))
    assert [row["mg"] for row in run(chain({"op": "embed", "field": "kw", "by": "mg"}), {LINK: data})] == [1, 2]


def test_embed_needs_every_hour_of_the_day() -> None:
    with pytest.raises(ChainError, match="none at"):
        run(chain({"op": "embed", "field": "kw"}), {LINK: rows(*hour_rows()[:12])})


def embedding_of(link: str, mg: int | None = None, start: str = "2026-06-01", end: str = "2026-07-18") -> dict:
    steps: list[dict[str, Any]] = [{"op": "embed", "field": "kw"}]
    if mg is not None:
        steps.append({"op": "map", "set": {"mg": str(mg)}})
    return {"source": f"{link}?start={start}&end={end}", "steps": steps}


def test_similarity_ranks_microgrids_by_their_profile() -> None:
    """Target MG11 (Athens, 4 days); candidates MG1 (Cluj) and MG12 (Athens). The closest is MG12."""
    sources = {
        "http://test/1?start=2026-06-01&end=2026-07-18": mg_rows(1, end="2026-07-18"),
        "http://test/12?start=2026-06-01&end=2026-07-18": mg_rows(12, end="2026-07-18"),
        "http://test/11?start=2026-07-15&end=2026-07-18": mg_rows(11, "2026-07-15", "2026-07-18"),
    }
    candidates = embedding_of("http://test/1", mg=1)
    candidates["steps"] += [
        {"op": "union", "with": embedding_of("http://test/12", mg=12)},
        {"op": "similarity", "with": embedding_of("http://test/11", start="2026-07-15")},
        {"op": "top", "n": 2, "by": "distance", "order": "asc"},
        {"op": "collect", "fields": ["mg", "distance"], "decimals": 1},
    ]
    result = run(candidates, sources)
    assert [row["mg"] for row in result] == [12, 1]
    assert result[0]["distance"] < result[1]["distance"] / 2


def test_cosine_similarity_sees_only_the_shape() -> None:
    """The 20 microgrids of the data set have almost the same daily shape: cosine cannot tell them apart."""
    sources = {"http://test/1?start=2026-06-01&end=2026-07-18": mg_rows(1, end="2026-07-18"),
               "http://test/11?start=2026-07-15&end=2026-07-18": mg_rows(11, "2026-07-15", "2026-07-18")}  # fmt: skip
    compare = embedding_of("http://test/1")
    compare["steps"].append(
        {"op": "similarity", "metric": "cosine", "with": embedding_of("http://test/11", start="2026-07-15")}
    )
    [row] = run(compare, sources)
    assert row["cosine"] > 0.99


def test_similarity_needs_one_reference_row() -> None:
    data = rows(*hour_rows(mg=1), *hour_rows(mg=2))
    step = {"op": "similarity", "with": {"source": LINK, "steps": [{"op": "embed", "field": "kw", "by": "mg"}]}}
    with pytest.raises(ChainError, match='the chain under "with" must give one row'):
        run(chain({"op": "embed", "field": "kw"}, step), {LINK: data})


def test_distance_and_cosine_in_expressions() -> None:
    data = rows({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    [row] = run(chain({"op": "map", "set": {"d": "distance(a, b)", "c": "cosine(a, b)"}}), {LINK: data})
    assert row["d"] == pytest.approx(math.sqrt(2)) and row["c"] == pytest.approx(0.0)


# ---------- the model store ----------


def test_the_store_keeps_a_card_per_model(store: ModelStore) -> None:
    train_mg1(store)
    loaded = store.load("mg1_demand")
    assert loaded.spec.lags == (1, 2, 24) and loaded.card == store.card("mg1_demand")
    assert [card["name"] for card in store.cards()] == ["mg1_demand"]
    with pytest.raises(ChainError, match="a model name may hold only"):
        store.card("../../etc/passwd")
