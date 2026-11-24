"""The ML operators that compare and score: evaluate (also on a model's card), embed and similarity, and the model
store. Training, prediction, forecast and adaptation have their own files: test_train.py, test_prepare.py,
test_forecast_recipe.py, test_finetune_columns.py. The data are the hourly readings of tests/resources/microgrids/.
"""

import math
from datetime import timedelta
from typing import Any

import pytest

from src.exceptions import ChainError
from src.store import ModelStore
from tests.utils import mg_rows, rows, run, t

LINK = "http://test/mg"
LAG = {"op": "lag", "field": "kw", "lags": [1, 24]}
INPUTS = {"op": "map", "set": {"hour_sin": "sin(2 * pi * ts.hour / 24)", "hour_cos": "cos(2 * pi * ts.hour / 24)",
                               "set": "'test' if ts >= '2026-07-18' else 'train'"}}  # fmt: skip
FEATURES = ["kw_lag_1", "kw_lag_24", "hour_sin", "hour_cos"]
TRAIN = {"op": "train", "model": "mg1_demand", "target": "kw", "features": FEATURES, "where": "set == 'train'"}


@pytest.fixture
def store(tmp_path) -> ModelStore:
    return ModelStore(tmp_path / "models")


def chain(*steps: dict[str, Any], source: str = LINK) -> dict[str, Any]:
    return {"source": source, "steps": list(steps)}


def train_mg1(store: ModelStore) -> None:
    run(chain(LAG, INPUTS, TRAIN), {LINK: mg_rows(1, "2026-07-01", "2026-07-18")}, models=store)


def test_an_unknown_model(store: ModelStore) -> None:
    with pytest.raises(ChainError, match=r"steps\[0\] predict: no model named nope"):
        run(chain({"op": "predict", "model": "nope"}), {LINK: rows()}, models=store)
    with pytest.raises(ChainError, match="no model named mg9_demand"):
        run(chain({"op": "finetune", "model": "mg9_demand", "as": "x"}), {LINK: rows()}, models=store)


# ---------- evaluate ----------


def test_evaluate_skips_nulls_and_gives_mae_rmse_mape() -> None:
    data = rows({"a": 10, "p": 12}, {"a": 20, "p": 18}, {"a": None, "p": 5}, {"a": 0, "p": 1})
    [scores] = run(chain({"op": "evaluate", "actual": "a", "predicted": "p"}), {LINK: data})
    assert scores == pytest.approx({"n": 3, "mae": 5 / 3, "rmse": math.sqrt(3), "mape": 15.0})


def test_evaluate_with_model_writes_the_scores_on_its_card(store: ModelStore) -> None:
    """train does not test: the test rows, predicted and evaluated with "model", give the card its quality."""
    train_mg1(store)
    assert store.card("mg1_demand")["mae"] is None
    [scores] = run(chain(LAG, INPUTS, TRAIN, {"op": "predict", "model": "mg1_demand", "as": "p"},
                         {"op": "filter", "where": "set == 'test'"},
                         {"op": "evaluate", "actual": "kw", "predicted": "p", "model": "mg1_demand"}),
                   {LINK: mg_rows(1, "2026-07-01", "2026-07-18")}, models=store)  # fmt: skip
    assert scores["model"] == "mg1_demand" and scores["n"] == 24
    card = store.card("mg1_demand")
    assert (card["mae"], card["rmse"], card["mape"]) == (scores["mae"], scores["rmse"], scores["mape"])
    assert card["evaluated_rows"] == 24
    assert (card["evaluated_from"], card["evaluated_to"]) == ("2026-07-18T00:00:00", "2026-07-18T23:00:00")


def test_evaluate_with_an_unknown_model(store: ModelStore) -> None:
    with pytest.raises(ChainError, match=r"steps\[0\] evaluate: no model named nope"):
        run(chain({"op": "evaluate", "actual": "a", "predicted": "p", "model": "nope"}),
            {LINK: rows({"a": 1, "p": 1})}, models=store)  # fmt: skip


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
    assert loaded.spec.features == ("kw_lag_1", "kw_lag_24", "hour_sin", "hour_cos")
    assert loaded.card == store.card("mg1_demand")
    assert [card["name"] for card in store.cards()] == ["mg1_demand"]
    with pytest.raises(ChainError, match="a model name may hold only"):
        store.card("../../etc/passwd")
