"""Sharing models between sandboxes: GET /models/{name}/export and the load operator (DESIGN.md sections 11, 17).

Each node has its own sandbox and its own model store. Node 12 trains a model; node 11 loads it by its link, makes
its inputs with the model's recipe, uses it, adapts it, and can share it on. The data are the hourly readings of
tests/resources/microgrids/.
"""

import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.exceptions import ChainError
from src.main import create_app
from src.ml.sharing import export_record
from src.store import ModelStore
from tests.conftest import SourceServer
from tests.nodes import Nodes, node_settings
from tests.utils import mg_rows, rows, run, source_rows, t

LINK = "http://test/mg"
MODEL_LINK = "http://node12/models/mg12_demand/export"
LAG = {"op": "lag", "field": "kw", "lags": [1, 24]}
INPUTS = {"op": "map", "set": {
    "hour_sin": "sin(2 * pi * ts.hour / 24)", "hour_cos": "cos(2 * pi * ts.hour / 24)",
    "weekend": "1 if ts.weekday() >= 5 else 0"}}  # fmt: skip
FEATURES = ["kw_lag_1", "kw_lag_24", "hour_sin", "hour_cos", "weekend"]
TRAIN = {"op": "train", "model": "mg12_demand", "target": "kw", "features": FEATURES}
PREPARE = {"op": "prepare", "model": "mg12_demand"}
LOAD = {"op": "load", "from": {"source": MODEL_LINK}}
SPLIT = {"op": "map", "set": {"set": "'test' if ts >= '2026-07-18' else 'train'"}}
MG11_LATER = mg_rows(11, "2026-07-18", "2026-07-21")  # the 18th only gives the lags
NOW = "2026-07-18"  # node 12's model is trained up to the end of this day


def chain(*steps: dict[str, Any], source: str = LINK) -> dict[str, Any]:
    return {"source": source, "steps": list(steps)}


def predictions(store: ModelStore, model: str) -> list[float]:
    """The predictions of a model on MG11's later days, its inputs made by its recipe."""
    replied = run(chain({"op": "prepare", "model": model}, {"op": "predict", "model": model, "as": "p"}),
                  {LINK: MG11_LATER}, models=store)  # fmt: skip
    return [row["p"] for row in replied]


def assert_close(first: list[float], second: list[float]) -> None:
    assert len(first) == len(second) > 0
    assert all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip(first, second, strict=True))


# ---------- in memory: one store per node ----------


@pytest.fixture
def store12(tmp_path: Path) -> ModelStore:
    store = ModelStore(tmp_path / "node12")
    run(chain(LAG, INPUTS, TRAIN), {LINK: mg_rows(12, end="2026-07-18")}, models=store)
    return store


@pytest.fixture
def store11(tmp_path: Path) -> ModelStore:
    return ModelStore(tmp_path / "node11")


def load(store: ModelStore, record: dict[str, Any], step: dict[str, Any] = LOAD) -> list[dict[str, Any]]:
    """Runs a chain whose only step is `step` (a load), with `record` as the reply of MODEL_LINK."""
    return run(chain(step), {LINK: MG11_LATER, MODEL_LINK: source_rows([record])}, models=store)


def test_the_export_is_the_card_with_the_numbers(store12: ModelStore) -> None:
    record = export_record(store12.load("mg12_demand"))
    assert record["format"] == "linear" and record["features"] == FEATURES and record["recipe"] == [LAG, INPUTS]
    assert len(record["coef"]) == len(FEATURES) and isinstance(record["intercept"], float)


def test_a_loaded_model_prepares_and_predicts_like_the_one_it_was_exported_from(
    store12: ModelStore, store11: ModelStore
) -> None:
    assert load(store11, export_record(store12.load("mg12_demand"))) == MG11_LATER.items  # the rows pass through
    assert_close(predictions(store11, "mg12_demand"), predictions(store12, "mg12_demand"))
    card, origin = store11.card("mg12_demand"), store12.card("mg12_demand")
    assert card["kind"] == "loaded" and card["source_model"] == "mg12_demand" and card["algo"] == "linear"
    for field in ("recipe", "features", "target", "interval_minutes", "rows", "skipped_rows", "first_ts", "last_ts"):
        assert card[field] == origin[field], field  # what it predicts from, and how it was trained at its origin


def test_a_loaded_model_forecasts_like_the_one_it_was_exported_from(store12: ModelStore, store11: ModelStore) -> None:
    load(store11, export_record(store12.load("mg12_demand")))
    step = chain({"op": "forecast", "model": "mg12_demand", "steps": 6, "as": "load"})
    on_11, on_12 = (run(step, {LINK: MG11_LATER}, models=store) for store in (store11, store12))
    assert_close([row["load"] for row in on_11], [row["load"] for row in on_12])


def test_as_names_the_loaded_model(store12: ModelStore, store11: ModelStore) -> None:
    load(store11, export_record(store12.load("mg12_demand")), {**LOAD, "as": "from_mg12"})
    assert store11.names() == ["from_mg12"]
    assert store11.card("from_mg12")["source_model"] == "mg12_demand"


def test_an_adapted_model_is_shared_as_one_linear_model_with_its_source_recipe(
    store12: ModelStore, store11: ModelStore
) -> None:
    finetune = {"op": "finetune", "model": "mg12_demand", "as": "mg11_from_mg12", "where": "set == 'train'"}
    run(chain(PREPARE, SPLIT, finetune), {LINK: mg_rows(11, "2026-07-14", "2026-07-18")}, models=store12)
    record = export_record(store12.load("mg11_from_mg12"))  # source + correction, added up
    assert record["kind"] == "adapted" and record["recipe"] == [LAG, INPUTS]
    load(store11, record)
    assert_close(predictions(store11, "mg11_from_mg12"), predictions(store12, "mg11_from_mg12"))


def test_the_record_is_the_model(store11: ModelStore) -> None:
    """Format "linear": intercept + sum(coef * features); here one feature, the value one hour before."""
    record = {"format": "linear", "name": "double", "target": "kw", "features": ["kw_lag_1"],
              "recipe": [{"op": "lag", "field": "kw", "lags": [1]}], "interval_minutes": 60,
              "coef": [2.0], "intercept": 1.0}  # fmt: skip
    load(store11, record)
    hourly = rows({"ts": t("2026-07-01T00:00"), "kw": 5.0}, {"ts": t("2026-07-01T01:00"), "kw": 7.0},
                  interval_minutes=60)  # fmt: skip
    [row] = run(chain({"op": "prepare", "model": "double"}, {"op": "predict", "model": "double", "as": "p"}),
                {LINK: hourly}, models=store11)  # fmt: skip
    assert row["p"] == 2.0 * 5.0 + 1.0


@pytest.mark.parametrize(("algo", "params"), [("gbr", {"n_estimators": 5}), ("mlp", {})])
def test_gbr_and_mlp_models_cannot_be_exported_yet(store12: ModelStore, algo: str, params: dict[str, int]) -> None:
    train = {**TRAIN, "model": f"mg12_{algo}", "algo": algo, "params": params}
    run(chain(LAG, INPUTS, train), {LINK: mg_rows(12, "2026-07-10", "2026-07-18")}, models=store12)
    message = f"mg12_{algo} is a {algo} model; only linear models can be exported for now"
    with pytest.raises(ChainError, match=message):
        export_record(store12.load(f"mg12_{algo}"))


def _valid_record() -> dict[str, Any]:
    return {"format": "linear", "name": "mg12_demand", "target": "kw", "interval_minutes": 60.0,
            "features": FEATURES, "recipe": [LAG, INPUTS], "coef": [0.8, 0.1, 5.0, -3.0, -20.0],
            "intercept": 12.0}  # fmt: skip


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(format="gbr"), "format: Input should be 'linear'"),
        (lambda r: r.pop("intercept"), "intercept: Field required"),
        (lambda r: r.pop("recipe"), "recipe: Field required"),
        (lambda r: r.pop("features"), "features: Field required"),
        (lambda r: r.update(coef=[1.0]), "coef must hold one number per feature: 5, not 1"),
        (lambda r: r["coef"].__setitem__(0, math.nan), "coef.0: must be a finite number"),
        (lambda r: r.update(features=["kw", *FEATURES[1:]]), "features must be different field names, without"),
        (lambda r: r.update(interval_minutes=0), "interval_minutes must be more than 0"),
        (lambda r: r.update(name="../models"), "a model name may hold only letters"),
        (lambda r: r["recipe"].append({"op": "filter", "where": "kw > 0"}),
         "the recipe of mg12_demand, recipe steps[2]: a recipe may only hold lag and map steps, not 'filter'"),
        (lambda r: r.update(recipe=[{"op": "map", "set": {"x": "().__class__"}}]), ".__class__ is not allowed"),
        (lambda r: r.update(recipe=[{"op": "lag", "field": "kw", "lags": [0]}]), "lags must be different whole"),
    ],
)  # fmt: skip
def test_a_record_that_is_not_a_model_is_refused(
    store11: ModelStore, change: Callable[[dict[str, Any]], Any], message: str
) -> None:
    record = _valid_record()
    change(record)
    with pytest.raises(ChainError) as error:
        load(store11, record)
    assert error.value.where == "steps[0] load"
    assert message in error.value.message
    assert store11.names() == []  # nothing was saved


def test_the_chain_under_from_must_give_one_record(store11: ModelStore) -> None:
    two = source_rows([_valid_record(), _valid_record()])
    with pytest.raises(ChainError, match='the chain under "from" must give one record, a model, not 2'):
        run(chain(LOAD), {LINK: MG11_LATER, MODEL_LINK: two}, models=store11)


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"op": "load", "from": {"source": "file:///etc/passwd"}}, "steps[0].from.source: source must be a link"),
        ({"op": "load", "from": {"source": MODEL_LINK}, "as": "../x"}, "steps[0] load: a model name may hold only"),
        ({"op": "load"}, "steps[0].from: missing"),
    ],
)
def test_load_is_checked_when_the_chain_is_parsed(store11: ModelStore, step: dict[str, Any], message: str) -> None:
    with pytest.raises(ChainError) as error:
        run(chain(step), {}, models=store11)
    assert str(error.value).startswith(message)


# ---------- over HTTP: two sandboxes, each with its own store ----------


@pytest.fixture
def node12(tmp_path: Path, source_server: SourceServer) -> Iterator[TestClient]:
    """Node 12's sandbox, with its demand model trained."""
    with TestClient(create_app(node_settings(tmp_path / "node12"))) as client:
        reply = client.post("/chains/run", json=chain(LAG, INPUTS, TRAIN, source=mg12_link(source_server, end=NOW)))
        assert reply.status_code == 200, reply.text
        yield client


@pytest.fixture
def node11(tmp_path: Path, node12: TestClient) -> Iterator[TestClient]:
    """Node 11's sandbox: it reaches node 12 at http://node12/..."""
    app = create_app(node_settings(tmp_path / "node11"), transport=Nodes({"node12": node12}))
    with TestClient(app) as client:
        yield client


def mg12_link(server: SourceServer, **days: str) -> str:
    return server.url("/microgrids/12/readings?" + "&".join(f"{key}={value}" for key, value in days.items()))


def mg11_link(server: SourceServer, start: str, end: str) -> str:
    return server.url(f"/microgrids/11/readings?start={start}&end={end}")


def test_a_model_card_gives_the_link_of_its_export(node12: TestClient) -> None:
    card = node12.get("/models/mg12_demand").json()
    assert card["export"] == "http://testserver/models/mg12_demand/export"
    [record] = node12.get(card["export"]).json()
    assert record["name"] == "mg12_demand" and record["format"] == "linear" and record["recipe"] == [LAG, INPUTS]
    assert [listed["export"] for listed in node12.get("/models").json()] == [card["export"]]


def test_another_sandbox_loads_prepares_and_predicts_in_one_request(
    node12: TestClient, node11: TestClient, source_server: SourceServer
) -> None:
    later = mg11_link(source_server, NOW, "2026-07-21")
    use = [PREPARE, {"op": "predict", "model": "mg12_demand", "as": "p"}]
    on_node11 = node11.post("/chains/run", json=chain(LOAD, *use, source=later))
    on_node12 = node12.post("/chains/run", json=chain(*use, source=later))
    assert on_node11.status_code == on_node12.status_code == 200, on_node11.text
    assert_close([row["p"] for row in on_node11.json()], [row["p"] for row in on_node12.json()])

    card = node11.get("/models/mg12_demand").json()
    assert card["kind"] == "loaded" and card["source_model"] == "mg12_demand" and card["recipe"] == [LAG, INPUTS]
    assert card["export"] == "http://testserver/models/mg12_demand/export"  # node 11 can share it on


def test_another_sandbox_adapts_and_scores_the_loaded_model_in_one_request(
    node11: TestClient, source_server: SourceServer
) -> None:
    """UC3 across nodes: node 11 has four days of readings; it starts from node 12's model."""
    finetune = {"op": "finetune", "model": "mg12_demand", "as": "mg11_from_mg12", "where": "set == 'train'"}
    test = [{"op": "predict", "model": "mg11_from_mg12", "as": "p"}, {"op": "filter", "where": "set == 'test'"},
            {"op": "evaluate", "actual": "kw", "predicted": "p", "model": "mg11_from_mg12"}]  # fmt: skip
    history = mg11_link(source_server, "2026-07-14", NOW)
    reply = node11.post("/chains/run", json=chain(LOAD, PREPARE, SPLIT, finetune, *test, source=history))
    assert reply.status_code == 200, reply.text
    [scores] = reply.json()
    assert scores["model"] == "mg11_from_mg12" and scores["n"] == 24
    card = node11.get("/models/mg11_from_mg12").json()
    assert card["kind"] == "adapted" and card["source_model"] == "mg12_demand" and card["rows"] == 3 * 24
    assert card["mae"] == scores["mae"]
    [record] = node11.get("/models/mg11_from_mg12/export").json()  # adapted from a linear model: shareable
    assert record["kind"] == "adapted" and record["mae"] == scores["mae"]


def test_check_lists_the_model_link_without_fetching_it(node11: TestClient, source_server: SourceServer) -> None:
    source = mg11_link(source_server, "2026-07-19", "2026-07-21")
    reply = node11.post("/chains/check", json=chain(LOAD, source=source))
    assert reply.json() == {"sources": [{"link": source, "raw": None}, {"link": MODEL_LINK, "raw": None}]}
    assert node11.get("/models").json() == []


def test_a_gbr_model_has_no_export(node12: TestClient, node11: TestClient, source_server: SourceServer) -> None:
    train = {**TRAIN, "model": "mg12_gbr", "algo": "gbr", "params": {"n_estimators": 5}}
    node12.post("/chains/run", json=chain(LAG, INPUTS, train, source=mg12_link(source_server, start="2026-07-10",
                                                                              end=NOW)))  # fmt: skip
    assert "export" not in node12.get("/models/mg12_gbr").json()
    reply = node12.get("/models/mg12_gbr/export")
    assert reply.status_code == 400
    assert reply.json() == {"error": "mg12_gbr is a gbr model; only linear models can be exported for now"}

    load_gbr = {"op": "load", "from": {"source": "http://node12/models/mg12_gbr/export"}}
    day = mg11_link(source_server, "2026-07-19", "2026-07-19")
    reply = node11.post("/chains/run", json=chain(load_gbr, source=day))
    assert reply.status_code == 400
    assert reply.json()["error"] == (
        "steps[0].from.source: http://node12/models/mg12_gbr/export: mg12_gbr is a gbr model; only linear models "
        "can be exported for now"
    )


def test_an_unknown_model_has_no_export(node12: TestClient) -> None:
    reply = node12.get("/models/nope/export")
    assert reply.status_code == 404 and reply.json() == {"error": "no model named nope"}


def test_with_the_policy_switched_on_the_model_link_must_be_allowed(
    client: TestClient, source_server: SourceServer
) -> None:
    reduce = {"op": "reduce", "window": "1h", "aggregate": {"kw": "mean"}}
    reply = client.post("/chains/run", json=chain(reduce, LOAD, source=source_server.url("/readings")))
    assert reply.status_code == 403
    assert reply.json() == {"rejected": f"steps[1].from.source: {MODEL_LINK} is not an allowed source"}
