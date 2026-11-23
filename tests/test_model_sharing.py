"""Sharing models between sandboxes: GET /models/{name}/export and the load operator (DESIGN.md section 11).

Each node has its own sandbox and its own model store. Node 12 trains a model; node 11 loads it by its link,
uses it, adapts it, and can share it on. The data are the hourly readings of tests/resources/microgrids/.
"""

import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from src.config import Settings
from src.exceptions import ChainError
from src.main import create_app
from src.ml.sharing import export_record
from src.store import ModelStore
from tests.conftest import SourceServer
from tests.utils import mg_rows, rows, run, source_rows, t

LINK = "http://test/mg"
MODEL_LINK = "http://node12/models/mg12_demand/export"
FIT = {"op": "fit", "model": "mg12_demand", "target": "kw", "lags": [1, 2, 24], "calendar": ["hour", "weekday"],
       "holdout": 24}  # fmt: skip
LOAD = {"op": "load", "from": {"source": MODEL_LINK}}
MG11_LATER = mg_rows(11, "2026-07-19", "2026-07-21")


def chain(*steps: dict[str, Any], source: str = LINK) -> dict[str, Any]:
    return {"source": source, "steps": list(steps)}


def predictions(store: ModelStore, model: str) -> list[float]:
    """The predictions of a model on MG11's readings of 2026-07-19 to 07-21."""
    replied = run(chain({"op": "predict", "model": model, "as": "p"}), {LINK: MG11_LATER}, models=store)
    return [row["p"] for row in replied]


def assert_close(first: list[float], second: list[float]) -> None:
    assert len(first) == len(second) > 0
    assert all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip(first, second, strict=True))


# ---------- in memory: one store per node ----------


@pytest.fixture
def store12(tmp_path: Path) -> ModelStore:
    store = ModelStore(tmp_path / "node12")
    run(chain(FIT), {LINK: mg_rows(12, end="2026-07-18")}, models=store)
    return store


@pytest.fixture
def store11(tmp_path: Path) -> ModelStore:
    return ModelStore(tmp_path / "node11")


def load(store: ModelStore, record: dict[str, Any], step: dict[str, Any] = LOAD) -> list[dict[str, Any]]:
    """Runs a chain whose only step is `step` (a load), with `record` as the reply of MODEL_LINK."""
    return run(chain(step), {LINK: MG11_LATER, MODEL_LINK: source_rows([record])}, models=store)


def test_a_loaded_model_predicts_like_the_model_it_was_exported_from(store12: ModelStore, store11: ModelStore) -> None:
    record = export_record(store12.load("mg12_demand"))
    assert record["format"] == "linear" and len(record["coef"]) == len(record["features"]) == 7

    assert load(store11, record) == MG11_LATER.items  # the rows pass through unchanged
    assert_close(predictions(store11, "mg12_demand"), predictions(store12, "mg12_demand"))

    card, origin = store11.card("mg12_demand"), store12.card("mg12_demand")
    assert card["kind"] == "loaded" and card["source_model"] == "mg12_demand" and card["algo"] == "linear"
    for field in ("features", "target", "lags", "calendar", "interval_minutes", "rows", "last_ts", "mae"):
        assert card[field] == origin[field], field  # what it predicts from, and how it was trained at its origin


def test_as_names_the_loaded_model(store12: ModelStore, store11: ModelStore) -> None:
    load(store11, export_record(store12.load("mg12_demand")), {**LOAD, "as": "from_mg12"})
    assert store11.names() == ["from_mg12"]
    assert store11.card("from_mg12")["source_model"] == "mg12_demand"


def test_an_adapted_model_is_exported_as_one_linear_model(store12: ModelStore, store11: ModelStore) -> None:
    finetune = {"op": "finetune", "model": "mg12_demand", "as": "mg11_from_mg12", "holdout": 24}
    run(chain(finetune), {LINK: mg_rows(11, "2026-07-15", "2026-07-18")}, models=store12)

    record = export_record(store12.load("mg11_from_mg12"))  # source + correction, added up
    load(store11, record)
    assert_close(predictions(store11, "mg11_from_mg12"), predictions(store12, "mg11_from_mg12"))


def test_the_record_is_the_model(store11: ModelStore) -> None:
    """Format "linear": intercept + sum(coef * features); here one feature, the value one hour before."""
    record = {"format": "linear", "name": "double", "target": "kw", "lags": [1], "calendar": [],
              "interval_minutes": 60, "coef": [2.0], "intercept": 1.0}  # fmt: skip
    load(store11, record)
    hourly = rows({"ts": t("2026-07-01T00:00"), "kw": 5.0}, {"ts": t("2026-07-01T01:00"), "kw": 7.0},
                  interval_minutes=60)  # fmt: skip
    [row] = run(chain({"op": "predict", "model": "double", "as": "p"}), {LINK: hourly}, models=store11)
    assert row["p"] == 2.0 * 5.0 + 1.0


@pytest.mark.parametrize(("algo", "params"), [("gbr", {"n_estimators": 5}), ("mlp", {})])
def test_gbr_and_mlp_models_cannot_be_exported_yet(store12: ModelStore, algo: str, params: dict[str, int]) -> None:
    fit = {**FIT, "model": f"mg12_{algo}", "algo": algo, "params": params}
    run(chain(fit), {LINK: mg_rows(12, "2026-07-10", "2026-07-18")}, models=store12)
    message = f"mg12_{algo} is a {algo} model; only linear models can be exported for now"
    with pytest.raises(ChainError, match=message):
        export_record(store12.load(f"mg12_{algo}"))


def _valid_record() -> dict[str, Any]:
    return {"format": "linear", "name": "mg12_demand", "target": "kw", "lags": [1, 2, 24],
            "calendar": ["hour", "weekday"], "interval_minutes": 60.0,
            "coef": [0.8, -0.1, 0.3, 1.0, -2.0, 0.5, 0.2], "intercept": 10.0}  # fmt: skip


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(format="gbr"), "format: Input should be 'linear'"),
        (lambda r: r.pop("intercept"), "intercept: Field required"),
        (lambda r: r.update(coef=r["coef"][:6]), "coef must hold one number per feature: 7"),
        (lambda r: r["coef"].__setitem__(0, math.nan), "coef.0: must be a finite number"),
        (lambda r: r.update(lags=[0, 2, 24]), "lags: lags must be different whole numbers from 1 to 168"),
        (lambda r: r.update(calendar=["hour", "hour"]), "calendar names must be different"),
        (lambda r: r.update(interval_minutes=0), "interval_minutes must be more than 0"),
        (lambda r: r.update(name="../models"), "a model name may hold only letters"),
    ],
)
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


class Nodes(httpx.BaseTransport):
    """Sends the links of the other sandboxes (http://<node>/...) to their apps; any other link to the network."""

    def __init__(self, nodes: dict[str, TestClient]) -> None:
        self._nodes = nodes
        self._network = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        node = self._nodes.get(request.url.host)
        if node is None:
            return self._network.handle_request(request)
        reply = node.get(request.url.raw_path.decode("ascii"))
        return httpx.Response(reply.status_code, headers={"content-type": reply.headers["content-type"]},
                              content=reply.content)  # fmt: skip

    def close(self) -> None:
        self._network.close()


def node_settings(folder: Path) -> Settings:
    return Settings(policy_enabled=False, model_dir=folder, fetch_timeout_seconds=5, max_records_per_source=10_000)


@pytest.fixture
def node12(tmp_path: Path, source_server: SourceServer) -> Iterator[TestClient]:
    """Node 12's sandbox, with its demand model trained."""
    with TestClient(create_app(node_settings(tmp_path / "node12"))) as client:
        reply = client.post("/chains/run", json=chain(FIT, source=mg12_link(source_server, end="2026-07-18")))
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
    assert record["name"] == "mg12_demand" and record["format"] == "linear"
    assert len(record["coef"]) == len(card["features"])
    assert [listed["export"] for listed in node12.get("/models").json()] == [card["export"]]


def test_another_sandbox_loads_the_model_by_its_link_and_uses_it(
    node12: TestClient, node11: TestClient, source_server: SourceServer
) -> None:
    later = mg11_link(source_server, "2026-07-19", "2026-07-21")
    predict = {"op": "predict", "model": "mg12_demand", "as": "p"}
    on_node11 = node11.post("/chains/run", json=chain(LOAD, predict, source=later))
    on_node12 = node12.post("/chains/run", json=chain(predict, source=later))
    assert on_node11.status_code == on_node12.status_code == 200, on_node11.text
    assert_close([row["p"] for row in on_node11.json()], [row["p"] for row in on_node12.json()])

    card = node11.get("/models/mg12_demand").json()
    assert card["kind"] == "loaded" and card["source_model"] == "mg12_demand"
    assert card["export"] == "http://testserver/models/mg12_demand/export"  # node 11 can share it on


def test_another_sandbox_adapts_the_loaded_model_in_one_request(
    node11: TestClient, source_server: SourceServer
) -> None:
    """UC3 across nodes: node 11 has four days of readings; it starts from node 12's model."""
    finetune = {"op": "finetune", "model": "mg12_demand", "as": "mg11_from_mg12", "holdout": 24}
    history = mg11_link(source_server, "2026-07-15", "2026-07-18")
    reply = node11.post("/chains/run", json=chain(LOAD, finetune, source=history))
    assert reply.status_code == 200, reply.text
    [summary] = reply.json()
    assert summary["kind"] == "adapted" and summary["source_model"] == "mg12_demand"
    assert summary["rows"] == 48 and summary["mae"] is not None
    [record] = node11.get("/models/mg11_from_mg12/export").json()  # adapted from a linear model: shareable
    assert record["kind"] == "adapted"


def test_check_lists_the_model_link_without_fetching_it(node11: TestClient, source_server: SourceServer) -> None:
    source = mg11_link(source_server, "2026-07-19", "2026-07-21")
    reply = node11.post("/chains/check", json=chain(LOAD, source=source))
    assert reply.json() == {"sources": [{"link": source, "raw": None}, {"link": MODEL_LINK, "raw": None}]}
    assert node11.get("/models").json() == []


def test_a_gbr_model_has_no_export(node12: TestClient, node11: TestClient, source_server: SourceServer) -> None:
    fit = {**FIT, "model": "mg12_gbr", "algo": "gbr", "params": {"n_estimators": 5}}
    node12.post("/chains/run", json=chain(fit, source=mg12_link(source_server, start="2026-07-10", end="2026-07-18")))
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
