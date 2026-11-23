"""The HTTP side of the ML operators: the operator catalogue, the models as assets, and the policy switch."""

from fastapi.testclient import TestClient

from tests.conftest import SourceServer

LAG = {"op": "lag", "field": "kw", "lags": [1, 24]}
INPUTS = {"op": "map", "set": {"hour_sin": "sin(2 * pi * ts.hour / 24)", "hour_cos": "cos(2 * pi * ts.hour / 24)"}}
TRAIN = {"op": "train", "model": "mg1_demand", "target": "kw", "features": ["kw_lag_1", "kw_lag_24", "hour_sin",
                                                                            "hour_cos"]}  # fmt: skip


def train(client: TestClient, server: SourceServer) -> None:
    reply = client.post("/chains/run", json={"source": server.url("/microgrids/1/readings?end=2026-07-18"),
                                             "steps": [LAG, INPUTS, TRAIN]})  # fmt: skip
    assert reply.status_code == 200, reply.text


def test_the_catalogue_lists_every_operator_with_its_schema(sandbox: TestClient) -> None:
    catalogue = {entry["op"]: entry for entry in sandbox.get("/operators").json()}
    assert [op for op, entry in catalogue.items() if entry["family"] == "stream"] == [
        "filter", "map", "reduce", "join", "union", "top", "lag", "collect"]  # fmt: skip
    assert [op for op, entry in catalogue.items() if entry["family"] == "ml"] == [
        "train", "prepare", "finetune", "load", "predict", "forecast", "evaluate", "embed", "similarity"]  # fmt: skip
    assert "Trains a model on columns of the rows" in catalogue["train"]["description"]
    assert {"model", "target", "features"} <= set(catalogue["train"]["schema"]["required"])
    assert "as" in catalogue["forecast"]["schema"]["properties"]  # the names a chain uses


def test_a_trained_model_is_listed_with_its_card(sandbox: TestClient, source_server: SourceServer) -> None:
    train(sandbox, source_server)
    assert [card["name"] for card in sandbox.get("/models").json()] == ["mg1_demand"]
    card = sandbox.get("/models/mg1_demand").json()
    assert card["features"] == ["kw_lag_1", "kw_lag_24", "hour_sin", "hour_cos"] and card["recipe"] == [LAG, INPUTS]
    assert card["last_ts"] == "2026-07-18T23:00:00"  # no rows kept back: train learns from all rows without "where"


def test_an_unknown_model(sandbox: TestClient, source_server: SourceServer) -> None:
    assert sandbox.get("/models/nope").status_code == 404
    reply = sandbox.post("/chains/run", json={"source": source_server.url("/microgrids/1/readings"),
                                              "steps": [{"op": "predict", "model": "nope"}]})  # fmt: skip
    assert reply.status_code == 400
    assert reply.json()["error"].startswith("steps[0] predict: no model named nope")


def test_with_the_policy_switched_off_raw_rows_can_be_returned(
    sandbox: TestClient, source_server: SourceServer
) -> None:
    chain = {"source": source_server.url("/readings"), "steps": [{"op": "top", "n": 1, "by": "kw"}]}
    assert sandbox.post("/chains/run", json=chain).status_code == 200
    assert sandbox.post("/chains/check", json=chain).json() == {"sources": [
        {"link": source_server.url("/readings"), "raw": None}]}  # fmt: skip


def test_with_the_policy_switched_on_ml_operators_need_reduced_rows(
    client: TestClient, source_server: SourceServer
) -> None:
    chain = {"source": source_server.url("/readings"), "steps": [TRAIN]}
    reply = client.post("/chains/run", json=chain)
    assert reply.status_code == 403
    assert reply.json() == {"rejected": "steps[0] train: not allowed on raw rows; reduce them with a window first"}
