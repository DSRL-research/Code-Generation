"""The HTTP side of the ML operators: the operator catalogue, the models as assets, and the policy switch."""

from fastapi.testclient import TestClient

from tests.conftest import SourceServer

FIT = {"op": "fit", "model": "mg1_demand", "target": "kw", "lags": [1, 24], "holdout": 24}


def train(client: TestClient, server: SourceServer) -> dict:
    reply = client.post("/chains/run", json={"source": server.url("/microgrids/1/readings?end=2026-07-18"),
                                             "steps": [FIT]})  # fmt: skip
    assert reply.status_code == 200, reply.text
    return reply.json()[0]


def test_the_catalogue_lists_every_operator_with_its_schema(sandbox: TestClient) -> None:
    catalogue = {entry["op"]: entry for entry in sandbox.get("/operators").json()}
    assert [op for op, entry in catalogue.items() if entry["family"] == "stream"] == [
        "filter", "map", "reduce", "join", "union", "top", "collect"]  # fmt: skip
    assert [op for op, entry in catalogue.items() if entry["family"] == "ml"] == [
        "fit", "finetune", "load", "predict", "forecast", "evaluate", "embed", "similarity"]  # fmt: skip
    assert "Trains a forecasting model" in catalogue["fit"]["description"]
    assert {"model", "target", "lags"} <= set(catalogue["fit"]["schema"]["required"])
    assert "as" in catalogue["forecast"]["schema"]["properties"]  # the names a chain uses


def test_a_trained_model_is_listed_with_its_card(sandbox: TestClient, source_server: SourceServer) -> None:
    summary = train(sandbox, source_server)
    assert summary["name"] == "mg1_demand" and summary["holdout_rows"] == 24
    assert [card["name"] for card in sandbox.get("/models").json()] == ["mg1_demand"]
    card = sandbox.get("/models/mg1_demand").json()
    assert card["features"] == ["lag_1", "lag_24", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
    assert card["last_ts"] == "2026-07-17T23:00:00"  # the last day is the holdout


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
    chain = {"source": source_server.url("/readings"), "steps": [FIT]}
    reply = client.post("/chains/run", json=chain)
    assert reply.status_code == 403
    assert reply.json() == {"rejected": "steps[0] fit: not allowed on raw rows; reduce them with a window first"}
