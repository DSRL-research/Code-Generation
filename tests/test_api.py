"""End to end: HTTP requests to the app, which fetches from the local source server (tests/conftest.py)."""

import pytest
from fastapi.testclient import TestClient

from src.exceptions import PolicyFileError
from src.main import create_app
from tests.conftest import SourceServer
from tests.utils import load_chain


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_example_a(client: TestClient, source_server: SourceServer) -> None:
    reply = client.post("/chains/run", json=load_chain("example_a", source_server.base_url))
    assert reply.status_code == 200, reply.text
    assert reply.json() == [
        {
            "charge_p10": 1179.8444,
            "charge": 1201.2534,
            "charge_p90": 1235.0528,
            "discharge_p10": -49.9416,
            "discharge": -16.7093,
            "discharge_p90": 132.3328,
        }
    ]


def test_example_b(client: TestClient, source_server: SourceServer) -> None:
    reply = client.post("/chains/run", json=load_chain("example_b", source_server.base_url))
    assert reply.status_code == 200, reply.text
    result = reply.json()
    assert [row["slot"] for row in result] == [48, 49, 50, 51, 52, 53, 54, 55, 72, 73]
    # the test readings have a baseline of 360 + slot (tests/utils.py reading_kw)
    assert result[0] == {
        "ts": "2020-02-25T12:00:00",
        "slot": 48,
        "load": 415.1,
        "baseline_kw": 408.0,
        "diff_kw": 7.1,
        "flex_kw": 7.1,
    }
    assert result[1] == {
        "ts": "2020-02-25T12:15:00",
        "slot": 49,
        "load": 397.0,
        "baseline_kw": 409.0,
        "diff_kw": -12.0,
        "flex_kw": 0,
    }
    assert result[8]["baseline_kw"] == 432.0
    assert [row["flex_kw"] for row in result] == [7.1, 0, 0.8, 17.0, 0, 0, 14.7, 0.7, 0, 0]


def test_check_lists_the_links_without_fetching(client: TestClient, source_server: SourceServer) -> None:
    before = source_server.requests["/readings"]
    reply = client.post("/chains/check", json=load_chain("example_b", source_server.base_url))
    assert reply.status_code == 200
    assert reply.json() == {
        "sources": [
            {"link": source_server.url("/files/demand_forecast"), "raw": False},
            {"link": source_server.url("/readings"), "raw": True},
        ]
    }
    assert source_server.requests["/readings"] == before


def test_a_refused_chain_fetches_nothing(client: TestClient, source_server: SourceServer) -> None:
    before = source_server.requests["/readings"]
    reply = client.post(
        "/chains/run",
        json={
            "source": source_server.url("/readings"),
            "steps": [
                {"op": "filter", "where": "kw > 400"},
                {"op": "reduce", "window": "15m", "aggregate": {"kw": "mean"}},
            ],
        },
    )
    assert reply.status_code == 403
    assert reply.json() == {"rejected": "steps[0] filter: on raw rows a filter may only read ['ts'], not ['kw']"}
    assert source_server.requests["/readings"] == before


@pytest.mark.parametrize(
    "path, steps, status, text",
    [
        ("/files/battery_flex", [{"op": "dump"}], 400, "steps[0]: unknown operator 'dump'"),
        ("/files/battery_flex", [{"op": "map", "set": {"x": "nope + 1"}}], 400, "steps[0] map: unknown field nope"),
        ("/files/battery_flex", [{"op": "map", "set": {"x": "charge / 0"}}], 400, "division by zero"),
        ("/files/battery_flex", [{"op": "map", "set": {"x": "exp(1000)"}}], 400, "math range error"),
        ("/files/battery_flex", [{"op": "map", "set": {"x": "charge * 1e308"}}], 400, "reply: x is inf"),
        ("/status/404", [], 400, "source: http"),
        ("/status/500", [], 502, "replied with status 500"),
        ("/slow", [], 502, "no reply after 0.5 s"),
    ],
)
def test_error_replies(
    client: TestClient, source_server: SourceServer, path: str, steps: list, status: int, text: str
) -> None:
    reply = client.post("/chains/run", json={"source": source_server.url(path), "steps": steps})
    assert reply.status_code == status, reply.text
    assert text in reply.json()["error"]


def test_a_body_that_is_not_json(client: TestClient) -> None:
    reply = client.post("/chains/run", content=b"{not json", headers={"content-type": "application/json"})
    assert reply.status_code == 400
    assert reply.json() == {"error": "the body must be a chain written as a JSON object"}


def test_no_body(client: TestClient) -> None:
    reply = client.post("/chains/run")
    assert reply.status_code == 400
    assert reply.json() == {"error": "the chain must be a JSON object"}


def test_the_app_does_not_start_with_a_policy_it_cannot_read(settings, tmp_path) -> None:
    broken = tmp_path / "policy.json"
    broken.write_text("{}")
    with pytest.raises(PolicyFileError), TestClient(create_app(settings.model_copy(update={"policy_file": broken}))):
        pass
