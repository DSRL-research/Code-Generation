"""UC1: each step of a workflow names a capability over a scope of microgrids. A microgrid that lacks the capability
gets it from generated code (a chain of the sandbox's operators) instead of breaking the workflow.

With 1, 2 and 3 microgrids (tests/scenario/ stands in for the orchestrator, its knowledge base and the LLM):
- MG2 has a demand forecast asset: the data service's demand_forecast file;
- MG1 and MG3 have only their load readings. Their forecast is derived: the generator writes the recipe once, for
  the first microgrid that lacks it; the recipe is checked, registered, and reused for the others.
"""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import SourceServer
from tests.scenario.generator import CannedGenerator, GenerationFailed
from tests.scenario.knowledge import Asset, KnowledgeBase
from tests.scenario.orchestrator import Orchestrator, RecipeFailed
from tests.scenario.sandbox import Sandbox, SandboxRefused
from tests.utils import load_json

SCOPES = pytest.mark.parametrize("mgs", [[1], [1, 2], [1, 2, 3]], ids=["1 MG", "2 MGs", "3 MGs"])
NEXT_4_HOURS = [f"2026-07-31T0{hour}:00:00" for hour in range(4)]  # the readings end on 2026-07-30


def orchestrator(sandbox: TestClient, server: SourceServer, mgs: list[int], **replies: dict) -> Orchestrator:
    knowledge = KnowledgeBase()
    for mg in mgs:
        knowledge.add_asset(Asset("load_readings", mg, "agg", server.url(f"/microgrids/{mg}/readings")))
    if 2 in mgs:
        knowledge.add_asset(Asset("demand_forecast", 2, "pred", server.url("/microgrids/2/demand_forecast")))
    return Orchestrator(Sandbox(sandbox), knowledge, CannedGenerator(replies))


@SCOPES
def test_demand_forecast_step_derives_it_where_it_is_missing(
    sandbox: TestClient, source_server: SourceServer, mgs: list[int]
) -> None:
    flow = orchestrator(sandbox, source_server, mgs)
    forecasts = flow.step("demand_forecast", mgs)

    derived = [mg for mg in mgs if mg != 2]
    for mg in derived:
        assert [row["ts"] for row in forecasts[mg]] == NEXT_4_HOURS
        assert all(200 < row["load"] < 900 for row in forecasts[mg])
        assert flow.knowledge.asset("demand_model", mg).ref == f"mg{mg}_demand_model"  # the trained model is kept
    if 2 in mgs:
        assert len(forecasts[2]) == 10 and forecasts[2][0]["load"] == 415.11  # MG2's own asset, as it is
    # the generator was asked once per missing capability, for the first microgrid that lacked it
    assert flow.generator.requests == [("demand_forecast", 1), ("demand_model", 1)]
    assert set(flow.knowledge.recipes) == {"demand_forecast", "demand_model"}


@SCOPES
def test_consumer_flexibility_step_builds_on_the_forecasts(
    sandbox: TestClient, source_server: SourceServer, mgs: list[int]
) -> None:
    flow = orchestrator(sandbox, source_server, mgs)
    flow.step("demand_forecast", mgs)
    flexibility = flow.step("consumer_flexibility", mgs)

    for mg in mgs:
        assert flexibility[mg], mg
        for row in flexibility[mg]:
            assert row["baseline_kw"] > 0
            assert row["flex_kw"] == pytest.approx(max(0, row["load"] - row["baseline_kw"]), abs=0.11)
    # the second step reused the models trained in the first: one training per derived microgrid
    trainings = [run for run in flow.sandbox.runs if "train" in run]
    assert len(trainings) == len([mg for mg in mgs if mg != 2])
    assert flow.generator.requests[-1] == ("consumer_flexibility", mgs[0])
    assert len(flow.generator.requests) == 3


def test_a_model_is_a_capability_of_its_own(sandbox: TestClient, source_server: SourceServer) -> None:
    flow = orchestrator(sandbox, source_server, [1])
    [card] = flow.step("demand_model", [1])[1]
    assert card["name"] == "mg1_demand_model" and card["kind"] == "trained" and card["mae"] < 40
    assert card["recipe"][0] == {"op": "lag", "field": "kw", "lags": [1, 2, 24]}  # how its inputs are made
    assert card["evaluated_rows"] == 8 * 24  # the test days: every 7th day of the month


# ---------- generated code is checked before it is registered ----------


def broken(change: dict, *, steps: list | None = None) -> dict:
    recipe = load_json("recipes/demand_model.json")
    if steps is not None:
        recipe["chain"]["steps"] = steps
    return {**recipe, **change}


def test_a_chain_with_an_operator_the_sandbox_does_not_offer_is_refused(
    sandbox: TestClient, source_server: SourceServer
) -> None:
    flow = orchestrator(sandbox, source_server, [1],
                        demand_model=broken({}, steps=[{"op": "automl", "model": "${model_name}"}]))  # fmt: skip
    with pytest.raises(GenerationFailed, match=r"\['automl'\]"):
        flow.step("demand_model", [1])
    assert flow.knowledge.recipe("demand_model") is None


def test_a_chain_the_sandbox_cannot_read_is_not_registered(sandbox: TestClient, source_server: SourceServer) -> None:
    wrong_features = [{"op": "train", "model": "${model_name}", "target": "kw", "features": "kw_lag_24"}]
    flow = orchestrator(sandbox, source_server, [1], demand_model=broken({}, steps=wrong_features))
    with pytest.raises(SandboxRefused) as refused:
        flow.step("demand_model", [1])
    assert refused.value.status == 400 and "steps[0].features" in refused.value.reply["error"]
    assert flow.knowledge.recipe("demand_model") is None


def test_a_chain_whose_output_breaks_the_capability_is_not_registered(
    sandbox: TestClient, source_server: SourceServer
) -> None:
    flow = orchestrator(sandbox, source_server, [1], demand_model=broken({"output_fields": ["model", "accuracy"]}))
    with pytest.raises(RecipeFailed, match=r"the output has no \['accuracy'\]"):
        flow.step("demand_model", [1])
    assert flow.knowledge.recipe("demand_model") is None
