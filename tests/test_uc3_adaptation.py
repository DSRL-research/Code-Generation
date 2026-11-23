"""UC3: a microgrid with little history (4 days) needs a demand model. The orchestrator looks for suitable
microgrids, and for each node picks the best of: training on its own few days, reusing the model of a similar
microgrid, or adapting that model to its days (tests/scenario/adaptation.py).

The scenario's "now" is the end of 2026-07-18: the source models are trained up to it, the target's history is
2026-07-15 to 2026-07-18, and the last day of it validates the candidates. The two days after it (2026-07-19 and
20) check the outcome.

- 1 MG:  MG3 alone: nothing to transfer, so it trains on its own days; the workflow does not break.
- 2 MGs: MG1 (Cluj) has a model; MG3 (Cluj, about the same size) reuses or adapts it.
- 3 MGs: MG1 (Cluj) and MG12 (Athens) have models; MG11 (Athens) is much closer to MG12. Trying only the closest
  source, it takes MG12's model. Trying both costs two more runs; the choice is still better than training locally,
  but one validation day can then favour a model adapted from the farther MG1.
"""

from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import SourceServer
from tests.scenario.adaptation import Adaptation, Decision
from tests.scenario.generator import CannedGenerator
from tests.scenario.knowledge import Asset, KnowledgeBase
from tests.scenario.orchestrator import Orchestrator
from tests.scenario.sandbox import Sandbox

NOW = "2026-07-18"


def setup(sandbox: TestClient, server: SourceServer, sources: list[int]) -> tuple[Sandbox, KnowledgeBase]:
    """The source microgrids, each with its readings up to NOW and a demand model trained on them (as in UC1)."""
    client, knowledge = Sandbox(sandbox), KnowledgeBase()
    for mg in sources:
        knowledge.add_asset(Asset("load_readings", mg, "agg", server.url(f"/microgrids/{mg}/readings?end={NOW}")))
    flow = Orchestrator(client, knowledge, CannedGenerator())
    for mg in sources:
        flow.provide("demand_model", mg)
    return client, knowledge


def adapt(
    sandbox: TestClient, server: SourceServer, target: int, sources: list[int], max_sources: int = 1
) -> tuple[Sandbox, Decision]:
    client, knowledge = setup(sandbox, server, sources)
    history = server.url(f"/microgrids/{target}/readings?start=2026-07-15&end={NOW}")
    decision = Adaptation(client, knowledge, max_sources).best_model(target, history, validation_from=NOW)
    assert knowledge.asset("demand_model", target).ref == decision.chosen.model
    assert decision.chosen.validation_mae == min(candidate.validation_mae for candidate in decision.candidates)
    return client, decision


def mae_on_the_next_two_days(client: Sandbox, server: SourceServer, target: int, model: str) -> float:
    chain: dict[str, Any] = {"source": server.url(f"/microgrids/{target}/readings?start={NOW}&end=2026-07-20"),
                             "steps": [
        {"op": "reduce", "window": "1h", "aggregate": {"kw": "mean"}},
        {"op": "predict", "model": model, "as": "kw_pred"},
        {"op": "filter", "where": "ts >= '2026-07-19'"},
        {"op": "evaluate", "actual": "kw", "predicted": "kw_pred"},
    ]}  # fmt: skip
    [scores] = client.run(chain)
    assert scores["n"] == 48
    return scores["mae"]


def local_model(decision: Decision) -> str:
    return next(candidate.model for candidate in decision.candidates if candidate.kind == "local")


def test_1_mg_with_nothing_to_transfer_trains_on_its_own_history(
    sandbox: TestClient, source_server: SourceServer
) -> None:
    _, decision = adapt(sandbox, source_server, target=3, sources=[])
    assert decision.ranking == []
    assert [candidate.kind for candidate in decision.candidates] == ["local"]
    assert decision.chosen.model == "mg3_demand_model_local"


def test_2_mgs_the_target_takes_the_knowledge_of_the_other(sandbox: TestClient, source_server: SourceServer) -> None:
    client, decision = adapt(sandbox, source_server, target=3, sources=[1])
    assert [entry["mg"] for entry in decision.ranking] == [1]
    assert [candidate.kind for candidate in decision.candidates] == ["local", "reused", "adapted"]
    assert decision.chosen.kind != "local" and decision.chosen.source_mg == 1
    # and on the two days after, the chosen model is better than the one trained on MG3's few days
    chosen = mae_on_the_next_two_days(client, source_server, 3, decision.chosen.model)
    assert chosen < mae_on_the_next_two_days(client, source_server, 3, local_model(decision))


def test_3_mgs_the_target_takes_the_model_of_the_closest_microgrid(
    sandbox: TestClient, source_server: SourceServer
) -> None:
    client, decision = adapt(sandbox, source_server, target=11, sources=[1, 12])
    assert [entry["mg"] for entry in decision.ranking] == [12, 1]  # Athens is closer to Athens
    assert decision.ranking[0]["distance"] < decision.ranking[1]["distance"] / 2
    assert [(candidate.kind, candidate.source_mg) for candidate in decision.candidates] == [
        ("local", None), ("reused", 12), ("adapted", 12)]  # fmt: skip
    assert decision.chosen.source_mg == 12
    chosen = mae_on_the_next_two_days(client, source_server, 11, decision.chosen.model)
    assert chosen < mae_on_the_next_two_days(client, source_server, 11, local_model(decision))


def test_3_mgs_trying_more_sources_costs_more_runs(sandbox: TestClient, source_server: SourceServer) -> None:
    client, decision = adapt(sandbox, source_server, target=11, sources=[1, 12], max_sources=2)
    assert len(decision.candidates) == 5  # local, and reused + adapted for each of the two sources
    assert decision.chosen.kind != "local"
    chosen = mae_on_the_next_two_days(client, source_server, 11, decision.chosen.model)
    assert chosen < mae_on_the_next_two_days(client, source_server, 11, local_model(decision))


def test_the_adapted_node_continues_the_workflow(sandbox: TestClient, source_server: SourceServer) -> None:
    """After UC3 gave MG11 a model, the UC1 step demand_forecast uses it: no training, no new model."""
    client, decision = adapt(sandbox, source_server, target=11, sources=[1, 12])
    knowledge = KnowledgeBase(assets=[
        Asset("load_readings", 11, "agg", source_server.url(f"/microgrids/11/readings?start=2026-07-15&end={NOW}")),
        Asset("demand_model", 11, "model", decision.chosen.model, derived_from="UC3"),
    ])  # fmt: skip
    trainings_before = sum("fit" in run for run in client.runs)
    forecast = Orchestrator(client, knowledge, CannedGenerator()).step("demand_forecast", [11])[11]
    assert [row["ts"] for row in forecast] == [f"2026-07-19T0{hour}:00:00" for hour in range(4)]
    assert sum("fit" in run for run in client.runs) == trainings_before
