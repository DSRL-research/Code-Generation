"""UC3: the best demand model for a microgrid with little history.

1. discovery: the demand models of the other microgrids (model assets of the knowledge base);
2. suitability: rank their microgrids by how close their load profile is to the target's (embed, similarity);
3. candidates: train on the target's own history; for the `max_sources` closest sources (1 by default), reuse
   their model as it is, or adapt it to the target's history (finetune). The ranking keeps the cost down: each
   source tried costs one evaluation and one training;
4. validation: every candidate is scored on the same day: the last day of the target's history;
5. the best candidate becomes the target's demand model in the knowledge base, with where it came from.
"""

from dataclasses import dataclass
from typing import Any

from tests.scenario.knowledge import Asset, KnowledgeBase
from tests.scenario.sandbox import Sandbox
from tests.scenario.templates import fill
from tests.utils import load_json

CAPABILITY = "demand_model"


@dataclass(frozen=True)
class Candidate:
    kind: str  # "local", "reused" or "adapted"
    model: str
    source_mg: int | None
    validation_mae: float


@dataclass(frozen=True)
class Decision:
    ranking: list[dict[str, Any]]  # the source microgrids, closest first: {"mg", "distance"}
    candidates: list[Candidate]
    chosen: Candidate


class Adaptation:
    def __init__(self, sandbox: Sandbox, knowledge: KnowledgeBase, max_sources: int = 1) -> None:
        self.sandbox = sandbox
        self.knowledge = knowledge
        self.max_sources = max_sources
        self.templates = load_json("recipes/adaptation.json")

    def best_model(self, target: int, history: str, validation_from: str) -> Decision:
        """`history` is the link to the target's few days; the last of them, from `validation_from`, validates."""
        sources = [asset for asset in self.knowledge.assets_of(CAPABILITY) if asset.mg != target]
        ranking = self._rank(history, sources)
        values = {"history": history, "validation_from": validation_from}
        candidates = [self._train(values, f"mg{target}_{CAPABILITY}_local", "local", None, None)]
        for entry in ranking[: self.max_sources]:
            source = next(asset for asset in sources if asset.mg == entry["mg"])
            candidates.append(self._reuse(values, source))
            candidates.append(self._train(values, f"mg{target}_{CAPABILITY}_from_mg{source.mg}", "adapted",
                                          source.ref, source.mg))  # fmt: skip
        chosen = min(candidates, key=lambda candidate: candidate.validation_mae)
        provenance = f"{chosen.kind}" + (f" from MG{chosen.source_mg}" if chosen.source_mg else "")
        self.knowledge.add_asset(Asset(CAPABILITY, target, "model", chosen.model, derived_from=provenance))
        return Decision(ranking, candidates, chosen)

    def _rank(self, history: str, sources: list[Asset]) -> list[dict[str, Any]]:
        """One chain: the embedding of each source microgrid, compared with the embedding of the target."""
        if not sources:
            return []
        embeddings = [self._embedding_of(source.mg) for source in sources]
        chain = embeddings[0]
        chain["steps"] += [{"op": "union", "with": other} for other in embeddings[1:]]
        chain["steps"] += [
            {
                "op": "similarity",
                "metric": "distance",
                "with": fill(self.templates["embedding"], {"readings": history}),
            },
            {"op": "top", "n": len(sources), "by": "distance", "order": "asc"},
            {"op": "collect", "fields": ["mg", "distance"], "decimals": 1},
        ]
        return self.sandbox.run(chain)

    def _embedding_of(self, mg: int) -> dict[str, Any]:
        readings = self.knowledge.asset("load_readings", mg)
        chain = fill(self.templates["embedding"], {"readings": readings.ref})
        chain["steps"].append({"op": "map", "set": {"mg": str(mg)}})
        return chain

    def _train(self, values: dict[str, Any], name: str, kind: str, model: str | None, source: int | None) -> Candidate:
        template = self.templates["local" if kind == "local" else "adapted"]
        [scores] = self.sandbox.run(fill(template, {**values, "model_name": name, "model": model}))
        return Candidate(kind, scores["model"], source, scores["mae"])  # the scores are also on the model's card

    def _reuse(self, values: dict[str, Any], source: Asset) -> Candidate:
        [scores] = self.sandbox.run(fill(self.templates["reused"], {**values, "model": source.ref}))
        return Candidate("reused", source.ref, source.mg, scores["mae"])
