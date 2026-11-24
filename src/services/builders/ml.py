from pydantic import BaseModel

from src.expressions import Expression
from src.ml.algorithms import check_parameters
from src.models.steps.ml import (
    EmbedJson,
    EvaluateJson,
    FinetuneJson,
    ForecastJson,
    LoadJson,
    PredictJson,
    PrepareJson,
    SimilarityJson,
    TrainJson,
)
from src.operations import (
    EmbedOperation,
    EvaluateOperation,
    FinetuneOperation,
    ForecastOperation,
    LoadOperation,
    Operation,
    PredictOperation,
    PrepareOperation,
    SimilarityOperation,
    TrainOperation,
)
from src.services.builders import Builder, Parsing
from src.store.model_store import check_name


class MlBuilders:
    """Model names and algorithm parameters are checked here, when the chain is parsed, before anything runs."""

    def __init__(self, parsing: Parsing) -> None:
        self._parsing = parsing

    def builders(self) -> dict[type[BaseModel], Builder]:
        return {
            TrainJson: self._train,
            PrepareJson: self._prepare,
            FinetuneJson: self._finetune,
            LoadJson: self._load,
            PredictJson: self._predict,
            ForecastJson: self._forecast,
            EvaluateJson: self._evaluate,
            EmbedJson: self._embed,
            SimilarityJson: self._similarity,
        }

    def _train(self, step: TrainJson, path: str) -> Operation:
        """The recipe is left empty here: the parser fills it in from the steps before (services/recipe.py)."""
        check_parameters(step.algo, step.params)
        return TrainOperation(check_name(step.model), step.target, tuple(step.features), self._where(step.where),
                              step.algo, dict(step.params))  # fmt: skip

    def _prepare(self, step: PrepareJson, path: str) -> Operation:
        return PrepareOperation(check_name(step.model))

    def _finetune(self, step: FinetuneJson, path: str) -> Operation:
        return FinetuneOperation(check_name(step.model), check_name(step.output), float(step.alpha),
                                 self._where(step.where))  # fmt: skip

    def _load(self, step: LoadJson, path: str) -> Operation:
        output = check_name(step.output) if step.output is not None else None
        return LoadOperation(self._parsing.nested_chain(step.chain, f"{path}from."), output)

    def _predict(self, step: PredictJson, path: str) -> Operation:
        return PredictOperation(check_name(step.model), step.output)

    def _forecast(self, step: ForecastJson, path: str) -> Operation:
        return ForecastOperation(check_name(step.model), step.steps, step.output)

    def _evaluate(self, step: EvaluateJson, path: str) -> Operation:
        model = check_name(step.model) if step.model is not None else None
        return EvaluateOperation(step.actual, step.predicted, model)

    def _embed(self, step: EmbedJson, path: str) -> Operation:
        by = (step.by,) if isinstance(step.by, str) else tuple(step.by)
        return EmbedOperation(step.field, by, step.normalize, step.output)

    def _similarity(self, step: SimilarityJson, path: str) -> Operation:
        reference = self._parsing.nested_chain(step.chain, f"{path}with.")
        return SimilarityOperation(step.field, step.metric, reference, step.output or step.metric)

    def _where(self, text: str | None) -> Expression | None:
        return self._parsing.expression(text, "where") if text is not None else None
