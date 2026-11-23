from pydantic import BaseModel

from src.ml.algorithms import check_parameters
from src.models.steps.ml import (
    EmbedJson,
    EvaluateJson,
    FinetuneJson,
    FitJson,
    ForecastJson,
    LoadJson,
    PredictJson,
    SimilarityJson,
)
from src.operations import (
    EmbedOperation,
    EvaluateOperation,
    FinetuneOperation,
    FitOperation,
    ForecastOperation,
    LoadOperation,
    Operation,
    PredictOperation,
    SimilarityOperation,
)
from src.services.builders import Builder, Parsing
from src.store.model_store import check_name


class MlBuilders:
    """Model names and algorithm parameters are checked here, when the chain is parsed, before anything runs."""

    def __init__(self, parsing: Parsing) -> None:
        self._parsing = parsing

    def builders(self) -> dict[type[BaseModel], Builder]:
        return {
            FitJson: self._fit,
            FinetuneJson: self._finetune,
            LoadJson: self._load,
            PredictJson: self._predict,
            ForecastJson: self._forecast,
            EvaluateJson: self._evaluate,
            EmbedJson: self._embed,
            SimilarityJson: self._similarity,
        }

    def _fit(self, step: FitJson, path: str) -> Operation:
        check_parameters(step.algo, step.params)
        return FitOperation(check_name(step.model), step.target, tuple(step.lags), tuple(step.calendar), step.algo,
                            dict(step.params), step.holdout)  # fmt: skip

    def _finetune(self, step: FinetuneJson, path: str) -> Operation:
        return FinetuneOperation(check_name(step.model), check_name(step.output), float(step.alpha), step.holdout)

    def _load(self, step: LoadJson, path: str) -> Operation:
        output = check_name(step.output) if step.output is not None else None
        return LoadOperation(self._parsing.nested_chain(step.chain, f"{path}from."), output)

    def _predict(self, step: PredictJson, path: str) -> Operation:
        return PredictOperation(check_name(step.model), step.output)

    def _forecast(self, step: ForecastJson, path: str) -> Operation:
        return ForecastOperation(check_name(step.model), step.steps, step.output)

    def _evaluate(self, step: EvaluateJson, path: str) -> Operation:
        return EvaluateOperation(step.actual, step.predicted)

    def _embed(self, step: EmbedJson, path: str) -> Operation:
        by = (step.by,) if isinstance(step.by, str) else tuple(step.by)
        return EmbedOperation(step.field, by, step.normalize, step.output)

    def _similarity(self, step: SimilarityJson, path: str) -> Operation:
        reference = self._parsing.nested_chain(step.chain, f"{path}with.")
        return SimilarityOperation(step.field, step.metric, reference, step.output or step.metric)
