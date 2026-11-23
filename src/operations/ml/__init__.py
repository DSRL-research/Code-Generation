"""ML operators: they train, use and compare models (DESIGN.md sections 11 and 17).

- train       train a model on columns made by the steps before, save it  (the training pipeline, step by step)
- prepare     make a model's inputs on new rows: run its recipe           (same inputs as in training)
- finetune    adapt another node's model to the local rows, save it      (transfer, UC3)
- load        save a model that another sandbox exported, by its link    (sharing models)
- predict     add the model's prediction to each row                     (backtest)
- forecast    the next N steps after the rows                            (demand forecasting, UC1)
- evaluate    errors of predictions against actual values                (validation; on the model's card)
- embed       one embedding per group: its hour-of-day profile           (characterization)
- similarity  compare each row's vector with a reference vector          (suitability, UC3)

The inputs of a model are columns made by stream operators before train: lag (earlier values) and map (anything
written by hand). They use the model store of the run (RunContext.models). The ML library pieces are in src/ml/.
"""

from src.operations.ml.embed import EmbedOperation
from src.operations.ml.evaluate import EvaluateOperation
from src.operations.ml.finetune import FinetuneOperation
from src.operations.ml.forecast import ForecastOperation
from src.operations.ml.load import LoadOperation
from src.operations.ml.predict import PredictOperation
from src.operations.ml.prepare import PrepareOperation
from src.operations.ml.similarity import SimilarityOperation
from src.operations.ml.train import TrainOperation

ML_OPERATIONS = (
    TrainOperation,
    PrepareOperation,
    FinetuneOperation,
    LoadOperation,
    PredictOperation,
    ForecastOperation,
    EvaluateOperation,
    EmbedOperation,
    SimilarityOperation,
)

__all__ = [
    "ML_OPERATIONS",
    "EmbedOperation",
    "EvaluateOperation",
    "FinetuneOperation",
    "ForecastOperation",
    "LoadOperation",
    "PredictOperation",
    "PrepareOperation",
    "SimilarityOperation",
    "TrainOperation",
]
