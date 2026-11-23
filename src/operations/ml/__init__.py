"""ML operators: they train, use and compare models (DESIGN.md section 11).

- fit         train a forecasting model on the rows, save it              (training pipeline)
- finetune    adapt another node's model to the local rows, save it      (transfer, UC3)
- load        save a model that another sandbox exported, by its link    (sharing models)
- predict     add the model's prediction to each row                     (backtest)
- forecast    the next N steps after the rows                            (demand forecasting, UC1)
- evaluate    errors of predictions against actual values                (validation)
- embed       one embedding per group: its hour-of-day profile           (characterization)
- similarity  compare each row's vector with a reference vector          (suitability, UC3)

They use the model store of the run (RunContext.models). The ML library pieces are in src/ml/.
"""

from src.operations.ml.embed import EmbedOperation
from src.operations.ml.evaluate import EvaluateOperation
from src.operations.ml.finetune import FinetuneOperation
from src.operations.ml.fit import FitOperation
from src.operations.ml.forecast import ForecastOperation
from src.operations.ml.load import LoadOperation
from src.operations.ml.predict import PredictOperation
from src.operations.ml.similarity import SimilarityOperation

ML_OPERATIONS = (
    FitOperation,
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
    "FitOperation",
    "ForecastOperation",
    "LoadOperation",
    "PredictOperation",
    "SimilarityOperation",
]
