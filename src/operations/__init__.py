"""The operators of a chain, in two families (DESIGN.md sections 5 and 11):

- stream/  filter, map, reduce, join, union, top, lag, collect: reshape rows
- ml/      train, prepare, finetune, load, predict, forecast, evaluate, embed, similarity: train, share, use and
           compare models

To add an operator: write its class in one of the families (a subclass of Operation) and add it to that family's
list, then add its JSON model (models/steps/), a builder (services/builders/), and a rule in policy.json.
See README.md, "Extend".
"""

from src.operations.base import CombinesChains, Operation, RecordsRecipe, RunContext
from src.operations.ml import (
    ML_OPERATIONS,
    EmbedOperation,
    EvaluateOperation,
    FinetuneOperation,
    ForecastOperation,
    LoadOperation,
    PredictOperation,
    PrepareOperation,
    SimilarityOperation,
    TrainOperation,
)
from src.operations.stream import (
    STREAM_OPERATIONS,
    CollectOperation,
    FilterOperation,
    JoinOperation,
    LagOperation,
    MapOperation,
    ReduceOperation,
    TopOperation,
    UnionOperation,
)

OPERATIONS: tuple[type[Operation], ...] = (*STREAM_OPERATIONS, *ML_OPERATIONS)

OPERATION_NAMES: tuple[str, ...] = tuple(operation.name for operation in OPERATIONS)

__all__ = [
    "ML_OPERATIONS",
    "OPERATIONS",
    "OPERATION_NAMES",
    "STREAM_OPERATIONS",
    "CollectOperation",
    "CombinesChains",
    "EmbedOperation",
    "EvaluateOperation",
    "FilterOperation",
    "FinetuneOperation",
    "ForecastOperation",
    "JoinOperation",
    "LagOperation",
    "LoadOperation",
    "MapOperation",
    "Operation",
    "PredictOperation",
    "PrepareOperation",
    "RecordsRecipe",
    "ReduceOperation",
    "RunContext",
    "SimilarityOperation",
    "TopOperation",
    "TrainOperation",
    "UnionOperation",
]
