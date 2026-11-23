"""The algorithms a chain may name in "fit", with the parameters it may set and their allowed ranges.

To add an algorithm, add an entry to ALGORITHMS. Parameters are checked when the chain is parsed, before anything
runs, so a policy can later limit them in the same way.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.exceptions import ChainError


@dataclass(frozen=True)
class Parameter:
    kind: type
    low: float
    high: float
    default: float


@dataclass(frozen=True)
class Algorithm:
    make: Callable[..., Any]
    parameters: dict[str, Parameter]


ALGORITHMS: dict[str, Algorithm] = {
    "linear": Algorithm(
        make=lambda alpha: Ridge(alpha=alpha),
        parameters={"alpha": Parameter(float, 0.0, 1e6, 1.0)},
    ),
    "gbr": Algorithm(
        make=lambda n_estimators, max_depth, learning_rate: GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=0
        ),
        parameters={
            "n_estimators": Parameter(int, 1, 1000, 100),
            "max_depth": Parameter(int, 1, 10, 3),
            "learning_rate": Parameter(float, 0.001, 1.0, 0.1),
        },
    ),
}


def check_parameters(algo: str, params: dict[str, Any]) -> dict[str, Any]:
    """The parameters with their defaults filled in. Raises ChainError for an unknown algorithm or parameter, or a
    value out of range."""
    algorithm = ALGORITHMS.get(algo)
    if algorithm is None:
        raise ChainError(f"unknown algo {algo!r}; the algorithms are {', '.join(ALGORITHMS)}")
    unknown = sorted(set(params) - set(algorithm.parameters))
    if unknown:
        raise ChainError(f"{algo} has no parameters {unknown}; it has {sorted(algorithm.parameters)}")
    values = {}
    for name, parameter in algorithm.parameters.items():
        value = params.get(name, parameter.default)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not parameter.low <= value <= parameter.high
        ):
            raise ChainError(f"{algo}: {name} must be a number from {parameter.low:g} to {parameter.high:g}")
        values[name] = parameter.kind(value)
    return values


def make_estimator(algo: str, params: dict[str, Any]) -> Any:
    """A new, untrained scikit-learn estimator."""
    return ALGORITHMS[algo].make(**check_parameters(algo, params))
