"""Forecast errors."""

import math
from typing import Any


def errors(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    """n, MAE, RMSE and MAPE (in %, over the actual values that are not 0). None when there are no values."""
    pairs = list(zip(actual, predicted, strict=True))
    if not pairs:
        return {"n": 0, "mae": None, "rmse": None, "mape": None}
    differences = [p - a for a, p in pairs]
    relative = [abs(p - a) / abs(a) for a, p in pairs if a != 0]
    return {
        "n": len(pairs),
        "mae": sum(abs(d) for d in differences) / len(pairs),
        "rmse": math.sqrt(sum(d * d for d in differences) / len(pairs)),
        "mape": 100 * sum(relative) / len(relative) if relative else None,
    }
