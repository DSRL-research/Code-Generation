"""The hour-of-day profile of a series: the mean value at each hour of the day.

It is the embedding of a microgrid in UC3: two microgrids with close profiles are good candidates to share a
model. In kW (normalize=False) the profile also carries the size of the microgrid; normalized it keeps only the
shape.
"""

import math
from datetime import datetime

from src.exceptions import ChainError
from src.models.rows import Row

HOURS = 24


def daily_profile(items: list[Row], field: str, normalize: bool) -> list[float]:
    """24 values. Normalized: minus their mean, divided by their norm."""
    by_hour: list[list[float]] = [[] for _ in range(HOURS)]
    for row in items:
        time, value = row.get("ts"), row.get(field)
        if not isinstance(time, datetime):
            raise ChainError("an embedding needs rows with a time (ts)")
        if isinstance(value, int | float) and not isinstance(value, bool):
            by_hour[time.hour].append(float(value))
    missing = [hour for hour, values in enumerate(by_hour) if not values]
    if missing:
        raise ChainError(f"an hour-of-day profile needs values of {field} at every hour; none at {missing}")
    profile = [sum(values) / len(values) for values in by_hour]
    if not normalize:
        return profile
    mean = sum(profile) / HOURS
    norm = math.sqrt(sum((value - mean) ** 2 for value in profile)) or 1.0
    return [(value - mean) / norm for value in profile]
