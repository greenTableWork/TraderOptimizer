from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from statistics import mean
from typing import Any


def full_window_slope_pct(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = mean(values)
    numerator = sum(
        (idx - x_mean) * (value - y_mean)
        for idx, value in enumerate(values)
    )
    denominator = sum((idx - x_mean) ** 2 for idx in range(len(values)))
    if denominator == 0 or values[0] == 0:
        return 0.0
    slope_per_bar = numerator / denominator
    return slope_per_bar * (len(values) - 1) / values[0]


def returns_from_bars(bars: Sequence[Any]) -> list[float]:
    output: list[float] = []
    for previous, current in zip(bars, bars[1:]):
        if previous.close:
            output.append(float(current.close) / float(previous.close) - 1.0)
    return output


def deltas(values: Sequence[float]) -> list[float]:
    return [current - previous for previous, current in zip(values, values[1:])]


def correlation(left: Sequence[float], right: Sequence[float]) -> float:
    count = min(len(left), len(right))
    if count < 2:
        return 0.0
    x_values = [float(value) for value in left[-count:]]
    y_values = [float(value) for value in right[-count:]]
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    )
    x_var = sum((value - x_mean) ** 2 for value in x_values)
    y_var = sum((value - y_mean) ** 2 for value in y_values)
    if x_var <= 0 or y_var <= 0:
        return 0.0
    return numerator / sqrt(x_var * y_var)
