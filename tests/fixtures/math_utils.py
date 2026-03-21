"""Mathematical utility functions for numerical computation."""
import math
from typing import Sequence


def moving_average(values: Sequence[float], window: int) -> list[float]:
    """Compute simple moving average with given window size."""
    if window <= 0:
        raise ValueError("Window must be positive")
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def standard_deviation(values: Sequence[float]) -> float:
    """Calculate population standard deviation."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def percentile(values: Sequence[float], p: float) -> float:
    """Calculate the p-th percentile (0-100) of a sequence."""
    if not values:
        raise ValueError("Cannot compute percentile of empty sequence")
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def normalize(values: Sequence[float]) -> list[float]:
    """Min-max normalize values to [0, 1] range."""
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        return [0.5] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
