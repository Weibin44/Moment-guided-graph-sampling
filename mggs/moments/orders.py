"""Order and backend validation for the current public interfaces."""
from __future__ import annotations
import numpy as np


def normalize_orders(orders, *, minimum=1) -> tuple[int, ...]:
    try:
        values = tuple(orders)
    except TypeError as error:
        raise ValueError("orders must be a nonempty collection of integers") from error
    if not values or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < minimum for k in values):
        raise ValueError(f"orders must contain integers >= {minimum}")
    return tuple(sorted(set(map(int, values))))


def validate_backend(backend: str, orders) -> str:
    if backend not in ("topology", "low_rank"):
        raise ValueError("backend must be topology or low_rank")
    if backend == "topology" and any(k not in (2, 3) for k in orders):
        raise ValueError("topology supports only orders 2 and 3")
    return backend
