"""Moment-distance objective, independent of candidate generation and graph edits."""
from __future__ import annotations
import numpy as np


def target_distances(current, deltas, target, scales, *, origin=None, euclidean=False):
    """Score proposed moments in stable order.

    The default sums squared distances in absolute moment coordinates.
    EXP3's historical protocol uses origin-relative coordinates and a Euclidean
    reduction; preserve that subtraction/reduction order for reproducibility.
    """
    components = []
    for k in current:
        after = current[k] + deltas[k]
        error = ((after - target[k]) / scales[k] if origin is None
                 else (after - origin[k]) / scales[k] - target[k])
        components.append(error)
    if euclidean:
        return np.linalg.norm(np.column_stack(components), axis=1)
    scores = np.zeros_like(components[0], dtype=np.float64)
    for error in components:
        scores += error ** 2
    return scores
