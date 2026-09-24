"""Results of moment-guided graph sampling."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EdgeEdit:
    operation: str
    u: int
    v: int


@dataclass(frozen=True)
class GraphSnapshot:
    edges: np.ndarray
    moments: dict[int, float]


@dataclass(frozen=True)
class SamplingResult:
    edges: np.ndarray
    edits: tuple[EdgeEdit, ...]
    initial_moments: dict[int, float]
    final_moments: dict[int, float]
    delta_moments: dict[int, float]
    metadata: dict
    snapshots: dict[int, GraphSnapshot]
