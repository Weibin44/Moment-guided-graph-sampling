"""Greedy edge deletion that preserves a fixed graph-moment reference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from mggs.moments.low_rank_delta import (
    low_rank_delta_m4_for_independent_deletions,
    low_rank_delta_moments_for_independent_deletions,
)
from mggs.moments.topology_delta import topology_delete_deltas
from mggs.moments._queries import deletion_deltas_from_state
from .updates import apply_moment_edit
from .target import target_distances


EPS = 1e-12


@dataclass(frozen=True)
class MomentPreservingSelection:
    indices: np.ndarray
    edges: np.ndarray


def moment_order(name: str) -> int:
    if len(name) < 2 or not name.startswith("m") or not name[1:].isdigit():
        raise ValueError(f"Invalid moment name: {name}")
    return int(name[1:])


def deletion_delta_m4(state, edges: np.ndarray) -> np.ndarray:
    return low_rank_delta_m4_for_independent_deletions(state, edges, validate=False)


def deletion_delta_m5(state, edges: np.ndarray) -> np.ndarray:
    return low_rank_delta_moments_for_independent_deletions(
        state, edges, orders=(5,), validate=False
    )[5]


def deletion_delta_m4_for_edge(state, u: int, v: int) -> float:
    return float(deletion_delta_m4(state, np.asarray([[u, v]], dtype=np.int64))[0])


def compute_delta_normalizer(
    state,
    include_m4: bool = False,
    include_m5: bool = False,
) -> dict[str, float]:
    """Compute the original per-order standard-deviation normalization."""

    edges = np.asarray(state.edge_list, dtype=np.int64)
    dm2, dm3 = topology_delete_deltas(state, edges)
    s2 = float(np.std(dm2))
    s3 = float(np.std(dm3))
    normalizer = {
        "m2": s2 if s2 > EPS else 1.0,
        "m3": s3 if s3 > EPS else 1.0,
    }
    if include_m4:
        scale = float(np.std(deletion_delta_m4(state, edges)))
        normalizer["m4"] = scale if scale > EPS else 1.0
    if include_m5:
        scale = float(np.std(deletion_delta_m5(state, edges)))
        normalizer["m5"] = scale if scale > EPS else 1.0
    return normalizer


def compute_candidate_deltas(
    state,
    edges: np.ndarray,
    include_m4: bool,
    include_m5: bool = False,
    *,
    backend: str = "hybrid",
) -> dict[str, np.ndarray]:
    orders = (2, 3) + ((4,) if include_m4 else ()) + ((5,) if include_m5 else ())
    return {f"m{k}": value for k, value in deletion_deltas_from_state(
        state, edges, orders=orders, backend=backend
    ).items()}


# Compatibility name for the shared persistent deletion search.
from .search import DeletionTrajectory as MomentPreservingSampler


def choose_moment_preserving_edge(
    state,
    reference_moments: dict[str, float],
    normalizer: dict[str, float],
    moment_names: tuple[str, ...],
    *,
    candidate_edges: np.ndarray | None = None,
    track_m4: bool = True,
) -> tuple[int, int, float, float, float, float]:
    """Choose the candidate deletion minimizing distance to fixed moments."""

    edges = np.asarray(
        state.edge_list if candidate_edges is None else candidate_edges,
        dtype=np.int64,
    )
    deltas = compute_candidate_deltas(
        state,
        edges,
        include_m4=("m4" in moment_names),
        include_m5=("m5" in moment_names),
    )
    score = target_distances(
        {name: float(getattr(state, name)) for name in moment_names},
        deltas, reference_moments, normalizer,
    )
    chosen = int(np.argmin(np.sqrt(score)))
    u, v = int(edges[chosen, 0]), int(edges[chosen, 1])
    if "m4" in deltas:
        dm4 = float(deltas["m4"][chosen])
    elif track_m4:
        dm4 = deletion_delta_m4_for_edge(state, u, v)
    else:
        dm4 = 0.0
    dm5 = float(deltas["m5"][chosen]) if "m5" in deltas else 0.0
    return u, v, float(deltas["m2"][chosen]), float(deltas["m3"][chosen]), dm4, dm5


def apply_moment_deletion(
    state,
    u: int,
    v: int,
    deltas: Mapping[str, float],
) -> None:
    """Delete one edge and apply its precomputed graph-moment deltas."""

    apply_moment_edit(state, "delete", u, v, deltas)


def moment_preserving_greedy(
    state,
    reference_moments: dict[str, float],
    normalizer: dict[str, float],
    *,
    budget: int,
    moment_names: tuple[str, ...] = ("m2", "m3", "m4"),
    backend: str = "hybrid",
) -> MomentPreservingSelection:
    """Repeatedly apply exact all-edge moment-preserving greedy deletion."""

    sampler = MomentPreservingSampler(state, reference_moments, normalizer,
                                     orders=tuple(moment_order(name) for name in moment_names), backend=backend)
    base_edges = sampler.base_edges
    selected = [sampler.step() for _ in range(min(max(0, int(budget)), sampler.remaining))]
    indices = np.asarray(selected, dtype=np.int64)
    return MomentPreservingSelection(indices=indices, edges=base_edges[indices])
