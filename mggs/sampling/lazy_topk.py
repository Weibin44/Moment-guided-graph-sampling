"""Validated lazy top-k accelerations for moment-preserving edge deletion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .moment_preserving import compute_candidate_deltas, MomentPreservingSampler, moment_order
from .target import target_distances
from .updates import apply_moment_edit


@dataclass(frozen=True)
class DeletionSelection:
    """Selected original edge indices and canonical edge pairs in order."""

    indices: np.ndarray
    edges: np.ndarray


def _edge_key(edge) -> tuple[int, int]:
    u, v = map(int, edge)
    return (u, v) if u < v else (v, u)


def _score_candidates(state, edges, reference, scales, moment_names):
    deltas = compute_candidate_deltas(
        state, edges, include_m4=("m4" in moment_names), include_m5=("m5" in moment_names)
    )
    scores = target_distances(
        {name: float(getattr(state, name)) for name in moment_names}, deltas, reference, scales,
    )
    return scores, deltas


def _apply_selected_delta(state, edge, deltas, position):
    apply_moment_edit(state, "delete", *_edge_key(edge), {
        name: float(values[position]) for name, values in deltas.items() if hasattr(state, name)
    })


def lazy_moment_preserving_topk(
    state,
    reference_moments: dict[str, float],
    normalizer: dict[str, float],
    *,
    budget: int,
    top_k: int = 5,
    refresh_interval: int | None = None,
    moment_names: tuple[str, ...] = ("m2", "m3", "m4"),
) -> DeletionSelection:
    """Refresh only the cached top-k candidates at each deletion.

    A positive ``refresh_interval`` additionally performs a full cache refresh
    at that interval. The score and update order match the validated experiment.
    """

    base_edges = np.asarray(state.edge_list, dtype=np.int64)
    base_index = {_edge_key(edge): index for index, edge in enumerate(base_edges)}
    initial_scores, _ = _score_candidates(
        state, base_edges, reference_moments, normalizer, moment_names
    )
    cache = {_edge_key(edge): float(score) for edge, score in zip(base_edges, initial_scores)}
    selected: list[int] = []
    top_k = max(1, int(top_k))
    refresh_interval = int(refresh_interval) if refresh_interval and refresh_interval > 0 else None

    for step in range(min(int(budget), len(base_edges))):
        current_edges = np.asarray(state.edge_list, dtype=np.int64)
        if refresh_interval is not None and step > 0 and step % refresh_interval == 0:
            refreshed_scores, _ = _score_candidates(
                state, current_edges, reference_moments, normalizer, moment_names
            )
            for edge, score in zip(current_edges, refreshed_scores):
                cache[_edge_key(edge)] = float(score)

        cached = np.asarray([cache[_edge_key(edge)] for edge in current_edges])
        count = min(top_k, len(current_edges))
        local = (
            np.argpartition(cached, count - 1)[:count]
            if count < len(current_edges)
            else np.arange(len(current_edges), dtype=np.int64)
        )
        candidates = current_edges[local]
        scores, deltas = _score_candidates(
            state, candidates, reference_moments, normalizer, moment_names
        )
        for edge, score in zip(candidates, scores):
            cache[_edge_key(edge)] = float(score)
        chosen_position = int(np.argmin(scores))
        chosen_edge = candidates[chosen_position]
        key = _edge_key(chosen_edge)
        selected.append(base_index[key])
        cache.pop(key, None)
        _apply_selected_delta(state, chosen_edge, deltas, chosen_position)

    indices = np.asarray(selected, dtype=np.int64)
    return DeletionSelection(indices=indices, edges=base_edges[indices])


def heap_lazy_moment_preserving_topk(
    state,
    reference_moments: dict[str, float],
    normalizer: dict[str, float],
    *,
    budget: int,
    top_k: int = 5,
    moment_names: tuple[str, ...] = ("m2", "m3", "m4"),
    backend: str = "hybrid",
) -> DeletionSelection:
    """Compatibility wrapper over the persistent heap sampler."""
    sampler = MomentPreservingSampler(
        state, reference_moments, normalizer, strategy="heap", top_k=top_k,
        orders=tuple(moment_order(name) for name in moment_names), backend=backend,
    )
    indices = np.asarray([
        sampler.step() for _ in range(min(max(0, int(budget)), sampler.remaining))
    ], dtype=np.int64)
    return DeletionSelection(indices=indices, edges=sampler.base_edges[indices])
