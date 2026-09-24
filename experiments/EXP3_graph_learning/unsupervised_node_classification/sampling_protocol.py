"""EXP3 compatibility protocol: progressive relative targets and legacy RNG.

This adapter preserves candidate order, operation caps, norm reduction and
add-first tie-breaking. Public sampling uses fixed absolute targets instead.
"""
from __future__ import annotations
import argparse
from typing import Any
import numpy as np

from mggs.moments import TopologyMomentState
from mggs.moments.topology_delta import topology_add_deltas, topology_delete_deltas
from mggs.sampling.moment_direction import relative_moment_delta, sample_edges, sample_non_edges
from mggs.sampling.target import target_distances
from mggs.sampling.updates import apply_moment_edit, run_steps
_MOMENT_EPSILON = 1e-12

Row = dict[str, Any]


def update_state(state: TopologyMomentState, edit: Row) -> None:
    apply_moment_edit(state, edit["op"], edit["u"], edit["v"],
                      {"m2": edit["dm2"], "m3": edit["dm3"]})

def candidate_edit(
    state: TopologyMomentState,
    edges: np.ndarray,
    deltas: tuple[np.ndarray, np.ndarray],
    operation: str,
    target: np.ndarray,
    original: tuple[float, float],
    current_distance: float,
) -> Row:
    """Return the candidate from one operation closest to the target."""
    dm2, dm3 = deltas
    distances = target_distances(
        {2: state.m2, 3: state.m3}, {2: dm2, 3: dm3},
        {2: target[0], 3: target[1]},
        {2: max(abs(original[0]), _MOMENT_EPSILON),
         3: max(abs(original[1]), _MOMENT_EPSILON)},
        origin={2: original[0], 3: original[1]}, euclidean=True,
    )
    index = int(np.argmin(distances))
    return {
        "op": operation,
        "u": int(edges[index, 0]),
        "v": int(edges[index, 1]),
        "dm2": float(dm2[index]),
        "dm3": float(dm3[index]),
        "improvement": current_distance - float(distances[index]),
    }

def best_target_edit(
    state: TopologyMomentState,
    target: np.ndarray,
    original: tuple[float, float],
    original_edges: set[tuple[int, int]],
    args: argparse.Namespace,
    allow_add: bool,
    allow_delete: bool,
) -> Row | None:
    additions = (
        sample_non_edges(state, args.candidate_add)
        if allow_add
        else np.empty((0, 2), dtype=np.int64)
    )
    deletions = (
        sample_edges(state, args.candidate_del)
        if allow_delete
        else np.empty((0, 2), dtype=np.int64)
    )
    # Never undo an earlier edit. This makes the operation count identical to
    # the final symmetric difference from the original graph.
    additions = np.asarray(
        [edge for edge in additions if tuple(edge) not in original_edges],
        dtype=np.int64,
    ).reshape(-1, 2)
    deletions = np.asarray(
        [edge for edge in deletions if tuple(edge) in original_edges],
        dtype=np.int64,
    ).reshape(-1, 2)
    current = relative_moment_delta(state.m2, state.m3, *original)
    current_distance = float(np.linalg.norm(current - target))
    candidates = []
    for operation, edges, delta_fn in (
        ("add", additions, topology_add_deltas),
        ("del", deletions, topology_delete_deltas),
    ):
        if len(edges):
            candidates.append(
                candidate_edit(
                    state, edges, delta_fn(state, edges), operation,
                    target, original, current_distance,
                )
            )
    return max(
        candidates,
        key=lambda candidate: float(candidate["improvement"]),
        default=None,
    )

def sample_target(
    base_state: TopologyMomentState,
    target: np.ndarray,
    add_budget: int,
    delete_budget: int,
    total_budget: int,
    args: argparse.Namespace,
) -> tuple[TopologyMomentState, int, int]:
    """Follow the target trajectory with exactly the requested number of edits."""
    state = base_state.copy()
    original_edges = set(base_state.edge_list)
    add_count = delete_count = 0
    def step(edit_index):
        nonlocal add_count, delete_count
        edit = best_target_edit(
            state,
            target * ((edit_index + 1) / total_budget),
            (base_state.m2, base_state.m3),
            original_edges,
            args,
            allow_add=add_count < add_budget,
            allow_delete=delete_count < delete_budget,
        )
        if edit is None:
            raise RuntimeError(
                f"Cannot complete exact edit budget: {edit_index}/{total_budget}."
            )
        update_state(state, edit)
        if edit["op"] == "add":
            add_count += 1
        else:
            delete_count += 1
    run_steps(total_budget, step)
    final_edit_count = len(original_edges ^ set(state.edge_list))
    if final_edit_count != total_budget:
        raise RuntimeError(
            f"Expected {total_budget} final edits, obtained {final_edit_count}."
        )
    return state, add_count, delete_count
