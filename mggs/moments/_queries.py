"""Shared state-based delta dispatch for calculators and experiment samplers."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .topology_delta import topology_add_deltas, topology_delete_deltas

from .low_rank_delta import (
    direct_trace_delta_moments_for_independent_deletions,
    low_rank_delta_moments_for_independent_deletions,
    low_rank_delta_moments_for_joint_edits,
)


def deletion_deltas_from_state(state, edges, *, orders=(2, 3), backend="hybrid"):
    """Score canonical, existing edges on a shared state without rebuilding it.

    Internal callers own edge validation. ``hybrid`` uses topology for m2/m3
    and low-rank for higher orders; ``topology`` never silently falls back.
    """
    requested = _orders(orders)
    if backend == "direct_trace":
        return direct_trace_delta_moments_for_independent_deletions(
            state, edges, orders=requested, validate=False
        )
    backend = _backend(backend)
    topology_orders = tuple(k for k in requested if k in (2, 3))
    if backend == "topology" and len(topology_orders) != len(requested):
        raise ValueError("The topology backend supports only moment orders 2 and 3.")
    values = {}
    if backend in ("hybrid", "topology") and topology_orders:
        dm2, dm3 = topology_delete_deltas(state, edges)
        values.update({k: value for k, value in ((2, dm2), (3, dm3)) if k in requested})
    low_rank_orders = requested if backend == "low_rank" else tuple(
        k for k in requested if k not in (2, 3)
    )
    if low_rank_orders:
        values.update(low_rank_delta_moments_for_independent_deletions(
            state, edges, orders=low_rank_orders, validate=False
        ))
    return values


def _orders(orders: Sequence[int]) -> tuple[int, ...]:
    result = tuple(sorted({int(k) for k in orders}))
    if not result or result[0] < 1:
        raise ValueError("orders must contain positive integers.")
    return result


def _backend(backend: str) -> str:
    value = str(backend).strip().lower().replace("-", "_")
    if value == "auto":  # Compatibility with historical experiment configurations.
        value = "hybrid"
    if value not in {"hybrid", "topology", "low_rank"}:
        raise ValueError("backend must be one of: hybrid, topology, low_rank.")
    return value


def addition_deltas_from_state(state, edges, *, orders=(2, 3), backend="hybrid"):
    """Compute independent additions on an existing state, without rebuilding it."""
    requested = _orders(orders)
    backend = _backend(backend)
    values: dict[int, np.ndarray] = {}

    topology_orders = tuple(k for k in requested if k in (2, 3))
    if backend == "topology" and len(topology_orders) != len(requested):
        raise ValueError("The topology backend supports only moment orders 2 and 3.")

    if backend in ("hybrid", "topology") and topology_orders:
        dm2, dm3 = topology_add_deltas(state, edges)
        if 2 in requested:
            values[2] = dm2
        if 3 in requested:
            values[3] = dm3

    low_rank_orders = (
        requested if backend == "low_rank"
        else tuple(k for k in requested if k not in (2, 3))
    )
    for order in low_rank_orders:
        values[order] = np.empty(len(edges), dtype=np.float64)
    for idx, edge in enumerate(edges):
        if low_rank_orders:
            delta = low_rank_delta_moments_for_joint_edits(
                state,
                additions=edge.reshape(1, 2),
                min_order=min(low_rank_orders),
                max_order=max(low_rank_orders),
                validate=False,
            )
            for order in low_rank_orders:
                values[order][idx] = delta[order]
    return values
