"""Exact delta moments from low-rank random-walk transition updates.

This module implements the "low-rank closed-walk moment delta" backend.  It is
separate from the fast local closed-form m2/m3 formulas in ``topology_delta.py``:
those formulas are specialized single-edge motif/topology updates, while this
backend handles joint edits, independent deletion scoring, and arbitrary orders.

The notation follows the low-rank moment-delta derivation:

    T = touched edit endpoints,
    C = [c_i] contains changed transition columns,
    R = [e_i] selects the touched columns, and
    P' = P + C R^T.

The transition convention here is column-stochastic:

    P = A D^{-1},  m_k = trace(P^k) / n.

For undirected graphs this has the same trace moments as D^{-1} A, but the
low-rank update changes columns, so the convention matters for implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
from scipy.sparse import csr_matrix

try:  # Optional acceleration; the Python fallback below is exact as well.
    from numba import njit
except Exception:  # pragma: no cover - depends on the local environment.
    njit = None


EdgeLike = Union[Sequence[int], np.ndarray]
EdgeBatch = Optional[Union[Iterable[EdgeLike], np.ndarray]]


@dataclass(frozen=True)
class LowRankUpdate:
    """Small representation of a transition update ``P' = P + C R^T``."""

    touched_nodes: np.ndarray
    column_delta: np.ndarray
    transition: csr_matrix

    @property
    def T(self) -> np.ndarray:
        """Touched endpoint set in paper notation."""

        return self.touched_nodes

    @property
    def C(self) -> np.ndarray:
        """Changed transition columns in paper notation."""

        return self.column_delta

    @property
    def P(self) -> csr_matrix:
        """Old transition matrix in paper notation."""

        return self.transition


def low_rank_delta_moments_for_joint_edits(
    state,
    additions: EdgeBatch = None,
    deletions: EdgeBatch = None,
    *,
    max_order: int = 5,
    min_order: int = 2,
    validate: bool = True,
) -> dict[int, float]:
    """Return exact moment deltas for a simultaneous batch of edge edits.

    Parameters
    ----------
    state:
        Current simple-undirected graph state.  The object must expose ``n``,
        ``nbrs`` and ``deg`` attributes, matching ``TopologyMomentState``.
    additions, deletions:
        Undirected edge batches applied simultaneously.  Each edge is canonical
        up to endpoint order.  Self-loops are rejected.
    max_order:
        Largest k for which delta m_k is returned.
    min_order:
        Smallest k returned.  The default mirrors the existing project usage,
        where m1 is usually zero and m2/m3 are the main control variables.
    validate:
        If true, reject duplicate edits, missing deletion edges, additions that
        already exist, and edges appearing in both add/delete batches.

    Notes
    -----
    Deleting the last edge incident to a node is allowed.  The resulting
    isolated node uses the same zero-column convention as the exact moment
    computation in ``utils.py``.
    """

    if max_order < 1:
        raise ValueError("max_order must be at least 1.")
    if min_order < 1 or min_order > max_order:
        raise ValueError("min_order must satisfy 1 <= min_order <= max_order.")

    add_edges = _normalize_edges(additions, state.n, "additions")
    del_edges = _normalize_edges(deletions, state.n, "deletions")
    _validate_edit_batch(state, add_edges, del_edges, validate=validate)

    if len(add_edges) == 0 and len(del_edges) == 0:
        return {k: 0.0 for k in range(min_order, max_order + 1)}

    update = build_low_rank_transition_update(
        state, add_edges, del_edges, validate=False
    )
    trace_deltas = trace_power_deltas_for_joint_low_rank_update(
        update.P, update.C, update.T, max_order
    )
    n = int(state.n)
    return {
        k: float(trace_deltas[k] / n)
        for k in range(min_order, max_order + 1)
    }


def low_rank_delta_moments_for_independent_deletions(
    state,
    deletions: EdgeBatch,
    *,
    orders: Union[int, Sequence[int]],
    validate: bool = True,
) -> dict[int, np.ndarray]:
    """Return exact ``delta m_k`` arrays for many single-edge deletions.

    This is the generalized endpoint-cache fast path.  For each candidate
    deletion ``(u, v)``, the touched set is ``T={u,v}``, so every matrix
    ``H_t = R^T P^t C`` can be reconstructed from endpoint transition values
    ``s_t(a,b)=(P^t)_{a,b}`` for ``a,b in {u,v}``.  Once those values are cached,
    the same trace-power series evaluates any requested order, such as
    ``m4``, ``m5`` or ``m6``.
    """

    order_tuple = _normalize_moment_orders(orders)
    del_edges = _normalize_edges(deletions, state.n, "deletions")
    empty = np.empty((0, 2), dtype=np.int64)
    _validate_edit_batch(state, empty, del_edges, validate=validate)

    if len(del_edges) == 0:
        return {
            order: np.empty(0, dtype=np.float64)
            for order in order_tuple
        }

    return _endpoint_cached_delta_moments_for_single_edge_deletions(
        state, del_edges, order_tuple
    )


def direct_trace_delta_moments_for_independent_deletions(
    state,
    deletions: EdgeBatch,
    *,
    orders: Union[int, Sequence[int]],
    validate: bool = True,
) -> dict[int, np.ndarray]:
    """Evaluate the left side ``Tr((P + C R^T)^k) - Tr(P^k)`` directly.

    The old sparse transition matrix and its traces are shared across all
    candidates.  Each deletion still forms its own exact ``P + C R^T`` and
    performs sparse matrix powers; no low-rank trace-series identity is used.
    Returned values are moment deltas, so trace differences are divided by n.
    """

    order_tuple = _normalize_moment_orders(orders)
    del_edges = _normalize_edges(deletions, state.n, "deletions")
    empty = np.empty((0, 2), dtype=np.int64)
    _validate_edit_batch(state, empty, del_edges, validate=validate)
    if len(del_edges) == 0:
        return {order: np.empty(0, dtype=np.float64) for order in order_tuple}

    max_order = max(order_tuple)
    transition = _column_transition_matrix(state)
    old_traces = _sparse_trace_powers(transition, max_order, order_tuple)
    out = {
        order: np.empty(len(del_edges), dtype=np.float64)
        for order in order_tuple
    }
    n = int(state.n)

    for edge_idx, edge in enumerate(del_edges):
        touched = np.asarray(sorted((int(edge[0]), int(edge[1]))), dtype=np.int64)
        one_deletion = np.asarray([edge], dtype=np.int64)
        new_neighbors = _updated_touched_neighbors(
            state, touched, empty, one_deletion
        )
        column_delta = _column_delta_matrix(state, touched, new_neighbors)
        row_idx, small_col_idx = np.nonzero(column_delta)
        update = csr_matrix(
            (
                column_delta[row_idx, small_col_idx],
                (row_idx, touched[small_col_idx]),
            ),
            shape=transition.shape,
        )
        updated = transition + update
        new_traces = _sparse_trace_powers(updated, max_order, order_tuple)
        for order in order_tuple:
            out[order][edge_idx] = (new_traces[order] - old_traces[order]) / n
    return out


def _sparse_trace_powers(
    matrix: csr_matrix,
    max_order: int,
    requested_orders: Sequence[int],
) -> dict[int, float]:
    """Return selected exact sparse trace powers without diagonalizing."""

    requested = set(int(order) for order in requested_orders)
    traces: dict[int, float] = {}
    power = matrix.copy()
    for order in range(1, max_order + 1):
        if order in requested:
            traces[order] = float(power.diagonal().sum())
        if order < max_order:
            power = power @ matrix
    return traces


def low_rank_delta_m4_for_independent_deletions(
    state,
    deletions: EdgeBatch,
    *,
    validate: bool = True,
) -> np.ndarray:
    """Return exact ``delta m4`` for many single-edge deletions.

    This is an m4-only fast path for scoring many candidate deletions in one
    unchanged graph state.  For a single edge ``(u, v)``, the touched set is
    ``T={u,v}``, so every small matrix ``H_t = R^T P^t C`` can be recovered
    from endpoint transition probabilities ``s_t(a,b)=(P^t)_{a,b}`` for
    ``a,b in {u,v}``.  The implementation caches these endpoint propagations
    once per endpoint and then evaluates the paper's exact trace formula.
    """

    return low_rank_delta_moments_for_independent_deletions(
        state, deletions, orders=(4,), validate=validate
    )[4]


def _endpoint_cached_delta_moments_for_single_edge_deletions(
    state,
    deletions: np.ndarray,
    orders: Sequence[int],
) -> dict[int, np.ndarray]:
    """Exact deletion deltas from cached endpoint transition values."""

    order_tuple = _normalize_moment_orders(orders)
    max_order = max(order_tuple)

    if njit is None:
        s_uu, s_vu, s_uv, s_vv = (
            _endpoint_transition_values_for_single_edge_deletions_python(
                state, deletions, max_order
            )
        )
    else:
        try:
            indptr, indices = _neighbor_csr_arrays(state)
            source_ptr, source_edge_indices, source_other_nodes, source_is_u = (
                _source_edge_incidence_arrays(int(state.n), deletions)
            )
            s_uu, s_vu, s_uv, s_vv = _endpoint_transition_values_numba(
                indptr,
                indices,
                np.asarray(state.deg, dtype=np.float64),
                np.asarray(deletions[:, 0], dtype=np.int64),
                np.asarray(deletions[:, 1], dtype=np.int64),
                source_ptr,
                source_edge_indices,
                source_other_nodes,
                source_is_u,
                max_order,
            )
        except Exception:
            s_uu, s_vu, s_uv, s_vv = (
                _endpoint_transition_values_for_single_edge_deletions_python(
                    state, deletions, max_order
                )
            )

    H_mats = _H_mats_from_endpoint_transition_values(
        np.asarray(state.deg, dtype=np.float64),
        np.asarray(deletions[:, 0], dtype=np.int64),
        np.asarray(deletions[:, 1], dtype=np.int64),
        s_uu,
        s_vu,
        s_uv,
        s_vv,
        max_order,
    )
    trace_deltas = _trace_power_deltas_from_independent_H_series(
        H_mats, max_order, order_tuple
    )
    n = float(state.n)
    return {
        order: np.asarray(trace_deltas[order], dtype=np.float64) / n
        for order in order_tuple
    }


def _endpoint_transition_values_for_single_edge_deletions_python(
    state,
    deletions: np.ndarray,
    max_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cache ``s_t(a,b)=(P^t)_{a,b}`` for deletion endpoints."""

    num_edges = len(deletions)
    if num_edges == 0:
        empty = np.empty((max_order + 1, 0), dtype=np.float64)
        return empty, empty.copy(), empty.copy(), empty.copy()

    edges = np.asarray(deletions, dtype=np.int64).reshape(-1, 2)
    deg = np.asarray(state.deg, dtype=np.float64)
    num_nodes = int(state.n)

    edge_indices_by_source: list[list[int]] = [[] for _ in range(num_nodes)]
    other_nodes_by_source: list[list[int]] = [[] for _ in range(num_nodes)]
    source_is_u_by_source: list[list[bool]] = [[] for _ in range(num_nodes)]
    for edge_idx, (u_raw, v_raw) in enumerate(edges):
        u = int(u_raw)
        v = int(v_raw)
        edge_indices_by_source[u].append(edge_idx)
        other_nodes_by_source[u].append(v)
        source_is_u_by_source[u].append(True)
        edge_indices_by_source[v].append(edge_idx)
        other_nodes_by_source[v].append(u)
        source_is_u_by_source[v].append(False)

    s_uu = np.zeros((max_order + 1, num_edges), dtype=np.float64)
    s_vu = np.zeros((max_order + 1, num_edges), dtype=np.float64)
    s_uv = np.zeros((max_order + 1, num_edges), dtype=np.float64)
    s_vv = np.zeros((max_order + 1, num_edges), dtype=np.float64)

    for source in range(num_nodes):
        edge_indices = edge_indices_by_source[source]
        if not edge_indices:
            continue

        idx_arr = np.asarray(edge_indices, dtype=np.int64)
        other_arr = np.asarray(other_nodes_by_source[source], dtype=np.int64)
        source_is_u = np.asarray(source_is_u_by_source[source], dtype=bool)

        y: dict[int, float] = {source: 1.0}
        for power in range(max_order + 1):
            self_value = float(y.get(source, 0.0))
            other_values = np.fromiter(
                (float(y.get(int(other), 0.0)) for other in other_arr),
                dtype=np.float64,
                count=len(other_arr),
            )

            u_side_idx = idx_arr[source_is_u]
            if len(u_side_idx):
                s_uu[power, u_side_idx] = self_value
                s_vu[power, u_side_idx] = other_values[source_is_u]

            v_side_idx = idx_arr[~source_is_u]
            if len(v_side_idx):
                s_vv[power, v_side_idx] = self_value
                s_uv[power, v_side_idx] = other_values[~source_is_u]

            if power < max_order:
                y = _propagate_endpoint_distribution(state, y, deg)

    return s_uu, s_vu, s_uv, s_vv


def _H_mats_from_endpoint_transition_values(
    deg: np.ndarray,
    u_nodes: np.ndarray,
    v_nodes: np.ndarray,
    s_uu: np.ndarray,
    s_vu: np.ndarray,
    s_uv: np.ndarray,
    s_vv: np.ndarray,
    max_order: int,
) -> np.ndarray:
    """Construct batched ``H_t = R^T P^t C`` matrices for edge deletions."""

    num_edges = len(u_nodes)
    H = np.zeros((max_order, num_edges, 2, 2), dtype=np.float64)
    du = deg[u_nodes]
    dv = deg[v_nodes]
    u_nonisolating = du > 1.0
    v_nonisolating = dv > 1.0
    u_isolating = ~u_nonisolating
    v_isolating = ~v_nonisolating

    for t in range(max_order):
        if np.any(u_nonisolating):
            u_denom = du[u_nonisolating] - 1.0
            H[t, u_nonisolating, 0, 0] = (
                s_uu[t + 1, u_nonisolating] - s_uv[t, u_nonisolating]
            ) / u_denom
            H[t, u_nonisolating, 1, 0] = (
                s_vu[t + 1, u_nonisolating] - s_vv[t, u_nonisolating]
            ) / u_denom
        if np.any(u_isolating):
            H[t, u_isolating, 0, 0] = -s_uv[t, u_isolating]
            H[t, u_isolating, 1, 0] = -s_vv[t, u_isolating]

        if np.any(v_nonisolating):
            v_denom = dv[v_nonisolating] - 1.0
            H[t, v_nonisolating, 0, 1] = (
                s_uv[t + 1, v_nonisolating] - s_uu[t, v_nonisolating]
            ) / v_denom
            H[t, v_nonisolating, 1, 1] = (
                s_vv[t + 1, v_nonisolating] - s_vu[t, v_nonisolating]
            ) / v_denom
        if np.any(v_isolating):
            H[t, v_isolating, 0, 1] = -s_uu[t, v_isolating]
            H[t, v_isolating, 1, 1] = -s_vu[t, v_isolating]

    return H


def _propagate_endpoint_distribution(
    state,
    y: Mapping[int, float],
    deg: np.ndarray,
) -> dict[int, float]:
    """Return ``P y`` for sparse endpoint distribution ``y``."""

    out: dict[int, float] = {}
    for col, value in y.items():
        d = float(deg[int(col)])
        if d <= 0.0 or value == 0.0:
            continue
        contribution = float(value) / d
        for row in state.nbrs[int(col)]:
            row_int = int(row)
            out[row_int] = out.get(row_int, 0.0) + contribution
    return out


def _neighbor_csr_arrays(state) -> tuple[np.ndarray, np.ndarray]:
    """Return adjacency lists as CSR arrays ordered by source column."""

    n = int(state.n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    for node in range(n):
        indptr[node + 1] = indptr[node] + len(state.nbrs[node])
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    for node in range(n):
        start = int(indptr[node])
        if indptr[node + 1] > start:
            indices[start:int(indptr[node + 1])] = np.asarray(
                sorted(int(x) for x in state.nbrs[node]), dtype=np.int64
            )
    return indptr, indices


def _source_edge_incidence_arrays(
    num_nodes: int,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return compact endpoint-to-candidate-edge incidence arrays."""

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    counts = np.zeros(num_nodes, dtype=np.int64)
    for u, v in edges:
        counts[int(u)] += 1
        counts[int(v)] += 1

    source_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=source_ptr[1:])
    fill = source_ptr[:-1].copy()
    total = int(source_ptr[-1])
    source_edge_indices = np.empty(total, dtype=np.int64)
    source_other_nodes = np.empty(total, dtype=np.int64)
    source_is_u = np.empty(total, dtype=np.int8)

    for edge_idx, (u_raw, v_raw) in enumerate(edges):
        u = int(u_raw)
        v = int(v_raw)
        pos = int(fill[u])
        source_edge_indices[pos] = edge_idx
        source_other_nodes[pos] = v
        source_is_u[pos] = 1
        fill[u] += 1

        pos = int(fill[v])
        source_edge_indices[pos] = edge_idx
        source_other_nodes[pos] = u
        source_is_u[pos] = 0
        fill[v] += 1

    return source_ptr, source_edge_indices, source_other_nodes, source_is_u


if njit is not None:

    @njit(cache=True)
    def _endpoint_transition_values_numba(
        indptr,
        indices,
        deg,
        u_nodes,
        v_nodes,
        source_ptr,
        source_edge_indices,
        source_other_nodes,
        source_is_u,
        max_power,
    ):
        num_edges = len(u_nodes)
        num_nodes = len(deg)
        s_uu = np.zeros((max_power + 1, num_edges), dtype=np.float64)
        s_vu = np.zeros((max_power + 1, num_edges), dtype=np.float64)
        s_uv = np.zeros((max_power + 1, num_edges), dtype=np.float64)
        s_vv = np.zeros((max_power + 1, num_edges), dtype=np.float64)

        y = np.zeros(num_nodes, dtype=np.float64)
        next_y = np.zeros(num_nodes, dtype=np.float64)
        active = np.empty(num_nodes, dtype=np.int64)
        next_active = np.empty(num_nodes, dtype=np.int64)

        for source in range(num_nodes):
            edge_start = source_ptr[source]
            edge_end = source_ptr[source + 1]
            if edge_start == edge_end:
                continue

            active_count = 1
            active[0] = source
            y[source] = 1.0

            for power in range(max_power + 1):
                self_value = y[source]
                for pos in range(edge_start, edge_end):
                    edge_idx = source_edge_indices[pos]
                    other = source_other_nodes[pos]
                    other_value = y[other]
                    if source_is_u[pos] == 1:
                        s_uu[power, edge_idx] = self_value
                        s_vu[power, edge_idx] = other_value
                    else:
                        s_vv[power, edge_idx] = self_value
                        s_uv[power, edge_idx] = other_value

                if power == max_power:
                    break

                next_count = 0
                for active_pos in range(active_count):
                    col = active[active_pos]
                    value = y[col]
                    if value != 0.0 and deg[col] > 0.0:
                        contribution = value / deg[col]
                        for ptr in range(indptr[col], indptr[col + 1]):
                            row = indices[ptr]
                            if next_y[row] == 0.0:
                                next_active[next_count] = row
                                next_count += 1
                            next_y[row] += contribution
                    y[col] = 0.0

                tmp_y = y
                y = next_y
                next_y = tmp_y
                tmp_active = active
                active = next_active
                next_active = tmp_active
                active_count = next_count

            for active_pos in range(active_count):
                y[active[active_pos]] = 0.0

        return s_uu, s_vu, s_uv, s_vv

else:
    _endpoint_transition_values_numba = None


def build_low_rank_transition_update(
    state,
    additions: EdgeBatch = None,
    deletions: EdgeBatch = None,
    *,
    validate: bool = True,
) -> LowRankUpdate:
    """Construct the exact low-rank update for a simultaneous edit batch."""

    add_edges = _normalize_edges(additions, state.n, "additions")
    del_edges = _normalize_edges(deletions, state.n, "deletions")
    _validate_edit_batch(state, add_edges, del_edges, validate=validate)

    touched = sorted(
        {int(x) for edge in np.vstack([add_edges, del_edges]) for x in edge}
    )
    if not touched:
        return LowRankUpdate(
            touched_nodes=np.empty(0, dtype=np.int64),
            column_delta=np.zeros((int(state.n), 0), dtype=np.float64),
            transition=_column_transition_matrix(state),
        )

    T = np.asarray(touched, dtype=np.int64)
    new_neighbors = _updated_touched_neighbors(state, T, add_edges, del_edges)
    C = _column_delta_matrix(state, T, new_neighbors)
    P = _column_transition_matrix(state)
    return LowRankUpdate(T, C, P)


def trace_power_deltas_for_joint_low_rank_update(
    transition: csr_matrix,
    column_delta: np.ndarray,
    touched_nodes: Union[Sequence[int], np.ndarray],
    max_order: int,
) -> dict[int, float]:
    """Return exact trace deltas for ``(P + C R^T)^k - P^k``.

    ``column_delta`` is ``C`` and ``touched_nodes`` encodes ``R`` as selected
    standard basis columns.  The implementation computes
    ``H_t = R^T P^t C`` using the old transition matrix, then evaluates the
    small-matrix trace series

        sum_l trace(S(z)^l) / l, where S(z) = z H_0 + z^2 H_1 + ...

    The coefficient relation is

        Delta trace(P^k) = k * [z^k] sum_l trace(S(z)^l) / l.
    """

    if max_order < 1:
        raise ValueError("max_order must be at least 1.")

    T = np.asarray(touched_nodes, dtype=np.int64)
    C = np.asarray(column_delta, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError("column_delta must be a 2-D matrix.")
    if C.shape[1] != len(T):
        raise ValueError("column_delta columns must match touched_nodes.")
    if C.shape[0] != transition.shape[0] or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition must be square and compatible with column_delta.")

    r = len(T)
    if r == 0:
        return {k: 0.0 for k in range(1, max_order + 1)}

    H_mats = _build_H_series_for_joint_edits(transition, C, T, max_order)
    return _trace_power_deltas_from_joint_H_series(H_mats, max_order)


def _build_H_series_for_joint_edits(
    transition: csr_matrix,
    column_delta: np.ndarray,
    touched_nodes: np.ndarray,
    max_order: int,
) -> list[np.ndarray]:
    """Compute ``H_t = R^T P^t C`` for t = 0, ..., max_order - 1."""

    out: list[np.ndarray] = []
    Y = np.asarray(column_delta, dtype=np.float64)
    for t in range(max_order):
        out.append(np.asarray(Y[touched_nodes, :], dtype=np.float64))
        if t + 1 < max_order:
            Y = transition @ Y
    return out


def _trace_power_deltas_from_joint_H_series(
    H_mats: Sequence[np.ndarray],
    max_order: int,
) -> dict[int, float]:
    """Evaluate trace deltas from ``H_t`` matrices by truncated products."""

    if len(H_mats) < max_order:
        raise ValueError("H_mats must contain H_0 through H_{max_order - 1}.")
    if max_order < 1:
        raise ValueError("max_order must be at least 1.")

    r = int(H_mats[0].shape[0])
    if r == 0:
        return {k: 0.0 for k in range(1, max_order + 1)}

    series = [np.zeros((r, r), dtype=np.float64) for _ in range(max_order + 1)]
    for degree in range(1, max_order + 1):
        mat = np.asarray(H_mats[degree - 1], dtype=np.float64)
        if mat.shape != (r, r):
            raise ValueError("all H_mats must be square with the same shape.")
        series[degree] = mat

    trace_coeff = np.zeros(max_order + 1, dtype=np.float64)
    power = [mat.copy() for mat in series]
    for ell in range(1, max_order + 1):
        for degree in range(1, max_order + 1):
            trace_coeff[degree] += float(np.trace(power[degree])) / float(ell)
        if ell < max_order:
            power = _truncated_matrix_poly_mul(power, series, max_order)

    return {
        k: float(k * trace_coeff[k])
        for k in range(1, max_order + 1)
    }


def _trace_power_deltas_from_independent_H_series(
    H_mats: np.ndarray,
    max_order: int,
    orders: Sequence[int],
) -> dict[int, np.ndarray]:
    """Vectorized trace deltas from batched ``H_t`` matrices.

    ``H_mats`` has shape ``(max_order, batch, r, r)``.  The implementation is
    the same trace-series used by ``_trace_power_deltas_from_joint_H_series``:

        Delta trace(P^k) = k * [z^k] sum_l trace(S(z)^l) / l,

    where ``S(z) = z H_0 + z^2 H_1 + ...``.
    """

    if max_order < 1:
        raise ValueError("max_order must be at least 1.")
    if H_mats.shape[0] < max_order:
        raise ValueError("H_mats must contain H_0 through H_{max_order - 1}.")

    H = np.asarray(H_mats, dtype=np.float64)
    if H.ndim < 3 or H.shape[-1] != H.shape[-2]:
        raise ValueError("H_mats must end with square matrix dimensions.")

    r = int(H.shape[-1])
    batch_shape = H.shape[1:-2]
    series = [
        np.zeros(batch_shape + (r, r), dtype=np.float64)
        for _ in range(max_order + 1)
    ]
    for degree in range(1, max_order + 1):
        mat = np.asarray(H[degree - 1], dtype=np.float64)
        if mat.shape[-2:] != (r, r):
            raise ValueError("all H_mats must be square with the same shape.")
        series[degree] = mat

    trace_coeff = [
        np.zeros(batch_shape, dtype=np.float64)
        for _ in range(max_order + 1)
    ]
    power = [mat.copy() for mat in series]
    for ell in range(1, max_order + 1):
        for degree in range(1, max_order + 1):
            trace_coeff[degree] += (
                np.trace(power[degree], axis1=-2, axis2=-1) / float(ell)
            )
        if ell < max_order:
            power = _truncated_batched_matrix_poly_mul(
                power, series, max_order
            )

    return {
        int(order): float(order) * trace_coeff[int(order)]
        for order in orders
    }


def _truncated_matrix_poly_mul(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
    max_degree: int,
) -> list[np.ndarray]:
    """Multiply matrix-polynomial coefficients up to ``max_degree``."""

    r = int(left[0].shape[0])
    out = [np.zeros((r, r), dtype=np.float64) for _ in range(max_degree + 1)]
    for i in range(1, max_degree + 1):
        Li = left[i]
        if not np.any(Li):
            continue
        for j in range(1, max_degree - i + 1):
            Rj = right[j]
            if np.any(Rj):
                out[i + j] += Li @ Rj
    return out


def _truncated_batched_matrix_poly_mul(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
    max_degree: int,
) -> list[np.ndarray]:
    """Multiply batched matrix-polynomial coefficients up to ``max_degree``."""

    r = int(left[0].shape[-1])
    batch_shape = left[0].shape[:-2]
    out = [
        np.zeros(batch_shape + (r, r), dtype=np.float64)
        for _ in range(max_degree + 1)
    ]
    for i in range(1, max_degree + 1):
        Li = left[i]
        if not np.any(Li):
            continue
        for j in range(1, max_degree - i + 1):
            Rj = right[j]
            if np.any(Rj):
                out[i + j] += Li @ Rj
    return out


def _column_transition_matrix(state) -> csr_matrix:
    """Build P = A D^{-1} as a sparse column-stochastic matrix."""

    n = int(state.n)
    deg = np.asarray(state.deg, dtype=np.float64)

    if hasattr(state, "edge_list"):
        edges = np.asarray(state.edge_list, dtype=np.int64)
        if edges.size == 0:
            return csr_matrix((n, n), dtype=np.float64)
        edges = edges.reshape(-1, 2)
        u = edges[:, 0]
        v = edges[:, 1]
        rows = np.concatenate([v, u])
        cols = np.concatenate([u, v])
        data = np.concatenate([1.0 / deg[u], 1.0 / deg[v]])
        return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for col in range(n):
        d = float(deg[col])
        if d <= 0.0:
            continue
        weight = 1.0 / d
        for row in state.nbrs[col]:
            rows.append(int(row))
            cols.append(col)
            data.append(weight)
    return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)


def _updated_touched_neighbors(
    state,
    touched_nodes: np.ndarray,
    add_edges: np.ndarray,
    del_edges: np.ndarray,
) -> Mapping[int, set[int]]:
    """Return post-edit neighbor sets for touched nodes only."""

    updated = {int(i): set(map(int, state.nbrs[int(i)])) for i in touched_nodes}
    for u, v in add_edges:
        updated[int(u)].add(int(v))
        updated[int(v)].add(int(u))
    for u, v in del_edges:
        updated[int(u)].discard(int(v))
        updated[int(v)].discard(int(u))
    return updated


def _column_delta_matrix(
    state,
    touched_nodes: np.ndarray,
    new_neighbors: Mapping[int, set[int]],
) -> np.ndarray:
    """Build U whose columns are p'_i - p_i for touched columns i."""

    n = int(state.n)
    deg = np.asarray(state.deg, dtype=np.float64)
    U = np.zeros((n, len(touched_nodes)), dtype=np.float64)
    for col_idx, node_raw in enumerate(touched_nodes):
        node = int(node_raw)
        old = set(map(int, state.nbrs[node]))
        new = new_neighbors[node]

        old_deg = float(deg[node])
        if old_deg > 0.0:
            old_weight = 1.0 / old_deg
            U[list(old), col_idx] -= old_weight

        new_deg = float(len(new))
        if new_deg > 0.0:
            new_weight = 1.0 / new_deg
            U[list(new), col_idx] += new_weight

    return U


def _normalize_edges(
    edges: EdgeBatch,
    num_nodes: int,
    name: str,
) -> np.ndarray:
    """Convert an edge batch into sorted int64 endpoint pairs."""

    if edges is None:
        return np.empty((0, 2), dtype=np.int64)

    arr = np.asarray(list(edges) if not isinstance(edges, np.ndarray) else edges, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(f"{name} must contain pairs of endpoints.")
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (num_edges, 2).")

    out = arr.copy()
    lo = np.minimum(out[:, 0], out[:, 1])
    hi = np.maximum(out[:, 0], out[:, 1])
    out[:, 0] = lo
    out[:, 1] = hi

    if np.any(out[:, 0] == out[:, 1]):
        raise ValueError(f"{name} cannot contain self-loops.")
    if np.any(out < 0) or np.any(out >= int(num_nodes)):
        raise ValueError(f"{name} contains node ids outside [0, num_nodes).")
    return out


def _normalize_moment_orders(orders: Union[int, Sequence[int]]) -> tuple[int, ...]:
    """Normalize and validate requested moment orders."""

    if isinstance(orders, (int, np.integer)):
        raw_orders = [int(orders)]
    else:
        raw_orders = [int(order) for order in orders]
    if not raw_orders:
        raise ValueError("orders must contain at least one moment order.")
    if any(order < 1 for order in raw_orders):
        raise ValueError("all moment orders must be positive.")
    return tuple(sorted(set(raw_orders)))


def _validate_edit_batch(
    state,
    add_edges: np.ndarray,
    del_edges: np.ndarray,
    *,
    validate: bool,
) -> None:
    """Validate simple-undirected simultaneous edge edits."""

    if not validate:
        return

    add_set = _edge_set(add_edges)
    del_set = _edge_set(del_edges)
    if len(add_set) != len(add_edges):
        raise ValueError("additions contains duplicate edges.")
    if len(del_set) != len(del_edges):
        raise ValueError("deletions contains duplicate edges.")
    overlap = add_set & del_set
    if overlap:
        raise ValueError(f"edges cannot be both added and deleted: {sorted(overlap)[:3]}")

    for u, v in add_set:
        if int(v) in state.nbrs[int(u)]:
            raise ValueError(f"addition edge already exists: {(u, v)}")
    for u, v in del_set:
        if int(v) not in state.nbrs[int(u)]:
            raise ValueError(f"deletion edge does not exist: {(u, v)}")


def _edge_set(edges: np.ndarray) -> set[tuple[int, int]]:
    return {(int(u), int(v)) for u, v in edges}


__all__ = [
    "LowRankUpdate",
    "build_low_rank_transition_update",
    "low_rank_delta_moments_for_joint_edits",
    "low_rank_delta_moments_for_independent_deletions",
    "low_rank_delta_m4_for_independent_deletions",
    "trace_power_deltas_for_joint_low_rank_update",
]
