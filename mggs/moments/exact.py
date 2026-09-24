"""Full-graph random-walk moment computation."""

from __future__ import annotations

import random

import numpy as np


def estimate_moments(
    edge_index,
    num_nodes,
    number_of_walks=10000,
    number_of_steps=4,
    method="random_walk",
):
    """Estimate moments by random walks or compute exact m1--m4.

    This preserves the original estimator under its public API name.
    """

    if method == "exact":
        return exact_moments_m1_to_m4(edge_index, num_nodes)

    edge_array = edge_index.cpu().numpy()
    neighbors = {i: [] for i in range(num_nodes)}
    for src, dst in edge_array.T:
        neighbors[src].append(dst)
    node_list = list(range(num_nodes))
    moments = np.zeros(number_of_steps)
    for _ in range(number_of_walks):
        node = random.choice(node_list)
        walk_node = node
        for step in range(number_of_steps):
            neighbor_list = neighbors[walk_node]
            if len(neighbor_list) == 0:
                break
            walk_node = random.choice(neighbor_list)
            if walk_node == node:
                moments[step] += 1
    moments = moments / number_of_walks
    return [moments[j] for j in range(len(moments))]


def exact_moments_m1_to_m4(edge_index, num_nodes):
    """Return exact ``[m1, m2, m3, m4]`` using the original sparse method."""

    values = _moments_from_symmetric(_symmetric_transition(edge_index, num_nodes), num_nodes, (1, 2, 3, 4))
    return [values[k] for k in (1, 2, 3, 4)]


def exact_m2_m3_from_state(state) -> tuple[float, float]:
    """Compute exact ``m2/m3`` directly from topology-state caches."""

    m2 = float(np.sum(state.inv_deg * state.W) / state.n)
    m3 = float(2.0 * np.sum(state.inv_deg * state.H) / state.n)
    return m2, m3


def normalized_adjacency_matrix_from_state(state, *, _sqrt=np.sqrt) -> np.ndarray:
    matrix = np.zeros((state.n, state.n), dtype=np.float64)
    for u, v in state.edge_list:
        if state.deg[u] > 0.0 and state.deg[v] > 0.0:
            weight = 1.0 / _sqrt(state.deg[u] * state.deg[v])
            matrix[u, v] = weight
            matrix[v, u] = weight
    return matrix


def exact_m4_from_state(state) -> float:
    """Compute exact ``m4``, using the maintained dense cache when available."""

    if getattr(state, "_use_dense_M", False):
        matrix_m = state._M_dense.copy()
        np.fill_diagonal(matrix_m, state.W)
        return float(np.sum((matrix_m * matrix_m) * np.outer(state.inv_deg, state.inv_deg)) / state.n)
    normalized = normalized_adjacency_matrix_from_state(state)
    squared = normalized @ normalized
    return float(np.sum(squared * squared) / state.n)


def _symmetric_transition(edge_index, num_nodes):
    from scipy.sparse import csr_matrix

    n = num_nodes
    edge_array = edge_index.cpu().numpy()
    rows, cols = edge_array[0], edge_array[1]

    degree = np.bincount(rows, minlength=n).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sqrt_degree = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)

    weights = inv_sqrt_degree[rows] * inv_sqrt_degree[cols]
    symmetric_transition = csr_matrix((weights, (rows, cols)), shape=(n, n))
    return symmetric_transition


def compute_moments(edges, num_nodes, *, orders=(2, 3)) -> dict[int, float]:
    """Exact tr(P**k)/n for requested positive integer orders.

    Uses the historical sparse low-order arithmetic; higher orders use sparse
    powers of D**(-1/2) A D**(-1/2). Isolated rows are zero and n stays fixed.
    """
    from ..io.graph import canonical_edges, edge_index_from_edges
    from .orders import normalize_orders

    orders = normalize_orders(orders)
    edges = canonical_edges(edges, num_nodes)
    edge_index = edge_index_from_edges(edges)
    matrix = _symmetric_transition(edge_index, num_nodes)
    return _moments_from_symmetric(matrix, num_nodes, orders)


def _moments_from_symmetric(matrix, num_nodes, orders):
    """Shared exact arithmetic, including the historical m2/m3/m4 reductions."""
    values = {1: 0.0}
    if 2 in orders:
        values[2] = float(matrix.power(2).sum()) / num_nodes
    if any(k >= 3 for k in orders):
        squared = matrix @ matrix
        if 3 in orders:
            values[3] = float(matrix.multiply(squared).sum()) / num_nodes
        if 4 in orders:
            values[4] = float(squared.power(2).sum()) / num_nodes
        if max(orders) > 4:
            power = squared
            for k in range(3, max(orders) + 1):
                power = power @ matrix
                if k in orders and k > 4:
                    values[k] = float(power.diagonal().sum()) / num_nodes
    return {k: values[k] for k in orders}
