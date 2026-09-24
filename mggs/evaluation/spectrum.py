"""Normalized-adjacency spectrum construction and comparison."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


@dataclass
class SpectrumStats:
    eigvals: np.ndarray
    m2: float
    m3: float
    m4: float
    num_components: int
    num_isolated: int


def count_components(edge_list, num_nodes: int) -> int:
    neighbors = [[] for _ in range(num_nodes)]
    for u, v in edge_list:
        neighbors[u].append(v)
        neighbors[v].append(u)
    seen = np.zeros(num_nodes, dtype=bool)
    count = 0
    for start in range(num_nodes):
        if seen[start]:
            continue
        count += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            for neighbor in neighbors[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
    return count


def normalized_adjacency_sparse_matrix(edge_list, num_nodes, edge_weights=None):
    edge_list = list(edge_list)
    weights = (
        np.ones(len(edge_list), dtype=np.float64)
        if edge_weights is None
        else np.asarray(edge_weights, dtype=np.float64).reshape(-1)
    )
    if len(weights) != len(edge_list):
        raise ValueError(f"Expected {len(edge_list)} edge weights, got {len(weights)}.")
    degree = np.zeros(num_nodes, dtype=np.float64)
    for (u, v), weight in zip(edge_list, weights):
        degree[u] += weight
        degree[v] += weight
    rows, cols, values = [], [], []
    for (u, v), edge_weight in zip(edge_list, weights):
        if degree[u] <= 0.0 or degree[v] <= 0.0:
            continue
        value = float(edge_weight) / math.sqrt(degree[u] * degree[v])
        rows.extend([u, v]); cols.extend([v, u]); values.extend([value, value])
    return csr_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes)), degree


def normalized_adjacency_spectrum(
    edge_list, num_nodes: int, *, edge_weights=None, eval_mode="exact", topk=64, moments=None
) -> SpectrumStats:
    edge_list = list(edge_list)
    matrix, degree = normalized_adjacency_sparse_matrix(edge_list, num_nodes, edge_weights)
    if eval_mode == "topk":
        k = min(int(topk), max(1, num_nodes - 2))
        if num_nodes <= 2 * k + 1:
            eigvals = np.linalg.eigvalsh(matrix.toarray())
        else:
            largest = eigsh(matrix, k=k, which="LA", return_eigenvectors=False, tol=1e-4)
            smallest = eigsh(matrix, k=k, which="SA", return_eigenvectors=False, tol=1e-4)
            eigvals = np.concatenate([np.sort(smallest), np.sort(largest)])
        moments = moments or {}
        m2, m3, m4 = (float(moments.get(name, np.nan)) for name in ("m2", "m3", "m4"))
    elif eval_mode == "exact":
        eigvals = np.linalg.eigvalsh(matrix.toarray())
        m2, m3, m4 = (float(np.mean(eigvals ** order)) for order in (2, 3, 4))
    else:
        raise ValueError(f"Unknown spectrum eval mode: {eval_mode}")
    return SpectrumStats(
        eigvals=eigvals, m2=m2, m3=m3, m4=m4,
        num_components=count_components(edge_list, num_nodes),
        num_isolated=int(np.sum(degree == 0.0)),
    )


def normalized_adjacency_matrix_from_state(state) -> np.ndarray:
    from ..moments.exact import normalized_adjacency_matrix_from_state as build_matrix
    return build_matrix(state, _sqrt=math.sqrt)


def matrix_after_delete(state, current_matrix: np.ndarray, u: int, v: int) -> np.ndarray:
    candidate = current_matrix.copy()
    candidate[u, v] = candidate[v, u] = 0.0
    if state.deg[u] > 1:
        for neighbor in state.nbrs[u]:
            if neighbor != v:
                value = 1.0 / math.sqrt((state.deg[u] - 1.0) * state.deg[neighbor])
                candidate[u, neighbor] = candidate[neighbor, u] = value
    if state.deg[v] > 1:
        for neighbor in state.nbrs[v]:
            if neighbor != u:
                value = 1.0 / math.sqrt((state.deg[v] - 1.0) * state.deg[neighbor])
                candidate[v, neighbor] = candidate[neighbor, v] = value
    return candidate


def spectrum_rmse(current: np.ndarray, original: np.ndarray) -> float:
    return float(np.sqrt(np.mean((current - original) ** 2)))
