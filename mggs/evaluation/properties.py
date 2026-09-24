"""Graph-property measurements shared by sampling experiments."""
from __future__ import annotations
import numpy as np
from ..moments.topology_state import TopologyMomentState
from .spectrum import normalized_adjacency_matrix_from_state, spectrum_rmse


def average_neighbor_inverse_degree(state: TopologyMomentState) -> float:
    """One-step random-walk expectation of endpoint reciprocal degree."""
    values = np.zeros(state.n, dtype=np.float64)
    for node in range(state.n):
        if state.deg[node] > 0:
            values[node] = np.mean(
                [1.0 / state.deg[neighbor] for neighbor in state.nbrs[node]]
            )
    return float(np.mean(values))


def absolute_mean_difference(
    current: np.ndarray,
    original: np.ndarray,
) -> float:
    """Absolute difference of graph-wide means, not mean nodewise error."""
    current = np.asarray(current, dtype=np.float64)
    original = np.asarray(original, dtype=np.float64)
    if current.shape != original.shape:
        raise ValueError("Node-aligned clustering arrays must have the same shape.")
    if len(original) == 0:
        return 0.0
    return float(abs(np.mean(current) - np.mean(original)))


def triangle_degree_weighted_clustering_coefficients(
    state: TopologyMomentState,
) -> np.ndarray:
    """Local closed-pair mass weighted by 1/sqrt(d_u d_w)."""

    coeffs = np.zeros(state.n, dtype=np.float64)
    inverse_sqrt_degree = np.zeros(state.n, dtype=np.float64)
    positive_degree = state.deg > 0
    inverse_sqrt_degree[positive_degree] = 1.0 / np.sqrt(
        state.deg[positive_degree]
    )
    for node in range(state.n):
        neighbors = list(state.nbrs[node])
        degree = len(neighbors)
        if degree < 2:
            continue
        closed_weight = 0.0
        for index, u in enumerate(neighbors):
            nbrs_u = state.nbrs[int(u)]
            for w in neighbors[index + 1:]:
                if int(w) in nbrs_u:
                    closed_weight += (
                        inverse_sqrt_degree[int(u)]
                        * inverse_sqrt_degree[int(w)]
                    )
        coeffs[node] = closed_weight / (degree * (degree - 1) / 2.0)
    return coeffs


def normalized_estrada_index(eigenvalues: np.ndarray) -> float:
    """Normalized Estrada index computed from normalized-adjacency eigenvalues."""
    return float(np.mean(np.exp(eigenvalues)))


def compute_property_snapshot(
    state: TopologyMomentState,
    properties: tuple[str, ...],
) -> dict[str, object]:
    requested = set(properties)
    snapshot: dict[str, object] = {}
    needs_eigenvalues = (
        "spectrum_rmse" in requested
        or "normalized_estrada_index_abs_error" in requested
    )
    if needs_eigenvalues:
        eigenvalues = np.linalg.eigvalsh(
            normalized_adjacency_matrix_from_state(state)
        )
        snapshot["eigenvalues"] = eigenvalues
    if "mean_neighbor_inverse_degree_abs_error" in requested:
        snapshot["average_neighbor_inverse_degree"] = (
            average_neighbor_inverse_degree(state)
        )
    if "mean_triangle_weighted_clustering_abs_error" in requested:
        snapshot["triangle_degree_weighted_clustering"] = (
            triangle_degree_weighted_clustering_coefficients(state)
        )
    if "normalized_estrada_index_abs_error" in requested:
        snapshot["normalized_estrada_index"] = normalized_estrada_index(eigenvalues)
    return snapshot


def property_distances(
    current: dict[str, object],
    original: dict[str, object],
    properties: tuple[str, ...],
) -> dict[str, float]:
    distances: dict[str, float] = {}
    requested = set(properties)
    if "spectrum_rmse" in requested:
        distances["spectrum_rmse"] = spectrum_rmse(
            np.asarray(current["eigenvalues"]),
            np.asarray(original["eigenvalues"]),
        )
    if "mean_neighbor_inverse_degree_abs_error" in requested:
        distances["mean_neighbor_inverse_degree_abs_error"] = abs(
            float(current["average_neighbor_inverse_degree"])
            - float(original["average_neighbor_inverse_degree"])
        )
    if "mean_triangle_weighted_clustering_abs_error" in requested:
        distances["mean_triangle_weighted_clustering_abs_error"] = (
            absolute_mean_difference(
                current["triangle_degree_weighted_clustering"],
                original["triangle_degree_weighted_clustering"],
            )
        )
    if "normalized_estrada_index_abs_error" in requested:
        distances["normalized_estrada_index_abs_error"] = abs(
            float(current["normalized_estrada_index"])
            - float(original["normalized_estrada_index"])
        )
    return distances


