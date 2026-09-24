"""Effective-resistance graph sparsification utilities.

The leverage score used here is ``w_e R_e``.  The module keeps both the
Spielman--Srivastava with-replacement reweighting sampler and a fixed-size
without-replacement edge subset sampler for fair same-edge-budget baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


EPS = 1e-15


@dataclass(frozen=True)
class EffectiveResistanceEstimate:
    resistances: np.ndarray
    leverage_scores: np.ndarray
    probabilities: np.ndarray
    leverage_sum: float
    method: str
    epsilon: float | None = None
    jl_dim: int | None = None


@dataclass(frozen=True)
class SparsifierSample:
    edges: np.ndarray
    weights: np.ndarray
    sample_budget: int
    unique_edges: int
    counts: np.ndarray
    with_replacement: bool = True


def canonical_edge_array(edges: Iterable[tuple[int, int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(edges) if not isinstance(edges, np.ndarray) else edges, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    arr = arr.reshape(-1, 2)
    out = np.column_stack([np.minimum(arr[:, 0], arr[:, 1]), np.maximum(arr[:, 0], arr[:, 1])])
    return out[out[:, 0] != out[:, 1]]


def _edge_weights(num_edges: int, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.ones(num_edges, dtype=np.float64)
    out = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(out) != num_edges:
        raise ValueError(f"Expected {num_edges} edge weights, got {len(out)}.")
    if np.any(out <= 0.0):
        raise ValueError("Effective-resistance sparsification requires positive edge weights.")
    return out


def weighted_laplacian_dense(
    num_nodes: int,
    edges: Iterable[tuple[int, int]] | np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    edge_arr = canonical_edge_array(edges)
    w = _edge_weights(len(edge_arr), weights)
    laplacian = np.zeros((int(num_nodes), int(num_nodes)), dtype=np.float64)
    for (u, v), weight in zip(edge_arr, w):
        laplacian[u, u] += weight
        laplacian[v, v] += weight
        laplacian[u, v] -= weight
        laplacian[v, u] -= weight
    return laplacian


def normalized_laplacian_dense(
    num_nodes: int,
    edges: Iterable[tuple[int, int]] | np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``L_norm = D^{-1/2} L D^{-1/2}``, degrees, and ``D^{-1/2}``."""
    edge_arr = canonical_edge_array(edges)
    w = _edge_weights(len(edge_arr), weights)
    degree = np.zeros(int(num_nodes), dtype=np.float64)
    for (u, v), weight in zip(edge_arr, w):
        degree[u] += weight
        degree[v] += weight
    inv_sqrt_degree = np.zeros(int(num_nodes), dtype=np.float64)
    positive = degree > 0.0
    inv_sqrt_degree[positive] = 1.0 / np.sqrt(degree[positive])

    laplacian = weighted_laplacian_dense(num_nodes, edge_arr, w)
    normalized = laplacian * inv_sqrt_degree[:, None] * inv_sqrt_degree[None, :]
    return normalized, degree, inv_sqrt_degree


def connected_components(
    num_nodes: int,
    edges: Iterable[tuple[int, int]] | np.ndarray,
) -> list[np.ndarray]:
    edge_arr = canonical_edge_array(edges)
    parent = np.arange(int(num_nodes), dtype=np.int64)
    rank = np.zeros(int(num_nodes), dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for u, v in edge_arr:
        union(int(u), int(v))

    groups: dict[int, list[int]] = {}
    for node in range(int(num_nodes)):
        groups.setdefault(find(node), []).append(node)
    return [np.asarray(nodes, dtype=np.int64) for nodes in groups.values()]


def exact_edge_effective_resistances(
    num_nodes: int,
    edges: Iterable[tuple[int, int]] | np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute exact edge effective resistances with the Laplacian pseudoinverse."""
    edge_arr = canonical_edge_array(edges)
    if len(edge_arr) == 0:
        return np.empty(0, dtype=np.float64)
    laplacian = weighted_laplacian_dense(num_nodes, edge_arr, weights)
    laplacian_pinv = np.linalg.pinv(laplacian, hermitian=True)
    diag = np.diag(laplacian_pinv)
    u = edge_arr[:, 0]
    v = edge_arr[:, 1]
    resistances = diag[u] + diag[v] - 2.0 * laplacian_pinv[u, v]
    return np.maximum(resistances, 0.0)


def exact_static_normalized_edge_resistances(
    num_nodes: int,
    edges: Iterable[tuple[int, int]] | np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute static normalized edge resistances on the original graph.

    For edge ``e=(u,v)``, the normalized edge vector is
    ``a_e = e_u/sqrt(d_u) - e_v/sqrt(d_v)`` using original degrees.  The
    leverage score is ``w_e * a_e^T L_norm^+ a_e``.
    """
    edge_arr = canonical_edge_array(edges)
    if len(edge_arr) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    w = _edge_weights(len(edge_arr), weights)
    normalized_laplacian, _, inv_sqrt_degree = normalized_laplacian_dense(
        num_nodes, edge_arr, w
    )
    laplacian_pinv = np.linalg.pinv(normalized_laplacian, hermitian=True)
    diag = np.diag(laplacian_pinv)
    u = edge_arr[:, 0]
    v = edge_arr[:, 1]
    resistances = (
        (inv_sqrt_degree[u] ** 2) * diag[u]
        + (inv_sqrt_degree[v] ** 2) * diag[v]
        - 2.0 * inv_sqrt_degree[u] * inv_sqrt_degree[v] * laplacian_pinv[u, v]
    )
    resistances = np.maximum(resistances, 0.0)
    leverage_scores = np.maximum(w * resistances, 0.0)
    return resistances, leverage_scores


def _solve_laplacian_zero_reference(
    laplacian: np.ndarray,
    rhs: np.ndarray,
    components: list[np.ndarray],
) -> np.ndarray:
    """Solve ``L x = rhs`` up to component-wise constants.

    The returned potentials use one zero-reference vertex per connected
    component.  Edge voltage differences are invariant to this choice.
    """
    rhs = np.asarray(rhs, dtype=np.float64)
    if rhs.ndim == 1:
        rhs = rhs.reshape(1, -1)
    out = np.zeros_like(rhs, dtype=np.float64)

    for nodes in components:
        if len(nodes) <= 1:
            continue
        free = nodes[:-1]
        sub_laplacian = laplacian[np.ix_(free, free)]
        sub_rhs = rhs[:, free].T
        try:
            solution = np.linalg.solve(sub_laplacian, sub_rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(sub_laplacian, sub_rhs, rcond=None)[0]
        out[:, free] = solution.T
    return out


def jl_dimension(num_nodes: int, epsilon: float) -> int:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    return max(1, int(math.ceil(24.0 * math.log(max(int(num_nodes), 2)) / (epsilon ** 2))))


def approximate_edge_effective_resistances_jl(
    num_nodes: int,
    edges: Iterable[tuple[int, int]] | np.ndarray,
    weights: np.ndarray | None = None,
    *,
    epsilon: float = 0.5,
    jl_dim: int | None = None,
    seed: int = 0,
    projection: str = "rademacher",
) -> np.ndarray:
    """Approximate edge effective resistances using JL-projected Laplacian solves.

    This implements the practical version of the Spielman--Srivastava sketch:
    form ``Q W^{1/2} B``, solve the projected right-hand sides against the
    Laplacian, then estimate ``R_uv`` by squared projected voltage differences.
    Dense direct solves are used because the citation/WebKB graphs in this
    workspace are small; this isolates the JL approximation from solver error.
    """
    edge_arr = canonical_edge_array(edges)
    m = len(edge_arr)
    if m == 0:
        return np.empty(0, dtype=np.float64)
    w = _edge_weights(m, weights)
    k = int(jl_dim) if jl_dim is not None else jl_dimension(num_nodes, epsilon)
    if k <= 0:
        raise ValueError("jl_dim must be positive.")

    rng = np.random.default_rng(seed)
    if projection == "rademacher":
        q = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=(k, m))
    elif projection == "gaussian":
        q = rng.normal(size=(k, m))
    else:
        raise ValueError(f"Unknown JL projection: {projection}")
    q /= math.sqrt(float(k))

    incidence = np.zeros((m, int(num_nodes)), dtype=np.float64)
    sqrt_w = np.sqrt(w)
    incidence[np.arange(m), edge_arr[:, 0]] = sqrt_w
    incidence[np.arange(m), edge_arr[:, 1]] = -sqrt_w
    projected_rhs = q @ incidence

    laplacian = weighted_laplacian_dense(num_nodes, edge_arr, w)
    components = connected_components(num_nodes, edge_arr)
    potentials = _solve_laplacian_zero_reference(laplacian, projected_rhs, components)

    diff = potentials[:, edge_arr[:, 0]] - potentials[:, edge_arr[:, 1]]
    resistances = np.sum(diff * diff, axis=0)
    return np.maximum(resistances, 0.0)


def approximate_static_normalized_edge_resistances_jl(
    num_nodes: int,
    edges: Iterable[tuple[int, int]] | np.ndarray,
    weights: np.ndarray | None = None,
    *,
    epsilon: float = 0.5,
    jl_dim: int | None = None,
    seed: int = 0,
    projection: str = "rademacher",
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate static normalized edge resistances via JL-projected solves."""
    edge_arr = canonical_edge_array(edges)
    m = len(edge_arr)
    if m == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    w = _edge_weights(m, weights)
    k = int(jl_dim) if jl_dim is not None else jl_dimension(num_nodes, epsilon)
    if k <= 0:
        raise ValueError("jl_dim must be positive.")

    normalized_laplacian, _, inv_sqrt_degree = normalized_laplacian_dense(
        num_nodes, edge_arr, w
    )

    rng = np.random.default_rng(seed)
    if projection == "rademacher":
        q = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=(k, m))
    elif projection == "gaussian":
        q = rng.normal(size=(k, m))
    else:
        raise ValueError(f"Unknown JL projection: {projection}")
    q /= math.sqrt(float(k))

    incidence = np.zeros((m, int(num_nodes)), dtype=np.float64)
    sqrt_w = np.sqrt(w)
    incidence[np.arange(m), edge_arr[:, 0]] = sqrt_w * inv_sqrt_degree[edge_arr[:, 0]]
    incidence[np.arange(m), edge_arr[:, 1]] = -sqrt_w * inv_sqrt_degree[edge_arr[:, 1]]
    projected_rhs = q @ incidence

    components = connected_components(num_nodes, edge_arr)
    potentials = _solve_laplacian_zero_reference(
        normalized_laplacian, projected_rhs, components
    )

    diff = (
        potentials[:, edge_arr[:, 0]] * inv_sqrt_degree[edge_arr[:, 0]]
        - potentials[:, edge_arr[:, 1]] * inv_sqrt_degree[edge_arr[:, 1]]
    )
    resistances = np.maximum(np.sum(diff * diff, axis=0), 0.0)
    leverage_scores = np.maximum(w * resistances, 0.0)
    return resistances, leverage_scores


def estimate_effective_resistances(
    num_nodes: int,
    edges: Iterable[tuple[int, int]] | np.ndarray,
    weights: np.ndarray | None = None,
    *,
    method: str = "exact",
    epsilon: float = 0.5,
    jl_dim: int | None = None,
    seed: int = 0,
    projection: str = "rademacher",
) -> EffectiveResistanceEstimate:
    edge_arr = canonical_edge_array(edges)
    w = _edge_weights(len(edge_arr), weights)
    if method == "exact":
        resistances = exact_edge_effective_resistances(num_nodes, edge_arr, w)
        used_dim = None
    elif method == "approx":
        resistances = approximate_edge_effective_resistances_jl(
            num_nodes,
            edge_arr,
            w,
            epsilon=epsilon,
            jl_dim=jl_dim,
            seed=seed,
            projection=projection,
        )
        used_dim = int(jl_dim) if jl_dim is not None else jl_dimension(num_nodes, epsilon)
    else:
        raise ValueError(f"Unknown effective-resistance method: {method}")
    probabilities, leverage_scores, leverage_sum = effective_resistance_probabilities(w, resistances)
    return EffectiveResistanceEstimate(
        resistances=resistances,
        leverage_scores=leverage_scores,
        probabilities=probabilities,
        leverage_sum=leverage_sum,
        method=method,
        epsilon=float(epsilon) if method == "approx" else None,
        jl_dim=used_dim,
    )


def estimate_static_normalized_effective_resistances(
    num_nodes: int,
    edges: Iterable[tuple[int, int]] | np.ndarray,
    weights: np.ndarray | None = None,
    *,
    method: str = "exact",
    epsilon: float = 0.5,
    jl_dim: int | None = None,
    seed: int = 0,
    projection: str = "rademacher",
) -> EffectiveResistanceEstimate:
    """Estimate leverage scores for the original normalized-Laplacian geometry."""
    edge_arr = canonical_edge_array(edges)
    w = _edge_weights(len(edge_arr), weights)
    if method == "exact":
        resistances, leverage_scores = exact_static_normalized_edge_resistances(
            num_nodes, edge_arr, w
        )
        used_dim = None
    elif method == "approx":
        resistances, leverage_scores = approximate_static_normalized_edge_resistances_jl(
            num_nodes,
            edge_arr,
            w,
            epsilon=epsilon,
            jl_dim=jl_dim,
            seed=seed,
            projection=projection,
        )
        used_dim = int(jl_dim) if jl_dim is not None else jl_dimension(num_nodes, epsilon)
    else:
        raise ValueError(f"Unknown static normalized ER method: {method}")
    probabilities, leverage_scores, leverage_sum = effective_resistance_probabilities(
        np.ones_like(leverage_scores), leverage_scores
    )
    return EffectiveResistanceEstimate(
        resistances=resistances,
        leverage_scores=leverage_scores,
        probabilities=probabilities,
        leverage_sum=leverage_sum,
        method=f"static_normalized_{method}",
        epsilon=float(epsilon) if method == "approx" else None,
        jl_dim=used_dim,
    )


def effective_resistance_probabilities(
    weights: np.ndarray,
    resistances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    resistances = np.asarray(resistances, dtype=np.float64).reshape(-1)
    if len(weights) != len(resistances):
        raise ValueError("weights and resistances must have the same length.")
    if len(weights) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), 0.0
    leverage_scores = np.maximum(weights * resistances, 0.0)
    leverage_sum = float(np.sum(leverage_scores))
    if leverage_sum <= EPS or not np.isfinite(leverage_sum):
        probabilities = np.full(len(weights), 1.0 / float(len(weights)), dtype=np.float64)
    else:
        probabilities = leverage_scores / leverage_sum
    return probabilities, leverage_scores, leverage_sum


def sample_sparsifier_from_probabilities(
    edges: Iterable[tuple[int, int]] | np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray,
    sample_budget: int,
    *,
    rng: np.random.Generator,
) -> SparsifierSample:
    edge_arr = canonical_edge_array(edges)
    w = _edge_weights(len(edge_arr), weights)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(p) != len(edge_arr):
        raise ValueError("probabilities must have one entry per edge.")
    q = int(sample_budget)
    if q <= 0 or len(edge_arr) == 0:
        return SparsifierSample(
            edges=np.empty((0, 2), dtype=np.int64),
            weights=np.empty(0, dtype=np.float64),
            sample_budget=max(q, 0),
            unique_edges=0,
            counts=np.zeros(len(edge_arr), dtype=np.int64),
            with_replacement=True,
        )

    p = np.maximum(p, 0.0)
    p_sum = float(np.sum(p))
    if p_sum <= EPS:
        p = np.full(len(edge_arr), 1.0 / float(len(edge_arr)), dtype=np.float64)
    else:
        p = p / p_sum
    sampled = rng.choice(len(edge_arr), size=q, replace=True, p=p)
    counts = np.bincount(sampled, minlength=len(edge_arr)).astype(np.int64)
    mask = counts > 0
    sampled_weights = counts[mask].astype(np.float64) * (w[mask] / (float(q) * p[mask]))
    return SparsifierSample(
        edges=edge_arr[mask],
        weights=sampled_weights,
        sample_budget=q,
        unique_edges=int(np.count_nonzero(mask)),
        counts=counts,
        with_replacement=True,
    )


def sample_edge_subset_without_replacement(
    edges: Iterable[tuple[int, int]] | np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray,
    keep_count: int,
    *,
    rng: np.random.Generator,
    reweight: bool = False,
) -> SparsifierSample:
    """Sample exactly ``keep_count`` distinct edges using leverage probabilities.

    By default the selected edges keep their original weights.  This is the
    fair edge-budget baseline against unweighted deletion methods.  Setting
    ``reweight=True`` applies the simple with-replacement scale
    ``w_e / (q p_e)`` to selected edges; that is useful as a diagnostic but is
    not an unbiased fixed-size PPS-without-replacement estimator.
    """
    edge_arr = canonical_edge_array(edges)
    w = _edge_weights(len(edge_arr), weights)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(p) != len(edge_arr):
        raise ValueError("probabilities must have one entry per edge.")
    q = int(keep_count)
    if q <= 0 or len(edge_arr) == 0:
        return SparsifierSample(
            edges=np.empty((0, 2), dtype=np.int64),
            weights=np.empty(0, dtype=np.float64),
            sample_budget=max(q, 0),
            unique_edges=0,
            counts=np.zeros(len(edge_arr), dtype=np.int64),
            with_replacement=False,
        )
    if q >= len(edge_arr):
        return SparsifierSample(
            edges=edge_arr.copy(),
            weights=w.copy(),
            sample_budget=len(edge_arr),
            unique_edges=len(edge_arr),
            counts=np.ones(len(edge_arr), dtype=np.int64),
            with_replacement=False,
        )

    p = np.maximum(p, 0.0)
    p_sum = float(np.sum(p))
    if p_sum <= EPS:
        p = np.full(len(edge_arr), 1.0 / float(len(edge_arr)), dtype=np.float64)
    else:
        p = p / p_sum
    if int(np.count_nonzero(p > 0.0)) < q:
        p = p + EPS
        p = p / float(np.sum(p))

    sampled = rng.choice(len(edge_arr), size=q, replace=False, p=p)
    sampled.sort()
    counts = np.zeros(len(edge_arr), dtype=np.int64)
    counts[sampled] = 1
    sampled_weights = w[sampled].copy()
    if reweight:
        sampled_weights = sampled_weights / (float(q) * p[sampled])
    return SparsifierSample(
        edges=edge_arr[sampled],
        weights=sampled_weights,
        sample_budget=q,
        unique_edges=q,
        counts=counts,
        with_replacement=False,
    )
