"""Conversions between canonical undirected edges and PyG edge indices."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def undirected_simple_edges(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    """Return sorted canonical edges from a possibly bidirectional edge index."""

    return sorted({
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in edge_index.cpu().numpy().T
        if int(u) != int(v)
    })


def edge_index_from_edges(
    edges: Iterable[tuple[int, int]], num_nodes: int | None = None
) -> torch.Tensor:
    """Build a bidirectional PyG edge index from undirected edges."""

    del num_nodes
    return undirected_edges_to_edge_index(edges)


def undirected_edges_to_edge_index(edges: Iterable[tuple[int, int]]) -> torch.Tensor:
    edge_list = [tuple(edge) for edge in edges]
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    arr = np.asarray(sorted(edge_list), dtype=np.int64)
    rows = np.concatenate([arr[:, 0], arr[:, 1]])
    cols = np.concatenate([arr[:, 1], arr[:, 0]])
    return torch.from_numpy(np.stack([rows, cols], axis=0)).long()


def validate_num_nodes(num_nodes: int) -> int:
    """Validate the fixed graph size, including isolated nodes."""
    if isinstance(num_nodes, (bool, np.bool_)) or not isinstance(num_nodes, (int, np.integer)) or num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer")
    return int(num_nodes)


def canonical_edges(edges, num_nodes: int, *, strict=True, sort=True) -> np.ndarray:
    """Validate simple undirected (E, 2) input and return a sorted copy.

    Unlike edge_index conversion, public moment/sampling inputs do not silently
    remove loops, duplicates, or truncate noninteger node identifiers.
    Internal compatibility adapters can retain historical casting and row order.
    """
    n = validate_num_nodes(num_nodes) if strict else int(num_nodes)
    array = np.asarray(edges if isinstance(edges, np.ndarray) else list(edges))
    if array.size == 0:
        if strict and array.shape not in ((0,), (0, 2)):
            raise ValueError("edges must have shape (E, 2)")
        return np.empty((0, 2), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("edges must have shape (E, 2)")
    if strict and array.dtype.kind not in "iu":
        raise ValueError("edge endpoints must be integers")
    array = array.astype(np.int64, copy=True)
    if np.any(array < 0) or np.any(array >= n):
        raise ValueError("edge endpoint is outside [0, num_nodes)")
    array.sort(axis=1)
    if np.any(array[:, 0] == array[:, 1]):
        raise ValueError("self-loops are not supported")
    unique = np.unique(array, axis=0)
    if len(unique) != len(array):
        raise ValueError("duplicate undirected edges are not supported")
    return unique if sort else array
