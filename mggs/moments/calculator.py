"""Read-only, cached moment changes for independent or joint edge edits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..io.graph import canonical_edges, edge_index_from_edges, validate_num_nodes
from ._queries import addition_deltas_from_state, deletion_deltas_from_state
from .low_rank_delta import low_rank_delta_moments_for_joint_edits
from .orders import normalize_orders, validate_backend
from .topology_state import TopologyMomentState


@dataclass(frozen=True)
class CandidateDeltaMoments:
    """Exact per-candidate moment changes.

    ``edges[i]`` corresponds to ``values[k][i]`` for every requested order
    ``k``.  Edges are stored canonically with the smaller endpoint first.
    """

    edges: np.ndarray
    values: dict[int, np.ndarray]

    def __getitem__(self, order: int) -> np.ndarray:
        return self.values[int(order)]

    @property
    def orders(self) -> tuple[int, ...]:
        return tuple(sorted(self.values))



class MomentDeltaCalculator:
    """Reusable exact delta calculator; queries never apply graph edits."""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int):
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_directed_edges].")
        self.state = TopologyMomentState(edge_index.long(), int(num_nodes))

    @classmethod
    def from_undirected_edges(cls, edges, num_nodes: int) -> "MomentDeltaCalculator":
        n = validate_num_nodes(num_nodes)
        return cls(edge_index_from_edges(canonical_edges(edges, n)), n)

    @property
    def num_nodes(self) -> int:
        return int(self.state.n)

    @property
    def edges(self) -> np.ndarray:
        return np.asarray(self.state.edge_list, dtype=np.int64).reshape(-1, 2)

    def deletion_deltas(
        self, candidate_edges=None, *, orders=(2, 3), backend="low_rank",
    ) -> CandidateDeltaMoments:
        """Return each existing edge's independent deletion effect."""
        orders = normalize_orders(orders, minimum=2)
        validate_backend(backend, orders)
        edges = self.edges if candidate_edges is None else canonical_edges(candidate_edges, self.num_nodes)
        missing = [tuple(edge) for edge in edges if tuple(edge) not in self.state.edge_to_idx]
        if missing:
            raise ValueError(f"deletion candidates are not current edges: {missing[:3]}")
        values = deletion_deltas_from_state(self.state, edges, orders=orders, backend=backend)
        return CandidateDeltaMoments(edges=edges, values=values)

    def addition_deltas(
        self, candidate_edges, *, orders=(2, 3), backend="low_rank",
    ) -> CandidateDeltaMoments:
        """Return each absent edge's independent addition effect."""
        orders = normalize_orders(orders, minimum=2)
        validate_backend(backend, orders)
        edges = canonical_edges(candidate_edges, self.num_nodes)
        existing = [tuple(edge) for edge in edges if tuple(edge) in self.state.edge_to_idx]
        if existing:
            raise ValueError(f"addition candidates already exist: {existing[:3]}")
        values = addition_deltas_from_state(self.state, edges, orders=orders, backend=backend)
        return CandidateDeltaMoments(edges=edges, values=values)

    def batch_edit_delta(
        self, *, additions=None, deletions=None, orders=(2, 3),
    ) -> dict[int, float]:
        """Return the combined effect, not a sum of independent deltas."""
        orders = normalize_orders(orders, minimum=2)
        additions = None if additions is None else canonical_edges(additions, self.num_nodes)
        deletions = None if deletions is None else canonical_edges(deletions, self.num_nodes)
        values = low_rank_delta_moments_for_joint_edits(
            self.state, additions=additions, deletions=deletions,
            min_order=min(orders), max_order=max(orders), validate=True,
        )
        return {k: values[k] for k in orders}
