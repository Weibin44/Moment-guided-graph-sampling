"""Random-edge and random-walk deletion baselines."""

from __future__ import annotations

import numpy as np

from mggs.moments.topology_delta import topology_delete_deltas
from mggs.sampling.moment_preserving import deletion_delta_m4_for_edge


def choose_random_edge(state, rng: np.random.Generator, *, track_m4: bool = True):
    index = int(rng.integers(len(state.edge_list)))
    u, v = state.edge_list[index]
    dm2, dm3 = topology_delete_deltas(state, np.asarray([[u, v]], dtype=np.int64))
    dm4 = deletion_delta_m4_for_edge(state, u, v) if track_m4 else 0.0
    return int(u), int(v), float(dm2[0]), float(dm3[0]), dm4


def choose_random_walk_edge(
    state,
    rng: np.random.Generator,
    walker: dict[str, int | None],
    *,
    track_m4: bool = True,
):
    current = walker.get("node")
    if current is None or len(state.nbrs[int(current)]) == 0:
        current = int(rng.choice(np.flatnonzero(state.deg > 0)))
    neighbors = tuple(state.nbrs[int(current)])
    nxt = int(neighbors[int(rng.integers(len(neighbors)))])
    walker["node"] = nxt
    u, v = int(current), nxt
    if u > v:
        u, v = v, u
    dm2, dm3 = topology_delete_deltas(state, np.asarray([[u, v]], dtype=np.int64))
    dm4 = deletion_delta_m4_for_edge(state, u, v) if track_m4 else 0.0
    return u, v, float(dm2[0]), float(dm3[0]), dm4
