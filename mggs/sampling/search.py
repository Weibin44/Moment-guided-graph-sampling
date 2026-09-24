"""Persistent candidate search; heap ordering is an internal implementation detail."""
from __future__ import annotations
import heapq
import numpy as np
from mggs.moments._queries import _backend, deletion_deltas_from_state, addition_deltas_from_state
from .target import target_distances
from .updates import apply_moment_edit

EPS = 1e-12


class DeletionTrajectory:
    """One persistent deletion trajectory, with interchangeable exact backends.

    ``greedy`` scores every remaining edge; ``heap`` refreshes only the cached
    top-k and deletes one edge. Heap scores are deliberately stale after edits,
    as in the original heuristic. Calling ``step`` across checkpoints retains
    the heap. Original edge indices also provide deterministic tie-breaking.
    """

    def __init__(self, state, reference, normalizer, *, orders=(2, 3),
                 backend="hybrid", strategy="greedy", top_k=25, seed=0,
                 initial_deltas=None):
        if strategy not in {"greedy", "heap", "random"}:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if backend != "direct_trace":
            backend = _backend(backend)
        self.orders = tuple(sorted(set(int(k) for k in orders)))
        if strategy != "random" and (not self.orders or min(self.orders) < 2):
            raise ValueError("Moment orders must be nonempty and at least 2")
        if backend == "topology" and any(k not in (2, 3) for k in self.orders):
            raise ValueError("Topology supports only m2/m3; use low_rank for higher orders")
        self.state, self.reference, self.normalizer = state, reference, normalizer
        self.backend, self.strategy, self.top_k = backend, strategy, int(top_k)
        self.base_edges = np.asarray(state.edge_list, dtype=np.int64).reshape(-1, 2)
        self.active = np.ones(len(self.base_edges), dtype=bool)
        self.remaining = len(self.base_edges)
        self.versions = np.zeros(self.remaining, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.edge_indices = ({tuple(edge): i for i, edge in enumerate(self.base_edges)}
                             if strategy == "random" else None)
        self.heap = []
        if strategy != "random":
            for k in self.orders:
                if not np.isfinite(normalizer[f"m{k}"]) or normalizer[f"m{k}"] <= 0:
                    raise ValueError("Normalizers must be finite and positive")
        if strategy == "heap" and self.remaining:
            scores, _ = self._score(self.base_edges, initial_deltas)
            self.heap = [(float(score), 0, i) for i, score in enumerate(scores)]
            heapq.heapify(self.heap)

    @classmethod
    def from_edges(cls, edges, num_nodes, *, orders=(2, 3), backend="hybrid",
                   strategy="greedy", top_k=25, normalization="graph_relative", seed=0):
        """All sampling preprocessing; callers can time this separately."""
        from mggs.moments import TopologyMomentState, exact_m2_m3_from_state, exact_moments_m1_to_m4
        from mggs.io.graph import edge_index_from_edges

        orders = tuple(sorted(set(int(k) for k in orders)))
        if not orders or any(k not in (2, 3, 4) for k in orders):
            raise ValueError("Trajectory initialization currently supports orders 2, 3 and 4")
        if normalization not in {"std", "graph_relative"}:
            raise ValueError(f"Unknown normalization: {normalization}")
        state = TopologyMomentState(None, int(num_nodes), canonical_edges=edges)
        reference, scales, initial = {}, {}, None
        if strategy != "random":
            state.m2, state.m3 = exact_m2_m3_from_state(state)
            if 4 in orders:
                edge_index = edge_index_from_edges(state.edge_list, state.n)
                state.m4 = exact_moments_m1_to_m4(edge_index, state.n)[3]
            reference = {f"m{k}": float(getattr(state, f"m{k}")) for k in orders}
            if normalization == "std":
                initial = deletion_deltas_from_state(
                    state, np.asarray(state.edge_list, dtype=np.int64).reshape(-1, 2),
                    orders=orders, backend=backend,
                )
            for k in orders:
                scale = float(np.std(initial[k])) if initial is not None else abs(reference[f"m{k}"])
                scales[f"m{k}"] = scale if scale > EPS else 1.0
        return cls(state, reference, scales, orders=orders, backend=backend,
                   strategy=strategy, top_k=top_k, seed=seed, initial_deltas=initial)

    def _score(self, edges, deltas=None):
        if deltas is None:
            deltas = deletion_deltas_from_state(self.state, edges, orders=self.orders, backend=self.backend)
        scores = target_distances(
            {k: float(getattr(self.state, f"m{k}")) for k in self.orders}, deltas,
            {k: self.reference[f"m{k}"] for k in self.orders},
            {k: self.normalizer[f"m{k}"] for k in self.orders},
        )
        return scores, deltas

    def step(self, *, candidate_batch_size: int = 0) -> int:
        """Delete one edge; optionally subsample greedy candidates (0 means all)."""
        if candidate_batch_size < 0:
            raise ValueError("candidate_batch_size must be nonnegative")
        if candidate_batch_size and self.strategy != "greedy":
            raise ValueError("Candidate subsampling applies only to greedy sampling")
        if not self.remaining:
            raise StopIteration("No remaining edges")
        if self.strategy == "random":
            u, v = self.state.edge_list[int(self.rng.integers(self.remaining))]
            chosen = self.edge_indices[(u, v)]
            self.state.apply_del(u, v)
            self.active[chosen] = False
            self.remaining -= 1
            # Random deletion does not need moment bookkeeping or a heap.
            return chosen
        if self.strategy == "heap":
            candidates = []
            while len(candidates) < min(self.top_k, self.remaining):
                _, version, index = heapq.heappop(self.heap)
                if self.active[index] and self.versions[index] == version:
                    candidates.append(index)
            candidates = np.asarray(candidates, dtype=np.int64)
        else:
            candidates = np.flatnonzero(self.active)
            if candidate_batch_size and candidate_batch_size < len(candidates):
                candidates = self.rng.choice(candidates, size=candidate_batch_size, replace=False)
        scores, deltas = self._score(self.base_edges[candidates])
        # Score first, original index second, independent of candidate pool order.
        tied = np.flatnonzero(scores == np.min(scores))
        position = int(tied[np.argmin(candidates[tied])])
        chosen = int(candidates[position])
        apply_moment_edit(self.state, "delete", *self.base_edges[chosen],
                              {f"m{k}": float(deltas[k][position]) for k in self.orders})
        self.active[chosen] = False
        self.remaining -= 1
        if self.strategy == "heap":
            for j, index in enumerate(candidates):
                if index != chosen:
                    self.versions[index] += 1
                    heapq.heappush(self.heap, (float(scores[j]), int(self.versions[index]), int(index)))
        return chosen




class GreedyEditSearch:
    """Addition/mixed search with bounded batches and no reversal of edits."""

    def __init__(self, state, original_edges, *, target, scales, backend,
                 operation, candidate_limit, seed):
        self.state = state
        self.original = set(map(tuple, original_edges))
        self.edited = set()
        self.target, self.scales = target, scales
        self.backend, self.operation = backend, operation
        self.limit = candidate_limit
        self.rng = np.random.default_rng(seed)

    def _non_edges(self):
        for u in range(self.state.n):
            for v in range(u + 1, self.state.n):
                edge = (u, v)
                if edge not in self.original and edge not in self.edited:
                    yield edge

    def _addition_candidates(self):
        if self.limit is None:
            yield from self._non_edges()
            return
        total = self.state.n * (self.state.n - 1) // 2
        remaining = total - len(self.original) - sum(e not in self.original for e in self.edited)
        count = min(self.limit, remaining)
        if count == remaining:
            yield from self._non_edges()
            return
        # Rejection is cheap for sparse graphs; a uniform reservoir handles dense
        # graphs without allocating a quadratic non-edge array.
        if remaining < total // 4:
            pool = []
            for index, edge in enumerate(self._non_edges()):
                if index < count:
                    pool.append(edge)
                else:
                    replace = int(self.rng.integers(index + 1))
                    if replace < count:
                        pool[replace] = edge
            yield from sorted(pool)
            return
        chosen = set()
        while len(chosen) < count:
            u, v = sorted(map(int, self.rng.integers(self.state.n, size=2)))
            edge = (u, v)
            if u != v and edge not in self.original and edge not in self.edited:
                chosen.add(edge)
        yield from sorted(chosen)

    def _batches(self, operation, batch_size=2048):
        if operation == "add":
            candidates = self._addition_candidates()
        else:
            candidates = sorted(self.original - self.edited)
            if self.limit is not None and len(candidates) > self.limit:
                indices = self.rng.choice(len(candidates), self.limit, replace=False)
                candidates = sorted(candidates[i] for i in indices)
        batch = []
        for edge in candidates:
            batch.append(edge)
            if len(batch) == batch_size:
                yield np.asarray(batch, dtype=np.int64)
                batch = []
        if batch:
            yield np.asarray(batch, dtype=np.int64)

    def step(self):
        from .types import EdgeEdit

        best = None
        operations = ("add", "delete") if self.operation == "mixed" else (self.operation,)
        current = {k: float(getattr(self.state, f"m{k}")) for k in self.target}
        for operation in operations:
            delta_fn = addition_deltas_from_state if operation == "add" else deletion_deltas_from_state
            for edges in self._batches(operation):
                deltas = delta_fn(self.state, edges, orders=tuple(self.target), backend=self.backend)
                scores = target_distances(current, deltas, self.target, self.scales)
                index = int(np.argmin(scores))
                u, v = map(int, edges[index])
                # Stable across batches: additions first, then canonical edge order.
                key = (float(scores[index]), operations.index(operation), u, v)
                if best is None or key < best[0]:
                    best = (key, EdgeEdit(operation, u, v), {
                        f"m{k}": float(deltas[k][index]) for k in self.target
                    })
        if best is None:
            raise RuntimeError("No eligible edit remains")
        _, edit, deltas = best
        apply_moment_edit(self.state, edit.operation, edit.u, edit.v, deltas)
        self.edited.add((edit.u, edit.v))
        return edit
