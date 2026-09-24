"""
Persistent augmentation state for moments-guided graph augmentation.

TopologyMomentState maintains graph structure (edges, degrees, W, H, M) incrementally
so that each add/delete operation is O(d) rather than O(N²).

M[i,j] = Σ_{w ∈ N(i)∩N(j)} 1/d_w  is kept dense for N≤5000 and updated
incrementally; for larger graphs it is recomputed on demand via _inter_sum.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix, diags as sp_diags


# ── numba-accelerated two-pointer intersection sum ────────────────────────────
try:
    import numba as _nb

    @_nb.njit(cache=True)
    def _inter_sum(a, b, inv_deg):
        """Σ_{k ∈ sorted(a) ∩ sorted(b)} inv_deg[k]  (two-pointer, O(|a|+|b|))"""
        s = 0.0; i = j = 0
        while i < len(a) and j < len(b):
            if   a[i] == b[j]: s += inv_deg[a[i]]; i += 1; j += 1
            elif a[i] <  b[j]: i += 1
            else:               j += 1
        return s

    @_nb.njit(cache=True)
    def _inter_sum_excl(a, b, inv_deg, excl):
        """Same but skip k == excl"""
        s = 0.0; i = j = 0
        while i < len(a) and j < len(b):
            if   a[i] == b[j]:
                if a[i] != excl: s += inv_deg[a[i]]
                i += 1; j += 1
            elif a[i] < b[j]:  i += 1
            else:               j += 1
        return s

    @_nb.njit(cache=True)
    def _large_graph_h_and_edge_m(edges, indptr, indices, edge_ids, inv_deg):
        """Compute H/M_uv by enumerating each degree-oriented triangle once."""
        h = np.zeros(len(inv_deg), dtype=np.float64)
        edge_m = np.zeros(len(edges), dtype=np.float64)
        for u in range(len(inv_deg)):
            for uv_pos in range(indptr[u], indptr[u + 1]):
                v = indices[uv_pos]
                uv_edge = edge_ids[uv_pos]
                i, i_end = indptr[u], indptr[u + 1]
                j, j_end = indptr[v], indptr[v + 1]
                while i < i_end and j < j_end:
                    if indices[i] == indices[j]:
                        w = indices[i]
                        uw_edge = edge_ids[i]
                        vw_edge = edge_ids[j]
                        edge_m[uv_edge] += inv_deg[w]
                        edge_m[uw_edge] += inv_deg[v]
                        edge_m[vw_edge] += inv_deg[u]
                        h[u] += inv_deg[v] * inv_deg[w]
                        h[v] += inv_deg[u] * inv_deg[w]
                        h[w] += inv_deg[u] * inv_deg[v]
                        i += 1; j += 1
                    elif indices[i] < indices[j]:
                        i += 1
                    else:
                        j += 1
        return h, edge_m

    _HAS_NUMBA = True

except ImportError:
    _HAS_NUMBA = False

    def _inter_sum(a, b, inv_deg):
        return sum(inv_deg[k] for k in set(a.tolist()) & set(b.tolist()))

    def _inter_sum_excl(a, b, inv_deg, excl):
        return sum(inv_deg[k] for k in (set(a.tolist()) & set(b.tolist())) - {int(excl)})

    def _large_graph_h_and_edge_m(edges, indptr, indices, edge_ids, inv_deg):
        h = np.zeros(len(inv_deg), dtype=np.float64)
        edge_m = np.zeros(len(edges), dtype=np.float64)
        for u in range(len(inv_deg)):
            for uv_pos in range(indptr[u], indptr[u + 1]):
                v, uv_edge = indices[uv_pos], edge_ids[uv_pos]
                u_neighbors = indices[indptr[u]:indptr[u + 1]]
                v_neighbors = indices[indptr[v]:indptr[v + 1]]
                for w in set(u_neighbors) & set(v_neighbors):
                    uw_edge = edge_ids[indptr[u] + list(u_neighbors).index(w)]
                    vw_edge = edge_ids[indptr[v] + list(v_neighbors).index(w)]
                    edge_m[uv_edge] += inv_deg[w]
                    edge_m[uw_edge] += inv_deg[v]
                    edge_m[vw_edge] += inv_deg[u]
                    h[u] += inv_deg[v] * inv_deg[w]
                    h[v] += inv_deg[u] * inv_deg[w]
                    h[w] += inv_deg[u] * inv_deg[v]
        return h, edge_m


# ──────────────────────────────────────────────────────────────────────────────

class TopologyMomentState:
    """
    Incremental state supporting exact topology-based ``m2/m3`` deltas.

    Neighbor, degree, ``W`` and ``H`` caches persist across edits, avoiding a
    full Python graph rebuild for every candidate. ``H`` is updated only for
    locally affected nodes. The edge list uses swap-with-last deletion for
    constant-time structural removal, and callers can track ``m2/m3`` using
    the exact edit deltas instead of recomputing full moments after each edit.
    """

    def __init__(self, edge_index, num_nodes, *, canonical_edges=None):
        self.n      = num_nodes
        self.device = edge_index.device if edge_index is not None else torch.device("cpu")

        # Convert PyG's usually bidirectional representation into one sorted
        # (min_node, max_node) pair per edge. The set removes duplicate
        # directions and the condition removes self-loops, ensuring every
        # downstream cache describes the same simple undirected graph.
        if canonical_edges is not None:
            canonical_array = np.asarray(canonical_edges, dtype=np.int64).reshape(-1, 2)
            self.edge_list = [tuple(edge) for edge in canonical_array.tolist()]
        else:
            ei = edge_index.cpu().numpy()
        if canonical_edges is None and ei.shape[1] > 0:
            source, target = ei[0].astype(np.int64), ei[1].astype(np.int64)
            valid = source != target
            canonical_array = np.column_stack((
                np.minimum(source[valid], target[valid]),
                np.maximum(source[valid], target[valid]),
            ))
            canonical_array = np.unique(canonical_array, axis=0)
            self.edge_list = [tuple(edge) for edge in canonical_array.tolist()]
        elif canonical_edges is None:
            self.edge_list = []
        # Map each edge to its current list position. This supports constant-
        # time lookup and is updated when swap-with-last changes list order.
        self.edge_to_idx = {e: i for i, e in enumerate(self.edge_list)}

        # Keep two synchronized neighbor representations for different jobs:
        # sets provide average constant-time edge/membership checks, while
        # sorted int32 arrays allow linear two-pointer intersections in Numba
        # and compact CSR-style traversal without Python set overhead.
        self.nbrs = [set() for _ in range(num_nodes)]
        for u, v in self.edge_list:
            self.nbrs[u].add(v)
            self.nbrs[v].add(u)
        self.nbrs_arr = [
            np.array(sorted(self.nbrs[i]), dtype=np.int32) for i in range(num_nodes)
        ]

        # Cache degree, inverse degree, and W[i] = sum_{j in N(i)} 1 / d_j.
        # The safe denominator prevents division by zero for isolated nodes;
        # their inverse degree is then explicitly reset to zero.
        self.deg     = np.array([float(len(self.nbrs[i])) for i in range(num_nodes)])
        _safe        = np.where(self.deg > 0, self.deg, 1.0)
        self.inv_deg = np.where(self.deg > 0, 1.0 / _safe, 0.0)
        self.W       = np.zeros(num_nodes)
        for i in range(num_nodes):
            nb = list(self.nbrs[i])
            if nb:
                self.W[i] = float(np.sum(self.inv_deg[nb]))

        # Build CSR adjacency and initialize H and M by sparse multiplication.
        # Doing this once is substantially cheaper than evaluating the same
        # neighborhood sums independently for every candidate edge.
        #   H[i] = (1/2) diag(A D⁻¹ A D⁻¹ A)[i]
        #   M[i,j] = [A D⁻¹ A][i,j] = Σ_{w∈N(i)∩N(j)} 1/d_w
        if self.edge_list:
            undirected_edges = np.asarray(self.edge_list, dtype=np.int64)
            rows_ei = np.concatenate([undirected_edges[:, 0], undirected_edges[:, 1]])
            cols_ei = np.concatenate([undirected_edges[:, 1], undirected_edges[:, 0]])
        else:
            undirected_edges = np.empty((0, 2), dtype=np.int64)
            rows_ei = np.empty(0, dtype=np.int64)
            cols_ei = np.empty(0, dtype=np.int64)
        A_sp    = csr_matrix(
            (np.ones(len(rows_ei), dtype=np.float64), (rows_ei, cols_ei)),
            shape=(num_nodes, num_nodes))
        # Dense M gives constant-time vectorized candidate lookup on citation-
        # sized graphs. Above 5000 nodes, an N² array is avoided; M values are
        # recovered from sorted-neighbor intersections only when requested.
        if num_nodes <= 5000:
            D_inv_sp   = sp_diags(self.inv_deg)
            T1         = A_sp @ D_inv_sp       # A D⁻¹
            M_sp       = T1 @ A_sp             # M = A D⁻¹ A
            self.H     = np.asarray((M_sp @ D_inv_sp @ A_sp).diagonal()).ravel() / 2.0
            self._M_dense = M_sp.toarray()     # float64, ~N²×8 bytes
            self._use_dense_M = True
            self._initial_edges = None
            self._initial_edge_M = None
        else:
            # Orient every edge from lower (degree, node-id) rank to higher.
            # Intersecting only forward neighborhoods enumerates each triangle
            # exactly once, including its contributions to all three edges.
            degrees_int = self.deg.astype(np.int64)
            u = undirected_edges[:, 0]
            v = undirected_edges[:, 1]
            u_first = (degrees_int[u] < degrees_int[v]) | (
                (degrees_int[u] == degrees_int[v]) & (u < v)
            )
            forward_u = np.where(u_first, u, v)
            forward_v = np.where(u_first, v, u)
            forward = csr_matrix(
                (
                    np.arange(len(undirected_edges), dtype=np.int64) + 1,
                    (forward_u, forward_v),
                ),
                shape=(num_nodes, num_nodes),
            )
            self.H, self._initial_edge_M = _large_graph_h_and_edge_m(
                undirected_edges,
                forward.indptr,
                forward.indices,
                forward.data - 1,
                self.inv_deg,
            )
            self._initial_edges = undirected_edges
            self._use_dense_M = False

        # m2 / m3 are initialized by the caller.
        self.m2 = None
        self.m3 = None

    # ── fast copy (snapshot restore) ─────────────────────────────────────────

    def copy(self):
        """Clone mutable graph caches for independent snapshot evaluation.

        NumPy arrays use compiled copies, avoiding reconstruction through the
        more expensive constructor. The sparse ``M`` dictionary is safe to
        share because sparse-mode edits never mutate it and lookups are
        recomputed from the copied neighbor structures.
        """
        s             = object.__new__(TopologyMomentState)
        s.n           = self.n
        s.device      = self.device
        s.nbrs        = [nb.copy() for nb in self.nbrs]
        s.nbrs_arr    = [a.copy()  for a  in self.nbrs_arr]
        s.edge_list   = list(self.edge_list)
        s.edge_to_idx = dict(self.edge_to_idx)
        s.deg         = self.deg.copy()
        s.inv_deg     = self.inv_deg.copy()
        s.W           = self.W.copy()
        s.H           = self.H.copy()
        s.m2          = self.m2
        s.m3          = self.m3
        s._use_dense_M = self._use_dense_M
        s._initial_edges = self._initial_edges
        s._initial_edge_M = self._initial_edge_M
        if self._use_dense_M:
            s._M_dense = self._M_dense.copy()
        return s

    # ── M lookup ─────────────────────────────────────────────────────────────

    def M_lookup(self, U, V):
        """Return Σ_{w∈N(U[i])∩N(V[i])} 1/d_w for each pair.

        These weighted common-neighbor sums are exactly the local quantities
        needed by the topology ``delta m3`` formulas.
        """
        if self._use_dense_M:
            return self._M_dense[U, V]
        if (
            self._initial_edge_M is not None
            and len(U) == len(self._initial_edges)
            and np.array_equal(U, self._initial_edges[:, 0])
            and np.array_equal(V, self._initial_edges[:, 1])
        ):
            return self._initial_edge_M
        return np.array([
            float(_inter_sum(self.nbrs_arr[int(u)], self.nbrs_arr[int(v)], self.inv_deg))
            for u, v in zip(U, V)
        ])

    # ── H incremental update ──────────────────────────────────────────────────

    def _delta_H_add(self, u, v):
        """Compute local ``H`` changes before adding ``(u, v)``.

        Only endpoints and their neighbors can change. Dense mode performs
        constant-time ``M`` lookups per affected node; sparse mode reconstructs
        the same values through Numba-compiled neighbor intersections.
        """
        inv    = self.inv_deg
        Nu_s, Nv_s = self.nbrs[u], self.nbrs[v]
        du, dv = self.deg[u], self.deg[v]
        delta  = {}

        s_C = (self._M_dense[u, v] if self._use_dense_M
               else _inter_sum(self.nbrs_arr[u], self.nbrs_arr[v], inv))
        if s_C:
            delta[u] = s_C / (dv + 1.0)
            delta[v] = s_C / (du + 1.0)

        coeff_u = 1.0 / (du * (du + 1.0)) if du > 0 else 0.0
        coeff_v = 1.0 / (dv * (dv + 1.0)) if dv > 0 else 0.0
        new_tri = 1.0 / ((du + 1.0) * (dv + 1.0))

        if self._use_dense_M:
            M = self._M_dense
            for i in Nu_s:
                d = -coeff_u * M[i, u]
                if v in self.nbrs[i]:
                    d += new_tri
                    d -= coeff_v * M[i, v]
                if d:
                    delta[i] = delta.get(i, 0.0) + d
            for i in Nv_s:
                if i in Nu_s or i == u:
                    continue
                d = -coeff_v * M[i, v]
                if d:
                    delta[i] = delta.get(i, 0.0) + d
        else:
            Na, Nb = self.nbrs_arr[u], self.nbrs_arr[v]
            for i in Nu_s:
                d = -coeff_u * _inter_sum(self.nbrs_arr[i], Na, inv)
                if v in self.nbrs[i]:
                    d += new_tri
                    d -= coeff_v * _inter_sum(self.nbrs_arr[i], Nb, inv)
                if d:
                    delta[i] = delta.get(i, 0.0) + d
            for i in Nv_s:
                if i in Nu_s or i == u:
                    continue
                d = -coeff_v * _inter_sum(self.nbrs_arr[i], Nb, inv)
                if d:
                    delta[i] = delta.get(i, 0.0) + d

        return delta

    def _delta_H_del(self, u, v):
        """Compute local ``H`` changes before deleting ``(u, v)``.

        The edge must still exist while these values are computed. Dense mode
        removes excluded endpoint contributions algebraically; sparse mode
        asks the intersection routine to skip the excluded endpoint.
        """
        inv    = self.inv_deg
        Nu_s, Nv_s = self.nbrs[u], self.nbrs[v]
        du, dv = self.deg[u], self.deg[v]
        delta  = {}

        s_C = (self._M_dense[u, v] if self._use_dense_M
               else _inter_sum(self.nbrs_arr[u], self.nbrs_arr[v], inv))
        if s_C:
            delta[u] = -s_C / dv
            delta[v] = -s_C / du

        coeff_u = 1.0 / (du * (du - 1.0)) if du > 1 else 0.0
        coeff_v = 1.0 / (dv * (dv - 1.0)) if dv > 1 else 0.0
        tri_uv  = 1.0 / (du * dv) if (du > 0 and dv > 0) else 0.0

        if self._use_dense_M:
            M = self._M_dense
            inv_v, inv_u = inv[v], inv[u]
            for i in Nu_s:
                if i == v:
                    continue
                # _inter_sum_excl(N(i), N(u), excl=v) = M[i,u] - inv[v]*(v∈N(i))
                v_in_Ni = v in self.nbrs[i]
                d = coeff_u * (M[i, u] - (inv_v if v_in_Ni else 0.0))
                if v_in_Ni:
                    d -= tri_uv
                    # i∈N(u) ⟹ u∈N(i), so excl=u always in intersection
                    d += coeff_v * (M[i, v] - inv_u)
                if d:
                    delta[i] = delta.get(i, 0.0) + d
            for i in Nv_s:
                if i == u or i in Nu_s:
                    continue
                # i∉N(u) ⟹ u∉N(i), so no excl correction needed
                d = coeff_v * M[i, v]
                if d:
                    delta[i] = delta.get(i, 0.0) + d
        else:
            Na, Nb = self.nbrs_arr[u], self.nbrs_arr[v]
            for i in Nu_s:
                if i == v:
                    continue
                d = coeff_u * _inter_sum_excl(self.nbrs_arr[i], Na, inv, np.int32(v))
                if v in self.nbrs[i]:
                    d -= tri_uv
                    d += coeff_v * _inter_sum_excl(self.nbrs_arr[i], Nb, inv, np.int32(u))
                if d:
                    delta[i] = delta.get(i, 0.0) + d
            for i in Nv_s:
                if i == u or i in Nu_s:
                    continue
                d = coeff_v * _inter_sum_excl(self.nbrs_arr[i], Nb, inv, np.int32(u))
                if d:
                    delta[i] = delta.get(i, 0.0) + d

        return delta

    # ── incremental apply ─────────────────────────────────────────────────────

    def apply_add(self, u, v):
        # Delta-H and the formulas below are defined relative to the old graph,
        # so compute them before changing degrees or neighbor containers.
        delta_h = self._delta_H_add(u, v)
        self._initial_edges = None
        self._initial_edge_M = None
        du, dv  = self.deg[u], self.deg[v]
        # Update only dense-M entries affected by the new endpoints and their
        # degree changes. Sparse mode deliberately skips this storage update
        # because its M lookups are reconstructed from current neighborhoods.
        if self._use_dense_M:
            Nu = self.nbrs_arr[u]          # old N(u), before inserting v
            Nv = self.nbrs_arr[v]          # old N(v), before inserting u
            inv_du1 = 1.0 / (du + 1)
            inv_dv1 = 1.0 / (dv + 1)
            self._M_dense[v, Nu] += inv_du1   # u new common nbr of (v,k), k∈N(u)
            self._M_dense[Nu, v] += inv_du1
            self._M_dense[u, Nv] += inv_dv1   # v new common nbr of (u,k), k∈N(v)
            self._M_dense[Nv, u] += inv_dv1
            if len(Nu) > 0:                    # du increases: pairs N(u)×N(u) lose 1/(du*(du+1))
                self._M_dense[np.ix_(Nu, Nu)] -= 1.0 / (du * (du + 1))
            if len(Nv) > 0:                    # dv increases: pairs N(v)×N(v) lose 1/(dv*(dv+1))
                self._M_dense[np.ix_(Nv, Nv)] -= 1.0 / (dv * (dv + 1))
        # Existing neighbors see a smaller reciprocal-degree contribution from
        # an endpoint after its degree increases; adjust W before adding the
        # new direct endpoint contributions.
        for k in self.nbrs[u]:
            self.W[k] -= 1.0 / (du * (du + 1))
        for k in self.nbrs[v]:
            self.W[k] -= 1.0 / (dv * (dv + 1))
        self.deg[u] += 1;  self.deg[v] += 1
        self.inv_deg[u] = 1.0 / self.deg[u]
        self.inv_deg[v] = 1.0 / self.deg[v]
        self.nbrs[u].add(v);  self.nbrs[v].add(u)
        # Insert at the sorted position so future two-pointer intersections
        # remain correct without sorting the complete neighbor array again.
        self.nbrs_arr[u] = np.insert(self.nbrs_arr[u],
                                     np.searchsorted(self.nbrs_arr[u], v), np.int32(v))
        self.nbrs_arr[v] = np.insert(self.nbrs_arr[v],
                                     np.searchsorted(self.nbrs_arr[v], u), np.int32(u))
        self.W[u] += self.inv_deg[v]
        self.W[v] += self.inv_deg[u]
        edge = (min(u, v), max(u, v))
        # Append is constant time; edge_to_idx records the new position for
        # later deletion without scanning the edge list.
        self.edge_to_idx[edge] = len(self.edge_list)
        self.edge_list.append(edge)
        for node, dh in delta_h.items():
            self.H[node] += dh

    def apply_del(self, u, v):
        # Compute all old-graph-dependent deltas before removing adjacency or
        # changing endpoint degrees.
        delta_h = self._delta_H_del(u, v)
        self._initial_edges = None
        self._initial_edge_M = None
        du, dv  = self.deg[u], self.deg[v]
        # Reverse only the dense-M contributions affected by removing the edge.
        # Sparse mode needs no stored-M maintenance because it recomputes M.
        if self._use_dense_M:
            Nu      = self.nbrs_arr[u]
            Nv      = self.nbrs_arr[v]
            Nu_excl = Nu[Nu != v]
            Nv_excl = Nv[Nv != u]
            inv_u   = self.inv_deg[u]
            inv_v   = self.inv_deg[v]
            self._M_dense[v, Nu_excl] -= inv_u
            self._M_dense[Nu_excl, v] -= inv_u
            self._M_dense[u, Nv_excl] -= inv_v
            self._M_dense[Nv_excl, u] -= inv_v
            if du > 1 and len(Nu_excl) > 0:
                self._M_dense[np.ix_(Nu_excl, Nu_excl)] += 1.0 / (du * (du - 1))
            if dv > 1 and len(Nv_excl) > 0:
                self._M_dense[np.ix_(Nv_excl, Nv_excl)] += 1.0 / (dv * (dv - 1))
        # Remove the endpoints' direct reciprocal-degree contributions from W
        # before changing degrees and neighbor sets.
        self.W[u] -= self.inv_deg[v]
        self.W[v] -= self.inv_deg[u]
        self.nbrs[u].discard(v);  self.nbrs[v].discard(u)
        iu = np.searchsorted(self.nbrs_arr[u], v)
        self.nbrs_arr[u] = np.delete(self.nbrs_arr[u], iu)
        iv = np.searchsorted(self.nbrs_arr[v], u)
        self.nbrs_arr[v] = np.delete(self.nbrs_arr[v], iv)
        edge = (min(u, v), max(u, v))
        idx  = self.edge_to_idx.pop(edge)
        last = self.edge_list[-1]
        # Replace the removed slot with the last edge, then pop the tail. This
        # avoids shifting every later list element; edge_to_idx is repaired for
        # the moved edge so both structures remain synchronized.
        if idx < len(self.edge_list) - 1:
            self.edge_list[idx] = last
            self.edge_to_idx[last] = idx
        self.edge_list.pop()
        self.deg[u] -= 1;  self.deg[v] -= 1
        self.inv_deg[u] = 1.0 / self.deg[u] if self.deg[u] > 0 else 0.0
        self.inv_deg[v] = 1.0 / self.deg[v] if self.deg[v] > 0 else 0.0
        for k in self.nbrs[u]:
            # After deletion, each remaining neighbor receives the endpoint's
            # larger reciprocal-degree contribution in its W cache.
            if self.deg[u] > 0:
                self.W[k] += 1.0 / (self.deg[u] * (self.deg[u] + 1))
        for k in self.nbrs[v]:
            if self.deg[v] > 0:
                self.W[k] += 1.0 / (self.deg[v] * (self.deg[v] + 1))
        for node, dh in delta_h.items():
            self.H[node] += dh

    # ── sampling ──────────────────────────────────────────────────────────────

    def sample_non_edges(self, k):
        # Rejection sampling uses set membership to discard self-loops and
        # existing edges cheaply. It returns candidate pairs for scoring and
        # does not mutate the graph.
        n, result = self.n, []
        while len(result) < k:
            U = np.random.randint(n, size=k * 4)
            V = np.random.randint(n, size=k * 4)
            for u, v in zip(U.tolist(), V.tolist()):
                if u != v and v not in self.nbrs[u]:
                    result.append((u, v))
                    if len(result) == k:
                        break
        return np.array(result, dtype=np.int64)

    def sample_edges(self, k):
        # Sampling list indices avoids copying the complete edge collection;
        # replace=False ensures one candidate edge appears at most once.
        E = len(self.edge_list)
        if E == 0:
            return np.empty((0, 2), dtype=np.int64)
        idxs = np.random.choice(E, min(k, E), replace=False)
        return np.array([self.edge_list[i] for i in idxs], dtype=np.int64)

    def to_edge_index(self):
        # PyG represents an undirected graph with both directions, so duplicate
        # every canonical edge in reverse while preserving the state device.
        if not self.edge_list:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)
        arr  = np.array(self.edge_list, dtype=np.int64)
        rows = np.concatenate([arr[:, 0], arr[:, 1]])
        cols = np.concatenate([arr[:, 1], arr[:, 0]])
        return torch.from_numpy(np.stack([rows, cols])).long().to(self.device)
