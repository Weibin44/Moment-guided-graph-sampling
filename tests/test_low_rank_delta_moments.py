import unittest

import numpy as np
import torch

from mggs.moments.low_rank_delta import (
    low_rank_delta_m4_for_independent_deletions,
    low_rank_delta_moments_for_independent_deletions,
    low_rank_delta_moments_for_joint_edits,
)
from mggs.moments.topology_delta import topology_delete_deltas
from mggs.moments.topology_state import TopologyMomentState
from reference_low_rank import global_spmv_delta_m4_for_single_edge_deletions


class SimpleState:
    def __init__(self, num_nodes, edges):
        self.n = int(num_nodes)
        self.edge_list = sorted(
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in edges
            if int(u) != int(v)
        )
        self.nbrs = [set() for _ in range(self.n)]
        for u, v in self.edge_list:
            self.nbrs[u].add(v)
            self.nbrs[v].add(u)
        self.deg = np.asarray([len(nb) for nb in self.nbrs], dtype=np.float64)


def canonical_edges(edges):
    return {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in edges
        if int(u) != int(v)
    }


def apply_batch(edges, additions=(), deletions=()):
    out = set(edges)
    for edge in canonical_edges(deletions):
        out.remove(edge)
    for edge in canonical_edges(additions):
        out.add(edge)
    return out


def transition_matrix(num_nodes, edges):
    P = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    deg = np.zeros(num_nodes, dtype=np.float64)
    for u, v in edges:
        deg[u] += 1.0
        deg[v] += 1.0
    for u, v in edges:
        P[v, u] = 1.0 / deg[u]
        P[u, v] = 1.0 / deg[v]
    return P


def brute_force_moment_deltas(num_nodes, old_edges, new_edges, max_order):
    old_P = transition_matrix(num_nodes, old_edges)
    new_P = transition_matrix(num_nodes, new_edges)
    return {
        k: (
            np.trace(np.linalg.matrix_power(new_P, k))
            - np.trace(np.linalg.matrix_power(old_P, k))
        ) / float(num_nodes)
        for k in range(1, max_order + 1)
    }


def topology_state(num_nodes, edges):
    """Build the production topology cache from an undirected edge set."""

    directed_edges = [
        directed
        for u, v in sorted(edges)
        for directed in ((u, v), (v, u))
    ]
    edge_index = torch.tensor(directed_edges, dtype=torch.long).T.contiguous()
    return TopologyMomentState(edge_index, num_nodes)


class LowRankClosedWalkDeltaMomentTest(unittest.TestCase):
    def test_delete_m2_m3_joint_independent_topology_match_ground_truth(self):
        """Cross-check every exact deletion backend for m2 and m3."""

        rng = np.random.default_rng(2026)
        for num_nodes in range(4, 11):
            all_pairs = [
                (u, v)
                for u in range(num_nodes)
                for v in range(u + 1, num_nodes)
            ]
            for _ in range(30):
                old_edges = {
                    edge for edge in all_pairs if rng.random() < 0.30
                }
                if not old_edges:
                    continue

                state = topology_state(num_nodes, old_edges)
                deletions = np.asarray(state.edge_list, dtype=np.int64)

                topology_m2, topology_m3 = topology_delete_deltas(
                    state, deletions
                )
                independent = low_rank_delta_moments_for_independent_deletions(
                    state, deletions, orders=(2, 3)
                )
                joint = {2: [], 3: []}
                ground_truth = {2: [], 3: []}
                for edge in deletions:
                    edge_tuple = tuple(edge)
                    delta = low_rank_delta_moments_for_joint_edits(
                        state,
                        deletions=[edge_tuple],
                        min_order=2,
                        max_order=3,
                    )
                    new_edges = apply_batch(old_edges, deletions=[edge_tuple])
                    exact = brute_force_moment_deltas(
                        num_nodes, old_edges, new_edges, 3
                    )
                    for order in (2, 3):
                        joint[order].append(delta[order])
                        ground_truth[order].append(exact[order])

                for order, topology in ((2, topology_m2), (3, topology_m3)):
                    expected = np.asarray(ground_truth[order], dtype=np.float64)
                    self.assertTrue(
                        np.allclose(topology, expected, rtol=0.0, atol=1e-12),
                        msg=f"topology m{order} mismatch",
                    )
                    self.assertTrue(
                        np.allclose(joint[order], expected, rtol=0.0, atol=1e-12),
                        msg=f"joint-edit m{order} mismatch",
                    )
                    self.assertTrue(
                        np.allclose(independent[order], expected, rtol=0.0, atol=1e-12),
                        msg=f"independent-deletion m{order} mismatch",
                    )

    def test_delete_m4_m5_joint_independent_match_ground_truth(self):
        """Cross-check both low-rank deletion paths for m4 and m5."""

        rng = np.random.default_rng(2027)
        for num_nodes in range(4, 11):
            all_pairs = [
                (u, v)
                for u in range(num_nodes)
                for v in range(u + 1, num_nodes)
            ]
            for _ in range(30):
                old_edges = {
                    edge for edge in all_pairs if rng.random() < 0.30
                }
                if not old_edges:
                    continue

                state = SimpleState(num_nodes, old_edges)
                deletions = np.asarray(state.edge_list, dtype=np.int64)
                independent = low_rank_delta_moments_for_independent_deletions(
                    state, deletions, orders=(4, 5)
                )
                joint = {4: [], 5: []}
                ground_truth = {4: [], 5: []}
                for edge in deletions:
                    edge_tuple = tuple(edge)
                    delta = low_rank_delta_moments_for_joint_edits(
                        state,
                        deletions=[edge_tuple],
                        min_order=4,
                        max_order=5,
                    )
                    new_edges = apply_batch(old_edges, deletions=[edge_tuple])
                    exact = brute_force_moment_deltas(
                        num_nodes, old_edges, new_edges, 5
                    )
                    for order in (4, 5):
                        joint[order].append(delta[order])
                        ground_truth[order].append(exact[order])

                for order in (4, 5):
                    expected = np.asarray(ground_truth[order], dtype=np.float64)
                    self.assertTrue(
                        np.allclose(joint[order], expected, rtol=0.0, atol=1e-12),
                        msg=f"joint-edit m{order} mismatch",
                    )
                    self.assertTrue(
                        np.allclose(independent[order], expected, rtol=0.0, atol=1e-12),
                        msg=f"independent-deletion m{order} mismatch",
                    )

    def test_random_batches_match_direct_trace_to_order_six(self):
        rng = np.random.default_rng(123)
        max_order = 6

        for num_nodes in range(4, 11):
            all_pairs = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
            for _ in range(80):
                old_edges = {
                    edge for edge in all_pairs
                    if rng.random() < 0.35
                }
                non_edges = sorted(set(all_pairs) - old_edges)
                old_edges_sorted = sorted(old_edges)

                del_count = int(rng.integers(0, min(3, len(old_edges_sorted)) + 1))
                add_count = int(rng.integers(0, min(3, len(non_edges)) + 1))
                deletions = [
                    old_edges_sorted[i]
                    for i in rng.choice(len(old_edges_sorted), size=del_count, replace=False)
                ] if del_count else []
                additions = [
                    non_edges[i]
                    for i in rng.choice(len(non_edges), size=add_count, replace=False)
                ] if add_count else []
                if not additions and not deletions:
                    continue

                state = SimpleState(num_nodes, old_edges)
                new_edges = apply_batch(old_edges, additions=additions, deletions=deletions)

                actual = low_rank_delta_moments_for_joint_edits(
                    state,
                    additions=additions,
                    deletions=deletions,
                    max_order=max_order,
                    min_order=1,
                )
                expected = brute_force_moment_deltas(
                    num_nodes, old_edges, new_edges, max_order
                )

                for order in range(1, max_order + 1):
                    self.assertAlmostEqual(actual[order], expected[order], places=12)

    def test_deleting_degree_one_endpoint_uses_zero_column_convention(self):
        num_nodes = 3
        old_edges = {(0, 1)}
        state = SimpleState(num_nodes, old_edges)

        actual = low_rank_delta_moments_for_joint_edits(
            state,
            deletions=[(0, 1)],
            max_order=5,
            min_order=1,
        )
        expected = brute_force_moment_deltas(num_nodes, old_edges, set(), 5)

        for order in range(1, 6):
            self.assertAlmostEqual(actual[order], expected[order], places=12)

    def test_cached_m4_deletion_path_matches_generic_low_rank(self):
        state = SimpleState(
            7,
            {
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 3),
                (3, 4),
                (3, 5),
                (4, 5),
                (5, 6),
            },
        )
        deletions = np.asarray(state.edge_list, dtype=np.int64)

        actual = low_rank_delta_m4_for_independent_deletions(state, deletions)
        expected = np.asarray(
            [
                low_rank_delta_moments_for_joint_edits(
                    state,
                    deletions=[tuple(edge)],
                    max_order=4,
                    min_order=4,
                )[4]
                for edge in deletions
            ],
            dtype=np.float64,
        )

        self.assertTrue(np.allclose(actual, expected, rtol=0.0, atol=1e-12))

    def test_endpoint_cached_deletion_path_matches_generic_to_order_six(self):
        state = SimpleState(
            8,
            {
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 3),
                (3, 4),
                (3, 5),
                (4, 5),
                (5, 6),
                (6, 7),
            },
        )
        deletions = np.asarray(state.edge_list, dtype=np.int64)
        orders = tuple(range(1, 7))

        actual = low_rank_delta_moments_for_independent_deletions(
            state, deletions, orders=orders
        )
        expected = {
            order: np.asarray(
                [
                    low_rank_delta_moments_for_joint_edits(
                        state,
                        deletions=[tuple(edge)],
                        max_order=6,
                        min_order=1,
                    )[order]
                    for edge in deletions
                ],
                dtype=np.float64,
            )
            for order in orders
        }

        for order in orders:
            self.assertTrue(
                np.allclose(actual[order], expected[order], rtol=0.0, atol=1e-12),
                msg=f"order {order} mismatch",
            )

    def test_endpoint_cached_degree_one_deletion_matches_direct_to_order_six(self):
        num_nodes = 4
        old_edges = {(0, 1), (1, 2)}
        state = SimpleState(num_nodes, old_edges)
        deletions = np.asarray([(0, 1), (1, 2)], dtype=np.int64)
        orders = tuple(range(1, 7))

        actual = low_rank_delta_moments_for_independent_deletions(
            state, deletions, orders=orders
        )
        expected_by_edge = []
        for edge in deletions:
            new_edges = apply_batch(old_edges, deletions=[tuple(edge)])
            expected_by_edge.append(
                brute_force_moment_deltas(num_nodes, old_edges, new_edges, 6)
            )

        for order in orders:
            expected = np.asarray(
                [row[order] for row in expected_by_edge], dtype=np.float64
            )
            self.assertTrue(
                np.allclose(actual[order], expected, rtol=0.0, atol=1e-12),
                msg=f"order {order} mismatch",
            )

    def test_endpoint_cached_m4_deletion_path_matches_old_spmv_path(self):
        rng = np.random.default_rng(321)
        for num_nodes in range(4, 13):
            all_pairs = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
            for _ in range(40):
                old_edges = {
                    edge for edge in all_pairs
                    if rng.random() < 0.25
                }
                if not old_edges:
                    continue
                state = SimpleState(num_nodes, old_edges)
                deletions = np.asarray(state.edge_list, dtype=np.int64)

                actual = low_rank_delta_m4_for_independent_deletions(state, deletions)
                expected = global_spmv_delta_m4_for_single_edge_deletions(state, deletions)

                self.assertTrue(np.allclose(actual, expected, rtol=0.0, atol=1e-12))

    def test_no_edits_return_zero_deltas(self):
        state = SimpleState(4, {(0, 1), (1, 2)})
        actual = low_rank_delta_moments_for_joint_edits(state, max_order=4)
        self.assertEqual(actual, {2: 0.0, 3: 0.0, 4: 0.0})

    def test_validation_rejects_invalid_batches(self):
        state = SimpleState(4, {(0, 1), (1, 2)})

        with self.assertRaises(ValueError):
            low_rank_delta_moments_for_joint_edits(state, additions=[(0, 1)])
        with self.assertRaises(ValueError):
            low_rank_delta_moments_for_joint_edits(state, deletions=[(0, 3)])
        with self.assertRaises(ValueError):
            low_rank_delta_moments_for_joint_edits(state, additions=[(0, 3), (3, 0)])
        with self.assertRaises(ValueError):
            low_rank_delta_moments_for_joint_edits(
                state, additions=[(0, 3)], deletions=[(3, 0)]
            )


if __name__ == "__main__":
    unittest.main()
