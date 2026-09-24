"""Backend, trajectory and reporting invariants for the sampling comparison."""

import unittest
from types import SimpleNamespace

import numpy as np

from experiments.EXP2_reporting import summarize_rows
from experiments.EXP2_graph_property_preservation import compute_property_snapshot, run_sampler_trajectory
from mggs.io.graph import edge_index_from_edges
from mggs.moments import exact_moments_m1_to_m4, TopologyMomentState
from mggs.sampling import MomentPreservingSampler


EDGES = np.asarray([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6)])


class SamplingComparisonTest(unittest.TestCase):
    def test_greedy_candidate_cap_and_seed(self):
        from unittest.mock import patch
        first = MomentPreservingSampler.from_edges(EDGES, 7, seed=12)
        second = MomentPreservingSampler.from_edges(EDGES, 7, seed=12)
        with patch.object(first, "_score", wraps=first._score) as score:
            for _ in range(len(EDGES)):
                remaining = first.remaining
                self.assertEqual(first.step(candidate_batch_size=3),
                                 second.step(candidate_batch_size=3))
                scored_edges = score.call_args.args[0]
                self.assertEqual(len(scored_edges), min(3, remaining))
                self.assertEqual(len(np.unique(scored_edges, axis=0)), len(scored_edges))

    def test_full_heap_matches_greedy_and_recomputed_moments(self):
        for orders in ((2, 3), (2, 3, 4)):
            for backend in ("hybrid", "low_rank", "direct_trace"):
                for normalization in ("std", "graph_relative"):
                    with self.subTest(orders=orders, backend=backend, normalization=normalization):
                        kwargs = dict(orders=orders, backend=backend, normalization=normalization)
                        greedy = MomentPreservingSampler.from_edges(EDGES, 7, **kwargs)
                        heap = MomentPreservingSampler.from_edges(EDGES, 7, strategy="heap", top_k=100, **kwargs)
                        for _ in range(5):
                            self.assertEqual(greedy.step(), heap.step())
                            actual = exact_moments_m1_to_m4(edge_index_from_edges(heap.state.edge_list, 7), 7)
                            for k in orders:
                                self.assertAlmostEqual(getattr(heap.state, f"m{k}"), actual[k-1], places=12)
                        self.assertEqual(len(heap.heap), heap.remaining)

    def test_backend_scores_match_before_and_after_deletions(self):
        for orders in ((2, 3), (2, 3, 4)):
            sampler = MomentPreservingSampler.from_edges(EDGES, 7, orders=orders)
            for _ in range(4):
                edges = np.asarray(sampler.state.edge_list)
                sampler.backend = "hybrid"
                topology_scores, topology_deltas = sampler._score(edges)
                sampler.backend = "low_rank"
                low_rank_scores, low_rank_deltas = sampler._score(edges)
                np.testing.assert_allclose(topology_scores, low_rank_scores, atol=1e-12)
                for k in orders:
                    np.testing.assert_allclose(topology_deltas[k], low_rank_deltas[k], atol=1e-12)
                sampler.backend = "direct_trace"
                trace_scores, trace_deltas = sampler._score(edges)
                np.testing.assert_allclose(topology_scores, trace_scores, atol=1e-12)
                for k in orders:
                    np.testing.assert_allclose(topology_deltas[k], trace_deltas[k], atol=1e-12)
                sampler.step()

    def test_checkpoint_evaluation_keeps_the_same_heap_trajectory(self):
        data = SimpleNamespace(edge_index=edge_index_from_edges(EDGES.tolist(), 7), num_nodes=7)
        properties = ("spectrum_rmse", "mean_neighbor_inverse_degree_abs_error")
        state = TopologyMomentState(None, 7, canonical_edges=EDGES)
        original = compute_property_snapshot(state, properties)
        kwargs = dict(orders=(2, 3, 4), backend="low_rank", strategy="heap", top_k=2,
                      seed=0, normalization="graph_relative", properties=properties,
                      original_properties=original)
        frequent = run_sampler_trajectory(data, checkpoints=[0, 1, 2, 3, 4], **kwargs)
        sparse = run_sampler_trajectory(data, checkpoints=[0, 4], **kwargs)
        for metric in properties:
            self.assertAlmostEqual(frequent[-1][metric], sparse[-1][metric], places=14)
        self.assertEqual(frequent[-1]["remaining_edges"], len(EDGES)-4)

    def test_random_trajectory_independent_of_moment_orders(self):
        first = MomentPreservingSampler.from_edges(EDGES, 7, orders=(2, 3), strategy="random", seed=8)
        second = MomentPreservingSampler.from_edges(EDGES, 7, orders=(2, 3, 4), strategy="random", seed=8)
        self.assertEqual([first.step() for _ in range(5)], [second.step() for _ in range(5)])

    def test_summary_does_not_merge_different_configs(self):
        common = dict(dataset="toy", method="exact", seed=0, step=1, initial_edges=8,
                      remaining_edges=7, remove_ratio=0.125, strategy="greedy", top_k=0,
                      normalization="graph_relative", error=1.0)
        rows = [dict(common, orders=orders, backend=backend)
                for orders in ("m2_m3", "m2_m3_m4") for backend in ("hybrid", "low_rank")]
        summary = summarize_rows(rows)
        self.assertEqual(len(summary), 4)
        self.assertTrue(all(row["n_runs"] == 1 for row in summary))

    def test_invalid_backend_orders_and_topk_fail(self):
        with self.assertRaises(ValueError):
            MomentPreservingSampler.from_edges(EDGES, 7, orders=(2, 3, 4), backend="topology")
        with self.assertRaises(ValueError):
            MomentPreservingSampler.from_edges(EDGES, 7, top_k=0)


if __name__ == "__main__":
    unittest.main()
