import unittest

import numpy as np

from mggs.sampling.baselines.effective_resistance import (
    approximate_edge_effective_resistances_jl,
    approximate_static_normalized_edge_resistances_jl,
    estimate_effective_resistances,
    estimate_static_normalized_effective_resistances,
    exact_static_normalized_edge_resistances,
    exact_edge_effective_resistances,
    sample_edge_subset_without_replacement,
    sample_sparsifier_from_probabilities,
)


class EffectiveResistanceTest(unittest.TestCase):
    def test_triangle_edges_have_two_thirds_resistance(self):
        edges = np.asarray([(0, 1), (1, 2), (0, 2)], dtype=np.int64)
        resistances = exact_edge_effective_resistances(3, edges)
        self.assertTrue(np.allclose(resistances, 2.0 / 3.0, atol=1e-12))

    def test_path_edges_have_unit_resistance(self):
        edges = np.asarray([(0, 1), (1, 2)], dtype=np.int64)
        resistances = exact_edge_effective_resistances(3, edges)
        self.assertTrue(np.allclose(resistances, 1.0, atol=1e-12))

    def test_leverage_scores_sum_to_laplacian_rank(self):
        edges = np.asarray([(0, 1), (1, 2), (0, 2), (3, 4)], dtype=np.int64)
        estimate = estimate_effective_resistances(6, edges, method="exact")
        self.assertAlmostEqual(float(np.sum(estimate.leverage_scores)), 3.0, places=12)
        self.assertAlmostEqual(float(np.sum(estimate.probabilities)), 1.0, places=12)

    def test_static_normalized_triangle_leverage_sum(self):
        edges = np.asarray([(0, 1), (1, 2), (0, 2)], dtype=np.int64)
        _, leverage = exact_static_normalized_edge_resistances(3, edges)
        self.assertTrue(np.allclose(leverage, 2.0 / 3.0, atol=1e-12))
        self.assertAlmostEqual(float(np.sum(leverage)), 2.0, places=12)

    def test_static_normalized_path_leverage_sum(self):
        edges = np.asarray([(0, 1), (1, 2)], dtype=np.int64)
        _, leverage = exact_static_normalized_edge_resistances(3, edges)
        self.assertTrue(np.allclose(leverage, 1.0, atol=1e-12))
        self.assertAlmostEqual(float(np.sum(leverage)), 2.0, places=12)

    def test_static_normalized_estimate_probabilities(self):
        edges = np.asarray([(0, 1), (1, 2), (0, 2), (3, 4)], dtype=np.int64)
        estimate = estimate_static_normalized_effective_resistances(6, edges, method="exact")
        self.assertAlmostEqual(float(np.sum(estimate.leverage_scores)), 3.0, places=12)
        self.assertAlmostEqual(float(np.sum(estimate.probabilities)), 1.0, places=12)

    def test_one_edge_sparsifier_keeps_original_weight(self):
        edges = np.asarray([(0, 1)], dtype=np.int64)
        weights = np.asarray([2.5], dtype=np.float64)
        probabilities = np.asarray([1.0], dtype=np.float64)
        sample = sample_sparsifier_from_probabilities(
            edges,
            weights,
            probabilities,
            sample_budget=7,
            rng=np.random.default_rng(123),
        )
        self.assertEqual(sample.unique_edges, 1)
        self.assertEqual(sample.sample_budget, 7)
        self.assertTrue(np.array_equal(sample.edges, edges))
        self.assertTrue(np.allclose(sample.weights, weights))

    def test_with_replacement_sampler_uses_paper_reweighting(self):
        edges = np.asarray([(0, 1), (1, 2), (2, 3)], dtype=np.int64)
        weights = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        probabilities = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
        q = 200
        sample = sample_sparsifier_from_probabilities(
            edges,
            weights,
            probabilities,
            sample_budget=q,
            rng=np.random.default_rng(5),
        )
        mask = sample.counts > 0
        expected_weights = sample.counts[mask] * (weights[mask] / (q * probabilities[mask]))
        self.assertTrue(sample.with_replacement)
        self.assertEqual(sample.sample_budget, q)
        self.assertEqual(int(np.sum(sample.counts)), q)
        self.assertTrue(np.array_equal(sample.edges, edges[mask]))
        self.assertTrue(np.allclose(sample.weights, expected_weights))

    def test_without_replacement_sampler_keeps_exact_unique_budget(self):
        edges = np.asarray([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], dtype=np.int64)
        weights = np.ones(len(edges), dtype=np.float64)
        probabilities = np.asarray([0.45, 0.25, 0.15, 0.10, 0.05], dtype=np.float64)
        sample = sample_edge_subset_without_replacement(
            edges,
            weights,
            probabilities,
            keep_count=3,
            rng=np.random.default_rng(42),
        )
        self.assertFalse(sample.with_replacement)
        self.assertEqual(sample.sample_budget, 3)
        self.assertEqual(sample.unique_edges, 3)
        self.assertEqual(len({tuple(edge) for edge in sample.edges}), 3)
        self.assertEqual(int(np.sum(sample.counts)), 3)
        self.assertTrue(np.all(sample.counts <= 1))
        self.assertTrue(np.allclose(sample.weights, 1.0))

    def test_without_replacement_sampler_full_budget_keeps_all_edges(self):
        edges = np.asarray([(0, 1), (1, 2), (2, 3)], dtype=np.int64)
        weights = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        probabilities = np.asarray([0.8, 0.2, 0.0], dtype=np.float64)
        sample = sample_edge_subset_without_replacement(
            edges,
            weights,
            probabilities,
            keep_count=3,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(np.array_equal(sample.edges, edges))
        self.assertTrue(np.allclose(sample.weights, weights))
        self.assertTrue(np.array_equal(sample.counts, np.ones(3, dtype=np.int64)))

    def test_jl_approximation_is_reasonable_on_small_graph(self):
        edges = np.asarray(
            [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
            dtype=np.int64,
        )
        exact = exact_edge_effective_resistances(4, edges)
        approx = approximate_edge_effective_resistances_jl(
            4,
            edges,
            epsilon=0.25,
            jl_dim=4096,
            seed=7,
        )
        self.assertTrue(np.allclose(approx, exact, rtol=0.10, atol=0.03))

    def test_static_normalized_jl_approximation_is_reasonable(self):
        edges = np.asarray(
            [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
            dtype=np.int64,
        )
        exact, _ = exact_static_normalized_edge_resistances(4, edges)
        approx, _ = approximate_static_normalized_edge_resistances_jl(
            4,
            edges,
            epsilon=0.25,
            jl_dim=4096,
            seed=11,
        )
        self.assertTrue(np.allclose(approx, exact, rtol=0.10, atol=0.03))


if __name__ == "__main__":
    unittest.main()
