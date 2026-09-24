import unittest

from mggs.evaluation import normalized_adjacency_spectrum
from mggs.io import edge_index_from_edges, undirected_simple_edges
from mggs.moments import (
    TopologyMomentState,
    exact_m2_m3_from_state,
    exact_m4_from_state,
)


class SharedGraphEvaluationTest(unittest.TestCase):
    def test_graph_conversion_round_trip(self):
        edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
        edge_index = edge_index_from_edges(edges, 4)
        self.assertEqual(undirected_simple_edges(edge_index), edges)
        self.assertEqual(tuple(edge_index.shape), (2, 2 * len(edges)))

    def test_state_moments_match_exact_spectrum(self):
        edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
        edge_index = edge_index_from_edges(edges, 4)
        state = TopologyMomentState(edge_index, 4)
        spectrum = normalized_adjacency_spectrum(edges, 4)
        m2, m3 = exact_m2_m3_from_state(state)
        self.assertAlmostEqual(m2, spectrum.m2, places=12)
        self.assertAlmostEqual(m3, spectrum.m3, places=12)
        self.assertAlmostEqual(exact_m4_from_state(state), spectrum.m4, places=12)


if __name__ == "__main__":
    unittest.main()
