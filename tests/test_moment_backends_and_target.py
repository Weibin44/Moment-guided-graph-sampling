import unittest

import numpy as np
import torch

from mggs import MomentDeltaCalculator
from mggs.moments import TopologyMomentState
from mggs.moments._queries import addition_deltas_from_state, deletion_deltas_from_state


EDGES = np.asarray([(0, 1), (1, 2), (0, 2), (2, 3)], dtype=np.int64)


def edge_index(edges=EDGES):
    directed = np.concatenate([edges, edges[:, ::-1]], axis=0)
    return torch.from_numpy(directed.T).long()


class MomentBackendAndTargetTest(unittest.TestCase):
    def test_topology_and_low_rank_match_for_m2_m3(self):
        calculator = MomentDeltaCalculator(edge_index(), 4)
        additions = np.asarray([(0, 3), (1, 3)], dtype=np.int64)
        deletions = np.asarray([(0, 1), (2, 3)], dtype=np.int64)

        for operation, candidates in (
            (calculator.addition_deltas, additions),
            (calculator.deletion_deltas, deletions),
        ):
            topology = operation(candidates, orders=(2, 3), backend="topology")
            low_rank = operation(candidates, orders=(2, 3), backend="low_rank")
            for order in (2, 3):
                self.assertTrue(np.allclose(topology[order], low_rank[order], atol=1e-12))

    def test_hybrid_preserves_legacy_auto_results(self):
        state = TopologyMomentState(edge_index(), 4)
        for operation, candidates in (
            (addition_deltas_from_state, np.array([(0, 3), (1, 3)])),
            (deletion_deltas_from_state, np.array([(0, 1), (2, 3)])),
        ):
            for orders in ((2, 3), (2, 4, 6)):
                with self.subTest(operation=operation.__name__, orders=orders):
                    hybrid = operation(state, candidates, orders=orders, backend="hybrid")
                    legacy = operation(state, candidates, orders=orders, backend="auto")
                    default = operation(state, candidates, orders=orders)
                    for order in orders:
                        np.testing.assert_array_equal(hybrid[order], legacy[order])
                        np.testing.assert_array_equal(hybrid[order], default[order])

    def test_topology_backend_rejects_higher_orders(self):
        calculator = MomentDeltaCalculator(edge_index(), 4)
        with self.assertRaises(ValueError):
            calculator.deletion_deltas(orders=(2, 4), backend="topology")

if __name__ == "__main__":
    unittest.main()
