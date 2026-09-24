import unittest

import numpy as np
import torch

from mggs.moments import TopologyMomentState, exact_moments_m1_to_m4
from mggs.sampling import (
    choose_directional_edit,
    compute_delta_normalizer,
    heap_lazy_moment_preserving_topk,
    lazy_moment_preserving_topk,
    moment_preserving_greedy,
)


EDGES = np.asarray(
    [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6)],
    dtype=np.int64,
)


def make_state():
    directed = np.concatenate([EDGES, EDGES[:, ::-1]], axis=0)
    edge_index = torch.from_numpy(directed.T).long()
    state = TopologyMomentState(edge_index, 7)
    _, state.m2, state.m3, state.m4 = exact_moments_m1_to_m4(edge_index, 7)
    return state


class SamplingApiTest(unittest.TestCase):
    def test_direction_selector_is_public_and_non_mutating(self):
        state = make_state()
        edit = choose_directional_edit(
            state, np.asarray([-1.0, 0.0]), state.m2, state.m3, 0, -1
        )
        self.assertEqual(edit["op"], "del")
        self.assertIn((edit["u"], edit["v"]), state.edge_to_idx)
        self.assertEqual(len(state.edge_list), len(EDGES))

    def test_lazy_and_heap_match_when_topk_covers_all_edges(self):
        first = make_state()
        reference = {name: float(getattr(first, name)) for name in ("m2", "m3", "m4")}
        scales = compute_delta_normalizer(first, include_m4=True)
        lazy = lazy_moment_preserving_topk(
            first, reference, scales, budget=3, top_k=len(EDGES)
        )

        second = make_state()
        heap = heap_lazy_moment_preserving_topk(
            second, reference, scales, budget=3, top_k=len(EDGES)
        )
        np.testing.assert_array_equal(lazy.indices, heap.indices)
        np.testing.assert_array_equal(lazy.edges, heap.edges)

        third = make_state()
        greedy = moment_preserving_greedy(
            third, reference, scales, budget=3, moment_names=("m2", "m3", "m4")
        )
        np.testing.assert_array_equal(lazy.indices, greedy.indices)


if __name__ == "__main__":
    unittest.main()
