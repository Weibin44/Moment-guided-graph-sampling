"""Golden trajectories captured before refactoring."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mggs.io import edge_index_from_edges
from mggs.moments import TopologyMomentState, exact_moments_m1_to_m4
from mggs.sampling import MomentPreservingSampler
from experiments.EXP3_graph_learning.unsupervised_node_classification.sampling_protocol import sample_target

EDGES = np.array([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6)])
FIXTURE = json.loads((Path(__file__).parent / 'fixtures/legacy_trajectories.json').read_text())


class ExperimentCompatibilityTest(unittest.TestCase):
    def test_exp2_legacy_deletion_trajectories(self):
        for case in FIXTURE['deletion']:
            kwargs = {k: v for k, v in case.items() if k != 'trace'}
            with self.subTest(**kwargs):
                sampler = MomentPreservingSampler.from_edges(EDGES, 7, seed=12, top_k=2, **kwargs)
                for expected in case['trace']:
                    actual = sampler.step(candidate_batch_size=3 if case['strategy'] == 'greedy' else 0)
                    self.assertEqual(actual, expected[0])
                    for k, value in zip(case['orders'], expected[1:]):
                        if value is None:
                            self.assertIsNone(getattr(sampler.state, f'm{k}', None))
                        else:
                            self.assertAlmostEqual(getattr(sampler.state, f'm{k}'), value, places=14)

    def test_exp3_progressive_targets_and_random_stream(self):
        base = TopologyMomentState(None, 7, canonical_edges=EDGES)
        base.m2, base.m3 = exact_moments_m1_to_m4(edge_index_from_edges(EDGES), 7)[1:3]
        for case in FIXTURE['progressive_target']:
            with self.subTest(seed=case['seed'], target=case['target'], add=case['add_budget']):
                np.random.seed(case['seed'])
                state, additions, deletions = sample_target(
                    base, np.array(case['target']), case['add_budget'], case['delete_budget'], 5,
                    SimpleNamespace(candidate_add=20, candidate_del=5),
                )
                edges, m2, m3, old_additions, old_deletions, next_random = case['result']
                self.assertEqual(state.edge_list, [tuple(edge) for edge in edges])
                self.assertEqual((additions, deletions), (old_additions, old_deletions))
                self.assertAlmostEqual(state.m2, m2, places=14)
                self.assertAlmostEqual(state.m3, m3, places=14)
                self.assertEqual(np.random.random(), next_random)
