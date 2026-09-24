"""Public contracts checked against independent dense traces and edit replay."""
import unittest
import numpy as np
from mggs import compute_moments, MomentDeltaCalculator
from mggs.sampling import sample_by_moments

EDGES = np.array([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)], dtype=np.int64)


def dense_moments(edges, n, orders):
    adjacency = np.zeros((n, n))
    for u, v in edges:
        adjacency[u, v] = adjacency[v, u] = 1
    degree = adjacency.sum(axis=1)
    transition = np.divide(adjacency, degree[:, None], out=np.zeros_like(adjacency), where=degree[:, None] > 0)
    return {k: float(np.trace(np.linalg.matrix_power(transition, k)) / n) for k in orders}


class CoreInterfacesTest(unittest.TestCase):
    def test_arbitrary_orders_empty_graph_and_isolated_nodes(self):
        for edges in (EDGES, np.empty((0, 2), dtype=int)):
            actual = compute_moments(edges, 7, orders=(6, 1, 3, 2, 4, 6, 9))
            expected = dense_moments(edges, 7, actual)
            self.assertEqual(tuple(actual), (1, 2, 3, 4, 6, 9))
            np.testing.assert_allclose(list(actual.values()), list(expected.values()), atol=1e-14)

    def test_candidate_and_joint_deltas_match_recomputation(self):
        calc = MomentDeltaCalculator.from_undirected_edges(EDGES, 7)
        for backend, orders in (("topology", (2, 3)), ("low_rank", (2, 3, 6))):
            before = dense_moments(EDGES, 7, orders)
            for operation, candidates in (("addition", [(0, 4), (5, 6)]), ("deletion", [(0, 1), (3, 4)])):
                result = getattr(calc, operation + '_deltas')(candidates, orders=orders, backend=backend)
                for index, edge in enumerate(result.edges):
                    changed = set(map(tuple, EDGES))
                    (changed.add if operation == "addition" else changed.remove)(tuple(edge))
                    after = dense_moments(changed, 7, orders)
                    for k in orders:
                        self.assertAlmostEqual(result[k][index], after[k] - before[k], places=13)
        joint = calc.batch_edit_delta(additions=[(0, 4)], deletions=[(0, 1)], orders=(6, 2, 6))
        changed = (set(map(tuple, EDGES)) - {(0, 1)}) | {(0, 4)}
        after = dense_moments(changed, 7, joint)
        before = dense_moments(EDGES, 7, joint)
        self.assertEqual(tuple(joint), (2, 6))
        for k in joint:
            self.assertAlmostEqual(joint[k], after[k] - before[k], places=13)
        np.testing.assert_array_equal(calc.edges, EDGES)
        self.assertEqual(calc.batch_edit_delta(orders=(2, 4, 6)), {2: 0.0, 4: 0.0, 6: 0.0})

    def test_default_and_explicit_original_target_have_identical_trajectories(self):
        for operation, strategy in (("delete", "greedy"), ("delete", "lazy"), ("add", "greedy"), ("mixed", "greedy")):
            for backend, orders in (("topology", (2, 3)), ("low_rank", (2, 4, 6))):
                with self.subTest(operation=operation, strategy=strategy, backend=backend):
                    kwargs = dict(budget=3, operation=operation, strategy=strategy, backend=backend,
                                  orders=orders, seed=17, top_k=2)
                    first = sample_by_moments(EDGES, 7, **kwargs)
                    second = sample_by_moments(EDGES, 7, target=compute_moments(EDGES, 7, orders=orders), **kwargs)
                    self.assertEqual(first.edits, second.edits)
                    self.assertEqual(first.final_moments, second.final_moments)

    def test_target_sampling_replays_and_checkpoints_do_not_change_search(self):
        for operation, strategy in (("delete", "lazy"), ("delete", "greedy"), ("add", "greedy"), ("mixed", "greedy")):
            with self.subTest(operation=operation, strategy=strategy):
                kwargs = dict(budget=4, operation=operation, strategy=strategy, top_k=2,
                              target={2: 0.2, 3: 0.1}, candidate_limit=None if strategy == "lazy" else 2, seed=5)
                plain = sample_by_moments(EDGES, 7, **kwargs)
                result = sample_by_moments(EDGES, 7, checkpoints=(0, 1, 2, 3, 4), **kwargs)
                self.assertEqual(plain.edits, result.edits)
                replay = set(map(tuple, EDGES))
                for count, edit in enumerate(result.edits, 1):
                    (replay.add if edit.operation == 'add' else replay.remove)((edit.u, edit.v))
                    snapshot = result.snapshots[count]
                    self.assertEqual(replay, set(map(tuple, snapshot.edges)))
                    expected = dense_moments(replay, 7, (2, 3))
                    for k in expected:
                        self.assertAlmostEqual(snapshot.moments[k], expected[k], places=13)
                self.assertEqual(len(replay ^ set(map(tuple, EDGES))), 4)
                for k in result.final_moments:
                    self.assertEqual(result.delta_moments[k], result.final_moments[k] - result.initial_moments[k])
                np.testing.assert_array_equal(result.snapshots[0].edges, EDGES)

    def test_one_step_matches_brute_force_best_objective(self):
        for operation in ('delete', 'add', 'mixed'):
            target = {2: 0.3, 3: 0.02}
            result = sample_by_moments(EDGES, 6, budget=1, target=target, operation=operation)
            base = set(map(tuple, EDGES))
            outcomes = []
            for u in range(6):
                for v in range(u + 1, 6):
                    edge = (u, v)
                    if operation == 'delete' and edge not in base or operation == 'add' and edge in base:
                        continue
                    after = dense_moments(base ^ {edge}, 6, target)
                    outcomes.append(sum(((after[k] - target[k]) / result.metadata['scale'][k]) ** 2 for k in target))
            actual = sum(((result.final_moments[k] - target[k]) / result.metadata['scale'][k]) ** 2 for k in target)
            self.assertAlmostEqual(actual, min(outcomes), places=12)

    def test_exhaustive_budget_and_dense_nonedge_sampling(self):
        for operation, budget in (("delete", len(EDGES)), ("add", 10-len(EDGES)), ("mixed", 10)):
            result = sample_by_moments(EDGES, 5, budget=budget, operation=operation, candidate_limit=2)
            self.assertEqual(len(set(map(tuple, EDGES)) ^ set(map(tuple, result.edges))), budget)
        dense = [(u, v) for u in range(8) for v in range(u+1, 8) if (u, v) not in ((0, 1), (2, 3), (4, 5))]
        self.assertEqual(len(sample_by_moments(dense, 8, budget=3, operation='add', candidate_limit=1).edges), 28)

    def test_public_validation_and_input_immutability(self):
        original = EDGES.copy()
        for bad in ([(0.5, 1)], [(0, 0)], [(0, 1), (1, 0)], [(-1, 0)], [(0, 7)]):
            with self.assertRaises(ValueError):
                compute_moments(bad, 7)
        for orders in ((), (2.5,), (True,), (0,)):
            with self.assertRaises(ValueError):
                compute_moments(EDGES, 7, orders=orders)
        for kwargs in (dict(budget=6), dict(budget=-1), dict(budget=1, backend='auto'),
                       dict(budget=1, backend='hybrid'), dict(budget=1, backend='topology', orders=(4,)),
                       dict(budget=1, target={2: float('nan')}), dict(budget=1, scale={2: 0, 3: 1}),
                       dict(budget=1, target={2: 0.2}, orders=(2, 3)),
                       dict(budget=1, strategy='lazy', operation='mixed')):
            with self.assertRaises(ValueError):
                sample_by_moments(EDGES, 7, **kwargs)
        empty = sample_by_moments(EDGES, 7, budget=0, checkpoints=(0,))
        self.assertEqual(empty.edits, ())
        np.testing.assert_array_equal(EDGES, original)
