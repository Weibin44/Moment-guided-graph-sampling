import unittest
import csv
import io
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from mggs import (
    EdgeDirectionSettings,
    MomentDeltaCalculator,
    compute_edge_direction_profile,
)
from mggs.moments import topology_add_deltas, topology_delete_deltas
from mggs import compute_moments
from mggs.cli import main


class MomentDeltaPublicApiTest(unittest.TestCase):
    def setUp(self):
        self.edges = np.asarray([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64)
        self.calculator = MomentDeltaCalculator.from_undirected_edges(self.edges, num_nodes=4)

    def test_topology_deletion_dispatch_is_unchanged(self):
        candidates = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
        expected_m2, expected_m3 = topology_delete_deltas(self.calculator.state, candidates)
        actual = self.calculator.deletion_deltas(candidates, orders=(2, 3), backend="topology")
        np.testing.assert_allclose(actual[2], expected_m2, rtol=0, atol=0)
        np.testing.assert_allclose(actual[3], expected_m3, rtol=0, atol=0)

    def test_topology_addition_dispatch_is_unchanged(self):
        candidates = np.asarray([[0, 3], [1, 3]], dtype=np.int64)
        expected_m2, expected_m3 = topology_add_deltas(self.calculator.state, candidates)
        actual = self.calculator.addition_deltas(candidates, orders=(2, 3), backend="topology")
        np.testing.assert_allclose(actual[2], expected_m2, rtol=0, atol=0)
        np.testing.assert_allclose(actual[3], expected_m3, rtol=0, atol=0)

    def test_mixed_orders_return_one_value_per_edge(self):
        actual = self.calculator.deletion_deltas(orders=(2, 3, 4, 5))
        self.assertEqual(actual.orders, (2, 3, 4, 5))
        for order in actual.orders:
            self.assertEqual(actual[order].shape, (len(self.edges),))

    def test_simultaneous_batch_edit(self):
        actual = self.calculator.batch_edit_delta(
            additions=[[1, 3]], deletions=[[0, 1]], orders=(2, 3, 4, 5)
        )
        self.assertEqual(tuple(actual), (2, 3, 4, 5))

    def test_edge_direction_profile_is_probability_distribution(self):
        settings = EdgeDirectionSettings(grid_step_deg=1.0)
        profile = compute_edge_direction_profile(
            self.edges, num_nodes=4, settings=settings
        )
        np.testing.assert_allclose(np.sum(profile.probability), 1.0)
        self.assertEqual(profile.angles_deg.shape, (len(self.edges),))
        self.assertEqual(profile.grid_deg.shape, profile.probability.shape)
        self.assertEqual(len(profile.grid_deg), 360)

    def test_cli_outputs_sorted_candidates_and_correct_deltas(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            graph, candidates, output = (root / name for name in ("graph.csv", "candidates.csv", "deltas.csv"))
            np.savetxt(graph, self.edges, fmt="%d", delimiter=",")
            candidates.write_text("u,v\n3,2\n1,0\n")
            for backend_args, orders in (([], (2, 3, 6)), (["--backend", "topology"], (2, 3))):
                argv = ["mggs", "delta-delete", "--edge-list", str(graph), "--num-nodes", "4",
                        "--candidates", str(candidates), "--orders", *map(str, orders),
                        "--output", str(output), *backend_args]
                with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                    main()
                with output.open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([(int(row['u']), int(row['v'])) for row in rows], [(0, 1), (2, 3)])
                before = compute_moments(self.edges, 4, orders=orders)
                for row in rows:
                    edge = (int(row['u']), int(row['v']))
                    remaining = [pair for pair in self.edges if tuple(sorted(pair)) != edge]
                    after = compute_moments(remaining, 4, orders=orders)
                    for order in orders:
                        self.assertAlmostEqual(float(row[f'delta_m{order}']), after[order] - before[order], places=13)


if __name__ == "__main__":
    unittest.main()
