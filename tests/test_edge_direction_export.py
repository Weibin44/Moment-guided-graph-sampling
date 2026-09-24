import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mggs import EdgeDirectionSettings, save_moment_change_fingerprint


class EdgeDirectionExportTest(unittest.TestCase):
    def test_exports_supported_formats_without_mutating_input_or_leaking_figures(self):
        edges = np.array([(0, 1), (0, 2), (1, 2), (2, 3)])
        original = edges.copy()
        figures = plt.get_fignums()
        settings = EdgeDirectionSettings(grid_step_deg=30)
        with tempfile.TemporaryDirectory() as directory:
            for suffix, signature in (("png", b"\x89PNG"), ("pdf", b"%PDF"), ("svg", b"<?xml")):
                with self.subTest(format=suffix):
                    path = Path(directory) / "nested" / f"fingerprint.{suffix}"
                    result = save_moment_change_fingerprint(edges, 5, path, settings=settings)
                    self.assertEqual(result, path)
                    self.assertTrue(path.read_bytes().startswith(signature))
        np.testing.assert_array_equal(edges, original)
        self.assertEqual(plt.get_fignums(), figures)

    def test_empty_graph_and_isolated_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_moment_change_fingerprint([], 3, Path(directory) / "empty.svg")
            self.assertIn(b"<svg", path.read_bytes())

    def test_invalid_format_and_render_failure_leave_no_open_figure(self):
        figures = plt.get_fignums()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_moment_change_fingerprint([(0, 1)], 2, Path(directory) / "fingerprint.txt")
            with self.assertRaises(ValueError):
                save_moment_change_fingerprint(
                    [(0, 1)], 2, Path(directory) / "fingerprint.png", color="invalid-color",
                )
        self.assertEqual(plt.get_fignums(), figures)
