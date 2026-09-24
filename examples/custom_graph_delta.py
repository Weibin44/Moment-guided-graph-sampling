"""Compute exact deletion delta moments for a user-provided graph."""

import numpy as np

from mggs import MomentDeltaCalculator


edges = np.asarray(
    [[0, 1], [1, 2], [2, 0], [2, 3]],
    dtype=np.int64,
)
calculator = MomentDeltaCalculator.from_undirected_edges(edges, num_nodes=4)
result = calculator.deletion_deltas(orders=(2, 3, 4, 5))

for idx, edge in enumerate(result.edges):
    print(tuple(edge), {order: result[order][idx] for order in result.orders})

