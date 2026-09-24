"""Slow independent reference paths used only by low-rank regression tests."""

import numpy as np

from mggs.moments.low_rank_delta import (
    _column_delta_matrix,
    _column_transition_matrix,
    _updated_touched_neighbors,
    trace_power_deltas_for_joint_low_rank_update,
)


def global_spmv_delta_m4_for_single_edge_deletions(state, deletions):
    """Old per-edge global sparse-matmul path retained as a test oracle."""

    empty = np.empty((0, 2), dtype=np.int64)
    transition = _column_transition_matrix(state)
    out = np.empty(len(deletions), dtype=np.float64)
    for idx, (u_raw, v_raw) in enumerate(deletions):
        touched = np.asarray([int(u_raw), int(v_raw)], dtype=np.int64)
        one_delete = np.asarray([[int(u_raw), int(v_raw)]], dtype=np.int64)
        new_neighbors = _updated_touched_neighbors(
            state, touched, empty, one_delete
        )
        column_delta = _column_delta_matrix(state, touched, new_neighbors)
        out[idx] = trace_power_deltas_for_joint_low_rank_update(
            transition, column_delta, touched, 4
        )[4] / float(state.n)
    return out
