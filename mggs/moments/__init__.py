"""Exact graph moments and delta-moment backends."""

from .topology_state import TopologyMomentState
from .topology_delta import (
    topology_add_deltas,
    topology_delete_deltas,
)
from .low_rank_delta import (
    LowRankUpdate,
    build_low_rank_transition_update,
    direct_trace_delta_moments_for_independent_deletions,
    low_rank_delta_m4_for_independent_deletions,
    low_rank_delta_moments_for_independent_deletions,
    low_rank_delta_moments_for_joint_edits,
    trace_power_deltas_for_joint_low_rank_update,
)
from .exact import (
    compute_moments,
    estimate_moments,
    exact_m2_m3_from_state,
    exact_m4_from_state,
    exact_moments_m1_to_m4,
)
from .calculator import CandidateDeltaMoments, MomentDeltaCalculator

# Recommended exports; explicit legacy imports above remain compatible.
__all__ = [
    "compute_moments",
    "MomentDeltaCalculator",
    "CandidateDeltaMoments",
]
