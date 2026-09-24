"""Public API for moment-guided graph sampling."""

from .moments import MomentDeltaCalculator, compute_moments
from .sampling import sample_by_moments
from .visualization import save_moment_change_fingerprint

# Historical explicit imports remain available for existing callers.
from .moments import (
    LowRankUpdate,
    CandidateDeltaMoments,
    TopologyMomentState,
    build_low_rank_transition_update,
    estimate_moments,
    exact_moments_m1_to_m4,
    low_rank_delta_moments_for_joint_edits,
    topology_add_deltas,
    topology_delete_deltas,
    trace_power_deltas_for_joint_low_rank_update,
)
from .visualization import (
    EdgeDirectionProfile,
    EdgeDirectionSettings,
    compute_edge_direction_profile,
    plot_moment_change_fingerprint,
    probability_to_area_radius,
)

# Recommended exports; explicit legacy imports above remain compatible.
__all__ = [
    "compute_moments",
    "MomentDeltaCalculator",
    "sample_by_moments",
    "save_moment_change_fingerprint",
]
