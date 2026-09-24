"""Moment-guided graph sampling strategies."""

from .moment_direction import (
    apply_edit,
    choose_directional_edit,
    choose_ray_constrained_edit,
    direction_record,
    relative_moment_delta,
)
from .moment_preserving import (
    MomentPreservingSelection,
    MomentPreservingSampler,
    apply_moment_deletion,
    choose_moment_preserving_edge,
    compute_candidate_deltas,
    compute_delta_normalizer,
    moment_preserving_greedy,
)
from .lazy_topk import (
    DeletionSelection,
    heap_lazy_moment_preserving_topk,
    lazy_moment_preserving_topk,
)

from .sampler import sample_by_moments
from .types import EdgeEdit, GraphSnapshot, SamplingResult

# Recommended exports; explicit legacy imports above remain compatible.
__all__ = [
    "sample_by_moments",
    "EdgeEdit",
    "GraphSnapshot",
    "SamplingResult",
]
