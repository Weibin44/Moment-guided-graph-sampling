"""Public target-moment sampling with explicit operation and search choices."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import numpy as np
from ..io.graph import canonical_edges, validate_num_nodes
from ..moments.exact import compute_moments
from ..moments.orders import normalize_orders, validate_backend
from ..moments.topology_state import TopologyMomentState
from .search import DeletionTrajectory, GreedyEditSearch
from .types import EdgeEdit, GraphSnapshot, SamplingResult
from .updates import run_steps


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _moment_mapping(values, orders, name, *, positive=False):
    if not isinstance(values, Mapping) or normalize_orders(values, minimum=2) != orders:
        raise ValueError(f"{name} keys must match orders")
    result = {k: float(values[k]) for k in orders}
    if any(not np.isfinite(v) or (positive and v <= 0) for v in result.values()):
        raise ValueError(f"{name} values must be finite" + (" and positive" if positive else ""))
    return result


def sample_by_moments(
    edges, num_nodes: int, *, budget: int,
    target: Mapping[int, float] | None = None,
    orders: Sequence[int] | None = None,
    operation: str = "delete", strategy: str = "greedy", top_k: int = 25,
    candidate_limit: int | None = None, scale: Mapping[int, float] | None = None,
    backend: str = "low_rank", seed: int = 0, checkpoints: Sequence[int] = (),
) -> SamplingResult:
    """Apply exactly budget edits minimizing distance to target moments.

    target=None means the original graph's moments, using the same computation
    as an explicitly supplied compute_moments result. Inputs are never mutated.
    Public backends are low_rank and topology; lazy search currently deletes only.
    Checkpoints observe one persistent trajectory and do not restart its RNG/heap.
    """
    n = validate_num_nodes(num_nodes)
    original_edges = canonical_edges(edges, n)
    budget = _integer(budget, "budget")
    top_k = _integer(top_k, "top_k", 1)
    seed = _integer(seed, "seed")
    if candidate_limit is not None:
        candidate_limit = _integer(candidate_limit, "candidate_limit", 1)
    orders = normalize_orders(
        orders if orders is not None else ((2, 3) if target is None else target), minimum=2,
    )
    validate_backend(backend, orders)
    if operation not in ("delete", "add", "mixed"):
        raise ValueError("operation must be delete, add or mixed")
    if strategy not in ("greedy", "lazy"):
        raise ValueError("strategy must be greedy or lazy")
    if strategy == "lazy" and (operation != "delete" or candidate_limit is not None):
        raise ValueError("lazy requires operation='delete' and candidate_limit=None")
    possible = n * (n - 1) // 2
    capacity = {"delete": len(original_edges), "add": possible - len(original_edges), "mixed": possible}[operation]
    if budget > capacity:
        raise ValueError(f"budget exceeds the {capacity} available distinct edits")
    checkpoints = tuple(sorted({_integer(k, "checkpoint") for k in checkpoints}))
    if any(k > budget for k in checkpoints):
        raise ValueError("checkpoints must not exceed budget")
    initial = compute_moments(original_edges, n, orders=orders)
    resolved_target = dict(initial) if target is None else _moment_mapping(target, orders, "target")
    scales = ({k: abs(v) if abs(v) > 1e-12 else 1.0 for k, v in initial.items()}
              if scale is None else _moment_mapping(scale, orders, "scale", positive=True))
    state = TopologyMomentState(None, n, canonical_edges=original_edges)
    for k, value in initial.items():
        setattr(state, f"m{k}", value)
    if operation == "delete":
        search = DeletionTrajectory(
            state, {f"m{k}": v for k, v in resolved_target.items()},
            {f"m{k}": v for k, v in scales.items()}, orders=orders,
            backend=backend, strategy="heap" if strategy == "lazy" else "greedy",
            top_k=top_k, seed=seed,
        )
        def step(index):
            chosen = search.step(candidate_batch_size=candidate_limit or 0)
            u, v = map(int, search.base_edges[chosen])
            return EdgeEdit("delete", u, v)
    else:
        search = GreedyEditSearch(
            state, original_edges, target=resolved_target, scales=scales,
            backend=backend, operation=operation, candidate_limit=candidate_limit, seed=seed,
        )
        def step(index):
            return search.step()
    edits, snapshots = [], {}

    def snapshot():
        kept = np.asarray(sorted(state.edge_list), dtype=np.int64).reshape(-1, 2)
        return GraphSnapshot(kept, compute_moments(kept, n, orders=orders))

    def observe(count, edit):
        edits.append(edit)
        if count in checkpoints:
            snapshots[count] = snapshot()

    if 0 in checkpoints:
        snapshots[0] = GraphSnapshot(original_edges.copy(), dict(initial))
    run_steps(budget, step, observe=observe)
    final = snapshots[budget] if budget in snapshots else snapshot()
    return SamplingResult(
        edges=final.edges, edits=tuple(edits), initial_moments=initial,
        final_moments=final.moments,
        delta_moments={k: final.moments[k] - initial[k] for k in orders},
        metadata=dict(target=resolved_target, scale=scales, orders=orders,
                      operation=operation, strategy=strategy, backend=backend,
                      budget=budget, edit_count=len(edits), seed=seed,
                      top_k=top_k, candidate_limit=candidate_limit, checkpoints=checkpoints),
        snapshots=snapshots,
    )
