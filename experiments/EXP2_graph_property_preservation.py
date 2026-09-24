"""Graph-property preservation with legacy probes or exact/heap/random comparisons.

Checkpoints evaluate only the requested properties. Spectrum-based metrics use
the exact normalized-adjacency eigenvalues.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mggs_matplotlib_cache")
os.environ.setdefault("PYG_HOME", "/tmp/mggs_pyg_cache")

from experiments.EXP2_reporting import plot_metric, summarize_rows, write_csv

from mggs.moments import (
    TopologyMomentState,
    exact_m2_m3_from_state,
    exact_m4_from_state,
    topology_delete_deltas,
)
from mggs.evaluation.properties import (
    average_neighbor_inverse_degree,
    absolute_mean_difference as mean_absolute_difference,
    triangle_degree_weighted_clustering_coefficients,
    normalized_estrada_index,
    compute_property_snapshot,
    property_distances,
)
from mggs.moments._queries import _backend
from mggs.paths import RECORDS_DIR
from mggs.sampling.moment_preserving import (
    apply_moment_deletion,
    choose_moment_preserving_edge,
    compute_delta_normalizer,
)

from mggs.datasets import canonical_dataset_name, load_graph_dataset
from mggs.evaluation.spectrum import (
    matrix_after_delete,
    normalized_adjacency_matrix_from_state,
    spectrum_rmse,
)
from mggs.io.graph import edge_index_from_edges, undirected_simple_edges


SPECTRAL_GREEDY_ORACLE = "spectral_greedy_oracle"
DEFAULT_METHODS = ("anchor_m2", "anchor_m3", "anchor_m2_m3", "random_edge")
NORMALIZATION_MODES = ("std", "graph_relative")
NORMALIZER_EPS = 1e-12


def checkpoint_steps(
    num_edges: int, max_remove_ratio: float, checkpoint_step: float
) -> list[int]:
    max_steps = int(math.floor(num_edges * max_remove_ratio))
    ratios = []
    value = 0.0
    while value <= max_remove_ratio + 1e-12:
        ratios.append(value)
        value += checkpoint_step
    steps = {0, max_steps}
    for ratio in ratios:
        steps.add(int(round(num_edges * float(ratio))))
    return sorted(step for step in steps if 0 <= step <= max_steps)


METHOD_MOMENTS = {
    "anchor_m2": ("m2",),
    "anchor_m3": ("m3",),
    "anchor_m4": ("m4",),
    "anchor_m2_m3": ("m2", "m3"),
    "anchor_m2_m4": ("m2", "m4"),
    "anchor_m3_m4": ("m3", "m4"),
    "anchor_m2_m3_m4": ("m2", "m3", "m4"),
}
PROPERTY_METRICS = (
    "spectrum_rmse",
    "mean_neighbor_inverse_degree_abs_error",
    "mean_triangle_weighted_clustering_abs_error",
    "normalized_estrada_index_abs_error",
)

def graph_relative_normalizer(
    original_moments: dict[str, float],
    moment_names: tuple[str, ...],
) -> dict[str, float]:
    """Scale every moment order by its magnitude on the original graph."""
    return {
        name: abs(original_moments[name])
        if abs(original_moments[name]) > NORMALIZER_EPS
        else 1.0
        for name in moment_names
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_candidate_edges(
    state: TopologyMomentState,
    rng: np.random.Generator,
    *,
    candidate_batch_ratio: float,
    candidate_batch_size: int,
) -> np.ndarray:
    edges = np.asarray(state.edge_list, dtype=np.int64)
    if len(edges) == 0:
        return edges
    if candidate_batch_size > 0:
        batch_size = min(int(candidate_batch_size), len(edges))
    elif candidate_batch_ratio < 1.0:
        batch_size = int(math.ceil(max(candidate_batch_ratio, 0.0) * len(edges)))
        batch_size = min(max(batch_size, 1), len(edges))
    else:
        return edges
    idx = rng.choice(len(edges), size=batch_size, replace=False)
    return edges[idx]


def choose_random_delete(
    state: TopologyMomentState,
    rng: np.random.Generator,
) -> tuple[int, int, float, float, float]:
    idx = int(rng.integers(len(state.edge_list)))
    u, v = state.edge_list[idx]
    dm2, dm3 = topology_delete_deltas(state, np.asarray([[u, v]], dtype=np.int64))
    return int(u), int(v), float(dm2[0]), float(dm3[0]), 0.0


def choose_spectral_greedy_oracle_delete(
    state: TopologyMomentState,
    original_eigvals: np.ndarray,
) -> tuple[int, int, float, float, float]:
    current_matrix = normalized_adjacency_matrix_from_state(state)
    best_idx = 0
    best_rmse = float("inf")

    for idx, (u, v) in enumerate(state.edge_list):
        candidate_matrix = matrix_after_delete(state, current_matrix, int(u), int(v))
        candidate_eigvals = np.linalg.eigvalsh(candidate_matrix)
        candidate_rmse = spectrum_rmse(candidate_eigvals, original_eigvals)
        if candidate_rmse < best_rmse:
            best_idx = idx
            best_rmse = candidate_rmse

    u, v = state.edge_list[best_idx]
    dm2, dm3 = topology_delete_deltas(state, np.asarray([[u, v]], dtype=np.int64))
    return int(u), int(v), float(dm2[0]), float(dm3[0]), 0.0


def append_checkpoint_row(
    rows: list[dict[str, float | int | str]],
    *,
    dataset: str,
    method: str,
    seed: int,
    step: int,
    initial_edges: int,
    state: TopologyMomentState,
    original_properties: dict[str, object],
    properties: tuple[str, ...],
    selection_time: float,
) -> None:
    current_properties = compute_property_snapshot(state, properties)
    row = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "step": step,
        "remove_ratio": step / initial_edges,
        "remaining_edges": len(state.edge_list),
        "initial_edges": initial_edges,
        "selection_time_sec": selection_time,
        "m2": float(state.m2),
        "m3": float(state.m3),
    }
    current_m4 = getattr(state, "m4", None)
    if current_m4 is not None:
        row["m4"] = float(current_m4)
    row.update(property_distances(current_properties, original_properties, properties))
    rows.append(row)


def run_method(
    *,
    dataset: str,
    data,
    method: str,
    seed: int,
    max_remove_ratio: float,
    checkpoint_step: float,
    candidate_batch_ratio: float,
    candidate_batch_size: int,
    properties: tuple[str, ...],
    normalization: str,
) -> list[dict[str, float | int | str]]:
    if method not in METHOD_MOMENTS and method not in {"random_edge", SPECTRAL_GREEDY_ORACLE}:
        raise ValueError(f"Unsupported method for property probe: {method}")
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    base_edges = undirected_simple_edges(data.edge_index)
    num_nodes = int(data.num_nodes)
    initial_edges = len(base_edges)
    checkpoint_sequence = checkpoint_steps(
        initial_edges, max_remove_ratio, checkpoint_step
    )
    checkpoints = set(checkpoint_sequence)
    max_steps = checkpoint_sequence[-1]

    state = TopologyMomentState(edge_index_from_edges(base_edges, num_nodes), num_nodes)
    state.m2, state.m3 = exact_m2_m3_from_state(state)
    moment_names = METHOD_MOMENTS.get(method)
    uses_m4 = moment_names is not None and "m4" in moment_names
    if uses_m4:
        state.m4 = exact_m4_from_state(state)
    original_moments = {"m2": float(state.m2), "m3": float(state.m3)}
    if uses_m4:
        original_moments["m4"] = float(state.m4)
    if moment_names is None:
        normalizer = {}
    elif normalization == "std":
        normalizer = compute_delta_normalizer(state, include_m4=uses_m4)
    elif normalization == "graph_relative":
        normalizer = graph_relative_normalizer(original_moments, moment_names)
    else:
        raise ValueError(f"Unknown normalization mode: {normalization}")
    original_properties = compute_property_snapshot(state, properties)
    original_eigvals = original_properties.get("eigenvalues")
    if method == SPECTRAL_GREEDY_ORACLE and original_eigvals is None:
        original_eigvals = np.linalg.eigvalsh(
            normalized_adjacency_matrix_from_state(state)
        )

    rows: list[dict[str, float | int | str]] = []
    selection_time = 0.0
    append_checkpoint_row(
        rows,
        dataset=dataset,
        method=method,
        seed=seed,
        step=0,
        initial_edges=initial_edges,
        state=state,
        original_properties=original_properties,
        properties=properties,
        selection_time=selection_time,
    )

    for step in range(1, max_steps + 1):
        select_start = time.perf_counter()
        if moment_names is not None:
            candidate_edges = sample_candidate_edges(
                state,
                rng,
                candidate_batch_ratio=candidate_batch_ratio,
                candidate_batch_size=candidate_batch_size,
            )
            u, v, dm2, dm3, dm4, _ = choose_moment_preserving_edge(
                state,
                original_moments,
                normalizer,
                moment_names,
                candidate_edges=candidate_edges,
                track_m4=False,
            )
        elif method == "random_edge":
            u, v, dm2, dm3, dm4 = choose_random_delete(state, rng)
        elif method == SPECTRAL_GREEDY_ORACLE:
            if original_eigvals is None:
                raise RuntimeError("Missing original eigenvalues for spectral greedy oracle.")
            u, v, dm2, dm3, dm4 = choose_spectral_greedy_oracle_delete(
                state, original_eigvals
            )
        else:
            raise ValueError(f"Unsupported method for property probe: {method}")
        deltas = {"m2": dm2, "m3": dm3}
        if uses_m4:
            deltas["m4"] = dm4
        apply_moment_deletion(state, u, v, deltas)
        selection_time += time.perf_counter() - select_start

        if step in checkpoints:
            append_checkpoint_row(
                rows,
                dataset=dataset,
                method=method,
                seed=seed,
                step=step,
                initial_edges=initial_edges,
                state=state,
                original_properties=original_properties,
                properties=properties,
                selection_time=selection_time,
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-samplers", action="store_true",
                        help="Compare all-edge exact greedy, lazy heap top-k, and random using public samplers.")
    parser.add_argument("--moment-groups", nargs="+", default=["2,3", "2,3,4"])
    parser.add_argument("--backend", type=_backend, choices=["hybrid", "topology", "low_rank"], default="low_rank")
    parser.add_argument("--top-ks", nargs="+", type=int, default=[5, 25, 50])
    parser.add_argument("--threads", type=int, default=None,
                        help="Optional CPU thread limit; omitted leaves runtime defaults unchanged")
    parser.add_argument("--datasets", nargs="+", default=["Texas", "Cora"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        choices=list(METHOD_MOMENTS) + ["random_edge", SPECTRAL_GREEDY_ORACLE],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-remove-ratio", type=float, default=0.90)
    parser.add_argument("--checkpoint-step", type=float, default=0.05)
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATION_MODES,
        default="std",
        help=(
            "Moment-coordinate scaling used by anchor methods: std uses the "
            "population standard deviation of all original-graph edge deltas; "
            "graph_relative uses the absolute original graph moment."
        ),
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        choices=PROPERTY_METRICS,
        default=None,
        help="Properties to evaluate. Omit to evaluate every available property.",
    )
    parser.add_argument(
        "--candidate-batch-ratio",
        type=float,
        default=1.0,
        help=(
            "Fraction of remaining edges scored per anchor step. Use 1.0 for "
            "exact full-candidate greedy; use e.g. 0.01 for scalable stochastic greedy."
        ),
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=None,
        help="Fixed number of remaining edges scored per anchor step. Overrides ratio when positive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RECORDS_DIR / "spectrum_sampling" / "property_probe",
    )
    args = parser.parse_args()
    if args.candidate_batch_size is None:
        args.candidate_batch_size = 2048 if args.compare_samplers else 0
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid ranges before creating output files or running datasets."""
    if not 0.0 <= args.max_remove_ratio < 1.0:
        raise ValueError("--max-remove-ratio must satisfy 0 <= value < 1.")
    if args.checkpoint_step <= 0.0:
        raise ValueError("--checkpoint-step must be positive.")
    if not 0.0 < args.candidate_batch_ratio <= 1.0:
        raise ValueError("--candidate-batch-ratio must satisfy 0 < value <= 1.")
    if args.candidate_batch_size < 0:
        raise ValueError("--candidate-batch-size must be nonnegative.")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed.")
    if (args.threads is not None and args.threads < 1) or any(k < 1 for k in args.top_ks):
        raise ValueError("Threads and top-k must be positive")
    if args.compare_samplers:
        groups = [tuple(sorted(set(map(int, value.split(","))))) for value in args.moment_groups]
        if any(not orders or any(k not in (2, 3, 4) for k in orders) for orders in groups):
            raise ValueError("Moment groups must contain orders 2, 3 and/or 4")
        if len(groups) != len(set(groups)) or len(args.top_ks) != len(set(args.top_ks)):
            raise ValueError("Duplicate moment groups or top-k values")
        if args.backend == "topology" and any(4 in orders for orders in groups):
            raise ValueError("Topology does not support m4; choose hybrid or low_rank")


def run_sampler_trajectory(data, *, orders, backend, strategy, top_k, seed,
                           normalization, checkpoints, properties, original_properties,
                           candidate_batch_size=0, candidate_batch_ratio=1.0):
    """Evaluate checkpoints on one persistent trajectory; evaluation is untimed."""
    from mggs.sampling import MomentPreservingSampler

    edges = np.asarray(undirected_simple_edges(data.edge_index), dtype=np.int64)
    start = time.perf_counter()
    sampler = MomentPreservingSampler.from_edges(
        edges, int(data.num_nodes), orders=orders, backend=backend, strategy=strategy,
        top_k=max(1, top_k), seed=seed, normalization=normalization,
    )
    preprocess = time.perf_counter() - start
    checkpoints = set(checkpoints)
    rows, sampling_time = [], 0.0
    for step in range(max(checkpoints) + 1):
        if step:
            start = time.perf_counter()
            batch_size = 0
            if strategy == "greedy":
                batch_size = candidate_batch_size or (
                    max(1, math.ceil(candidate_batch_ratio * sampler.remaining))
                    if candidate_batch_ratio < 1 else 0)
            sampler.step(candidate_batch_size=batch_size)
            sampling_time += time.perf_counter() - start
        if step in checkpoints:
            current = original_properties if step == 0 else compute_property_snapshot(sampler.state, properties)
            row = dict(seed=seed, step=step, remove_ratio=step/len(edges), initial_edges=len(edges),
                       remaining_edges=sampler.remaining, preprocess_sec=preprocess,
                       selection_time_sec=sampling_time, total_time_sec=preprocess+sampling_time)
            row.update(property_distances(current, original_properties, properties))
            rows.append(row)
            print(f"[checkpoint] {strategy} K={top_k} seed={seed} deleted={step}/{max(checkpoints)}", flush=True)
    return rows


def compare_samplers(args, properties):
    """Reuse EXP2 metrics and plotting for exact/heap/random comparisons."""
    from threadpoolctl import threadpool_limits

    groups = [tuple(sorted(set(map(int, value.split(","))))) for value in args.moment_groups]
    subsampled = args.candidate_batch_size > 0 or args.candidate_batch_ratio < 1
    greedy_method = "sampled_greedy" if subsampled else "exact"
    method_order = [greedy_method] + [f"heap_k{k}" for k in args.top_ks] + ["random_edge"]
    labels = {"random_edge": "Random"}
    labels.update({f"heap_k{k}": f"Lazy heap K={k}" for k in args.top_ks})
    all_rows = []
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        for name in args.datasets:
            dataset = canonical_dataset_name(name)
            data = load_graph_dataset(dataset)
            edges = undirected_simple_edges(data.edge_index)
            if not edges:
                raise ValueError(f"{dataset} has no edges")
            checkpoints = checkpoint_steps(len(edges), args.max_remove_ratio, args.checkpoint_step)
            original_state = TopologyMomentState(None, int(data.num_nodes), canonical_edges=edges)
            original = compute_property_snapshot(original_state, properties)
            del original_state
            # One random trajectory per seed, reused across moment groups.
            random_rows = []
            for seed in args.seeds:
                random_rows.extend(run_sampler_trajectory(
                    data, orders=groups[0], backend="hybrid", strategy="random", top_k=0, seed=seed,
                    normalization=args.normalization, checkpoints=checkpoints,
                    properties=properties, original_properties=original,
                ))
            for orders in groups:
                group_name = "_".join(f"m{k}" for k in orders)
                actual_backend = args.backend if args.backend != "hybrid" else (
                    "hybrid" if 4 in orders else "topology")
                for method, strategy, k in [(greedy_method, "greedy", 0)] + [
                    (f"heap_k{k}", "heap", k) for k in args.top_ks
                ] + [("random_edge", "random", 0)]:
                    print(f"[preservation] {dataset} {group_name} {actual_backend} {method}", flush=True)
                    rows = random_rows if strategy == "random" else run_sampler_trajectory(
                        data, orders=orders, backend=args.backend, strategy=strategy, top_k=k,
                        seed=args.seeds[0], normalization=args.normalization, checkpoints=checkpoints,
                        properties=properties, original_properties=original,
                        candidate_batch_size=args.candidate_batch_size,
                        candidate_batch_ratio=args.candidate_batch_ratio,
                    )
                    all_rows.extend(dict(row, dataset=dataset, method=method, orders=group_name,
                                         backend="none" if strategy == "random" else actual_backend,
                                         strategy=strategy, top_k=k, normalization=args.normalization)
                                    for row in rows)
                    write_csv(args.output_dir / "checkpoint_metrics.csv", all_rows)
    summary = summarize_rows(all_rows, properties)
    write_csv(args.output_dir / "summary_metrics.csv", summary)
    for orders in groups:
        group_name = "_".join(f"m{k}" for k in orders)
        labels[greedy_method] = f"anchor_{group_name}"
        rows = [row for row in summary if row["orders"] == group_name]
        for metric in properties:
            plot_metric(rows, args.output_dir / "figures" / group_name / f"{metric}_vs_remove_ratio.png",
                        metric, method_order=method_order, method_labels=labels)


def main() -> None:
    args = parse_args()
    validate_args(args)
    selected_properties = tuple(args.properties or PROPERTY_METRICS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    config["resolved_properties"] = list(selected_properties)
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    if args.compare_samplers:
        compare_samplers(args, selected_properties)
        return

    all_rows: list[dict[str, float | int | str]] = []
    datasets = [canonical_dataset_name(name) for name in args.datasets]
    for dataset in datasets:
        print(f"[dataset] loading {dataset}", flush=True)
        data = load_graph_dataset(dataset)
        base_edges = undirected_simple_edges(data.edge_index)
        print(
            f"[dataset] {dataset}: nodes={int(data.num_nodes)} "
            f"undirected_edges={len(base_edges)}",
            flush=True,
        )
        for method in args.methods:
            stochastic_anchor = (
                method in METHOD_MOMENTS
                and (args.candidate_batch_size > 0 or args.candidate_batch_ratio < 1.0)
            )
            method_seeds = (
                args.seeds
                if method == "random_edge" or stochastic_anchor
                else args.seeds[:1]
            )
            for seed in method_seeds:
                print(
                    f"[run] dataset={dataset} method={method} seed={seed} "
                    f"normalization={args.normalization}",
                    flush=True,
                )
                all_rows.extend(
                    run_method(
                        dataset=dataset,
                        data=data,
                        method=method,
                        seed=seed,
                        max_remove_ratio=args.max_remove_ratio,
                        checkpoint_step=args.checkpoint_step,
                        candidate_batch_ratio=args.candidate_batch_ratio,
                        candidate_batch_size=args.candidate_batch_size,
                        properties=selected_properties,
                        normalization=args.normalization,
                    )
                )

    checkpoint_path = args.output_dir / "checkpoint_metrics.csv"
    write_csv(checkpoint_path, all_rows)
    summary_rows = summarize_rows(
        all_rows, (*selected_properties, "selection_time_sec", "m2", "m3")
    )
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows)

    figures_dir = args.output_dir / "figures"
    for metric in selected_properties:
        plot_metric(
            summary_rows,
            figures_dir / f"{metric}_vs_remove_ratio.png",
            metric,
            method_order=args.methods,
        )

    print(f"[done] wrote {checkpoint_path}", flush=True)
    print(f"[done] wrote {args.output_dir / 'summary_metrics.csv'}", flush=True)
    print(f"[done] wrote figures under {figures_dir}", flush=True)


if __name__ == "__main__":
    main()
