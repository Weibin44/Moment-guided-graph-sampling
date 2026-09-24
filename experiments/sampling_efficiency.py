"""Benchmark exact greedy and lazy heap sampling with interchangeable backends.

Runs the current efficiency experiment; --benchmark is an optional compatibility flag.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mggs_matplotlib_cache")
os.environ.setdefault("PYG_HOME", "/tmp/mggs_pyg_cache")

from experiments.EXP2_reporting import write_csv
from mggs.datasets import load_graph_dataset
from mggs.io.graph import undirected_simple_edges
from mggs.sampling import MomentPreservingSampler


def parse_benchmark_args() -> tuple[argparse.Namespace, list[tuple[int, ...]]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--delete-budget", type=int, default=100)
    parser.add_argument("--direct-trace-delete-budget", type=int, default=2,
                        help="Separate deletion budget for the direct-trace baseline")
    parser.add_argument("--moment-groups", nargs="+", default=["2,3", "2,3,4"])
    parser.add_argument("--top-ks", nargs="+", type=int, default=[5, 25, 50],
                        help="Candidate counts for heap methods; exact methods run once per moment group")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--normalization", choices=["graph_relative", "std"], default="graph_relative")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        groups = [tuple(sorted(set(map(int, value.split(","))))) for value in args.moment_groups]
    except ValueError:
        parser.error("Moment groups must be comma-separated integers, e.g. 2,3 2,3,4")
    if any(not orders or any(k not in (2, 3, 4) for k in orders) for orders in groups):
        parser.error("Moment groups must contain orders 2, 3 and/or 4")
    if min(args.delete_budget, args.direct_trace_delete_budget, args.threads, *args.top_ks) < 1:
        parser.error("Budget, threads and top-k must be positive")
    if len(groups) != len(set(groups)) or len(args.top_ks) != len(set(args.top_ks)):
        parser.error("Duplicate moment groups or top-k values")
    return args, groups


def make_sampler(edges, num_nodes, orders, variant, normalization):
    strategy, backend, top_k = variant
    return MomentPreservingSampler.from_edges(
        edges, num_nodes, orders=orders, backend=backend, strategy=strategy,
        top_k=max(1, top_k), normalization=normalization,
    )


def time_sampling(edges, num_nodes, orders, variant, normalization, budget):
    """Time one fresh trajectory; exclude reporting and sampler disposal."""
    started = time.perf_counter()
    sampler = make_sampler(edges, num_nodes, orders, variant, normalization)
    prepared = time.perf_counter()
    for _ in range(budget):
        sampler.step()
    finished = time.perf_counter()
    return dict(preprocess_sec=prepared - started, sampling_sec=finished - prepared,
                total_sec=finished - started, mean_step_ms=1000 * (finished - prepared) / budget)


def method_metadata(orders, variant):
    strategy, backend, top_k = variant
    if backend == "auto":  # Historical benchmark configurations.
        backend = "hybrid"
    actual_backend = ("hybrid" if 4 in orders else "topology") if backend == "hybrid" else backend
    label = {"low_rank": "Low-rank", "hybrid": "Hybrid", "topology": "Topology",
             "direct_trace": "Direct trace"}[actual_backend]
    if strategy == "heap":
        label = f"Lazy heap {label} (K={top_k})"
    return dict(orders=",".join(map(str, orders)), backend=actual_backend,
                strategy=strategy, top_k=top_k, method=label)


def benchmark_main() -> None:
    args, groups = parse_benchmark_args()
    variants = [("greedy", "direct_trace", 0)] + [
                (strategy, backend, k) for strategy in ("greedy", "heap")
                for backend in ("hybrid", "low_rank")
                for k in (args.top_ks if strategy == "heap" else [0])]
    torch.set_num_threads(args.threads)
    data = load_graph_dataset(args.dataset)
    edges = np.asarray(undirected_simple_edges(data.edge_index), dtype=np.int64)
    num_nodes = int(data.num_nodes)
    if max(args.delete_budget, args.direct_trace_delete_budget) > len(edges):
        raise SystemExit("Delete budget exceeds the number of undirected edges")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), num_nodes=num_nodes, num_edges=len(edges),
                  timing="preprocess + selection and graph updates; excludes loading, JIT and evaluation",
                  backend_note="hybrid uses topology for m2/m3 and low_rank for m4")
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    rows = []
    total_runs = len(groups) * len(variants)
    print(f"[benchmark] {total_runs} runs; deletions: direct trace={args.direct_trace_delete_budget}, "
          f"other methods={args.delete_budget}; "
          f"heap K={args.top_ks}", flush=True)
    with threadpool_limits(limits=args.threads):
        # K changes the candidate count, not the code path that needs warming.
        toy = np.asarray([(0, 1), (0, 2), (1, 2), (2, 3)], dtype=np.int64)
        warm_variants = dict.fromkeys((strategy, backend) for strategy, backend, _ in variants)
        for orders in groups:
            for strategy, backend in warm_variants:
                time_sampling(toy, 4, orders, (strategy, backend, args.top_ks[0]),
                              args.normalization, budget=1)
        for orders in groups:
            for variant in variants:
                budget = args.direct_trace_delete_budget if variant[1] == "direct_trace" else args.delete_budget
                timing = time_sampling(edges, num_nodes, orders, variant,
                                       args.normalization, budget)
                row = dict(dataset=args.dataset, **method_metadata(orders, variant),
                           normalization=args.normalization, delete_budget=budget, **timing)
                rows.append(row)
                write_csv(args.output_dir / "timings.csv", rows)
                print(f"[timing {len(rows)}/{total_runs}] "
                      f"{row['orders']} {row['method']} deleted={budget}: {row['total_sec']:.4f}s", flush=True)


def main() -> None:
    benchmark_main()


if __name__ == "__main__":
    main()
