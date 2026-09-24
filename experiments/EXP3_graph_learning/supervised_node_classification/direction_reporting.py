"""Evaluate moment-delta edge directions against matched random deletion."""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from mggs.io import load_csv_rows, ordered_fieldnames, write_csv_rows
from experiments.EXP3_graph_learning.supervised_node_classification.supervised_common import resolve_run_name, run_root
from experiments.EXP3_graph_learning.supervised_node_classification.EXP3_plot_edge_direction_sector import (
    plot_direction_ratio_sector,
)


def paired_results(metadata, eval_rows):
    metadata_by_id = {row["snapshot_id"]: row for row in metadata}
    eval_by_snapshot = defaultdict(list)
    for row in eval_rows:
        eval_by_snapshot[row["snapshot_id"]].append(row)

    random_by_budget_repeat_seed = {}
    for snapshot_id, rows in eval_by_snapshot.items():
        meta = metadata_by_id[snapshot_id]
        if meta["method"] != "random":
            continue
        budget = int(meta["target_budget"])
        repeat_id = int(meta["repeat_id"])
        for row in rows:
            key = (budget, repeat_id, int(row["seed"]))
            random_by_budget_repeat_seed[key] = float(row["test_acc"]) * 100.0

    paired = []
    for snapshot_id, rows in eval_by_snapshot.items():
        meta = metadata_by_id[snapshot_id]
        if meta["method"] != "moment_edge_type":
            continue
        budget = int(meta["target_budget"])
        repeat_id = int(meta["repeat_id"])
        for row in rows:
            seed = int(row["seed"])
            random_key = (budget, repeat_id, seed)
            if random_key not in random_by_budget_repeat_seed:
                continue
            moment_accuracy = float(row["test_acc"]) * 100.0
            random_accuracy = random_by_budget_repeat_seed[random_key]
            paired.append(
                {
                    "dataset": meta["dataset"],
                    "snapshot_id": snapshot_id,
                    "direction_id": int(meta["direction_id"]),
                    "target_budget": budget,
                    "budget_ratio": float(meta["budget_ratio"]),
                    "realized_budget_ratio": float(meta["realized_budget_ratio"]),
                    "edge_repeat": int(meta["repeat_id"]),
                    "seed": seed,
                    "moment_test_acc": moment_accuracy,
                    "random_test_acc_mean": random_accuracy,
                    "delta_vs_random": moment_accuracy - random_accuracy,
                    "rel_delta_m2": float(meta["rel_delta_m2"]),
                    "rel_delta_m3": float(meta["rel_delta_m3"]),
                }
            )
    return paired, random_by_budget_repeat_seed

def summarize_groups(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summaries = []
    for group, rows_in_group in sorted(grouped.items()):
        values = np.asarray([float(row["delta_vs_random"]) for row in rows_in_group])
        moment_values = np.asarray([float(row["moment_test_acc"]) for row in rows_in_group])
        random_values = np.asarray([float(row["random_test_acc_mean"]) for row in rows_in_group])
        summary = dict(zip(keys, group))
        summary.update(
            {
                "delta_vs_random_mean": float(values.mean()),
                "delta_vs_random_std": float(values.std(ddof=0)),
                "ci95": float(1.96 * values.std(ddof=0) / math.sqrt(len(values))),
                "moment_test_acc_mean": float(moment_values.mean()),
                "moment_test_acc_std": float(moment_values.std(ddof=0)),
                "random_test_acc_mean": float(random_values.mean()),
                "random_test_acc_std": float(random_values.std(ddof=0)),
                "n_paired": len(values),
            }
        )
        summaries.append(summary)
    return summaries

def merge_eval_shards(output_root):
    """Merge disjoint evaluation shard CSVs into the canonical result file."""
    shard_paths = sorted(output_root.glob("eval_results.shard_*_of_*.csv"))
    eval_path = output_root / "eval_results.csv"
    if not shard_paths:
        if not eval_path.exists():
            raise FileNotFoundError(f"Missing evaluation results under {output_root}")
        return eval_path

    rows_by_key = {}
    for shard_path in shard_paths:
        for row in load_csv_rows(shard_path):
            key = (row["snapshot_id"], int(row["seed"]))
            if key in rows_by_key:
                raise ValueError(f"Duplicate sharded evaluation row: {key}")
            rows_by_key[key] = row

    metadata = load_csv_rows(output_root / "metadata.csv")
    with (output_root / "config.json").open() as handle:
        config = json.load(handle)
    max_snapshots = int(config.get("max_eval_snapshots", 0))
    evaluated_snapshot_count = (
        min(max_snapshots, len(metadata)) if max_snapshots > 0 else len(metadata)
    )
    expected_count = evaluated_snapshot_count * len(config["eval_seeds"])
    if len(rows_by_key) != expected_count:
        raise ValueError(
            f"Incomplete evaluation shards: found {len(rows_by_key)} rows, "
            f"expected {expected_count}."
        )
    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (int(row["snapshot_idx"]), int(row["seed"])),
    )
    write_csv_rows(eval_path, rows, ordered_fieldnames(rows))
    print(f"[merge] saved {len(rows)} evaluation rows to {eval_path}")
    return eval_path

def analyze(args):
    args.run_name = resolve_run_name(args.dataset, args.run_name, args.split)
    output_root = run_root(args.dataset, args.run_name)
    eval_path = merge_eval_shards(output_root)
    metadata = load_csv_rows(output_root / "metadata.csv")
    paired, random_by_budget_repeat_seed = paired_results(
        metadata, load_csv_rows(eval_path)
    )
    if not paired:
        raise ValueError("No paired moment-edge and random results were found.")
    direction_ratio = summarize_groups(
        paired,
        [
            "direction_id",
            "budget_ratio",
            "target_budget",
            "realized_budget_ratio",
        ],
    )
    direction = summarize_groups(paired, ["direction_id"])
    random_summary = []
    for budget in sorted({key[0] for key in random_by_budget_repeat_seed}):
        values = np.asarray([
            value
            for (row_budget, _, _), value in random_by_budget_repeat_seed.items()
            if row_budget == budget
        ])
        random_summary.append(
            {
                "target_budget": budget,
                "test_acc_mean": float(values.mean()),
                "test_acc_std": float(values.std(ddof=0)),
                "n_evaluations": len(values),
            }
        )
    write_csv_rows(output_root / "paired_results.csv", paired)
    write_csv_rows(output_root / "direction_ratio_summary.csv", direction_ratio)
    write_csv_rows(output_root / "direction_summary.csv", direction)
    write_csv_rows(output_root / "random_baseline_summary.csv", random_summary)
    with (output_root / "config.json").open() as handle:
        config = json.load(handle)
    plot_direction_ratio_sector(
        direction_ratio,
        output_root / "figures" / "direction_ratio_sector_map.pdf",
        total_directions=int(config["directions"]),
        budget_ratios=config["budget_ratios"],
    )
    print(f"[analyze] results saved to {output_root}")
