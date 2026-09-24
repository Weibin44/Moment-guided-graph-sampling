"""Evaluate moment-delta edge directions against matched random deletion."""

from __future__ import annotations

import json
import math
import shutil

import numpy as np

from mggs.datasets import canonical_dataset_name
from mggs.io import ordered_fieldnames, write_csv_rows
from mggs.moments import TopologyMomentState, estimate_moments, topology_delete_deltas
from mggs.sampling.moment_direction import relative_moment_delta
from experiments.EXP3_graph_learning.supervised_node_classification.supervised_common import base_graph_from_data, load_planetoid, protected_delete_edges, run_root, save_snapshot


DELETE_LABEL_SCOPE = "protect_train_same_label"


def classify_original_edges(base_state, directions, original_m2, original_m3):
    """Assign each original edge to the nearest normalized moment-delta angle."""
    edges = np.asarray(base_state.edge_list, dtype=np.int64)
    delta_m2, delta_m3 = topology_delete_deltas(base_state, edges)
    relative_deltas = np.column_stack(
        [
            delta_m2 / max(abs(original_m2), 1e-12),
            delta_m3 / max(abs(original_m3), 1e-12),
        ]
    )
    angles = np.mod(
        np.arctan2(relative_deltas[:, 1], relative_deltas[:, 0]),
        2.0 * math.pi,
    )
    direction_ids = np.floor(
        angles * directions / (2.0 * math.pi) + 0.5
    ).astype(int) % directions
    return edges, direction_ids, relative_deltas

def filtered_direction_pools(args, data, base_state, edges, direction_ids):
    allowed = protected_delete_edges(data, base_state)
    pools = {}
    for direction_id in range(args.directions):
        pools[direction_id] = np.asarray(
            [
                edge
                for edge, assigned in zip(edges, direction_ids)
                if assigned == direction_id and tuple(edge) in allowed
            ],
            dtype=np.int64,
        ).reshape(-1, 2)
    return pools, allowed

def sampled_snapshot(
    args,
    base_edge_index,
    num_nodes,
    original_m2,
    original_m3,
    selected_edges,
    direction_id,
    budget,
    budget_ratio,
    realized_budget_ratio,
    repeat_id,
    run_seed,
    method,
    snapshots_dir,
    snapshot_idx,
):
    state = TopologyMomentState(base_edge_index, num_nodes)
    for u, v in selected_edges:
        state.apply_del(int(u), int(v))
    edge_index = state.to_edge_index().cpu()
    _, m2, m3, _ = estimate_moments(edge_index, num_nodes, method="exact")
    if direction_id >= 0:
        prefix = f"dir{direction_id:02d}"
    else:
        prefix = "random"
    relative_delta = relative_moment_delta(
        m2, m3, original_m2, original_m3
    )
    ratio_tag = (
        f"ratio{int(round(float(budget_ratio) * 1_000_000)):06d}"
        if direction_id >= 0
        else "ratio_random"
    )
    return save_snapshot(
        edge_index=edge_index,
        snapshot_id="_".join([
            prefix, ratio_tag, f"bud{budget:05d}", f"rep{repeat_id:02d}"
        ]),
        snapshots_dir=snapshots_dir,
        metadata={
            "dataset": canonical_dataset_name(args.dataset),
            "split": args.split,
            "direction_id": direction_id,
            "m2": float(m2),
            "m3": float(m3),
            "rel_delta_m2": float(relative_delta[0]),
            "rel_delta_m3": float(relative_delta[1]),
            "snapshot_idx": snapshot_idx,
            "target_budget": budget,
            "budget_ratio": budget_ratio,
            "realized_budget_ratio": realized_budget_ratio,
            "repeat_id": repeat_id,
            "run_seed": run_seed,
            "method": method,
        },
    )

def generate_snapshots(args):
    output_root = run_root(args.dataset, args.run_name)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    snapshots_dir = output_root / "snapshots"
    snapshots_dir.mkdir(parents=True)

    dataset = load_planetoid(args.dataset, args.dataset_root, args.split)
    data = dataset[0]
    base_state, base_edge_index, original_m2, original_m3 = base_graph_from_data(data)
    edges, direction_ids, relative_deltas = classify_original_edges(
        base_state, args.directions, original_m2, original_m3
    )
    direction_pools, allowed = filtered_direction_pools(
        args, data, base_state, edges, direction_ids
    )
    # The matched random control samples from exactly the same protected pool.
    baseline_pool = np.asarray(sorted(allowed), dtype=np.int64).reshape(-1, 2)
    budget_ratios = sorted({float(ratio) for ratio in args.budget_ratios})
    if not budget_ratios or budget_ratios[0] <= 0.0 or budget_ratios[-1] > 1.0:
        raise ValueError("--budget-ratios must be in the interval (0, 1].")

    class_rows = []
    allowed_mask = np.asarray([tuple(edge) in allowed for edge in edges])
    requested_directions = (
        set(args.direction_ids)
        if args.direction_ids is not None
        else set(range(args.directions))
    )
    direction_ratio_budgets = {}
    for direction_id, pool in direction_pools.items():
        if direction_id not in requested_directions or len(pool) == 0:
            continue
        ratio_budgets = []
        for ratio in budget_ratios:
            budget = min(len(pool), int(math.ceil(ratio * len(pool))))
            realized_ratio = budget / len(pool)
            ratio_budgets.append(
                {
                    "budget_ratio": ratio,
                    "target_budget": budget,
                    "realized_budget_ratio": realized_ratio,
                }
            )
        direction_ratio_budgets[direction_id] = ratio_budgets
    valid_directions = [
        direction_id
        for direction_id, ratio_budgets in direction_ratio_budgets.items()
        if ratio_budgets
    ]
    actual_budgets = sorted(
        {
            row["target_budget"]
            for rows in direction_ratio_budgets.values()
            for row in rows
        }
    )
    if actual_budgets and actual_budgets[-1] > len(baseline_pool):
        raise ValueError("A direction-derived budget exceeds the random pool size.")
    for direction_id, pool in direction_pools.items():
        all_direction_mask = direction_ids == direction_id
        filtered_mask = all_direction_mask & allowed_mask
        class_rows.append(
            {
                "direction_id": direction_id,
                "angle_rad": 2.0 * math.pi * direction_id / args.directions,
                "all_edge_count": int(all_direction_mask.sum()),
                "filtered_edge_count": len(pool),
                "included": int(direction_id in valid_directions),
                "available_budget_ratios": " ".join(
                    f"{row['budget_ratio']:.12g}"
                    for row in direction_ratio_budgets.get(direction_id, [])
                ),
                "ratio_budget_mapping": " ".join(
                    f"{row['budget_ratio']:.12g}:{row['target_budget']}"
                    for row in direction_ratio_budgets.get(direction_id, [])
                ),
                "mean_rel_delta_m2": float(relative_deltas[filtered_mask, 0].mean()) if filtered_mask.any() else np.nan,
                "mean_rel_delta_m3": float(relative_deltas[filtered_mask, 1].mean()) if filtered_mask.any() else np.nan,
            }
        )
    write_csv_rows(output_root / "edge_direction_classes.csv", class_rows)

    if not valid_directions:
        raise ValueError(
            "No requested direction contains eligible edges."
        )
    excluded = sorted(requested_directions - set(valid_directions))
    metadata = []
    snapshot_idx = 0
    for direction_id in valid_directions:
        pool = direction_pools[direction_id]
        for repeat_id in range(args.graph_repeats):
            run_seed = args.seed + direction_id * 1_000_000 + repeat_id
            rng = np.random.default_rng(run_seed)
            ordered_edges = pool[rng.permutation(len(pool))]
            for ratio_budget in direction_ratio_budgets[direction_id]:
                budget = ratio_budget["target_budget"]
                selected = ordered_edges[:budget]
                metadata.append(
                    sampled_snapshot(
                        args,
                        base_edge_index,
                        data.num_nodes,
                        original_m2,
                        original_m3,
                        selected,
                        direction_id,
                        budget,
                        ratio_budget["budget_ratio"],
                        ratio_budget["realized_budget_ratio"],
                        repeat_id,
                        run_seed,
                        "moment_edge_type",
                        snapshots_dir,
                        snapshot_idx,
                    )
                )
                snapshot_idx += 1

    for repeat_id in range(args.graph_repeats):
        run_seed = args.random_seed + repeat_id
        rng = np.random.default_rng(run_seed)
        ordered_edges = baseline_pool[rng.permutation(len(baseline_pool))]
        for budget in actual_budgets:
            selected = ordered_edges[:budget]
            metadata.append(
                sampled_snapshot(
                    args,
                    base_edge_index,
                    data.num_nodes,
                    original_m2,
                    original_m3,
                    selected,
                    -1,
                    budget,
                    np.nan,
                    np.nan,
                    repeat_id,
                    run_seed,
                    "random",
                    snapshots_dir,
                    snapshot_idx,
                )
            )
            snapshot_idx += 1

    if not any(row["method"] == "moment_edge_type" for row in metadata):
        raise ValueError("No eligible direction-ratio snapshots were generated.")
    write_csv_rows(
        output_root / "metadata.csv", metadata, ordered_fieldnames(metadata)
    )
    config = vars(args).copy()
    config.update(
        {
            "dataset": canonical_dataset_name(args.dataset),
            "split": args.split,
            "experiment": "static_moment_edge_types_ratio_vs_random",
            "delete_label_scope": DELETE_LABEL_SCOPE,
            "random_baseline_scope": "same_filter",
            "base_num_edges": len(base_state.edge_list),
            "moment_candidate_edge_count": len(allowed),
            "random_candidate_edge_count": len(baseline_pool),
            "valid_direction_ids": valid_directions,
            "excluded_direction_ids": excluded,
            "direction_ratio_budgets": direction_ratio_budgets,
            "nested_budget_sampling": True,
            "comparison_pairing": "actual_budget_repeat_seed",
            "orig_m2": original_m2,
            "orig_m3": original_m3,
        }
    )
    with (output_root / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True, default=str)
    print(f"[generate] saved {len(metadata)} snapshots to {output_root}")
