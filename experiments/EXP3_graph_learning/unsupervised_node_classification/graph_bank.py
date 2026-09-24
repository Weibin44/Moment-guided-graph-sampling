"""Resumable graph-bank construction; cache schema and seeds remain stable."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import numpy as np

import json
import os
import torch
from mggs.io import load_csv_rows as read_rows
from mggs.io.results import write_rows_atomic, write_json_atomic
from mggs.moments import TopologyMomentState
from mggs.sampling.moment_direction import relative_moment_delta
from .sampling_protocol import sample_target

Row = dict[str, Any]


def prepare_graph_bank(
    output_dir: Path,
    row: Row,
    base_state: TopologyMomentState,
    args: argparse.Namespace,
) -> list[Row]:
    """Create or resume a reusable bank of target-conditioned graphs."""
    bank_dir = output_dir / "graph_bank" / str(row["point_id"])
    bank_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bank_dir / "manifest.csv"
    config_path = bank_dir / "config.json"
    config = {
        "version": 1,
        "dataset": args.dataset,
        "point_id": row["point_id"],
        "target_m2": float(row["target_m2"]),
        "target_m3": float(row["target_m3"]),
        "add_budget": int(row["add_budget"]),
        "delete_budget": int(row["delete_budget"]),
        "total_budget": int(row["total_budget"]),
        "candidate_add": args.candidate_add,
        "candidate_del": args.candidate_del,
        "sampling_seed": args.sampling_seed,
        "graph_bank_size": args.graph_bank_size,
    }
    if config_path.exists():
        with config_path.open(encoding="utf-8") as file:
            cached = json.load(file)
        cached_size = int(cached.pop("graph_bank_size", -1))
        config_without_size = {
            key: value for key, value in config.items() if key != "graph_bank_size"
        }
        if cached != config_without_size or cached_size < 1:
            raise ValueError(
                f"Incompatible graph-bank cache at {bank_dir}. "
                "Use a new output directory."
            )
        if cached_size > args.graph_bank_size:
            raise ValueError(
                f"Graph bank at {bank_dir} was created with size "
                f"{cached_size}, larger than requested size {args.graph_bank_size}."
            )
        if cached_size < args.graph_bank_size:
            write_json_atomic(config_path, config)
    else:
        write_json_atomic(config_path, config)

    entries = read_rows(manifest_path) if manifest_path.exists() else []
    if len(entries) > args.graph_bank_size:
        raise ValueError(f"Graph bank at {bank_dir} is larger than requested.")
    for index, entry in enumerate(entries):
        if int(entry["bank_id"]) != index:
            raise ValueError(f"Non-contiguous graph-bank manifest: {manifest_path}")
        if not (bank_dir / str(entry["snapshot"])).is_file():
            raise FileNotFoundError(bank_dir / str(entry["snapshot"]))

    original_edges = set(base_state.edge_list)
    target = relative_moment_delta(
        float(row["target_m2"]),
        float(row["target_m3"]),
        base_state.m2,
        base_state.m3,
    )
    for bank_id in range(len(entries), args.graph_bank_size):
        sampling_seed = (
            args.sampling_seed
            + (int(row["target_id"]) + 1) * 1_000_003
            + bank_id
        )
        np.random.seed(sampling_seed)
        sampled, add_count, delete_count = sample_target(
            base_state, target, int(row["add_budget"]),
            int(row["delete_budget"]), int(row["total_budget"]), args,
        )
        actual = relative_moment_delta(
            sampled.m2, sampled.m3, base_state.m2, base_state.m3
        )
        snapshot = f"graph_{bank_id:03d}.pt"
        temporary = bank_dir / f".{snapshot}.tmp"
        torch.save({"edge_index": sampled.to_edge_index().cpu()}, temporary)
        os.replace(temporary, bank_dir / snapshot)
        entries.append(
            {
                "bank_id": bank_id,
                "sampling_seed": sampling_seed,
                "m2": float(sampled.m2),
                "m3": float(sampled.m3),
                "target_distance": float(np.linalg.norm(actual - target)),
                "add_count": add_count,
                "delete_count": delete_count,
                "edit_count": len(original_edges ^ set(sampled.edge_list)),
                "snapshot": snapshot,
            }
        )
        write_rows_atomic(manifest_path, entries)
        print(
            f"[bank] {row['point_id']} {bank_id + 1}/{args.graph_bank_size} "
            f"m2={sampled.m2:.6f} m3={sampled.m3:.6f}",
            flush=True,
        )

    return entries
