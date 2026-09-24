"""Target grids, edit budgets and dataset initialization for EXP3."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from mggs.datasets import canonical_dataset_name
from mggs.io import load_csv_rows as read_rows, write_csv_rows as write_rows
from mggs.moments import TopologyMomentState, estimate_moments

_MOMENT_EPSILON = 1e-12
_TARGET_TYPES = {
    "point_id": str,
    "target_id": int,
    "target_m2": float,
    "target_m3": float,
    "add_budget": int,
    "delete_budget": int,
    "total_budget": int,
}

Row = dict[str, Any]


def load_graph(name: str, root: Path):
    dataset = Planetoid(
        str(root), canonical_dataset_name(name), transform=T.NormalizeFeatures()
    )
    data = dataset[0]
    state = TopologyMomentState(data.edge_index.cpu(), data.num_nodes)
    edge_index = state.to_edge_index().cpu()
    _, m2, m3, _ = estimate_moments(edge_index, data.num_nodes, method="exact")
    state.m2, state.m3 = float(m2), float(m3)
    data.edge_index = edge_index
    return data, state

def rectangular_grid_targets(
    args: argparse.Namespace, base_state: TopologyMomentState
) -> list[tuple[float, float]]:
    """Return a fixed Cartesian grid centered on the original moments."""
    m2_span = base_state.m2 * args.m2_relative_range
    m3_span = base_state.m3 * args.m3_relative_range
    m2_values = np.linspace(
        max(0.0, base_state.m2 - m2_span),
        min(1.0, base_state.m2 + m2_span),
        args.grid_m2_count,
    )
    m3_values = np.linspace(
        max(0.0, base_state.m3 - m3_span),
        min(0.25, base_state.m3 + m3_span),
        args.grid_m3_count,
    )
    targets = [
        (float(m2), float(m3))
        for m3 in m3_values
        for m2 in m2_values
    ]
    invalid = [(m2, m3) for m2, m3 in targets if m3 > m2 + _MOMENT_EPSILON]
    if invalid:
        raise ValueError(
            "The requested rectangular grid contains invalid moments with m3 > m2."
        )
    if args.include_origin_target:
        targets.append((base_state.m2, base_state.m3))
    return targets

def resolve_budgets(args: argparse.Namespace, edge_count: int) -> tuple[int, int, int]:
    if args.edit_budget is not None:
        return args.edit_budget, args.edit_budget, args.edit_budget
    if args.add_budget is not None:
        return (
            args.add_budget,
            args.delete_budget,
            args.add_budget + args.delete_budget,
        )
    if args.add_budget_ratio is not None:
        add_budget = round(edge_count * args.add_budget_ratio)
        delete_budget = round(edge_count * args.delete_budget_ratio)
        return add_budget, delete_budget, add_budget + delete_budget
    combined_budget = max(1, round(edge_count * args.max_edit_ratio))
    return combined_budget, combined_budget, combined_budget

def select_points(args: argparse.Namespace, rows: list[Row]) -> list[Row]:
    """Use the same target filtering for generated and prepared coordinates."""
    available = {int(row["target_id"]) for row in rows}
    if any(target_id < 0 for target_id in available):
        raise ValueError("Target IDs must be non-negative.")
    requested = set(args.target_indices) if args.target_indices is not None else available
    if requested - available:
        raise ValueError(f"Target indices not found: {sorted(requested - available)}")
    selected = [row for row in rows if int(row["target_id"]) in requested]
    if not selected:
        raise ValueError("No requested points found.")
    return selected

def prepare_targets(
    args: argparse.Namespace,
    output_dir: Path,
    base_state: TopologyMomentState,
) -> list[Row]:
    """Load or generate targets and budgets, without constructing graph snapshots."""
    if args.prepared_points is not None:
        rows = [
            {key: convert(row[key]) for key, convert in _TARGET_TYPES.items()}
            for row in read_rows(args.prepared_points)
        ]
    else:
        targets = (
            [(args.target_m2, args.target_m3)]
            if args.target_m2 is not None
            else rectangular_grid_targets(args, base_state)
        )
        add, delete, total = resolve_budgets(args, len(base_state.edge_list))
        rows = [
            {
                "point_id": f"target_{target_id:03d}",
                "target_id": target_id,
                "target_m2": m2,
                "target_m3": m3,
                "add_budget": add,
                "delete_budget": delete,
                "total_budget": total,
            }
            for target_id, (m2, m3) in enumerate(targets)
        ]
    rows = select_points(args, rows)
    write_rows(output_dir / "points.csv", rows)
    return rows
