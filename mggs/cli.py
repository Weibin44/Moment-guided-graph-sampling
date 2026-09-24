"""Command-line access to reusable MGGS functionality."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .moments import MomentDeltaCalculator


def _read_edges(path: Path) -> np.ndarray:
    delimiter = "," if path.suffix.lower() == ".csv" else None
    try:
        edges = np.loadtxt(path, delimiter=delimiter, dtype=np.int64)
    except ValueError:
        edges = np.loadtxt(path, delimiter=delimiter, dtype=np.int64, skiprows=1)
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


def _delta_delete(args: argparse.Namespace) -> None:
    graph_edges = _read_edges(args.edge_list)
    candidates = graph_edges if args.candidates is None else _read_edges(args.candidates)
    calculator = MomentDeltaCalculator.from_undirected_edges(graph_edges, args.num_nodes)
    result = calculator.deletion_deltas(candidates, orders=args.orders, backend=args.backend)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["u", "v", *[f"delta_m{k}" for k in result.orders]]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (u, v) in enumerate(result.edges):
            row = {"u": int(u), "v": int(v)}
            row.update({f"delta_m{k}": float(result[k][idx]) for k in result.orders})
            writer.writerow(row)
    print(f"wrote {len(result.edges)} edge deltas to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mggs", description="Moment-Guided Graph Sampling tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    delta = subparsers.add_parser(
        "delta-delete",
        help="compute exact per-edge deletion delta moments for an edge-list graph",
    )
    delta.add_argument("--edge-list", type=Path, required=True, help="CSV or whitespace u,v edge list")
    delta.add_argument("--num-nodes", type=int, required=True)
    delta.add_argument("--candidates", type=Path, default=None, help="optional candidate edge list")
    delta.add_argument("--orders", nargs="+", type=int, default=[2, 3])
    delta.add_argument("--backend", choices=("topology", "low_rank"), default="low_rank")
    delta.add_argument("--output", type=Path, required=True)
    delta.set_defaults(handler=_delta_delete)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
