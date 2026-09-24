"""Evaluate moment-delta edge directions against matched random deletion."""
from __future__ import annotations
import argparse
from pathlib import Path
from mggs.paths import DATASET_DIR

def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="Cora", help="Cora, CiteSeer, or PubMed")
    parser.add_argument("--split", default="public", choices=["public", "full"])
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--stage",
        choices=["all", "generate", "eval", "analyze"],
        default="all",
    )
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--direction-ids", type=int, nargs="+", default=None)
    parser.add_argument(
        "--budget-ratios",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15],
        help="Fractions deleted independently within each direction pool.",
    )
    parser.add_argument("--graph-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--max-eval-snapshots", type=int, default=0)
    parser.add_argument("--overwrite-eval", action="store_true")
    parser.add_argument("--eval-num-shards", type=int, default=1)
    parser.add_argument("--eval-shard-index", type=int, default=0)
    return parser
