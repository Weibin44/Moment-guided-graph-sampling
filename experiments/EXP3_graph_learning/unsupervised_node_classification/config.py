"""Exact target-moment graph banks for GCL node classification.

Train with an original-graph view and a bank-sampled view, then evaluate
unmasked embeddings on the original graph. Targets form a rectangular
grid or are specified explicitly as one (m2, m3) pair."""
from __future__ import annotations
import argparse
from pathlib import Path

from mggs.datasets import canonical_dataset_name
_REPO_ROOT = Path(__file__).resolve().parents[3]

def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--data-root", type=Path, default=_REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--grid-m2-count", type=int, default=5)
    parser.add_argument("--grid-m3-count", type=int, default=4)
    parser.add_argument("--m2-relative-range", type=float, default=0.5)
    parser.add_argument("--m3-relative-range", type=float, default=0.5)
    parser.add_argument("--include-origin-target", action="store_true")
    parser.add_argument("--target-m2", type=float)
    parser.add_argument("--target-m3", type=float)
    parser.add_argument("--prepared-points", type=Path)
    preparation = parser.add_mutually_exclusive_group()
    preparation.add_argument("--prepare-only", action="store_true")
    preparation.add_argument("--prepare-bank-only", action="store_true")
    parser.add_argument("--target-indices", type=int, nargs="+")
    parser.add_argument("--defer-analysis", action="store_true")
    parser.add_argument("--max-edit-ratio", type=float, default=0.3)
    parser.add_argument("--edit-budget", type=int)
    parser.add_argument("--add-budget", type=int)
    parser.add_argument("--delete-budget", type=int)
    parser.add_argument("--add-budget-ratio", type=float)
    parser.add_argument("--delete-budget-ratio", type=float)
    parser.add_argument("--candidate-add", type=int, default=2000)
    parser.add_argument("--candidate-del", type=int, default=-1)
    parser.add_argument("--sampling-seed", type=int, default=15)
    parser.add_argument("--graph-bank-size", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[15])
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--pf", type=float, default=0.4)
    parser.add_argument("--eval-epochs", type=int, default=5000)
    parser.add_argument("--eval-learning-rate", type=float, default=0.001)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument(
        "--eval-weight-decays", type=float, nargs="+",
        default=[0.0, 0.001, 0.005, 0.01, 0.1],
    )
    return parser

def validate_args(args: argparse.Namespace) -> None:
    args.dataset = canonical_dataset_name(args.dataset)
    if args.dataset not in {"Cora", "CiteSeer"}:
        raise ValueError("Only Cora and CiteSeer are supported.")
    if (args.target_m2 is None) != (args.target_m3 is None):
        raise ValueError("--target-m2 and --target-m3 must be provided together.")
    if args.target_m2 is not None:
        if not (
            0.0 <= args.target_m2 <= 1.0
            and 0.0 <= args.target_m3 <= min(args.target_m2, 0.25)
        ):
            raise ValueError("Target must satisfy 0 <= m2 <= 1 and 0 <= m3 <= min(m2, 0.25).")
        if args.include_origin_target or args.prepared_points is not None:
            raise ValueError(
                "Explicit targets cannot be combined with origin-target or prepared points."
            )
    if args.target_indices is not None and min(args.target_indices) < 0:
        raise ValueError("Target indices must be non-negative.")
    if args.prepare_only and args.prepared_points is not None:
        raise ValueError("Cannot combine --prepare-only and --prepared-points.")
    if args.prepared_points is not None:
        args.prepared_points = args.prepared_points.expanduser().resolve()
        if not args.prepared_points.is_file():
            raise FileNotFoundError(args.prepared_points)

    for add, delete in (
        (args.add_budget, args.delete_budget),
        (args.add_budget_ratio, args.delete_budget_ratio),
    ):
        if (add is None) != (delete is None):
            raise ValueError("Addition and deletion budgets must be provided together.")
    budget_options = (args.edit_budget, args.add_budget, args.add_budget_ratio)
    if sum(value is not None for value in budget_options) > 1:
        raise ValueError("Use only one of edit, exact add/delete, or ratio budgets.")
    exact_budgets = (args.edit_budget, args.add_budget, args.delete_budget)
    if any(value is not None and value < 0 for value in exact_budgets):
        raise ValueError("Edit budgets must be non-negative.")
    ratios = (args.add_budget_ratio, args.delete_budget_ratio)
    if any(value is not None and not 0.0 <= value <= 1.0 for value in ratios):
        raise ValueError("Add/delete budget ratios must be in [0, 1].")
    if not 0.0 < args.max_edit_ratio <= 1.0:
        raise ValueError("Require 0 < max-edit-ratio <= 1.")
    # The shared non-edge sampler requires a positive count; only deletion supports -1.
    if args.candidate_add < 1 or (args.candidate_del != -1 and args.candidate_del < 1):
        raise ValueError("Candidate counts must be positive; candidate-del also accepts -1.")
    for name in ("grid_m2_count", "grid_m3_count", "log_interval", "epochs",
                 "eval_epochs", "eval_interval", "hidden_dim",
                 "m2_relative_range", "m3_relative_range", "learning_rate", "eval_learning_rate"):
        if not getattr(args, name) > 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.graph_bank_size < 2:
        raise ValueError("--graph-bank-size must be at least 2.")
    if args.eval_epochs < args.eval_interval:
        raise ValueError("--eval-epochs must be at least --eval-interval.")
    if not 0.0 <= args.pf <= 1.0:
        raise ValueError("--pf must be in [0, 1].")
    if not args.seeds:
        raise ValueError("At least one training seed is required.")
    if not args.eval_weight_decays or any(not decay >= 0.0 for decay in args.eval_weight_decays):
        raise ValueError("Evaluation weight decays must be non-negative.")
