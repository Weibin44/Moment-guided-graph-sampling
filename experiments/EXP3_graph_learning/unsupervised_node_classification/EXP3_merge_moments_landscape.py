"""Merge multi-GPU moments-landscape shards and create final plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mggs.io import load_csv_rows as read_rows

from .reporting import analyze
from mggs.io.results import write_json
from mggs.io import write_csv_rows as write_rows

Row = dict[str, Any]


def ratio_name(ratio: float) -> str:
    return f"ratio_{ratio:g}"


def merge_configuration(
    root: Path,
    dataset: str,
    mode: str,
    ratio: float,
    num_targets: int,
    num_seeds: int,
    target_indices: list[int] | None = None,
) -> list[Row]:
    output_dir = root / dataset / mode / ratio_name(ratio)
    selected_targets = target_indices or list(range(num_targets))
    analysis_dir = output_dir
    if target_indices is not None:
        subset_name = "_".join(f"target_{index:03d}" for index in selected_targets)
        analysis_dir = output_dir / "subsets" / subset_name
    shard_paths = [
        output_dir / "shards" / f"target_{index:03d}" / "evaluations.csv"
        for index in selected_targets
    ]
    missing_shards = [str(path) for path in shard_paths if not path.is_file()]
    if missing_shards:
        raise RuntimeError(
            f"{output_dir}: missing evaluation shards: {missing_shards}"
        )

    evaluations: list[Row] = []
    for path in shard_paths:
        evaluations.extend(read_rows(path))
    counts: dict[str, int] = {}
    for row in evaluations:
        point_id = str(row["point_id"])
        counts[point_id] = counts.get(point_id, 0) + 1
    expected = {f"target_{index:03d}" for index in selected_targets}
    observed_points = set(counts)
    if observed_points != expected:
        raise RuntimeError(
            f"{output_dir}: missing/unexpected points: "
            f"{sorted(expected ^ observed_points)}"
        )
    invalid = {key: count for key, count in counts.items() if count != num_seeds}
    if invalid:
        raise RuntimeError(f"{output_dir}: invalid seed counts: {invalid}")

    evaluations.sort(
        key=lambda row: (
            int(row["target_id"]),
            int(row["seed"]),
        )
    )
    write_rows(analysis_dir / "evaluations.csv", evaluations)
    analyze(
        SimpleNamespace(dataset=dataset),
        analysis_dir,
        evaluations,
    )
    return read_rows(analysis_dir / "summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer"])
    parser.add_argument("--modes", nargs="+", default=["static", "dynamic"])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    parser.add_argument("--num-targets", type=int, default=20)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument(
        "--target-indices",
        type=int,
        nargs="+",
        default=None,
        help="Merge only these target shards; default merges every target.",
    )
    args = parser.parse_args()
    if args.target_indices is not None:
        invalid = [
            index
            for index in args.target_indices
            if index < 0 or index >= args.num_targets
        ]
        if invalid:
            parser.error(f"target indices out of range: {invalid}")

    for dataset in args.datasets:
        for mode in args.modes:
            mode_root = args.root / dataset / mode
            aggregate_root = mode_root
            if args.target_indices is not None:
                subset_name = "_".join(
                    f"target_{index:03d}" for index in args.target_indices
                )
                aggregate_root = mode_root / "subsets" / subset_name
            all_points: list[Row] = []
            best_by_test: list[Row] = []
            best_by_validation: list[Row] = []
            for ratio in args.ratios:
                rows = merge_configuration(
                    args.root,
                    dataset,
                    mode,
                    ratio,
                    args.num_targets,
                    args.num_seeds,
                    target_indices=args.target_indices,
                )
                for row in rows:
                    all_points.append(
                        {"dataset": dataset, "mode": mode, "edit_ratio": ratio, **row}
                    )
                best_by_test.append(
                    {
                        "dataset": dataset,
                        "mode": mode,
                        "edit_ratio": ratio,
                        **max(
                            rows,
                            key=lambda row: float(row["test_accuracy_mean"]),
                        ),
                    }
                )
                best_by_validation.append(
                    {
                        "dataset": dataset,
                        "mode": mode,
                        "edit_ratio": ratio,
                        **max(
                            rows,
                            key=lambda row: float(row["val_accuracy_mean"]),
                        ),
                    }
                )

            write_rows(aggregate_root / "all_budget_points.csv", all_points)
            write_rows(
                aggregate_root / "budget_best_by_test_oracle.csv", best_by_test
            )
            write_rows(
                aggregate_root / "budget_best_by_validation.csv",
                best_by_validation,
            )
            write_json(
                aggregate_root / "best_overall_by_test_oracle.json",
                max(best_by_test, key=lambda row: float(row["test_accuracy_mean"])),
            )
            write_json(
                aggregate_root / "best_overall_by_validation.json",
                max(
                    best_by_validation,
                    key=lambda row: float(row["val_accuracy_mean"]),
                ),
            )

    phase_name = "_".join(args.modes)
    if args.target_indices is not None:
        phase_name += "_" + "_".join(
            f"target_{index:03d}" for index in args.target_indices
        )
    write_json(
        args.root / f"completed_{phase_name}.json",
        {"status": "complete", **vars(args)},
    )


if __name__ == "__main__":
    main()
