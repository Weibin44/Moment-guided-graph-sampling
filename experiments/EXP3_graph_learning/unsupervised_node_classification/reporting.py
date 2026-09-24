"""Aggregate EXP3 seed results and export the moment landscape."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import numpy as np

from collections import defaultdict
from mggs.datasets import canonical_dataset_name
from mggs.io import write_csv_rows as write_rows
from mggs.io.results import write_json
from mggs.sampling.moment_direction import relative_moment_delta
from .targets import _TARGET_TYPES

Row = dict[str, Any]


def analyze(args: argparse.Namespace, output_dir: Path, evaluations: list[Row]) -> None:
    if not evaluations:
        raise ValueError("No augmented targets to analyze.")
    if len({int(row["encoder_epoch"]) for row in evaluations}) != 1:
        raise ValueError("Expected evaluations from one encoder epoch.")
    reference = evaluations[0]
    base = {key: float(reference[key]) for key in ("base_m2", "base_m3", "base_edge_count")}
    original_moments = (base["base_m2"], base["base_m3"])
    grouped: defaultdict[str, list[Row]] = defaultdict(list)
    for row in evaluations:
        grouped[str(row["point_id"])].append(row)
    summary: list[Row] = []
    for point_rows in grouped.values():
        first = point_rows[0]
        result = {
            **{key: convert(first[key]) for key, convert in _TARGET_TYPES.items()},
            **base,
            "encoder_epoch": int(first["encoder_epoch"]),
            "num_seeds": len(point_rows),
        }
        for source, mean_key, std_key in (
            ("test_accuracy", "test_accuracy_mean", "test_accuracy_std"),
            ("val_accuracy", "val_accuracy_mean", "val_accuracy_std"),
            ("observed_m2", "m2", "m2_std"),
            ("observed_m3", "m3", "m3_std"),
            ("observed_add_count", "add_count", None),
            ("observed_delete_count", "delete_count", None),
        ):
            values = np.array([float(row[source]) for row in point_rows])
            result[mean_key] = float(values.mean())
            if std_key is not None:
                result[std_key] = float(values.std())
        summary.append(result)
    for row in summary:
        actual = relative_moment_delta(
            float(row["m2"]), float(row["m3"]), *original_moments
        )
        target = relative_moment_delta(
            float(row["target_m2"]), float(row["target_m3"]), *original_moments
        )
        row["rel_delta_m2"] = float(actual[0])
        row["rel_delta_m3"] = float(actual[1])
        row["target_distance"] = float(np.linalg.norm(actual - target))
        row["edge_count"] = (
            base["base_edge_count"]
            + float(row["add_count"])
            - float(row["delete_count"])
        )
    summary.sort(key=lambda row: int(row["target_id"]))
    write_rows(output_dir / "summary.csv", summary)
    best = dict(max(summary, key=lambda row: float(row["test_accuracy_mean"])))
    write_json(output_dir / "best_point_by_test_oracle.json", best)
    best_by_validation = dict(
        max(summary, key=lambda row: float(row["val_accuracy_mean"]))
    )
    write_json(output_dir / "best_point_by_validation.json", best_by_validation)
    for name, key in (("test", "test_accuracy_mean"), ("validation", "val_accuracy_mean")):
        plot_landscape(
            summary, output_dir / f"{name}_accuracy_landscape.png", args.dataset,
            metric_key=key, metric_label=f"{name.capitalize()} accuracy",
            original_moments=original_moments,
        )
    print(
        f"[best:test-oracle] {best['point_id']} "
        f"accuracy={best['test_accuracy_mean']:.4f} "
        f"moments=({float(best['m2']):.6f}, {float(best['m3']):.6f}); "
        f"[best:validation] {best_by_validation['point_id']}",
        flush=True,
    )

def plot_landscape(
    rows: list[Row],
    output_path: Path,
    dataset: str,
    metric_key: str,
    metric_label: str,
    original_moments: tuple[float, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    grouped: defaultdict[tuple[float, float], list[float]] = defaultdict(list)
    for row in rows:
        coordinate = (
            round(float(row["m2"]), 12),
            round(float(row["m3"]), 12),
        )
        grouped[coordinate].append(float(row[metric_key]) * 100.0)
    unique = {
        coordinate: float(np.mean(values)) for coordinate, values in grouped.items()
    }
    x = np.array([point[0] for point in unique])
    y = np.array([point[1] for point in unique])
    z = np.array(list(unique.values()))

    figure, axis = plt.subplots(figsize=(7.4, 5.8))
    can_interpolate = (
        len(unique) >= 3
        and np.linalg.matrix_rank(np.column_stack((x, y)) - [x.mean(), y.mean()]) >= 2
    )
    if can_interpolate:
        triangulation = tri.Triangulation(x, y)
        colors = axis.tricontourf(triangulation, z, levels=24, cmap="viridis")
        axis.scatter(x, y, c=z, cmap="viridis", edgecolors="black", s=20)
    else:
        colors = axis.scatter(x, y, c=z, cmap="viridis", edgecolors="black", s=45)
    best = max(rows, key=lambda row: float(row[metric_key]))
    axis.scatter(
        original_moments[0],
        original_moments[1],
        marker="*",
        color="dodgerblue",
        edgecolors="black",
        s=140,
        label="Original moments",
    )
    axis.scatter(
        float(best["m2"]),
        float(best["m3"]),
        marker="*",
        color="red",
        edgecolors="black",
        s=140,
        label=f"Best {metric_label.lower()}",
    )
    axis.set_xlabel("$m_2$")
    axis.set_ylabel("$m_3$")
    epoch = int(rows[0]["encoder_epoch"])
    axis.set_title(
        f"{canonical_dataset_name(dataset)}: {metric_label} Landscape at epoch {epoch}"
    )
    axis.legend()
    figure.colorbar(colors, ax=axis, label=f"{metric_label} (%)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
