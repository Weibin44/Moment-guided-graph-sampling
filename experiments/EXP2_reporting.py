"""CSV aggregation and plotting for the Exp2 property-preservation experiment."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METHOD_ORDER = (
    "anchor_m2",
    "anchor_m3",
    "anchor_m4",
    "anchor_m2_m3",
    "anchor_m2_m4",
    "anchor_m3_m4",
    "anchor_m2_m3_m4",
    "random_edge",
    "spectral_greedy_oracle",
)

GROUP_COLUMNS = {"dataset", "method", "remove_ratio"}
CONFIG_COLUMNS = ("orders", "backend", "strategy", "top_k", "normalization")
NON_METRIC_COLUMNS = GROUP_COLUMNS | {"seed", "step", "initial_edges", "remaining_edges"}
NON_METRIC_COLUMNS.update(CONFIG_COLUMNS)
METRIC_LABELS = {
    "spectrum_rmse": "Spectral RMSE",
    "mean_neighbor_inverse_degree_abs_error": "Neighbor Inverse-Degree Error",
    "mean_triangle_weighted_clustering_abs_error": (
        "Triangle-Weighted Clustering Error"
    ),
    "normalized_estrada_index_abs_error": "Normalized Estrada Index Error",
}


def write_csv(path: Path, rows) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows, metrics: Iterable[str] | None = None):
    """Aggregate numeric checkpoint metrics over seeds.

    When ``metrics`` is omitted, numeric columns are discovered from the rows so
    new graph-property metrics require no reporting changes.
    """
    grouped = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["method"]), float(row["remove_ratio"])) + tuple(
            str(row.get(column, "")) for column in CONFIG_COLUMNS
        )
        grouped.setdefault(key, []).append(row)
    summary = []
    for key, group in sorted(grouped.items()):
        dataset, method, ratio = key[:3]
        result = {
            "dataset": dataset,
            "method": method,
            "remove_ratio": ratio,
            "n_runs": len(group),
            "remaining_edges_mean": float(
                np.mean([float(row["remaining_edges"]) for row in group])
            ),
        }
        result.update({column: group[0][column] for column in CONFIG_COLUMNS if column in group[0]})
        candidate_metrics = list(metrics or ())
        candidate_metrics.extend(
            key
            for row in group
            for key in row
            if key not in NON_METRIC_COLUMNS and key not in candidate_metrics
        )
        for metric in candidate_metrics:
            if any(metric not in row or row[metric] in {"", None} for row in group):
                continue
            try:
                values = np.asarray([float(row[metric]) for row in group])
            except (TypeError, ValueError):
                continue
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = float(np.std(values))
        summary.append(result)
    return summary


def plot_metric(
    summary_rows,
    out_path: Path,
    metric: str,
    exclude_methods: set[str] | None = None,
    method_order: Iterable[str] | None = None,
    method_labels: dict[str, str] | None = None,
) -> None:
    datasets = sorted({str(row["dataset"]) for row in summary_rows})
    excluded = exclude_methods or set()
    present = {str(row["method"]) for row in summary_rows}
    ordered = list(method_order or DEFAULT_METHOD_ORDER)
    ordered.extend(sorted(present - set(ordered)))
    methods = [
        method for method in ordered
        if method in present and method not in excluded
    ]
    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5.2 * len(datasets), 3.8),
        squeeze=False,
    )
    for axis, dataset in zip(axes[0], datasets):
        for method in methods:
            rows = sorted(
                (
                    row for row in summary_rows
                    if row["dataset"] == dataset and row["method"] == method
                ),
                key=lambda row: float(row["remove_ratio"]),
            )
            if not rows:
                continue
            x = np.asarray([float(row["remove_ratio"]) for row in rows])
            y = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
            error = np.asarray([float(row[f"{metric}_std"]) for row in rows])
            line, = axis.plot(x, y, marker="o", linewidth=1.8, markersize=3.0,
                              label=(method_labels or {}).get(method, method))
            if np.any(error > 0):
                axis.fill_between(x, y - error, y + error, alpha=0.14, color=line.get_color())
        axis.set_title(dataset, fontsize=16)
        axis.set_xlabel("Removed edge ratio", fontsize=12)
        axis.tick_params(axis="both", labelsize=12)
        axis.grid(True, alpha=0.25)
    axes[0][0].set_ylabel(METRIC_LABELS.get(metric, metric.replace("_", " ")), fontsize=12)
    axes[0][-1].legend(frameon=False, fontsize=10)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=220)
    figure.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
