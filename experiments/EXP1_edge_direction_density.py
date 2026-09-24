"""Generate the paper's area-normalized moment-change fingerprint panel.

All per-graph calculations use the public edge-direction visualization API.
This module owns only dataset loading, aggregation, panel layout, and outputs.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mggs_matplotlib_cache")
os.environ.setdefault("PYG_HOME", "/tmp/mggs_pyg_cache")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch_geometric.datasets import TUDataset

from mggs.datasets import load_graph_dataset
from mggs.io.graph import undirected_simple_edges
from mggs.paths import DATASET_DIR, RECORDS_DIR
from mggs.visualization.edge_direction import (
    EdgeDirectionProfile,
    EdgeDirectionSettings,
    SCALE_MODES,
    compute_edge_direction_profile,
    probability_to_area_radius,
)

SINGLE_GRAPH_DATASETS = ("Cora", "CiteSeer", "Texas", "Cornell", "Photo", "Computers")
MULTIGRAPH_DATASETS = ("PROTEINS", "ENZYMES")
SNAP_DATASETS = ("ego-Facebook", "ego-Twitter")
REFERENCE_GRAPH_SPECS = (
    ("Complete", 20),
    ("Cycle", 60),
    ("Path", 60),
    ("Lollipop", 30),
    ("House", 5),
    ("Bull", 5),
    ("Friendship", 41),
    ("Balanced binary tree", 31),
)
REAL_GRAPH_COLUMNS = (
    ("Citation", ("Cora", "CiteSeer")),
    ("Web pages", ("Texas", "Cornell")),
    ("Co-purchase", ("Photo", "Computers")),
    ("Biological", MULTIGRAPH_DATASETS),
    ("Social", SNAP_DATASETS),
)
CLASSIC_GRAPH_ORDER = (
    "Cycle",
    "Bull",
    "Friendship",
    "Complete",
    "Lollipop",
    "House",
    "Balanced binary tree",
)
GRAPH_COLORS = {
    "Cora": "#174A7E",
    "CiteSeer": "#7DD3FC",
    "Texas": "#B91C1C",
    "Cornell": "#FB7185",
    "Photo": "#D97706",
    "Computers": "#FBBF24",
    "PROTEINS": "#0E7490",
    "ENZYMES": "#22D3EE",
    "ego-Facebook": "#6D28D9",
    "ego-Twitter": "#C084FC",
    "Cycle": "#CCBB44",
    "Bull": "#66CCEE",
    "Friendship": "#555555",
    "Complete": "#EE6677",
    "Lollipop": "#AA3377",
    "House": "#228833",
    "Balanced binary tree": "#4477AA",
}
SNAP_URLS = {
    "ego-Facebook": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    "ego-Twitter": "https://snap.stanford.edu/data/twitter_combined.txt.gz",
}


def _reference_edges(name: str, n: int) -> list[tuple[int, int]]:
    """Construct a classic graph as canonical undirected edges."""
    if name == "Complete":
        return [(u, v) for u in range(n) for v in range(u + 1, n)]
    if name == "Path":
        return [(u, u + 1) for u in range(n - 1)]
    if name == "Cycle":
        return _reference_edges("Path", n) + [(0, n - 1)]
    if name == "Lollipop":
        clique_size = n // 2
        edges = [
            (u, v)
            for u in range(clique_size)
            for v in range(u + 1, clique_size)
        ]
        edges.extend((u, u + 1) for u in range(clique_size, n - 1))
        edges.append((clique_size - 1, clique_size))
        return edges
    if name == "House":
        return [(0, 1), (1, 2), (2, 3), (0, 3), (2, 4), (3, 4)]
    if name == "Bull":
        return [(0, 1), (1, 2), (0, 2), (0, 3), (1, 4)]
    if name == "Friendship":
        return [
            edge
            for blade in range(20)
            for edge in (
                (0, 2 * blade + 1),
                (0, 2 * blade + 2),
                (2 * blade + 1, 2 * blade + 2),
            )
        ]
    if name == "Balanced binary tree":
        return [
            (parent, child)
            for parent in range(15)
            for child in (2 * parent + 1, 2 * parent + 2)
        ]
    raise ValueError(f"Unknown reference graph: {name}")


# Dataset loading and aggregation ------------------------------------------


def _load_snap(name: str) -> tuple[np.ndarray, int]:
    """Load and relabel an official SNAP combined ego-network."""
    url = SNAP_URLS[name]
    raw = DATASET_DIR / name / "raw" / Path(url).name
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        urllib.request.urlretrieve(url, raw)
    edges = set()
    with gzip.open(raw, "rt", encoding="utf-8") as f:
        for line in f:
            x = line.split()
            if len(x) == 2:
                u, v = map(int, x)
                if u != v:
                    edges.add((min(u, v), max(u, v)))
    a = np.asarray(sorted(edges), dtype=np.int64)
    nodes, inv = np.unique(a.reshape(-1), return_inverse=True)
    return inv.reshape(-1, 2), len(nodes)


def compute_single_graph_profiles(
    settings: EdgeDirectionSettings,
) -> dict[str, EdgeDirectionProfile]:
    """Compute profiles for ordinary, SNAP, and classic single graphs."""
    profiles: dict[str, EdgeDirectionProfile] = {}
    for name in SINGLE_GRAPH_DATASETS:
        data = load_graph_dataset(name)
        profiles[name] = compute_edge_direction_profile(
            undirected_simple_edges(data.edge_index),
            int(data.num_nodes),
            settings=settings,
        )
        del data
        gc.collect()
    for name in SNAP_DATASETS:
        start = time.perf_counter()
        edges, num_nodes = _load_snap(name)
        print(
            f"[snap] {name}: {num_nodes:,} nodes, {len(edges):,} undirected edges; "
            f"loaded in {time.perf_counter() - start:.1f}s",
            flush=True,
        )
        start = time.perf_counter()
        profiles[name] = compute_edge_direction_profile(
            edges,
            num_nodes,
            settings=settings,
        )
        print(
            f"[snap] {name}: profile in {time.perf_counter() - start:.1f}s",
            flush=True,
        )
        del edges
        gc.collect()
    for name, num_nodes in REFERENCE_GRAPH_SPECS:
        profiles[name] = compute_edge_direction_profile(
            _reference_edges(name, num_nodes),
            num_nodes,
            settings=settings,
        )
    return profiles


def compute_multigraph_profiles(
    settings: EdgeDirectionSettings,
) -> dict[str, list[EdgeDirectionProfile]]:
    """Compute one profile per usable graph in each graph collection."""
    profiles_by_dataset: dict[str, list[EdgeDirectionProfile]] = {}
    for name in MULTIGRAPH_DATASETS:
        ds = TUDataset(root=str(DATASET_DIR / name), name=name)
        graph_profiles: list[EdgeDirectionProfile] = []
        for graph in ds:
            edges = undirected_simple_edges(graph.edge_index)
            if edges:
                profile = compute_edge_direction_profile(
                    edges,
                    int(graph.num_nodes),
                    settings=settings,
                )
                if np.any(np.isfinite(profile.angles_deg)):
                    graph_profiles.append(profile)
        profiles_by_dataset[name] = graph_profiles
        del ds
        gc.collect()
    return profiles_by_dataset


# Figure rendering ----------------------------------------------------------


def _style_polar_axis(ax: plt.Axes, radial_limit: float) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids(np.arange(0, 360, 45))
    ax.set_ylim(0, radial_limit)
    ax.set_yticklabels([])
    ax.grid(alpha=0.25)


def _save_figure(fig: plt.Figure, output_base: Path) -> None:
    """Save matching raster, publication, and PowerPoint-editable formats."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")

    polar_patches = [
        patch
        for axis in fig.axes
        if axis.name == "polar"
        for patch in axis.patches
    ]
    original_rasterized_states = [
        patch.get_rasterized() for patch in polar_patches
    ]
    try:
        for patch in polar_patches:
            patch.set_rasterized(True)
        fig.savefig(
            output_base.with_suffix(".pdf"),
            dpi=240,
            bbox_inches="tight",
        )
    finally:
        for patch, state in zip(polar_patches, original_rasterized_states):
            patch.set_rasterized(state)
    plt.close(fig)


def plot_real_graph_panel(
    single_profiles: dict[str, EdgeDirectionProfile],
    multigraph_profiles: dict[str, list[EdgeDirectionProfile]],
    output_base: Path,
) -> None:
    """Render the 2-by-5 real-world graph comparison panel."""
    first = next(iter(single_profiles.values()))
    step = float(first.grid_deg[1] - first.grid_deg[0])
    theta = np.deg2rad(first.grid_deg)
    width = np.deg2rad(step)
    radii = {
        name: probability_to_area_radius(profile.probability, step)
        for name, profile in single_profiles.items()
    }
    bands = {}
    for name, profiles in multigraph_profiles.items():
        a = np.asarray([p.probability for p in profiles])
        radii[name] = probability_to_area_radius(np.mean(a, axis=0), step)
        bands[name] = (
            probability_to_area_radius(np.quantile(a, 0.25, axis=0), step),
            probability_to_area_radius(np.quantile(a, 0.75, axis=0), step),
        )
    real_names = [name for _, names in REAL_GRAPH_COLUMNS for name in names]
    radial_limit = max(float(np.max(radii[name])) for name in real_names)
    fig = plt.figure(figsize=(21, 8.9))
    grid = fig.add_gridspec(
        2, 5, left=0.027, right=0.973, bottom=0.065, top=0.865,
        wspace=0.24, hspace=0.40,
    )
    column_centers = [
        (grid[:, col].get_position(fig).x0 + grid[:, col].get_position(fig).x1) / 2
        for col in range(len(REAL_GRAPH_COLUMNS))
    ]
    background_edges = [column_centers[0] - 0.097] + [
        (left + right) / 2
        for left, right in zip(column_centers[:-1], column_centers[1:])
    ] + [column_centers[-1] + 0.097]
    axes: dict[str, plt.Axes] = {}
    for col, (domain, names) in enumerate(REAL_GRAPH_COLUMNS):
        domain_color = GRAPH_COLORS[names[0]]
        center = column_centers[col]
        fig.add_artist(Rectangle(
            (background_edges[col], 0.012),
            background_edges[col + 1] - background_edges[col], 0.976,
            transform=fig.transFigure, facecolor=domain_color,
            edgecolor="none", alpha=0.065, antialiased=False, zorder=-1,
        ))
        fig.text(
            center, 0.958, domain, ha="center", va="center",
            fontsize=20, fontweight="bold", color=domain_color,
        )
        for row, name in enumerate(names):
            ax = fig.add_subplot(grid[row, col], projection="polar")
            axes[name] = ax
            ax.set_facecolor("none")
            suffix = " (mean + IQR)" if name in multigraph_profiles else ""
            ax.set_title(f"{name}{suffix}", pad=16, fontsize=14)

    for name, ax in axes.items():
        if name in bands:
            lo, hi = bands[name]
            ax.bar(
                theta,
                hi - lo,
                bottom=lo,
                width=width,
                align="edge",
                color=GRAPH_COLORS[name],
                edgecolor="none",
                alpha=0.30,
                zorder=1,
            )
        ax.bar(
            theta,
            radii[name],
            width=width,
            align="edge",
            color=GRAPH_COLORS[name],
            edgecolor=GRAPH_COLORS[name],
            alpha=0.90,
            linewidth=0.7,
            zorder=2,
        )
        _style_polar_axis(ax, radial_limit)
    _save_figure(fig, output_base)


def plot_classic_graph_panel(
    single_profiles: dict[str, EdgeDirectionProfile],
    output_base: Path,
) -> None:
    """Render classic graph distributions as a standalone editable panel."""
    first_profile = next(iter(single_profiles.values()))
    step_deg = float(first_profile.grid_deg[1] - first_profile.grid_deg[0])
    theta = np.deg2rad(first_profile.grid_deg)
    width = np.deg2rad(step_deg)
    radii = {
        name: probability_to_area_radius(
            single_profiles[name].probability,
            step_deg,
        )
        for name in CLASSIC_GRAPH_ORDER
    }
    radial_limit = max(float(np.max(radius)) for radius in radii.values())

    figure = plt.figure(figsize=(12, 8.5))
    # Keep explicit top clearance so SVG-to-EMF conversion does not clip 90°.
    axis = figure.add_axes([0.21, 0.29, 0.58, 0.60], projection="polar")
    for zorder, name in enumerate(CLASSIC_GRAPH_ORDER, start=2):
        axis.bar(
            theta,
            radii[name],
            width=width,
            align="edge",
            color=GRAPH_COLORS[name],
            edgecolor=GRAPH_COLORS[name],
            alpha=0.65,
            linewidth=1,
            label=name,
            zorder=zorder,
        )
    _style_polar_axis(axis, radial_limit)
    axis.set_thetagrids(
        np.arange(0, 360, 45),
        labels=("0°", "45°", "", "135°", "180°", "225°", "270°", "315°"),
    )

    angle_label_axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    angle_label_axis.set_axis_off()
    angle_label_axis.text(0.5, 0.91, "90°", ha="center", va="center")

    # Use ordinary figure artists instead of Matplotlib's Legend container.
    # LibreOffice otherwise drops or clips legend rows during SVG-to-EMF export.
    legend_axis = figure.add_axes([0.05, 0.06, 0.90, 0.12])
    legend_axis.set_axis_off()
    legend_x = (0.055, 0.170, 0.280, 0.415, 0.540, 0.655, 0.775)
    for x, name in zip(legend_x, CLASSIC_GRAPH_ORDER):
        legend_axis.add_patch(
            Rectangle(
                (x, 0.43),
                0.022,
                0.14,
                transform=legend_axis.transAxes,
                facecolor=GRAPH_COLORS[name],
                edgecolor="none",
            )
        )
        legend_axis.text(
            x + 0.028,
            0.50,
            name,
            transform=legend_axis.transAxes,
            ha="left",
            va="center",
            fontsize=9,
        )
    _save_figure(figure, output_base)


# Experiment entry point ---------------------------------------------------


def run(args: argparse.Namespace) -> None:
    output_dir = (
        args.output_dir
        or RECORDS_DIR
        / "spectrum_sampling"
        / f"edge_direction_density_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = EdgeDirectionSettings(
        scale_mode=args.scale_mode,
        grid_step_deg=args.density_grid_step_deg,
        bandwidth_deg=args.kde_bandwidth_deg,
    )
    single_profiles = compute_single_graph_profiles(settings)
    multigraph_profiles = compute_multigraph_profiles(settings)
    figures_dir = output_dir / "figures"
    plot_real_graph_panel(
        single_profiles,
        multigraph_profiles,
        figures_dir / "real_graphs_edge_direction_rose",
    )
    plot_classic_graph_panel(
        single_profiles,
        figures_dir / "classic_graphs_edge_direction_rose",
    )
    summary = {
        "datasets": list(SINGLE_GRAPH_DATASETS),
        "multigraph_datasets": list(MULTIGRAPH_DATASETS),
        "snap_datasets": list(SNAP_DATASETS),
        "reference_graphs": [name for name, _ in REFERENCE_GRAPH_SPECS],
        "scale_mode": args.scale_mode,
        "density_grid_step_deg": args.density_grid_step_deg,
        "kde_bandwidth_deg": args.kde_bandwidth_deg,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str)
    )
    print(f"[done] wrote edge density results to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scale-mode", choices=SCALE_MODES, default="graph_relative")
    parser.add_argument("--density-grid-step-deg", type=float, default=0.5)
    parser.add_argument("--kde-bandwidth-deg", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    torch.set_num_threads(1)
    run(parse_args())


if __name__ == "__main__":
    main()
