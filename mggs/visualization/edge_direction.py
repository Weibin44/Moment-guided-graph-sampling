"""Moment-change fingerprint computation and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..moments import (
    TopologyMomentState,
    exact_m2_m3_from_state,
    topology_delete_deltas,
)


EPS = 1e-15
SCALE_MODES = ("graph_relative", "std")


@dataclass(frozen=True)
class EdgeDirectionProfile:
    """Numerical result underlying a moment-change fingerprint."""

    edges: np.ndarray
    delta_m2: np.ndarray
    delta_m3: np.ndarray
    scaled_delta_m2: np.ndarray
    scaled_delta_m3: np.ndarray
    angles_deg: np.ndarray
    lengths: np.ndarray
    grid_deg: np.ndarray
    probability: np.ndarray
    scale_mode: str
    m2_scale: float
    m3_scale: float
    original_m2: float
    original_m3: float


@dataclass(frozen=True)
class EdgeDirectionSettings:
    """Numerical settings for an edge-direction profile."""

    scale_mode: str = "graph_relative"
    grid_step_deg: float = 0.5
    bandwidth_deg: float = 3.0
    chunk_size: int = 2048


def _canonical_edges(edges: Iterable[Sequence[int]] | np.ndarray) -> np.ndarray:
    source = edges if hasattr(edges, "shape") else list(edges)
    array = np.asarray(source, dtype=np.int64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("edges must be a two-dimensional array")
    if array.shape[0] == 2 and array.shape[1] != 2:
        array = array.T
    if array.shape[1] != 2:
        raise ValueError("edges must have shape (E, 2) or (2, E)")
    array = np.sort(array, axis=1)
    array = array[array[:, 0] != array[:, 1]]
    return np.unique(array, axis=0)


def _scale(values: np.ndarray, original: float, mode: str) -> float:
    if mode == "graph_relative":
        scale = abs(float(original))
    elif mode == "std":
        finite = values[np.isfinite(values)]
        scale = float(np.std(finite)) if len(finite) else 0.0
    else:
        raise ValueError(f"Unknown scale mode: {mode}. Use one of {SCALE_MODES}.")
    return scale if scale > EPS else 1.0


def _count_profile(
    angles_deg: np.ndarray,
    *,
    grid_step_deg: float,
    bandwidth_deg: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if grid_step_deg <= 0 or bandwidth_deg <= 0:
        raise ValueError("grid_step_deg and bandwidth_deg must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    grid = np.arange(0.0, 360.0, grid_step_deg, dtype=np.float64)
    angles = np.asarray(angles_deg, dtype=np.float64)
    angles = angles[np.isfinite(angles)] % 360.0
    density = np.zeros(len(grid), dtype=np.float64)
    for start in range(0, len(angles), chunk_size):
        batch = angles[start : start + chunk_size]
        diff = (grid[:, None] - batch[None, :] + 180.0) % 360.0 - 180.0
        kernel = np.exp(-0.5 * (diff / bandwidth_deg) ** 2)
        density += kernel @ np.ones(len(batch), dtype=np.float64)
    total = float(np.sum(density))
    return grid, density / max(total, EPS)


def compute_edge_direction_profile(
    edges: Iterable[Sequence[int]] | np.ndarray,
    num_nodes: int,
    *,
    settings: EdgeDirectionSettings | None = None,
) -> EdgeDirectionProfile:
    """Compute the equal-edge-weight circular profile ``p_G`` for a graph.

    The input is interpreted as a simple undirected graph. Duplicate directed
    entries are collapsed and self-loops are ignored.
    """
    settings = settings or EdgeDirectionSettings()
    canonical_edges = _canonical_edges(edges)
    state = TopologyMomentState(None, num_nodes, canonical_edges=canonical_edges)
    original_m2, original_m3 = exact_m2_m3_from_state(state)
    delta_m2, delta_m3 = topology_delete_deltas(state, canonical_edges)
    m2_scale = _scale(delta_m2, original_m2, settings.scale_mode)
    m3_scale = _scale(delta_m3, original_m3, settings.scale_mode)
    scaled_m2 = delta_m2 / m2_scale
    scaled_m3 = delta_m3 / m3_scale
    lengths = np.hypot(scaled_m2, scaled_m3)
    angles = (np.rad2deg(np.arctan2(scaled_m3, scaled_m2)) + 360.0) % 360.0
    angles = np.where(lengths > EPS, angles, np.nan)
    grid, probability = _count_profile(
        angles,
        grid_step_deg=settings.grid_step_deg,
        bandwidth_deg=settings.bandwidth_deg,
        chunk_size=settings.chunk_size,
    )
    return EdgeDirectionProfile(
        edges=canonical_edges,
        delta_m2=delta_m2,
        delta_m3=delta_m3,
        scaled_delta_m2=scaled_m2,
        scaled_delta_m3=scaled_m3,
        angles_deg=angles,
        lengths=lengths,
        grid_deg=grid,
        probability=probability,
        scale_mode=settings.scale_mode,
        m2_scale=m2_scale,
        m3_scale=m3_scale,
        original_m2=float(original_m2),
        original_m3=float(original_m3),
    )


def plot_moment_change_fingerprint(
    profile: EdgeDirectionProfile,
    *,
    ax=None,
    title: str | None = None,
    color: str = "#174A7E",
):
    """Plot ``p_G`` with radius chosen so sector area represents probability."""
    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(subplot_kw={"projection": "polar"})
    else:
        figure = ax.figure
    step_deg = float(profile.grid_deg[1] - profile.grid_deg[0])
    step_rad = np.deg2rad(step_deg)
    radii = probability_to_area_radius(profile.probability, step_deg)
    ax.bar(
        np.deg2rad(profile.grid_deg),
        radii,
        width=step_rad,
        align="edge",
        color=color,
        edgecolor=color,
        linewidth=0.35,
    )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_yticklabels([])
    ax.grid(alpha=0.25)
    if title is not None:
        ax.set_title(title)
    return figure, ax


def probability_to_area_radius(
    probability: np.ndarray,
    angular_width_deg: float,
) -> np.ndarray:
    """Convert sector probabilities to radii whose areas encode probability."""
    if angular_width_deg <= 0:
        raise ValueError("angular_width_deg must be positive")
    width_rad = np.deg2rad(float(angular_width_deg))
    return np.sqrt(2.0 * np.asarray(probability, dtype=np.float64) / width_rad)


def save_moment_change_fingerprint(
    edges: Iterable[Sequence[int]] | np.ndarray,
    num_nodes: int,
    output_path: str | Path,
    *,
    settings: EdgeDirectionSettings | None = None,
    title: str | None = None,
    color: str = "#174A7E",
    dpi: float = 220,
) -> Path:
    """Compute and save one EXP1-style moment-change fingerprint.

    Uses the same moment scaling, circular smoothing and area-proportional
    sectors as EXP1. Duplicate directions collapse to undirected edges and
    self-loops are ignored, as in ``compute_edge_direction_profile``.
    ``num_nodes`` includes isolated nodes. Zero-length deltas contribute no
    direction; a graph with no nonzero deltas produces an empty moment-change fingerprint.

    The suffix selects PNG, SVG or PDF. Parent directories are created, an
    existing output is overwritten, and the figure is closed after saving.
    Returns the output path, without opening an interactive window.
    """
    import matplotlib.pyplot as plt

    path = Path(output_path).expanduser()
    suffix = path.suffix.lower()
    if suffix not in {".png", ".svg", ".pdf"}:
        raise ValueError("output_path must end in .png, .svg or .pdf")
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be finite and positive")
    settings = settings or EdgeDirectionSettings()
    if not np.isfinite(settings.grid_step_deg) or not 0 < settings.grid_step_deg < 360:
        raise ValueError("grid_step_deg must be finite and between 0 and 360")
    if not np.isfinite(settings.bandwidth_deg) or settings.bandwidth_deg <= 0:
        raise ValueError("bandwidth_deg must be finite and positive")
    if not isinstance(num_nodes, (int, np.integer)) or num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer")
    canonical_edges = _canonical_edges(edges)
    if np.any(canonical_edges < 0) or np.any(canonical_edges >= num_nodes):
        raise ValueError("edge endpoint is outside [0, num_nodes)")
    profile = compute_edge_direction_profile(
        canonical_edges, num_nodes, settings=settings,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(subplot_kw={"projection": "polar"})
    try:
        plot_moment_change_fingerprint(profile, ax=ax, title=title, color=color)
        figure.savefig(path, format=suffix[1:], dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
    return path
