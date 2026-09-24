"""Plot the supervised edge-direction result from an existing summary CSV."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from mggs.io import load_csv_rows


def plot_direction_ratio_sector(
    rows,
    output_path,
    total_directions=8,
    budget_ratios=None,
):
    """Draw one polar map: angle is moment direction, radius is deletion ratio."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Patch

    if not rows:
        raise ValueError("The direction-ratio summary is empty.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ratios = sorted(
        {float(row["budget_ratio"]) for row in rows}
        if budget_ratios is None
        else {float(ratio) for ratio in budget_ratios}
    )
    direction_ids = list(range(total_directions))
    direction_index = {value: index for index, value in enumerate(direction_ids)}
    ratio_index = {round(value, 12): index for index, value in enumerate(ratios)}
    values = np.full((total_directions, len(ratios)), np.nan)
    for row in rows:
        direction_id = int(row["direction_id"])
        ratio = round(float(row["budget_ratio"]), 12)
        if direction_id in direction_index and ratio in ratio_index:
            values[direction_index[direction_id], ratio_index[ratio]] = float(
                row["delta_vs_random_mean"]
            )

    color_limit = max(float(np.nanmax(np.abs(values))), 1e-6)
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#dddddd")

    figure, axis = plt.subplots(
        figsize=(8.0, 7.2), subplot_kw={"projection": "polar"}
    )
    sector_width = 2.0 * math.pi / total_directions
    for direction_id in direction_ids:
        theta = direction_id * sector_width
        for ratio_level in range(len(ratios)):
            value = values[direction_id, ratio_level]
            axis.bar(
                theta,
                1.0,
                width=sector_width,
                bottom=float(ratio_level),
                align="center",
                color="#dddddd" if np.isnan(value) else cmap(norm(value)),
                edgecolor="none",
            )

    outer_radius = float(len(ratios))
    full_circle = np.linspace(0.0, 2.0 * math.pi, 721)
    for direction_id in direction_ids:
        boundary = (direction_id - 0.5) * sector_width
        axis.plot(
            [boundary, boundary], [0.0, outer_radius], color="#333333", linewidth=0.75
        )
    for radius in range(1, len(ratios)):
        axis.plot(
            full_circle,
            np.full_like(full_circle, float(radius)),
            color="#333333",
            linewidth=0.75,
        )

    # Four arrows make the moment coordinate system explicit.
    for theta in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
        axis.annotate(
            "",
            xy=(theta, outer_radius + 0.08),
            xytext=(theta, 0.0),
            arrowprops={"arrowstyle": "-|>", "color": "#222222", "lw": 1.15},
            annotation_clip=False,
            zorder=6,
        )

    axis.set_xticks([])
    axis.set_ylim(0.0, outer_radius)
    axis.grid(False)
    axis.spines["polar"].set_color("#222222")
    axis.spines["polar"].set_linewidth(0.9)

    cardinal_labels = {}
    if total_directions % 4 == 0:
        quarter = total_directions // 4
        cardinal_labels = {
            0: (r"+$\Delta m_2$", (8, 15), (8, -12), "left", "left"),
            quarter: (r"+$\Delta m_3$", (12, 18), (-12, 4), "left", "right"),
            2 * quarter: (r"−$\Delta m_2$", (-8, 15), (-8, -12), "right", "right"),
            3 * quarter: (r"−$\Delta m_3$", (12, -18), (-12, -4), "left", "right"),
        }

    # Diagonal direction labels remain radial. Cardinal labels are split across
    # the arrow tip from their red moment-axis labels to prevent crowding.
    direction_radius = outer_radius + 0.30
    for direction_id in direction_ids:
        theta = direction_id * sector_width
        direction_text = f"D{direction_id}\n{360.0 * direction_id / total_directions:g}°"
        if direction_id in cardinal_labels:
            sign, direction_offset, sign_offset, direction_ha, sign_ha = cardinal_labels[
                direction_id
            ]
            anchor = (theta, outer_radius + 0.10)
            axis.annotate(
                direction_text,
                xy=anchor,
                xytext=direction_offset,
                textcoords="offset points",
                ha=direction_ha,
                va="center",
                fontsize=9.5,
                annotation_clip=False,
            )
            axis.annotate(
                sign,
                xy=anchor,
                xytext=sign_offset,
                textcoords="offset points",
                ha=sign_ha,
                va="center",
                color="#c62828",
                fontweight="bold",
                fontsize=11,
                annotation_clip=False,
            )
            continue
        cosine, sine = math.cos(theta), math.sin(theta)
        horizontal = "left" if cosine > 0.2 else "right" if cosine < -0.2 else "center"
        vertical = "bottom" if sine > 0.2 else "top" if sine < -0.2 else "center"
        axis.text(
            theta,
            direction_radius,
            direction_text,
            ha=horizontal,
            va=vertical,
            fontsize=9.5,
            clip_on=False,
        )

    # Ring labels communicate the ratio directly; ring widths reflect equal spacing.
    axis.set_yticks(
        np.arange(len(ratios), dtype=float) + 0.5,
        [f"{ratio:.0%}" for ratio in ratios],
    )
    axis.set_rlabel_position(22.5)
    for label in axis.get_yticklabels():
        label.set_fontsize(8.5)
        label.set_bbox(
            {"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.3}
        )
    axis.text(
        math.radians(22.5),
        outer_radius + 0.02,
        "deletion ratio",
        ha="left",
        va="bottom",
        fontsize=8.5,
        clip_on=False,
    )

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    figure.colorbar(
        scalar_mappable,
        ax=axis,
        pad=0.16,
        shrink=0.82,
        label=r"$\Delta$ test accuracy (pp)",
    )
    if np.isnan(values).any():
        axis.legend(
            handles=[Patch(facecolor="#dddddd", label="Not evaluated")],
            loc="lower left",
            bbox_to_anchor=(-0.10, -0.08),
            frameon=False,
        )
    figure.subplots_adjust(left=0.08, right=0.82, top=0.91, bottom=0.10)
    output_stem = output_path.with_suffix("")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight"
    )
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--budget-ratios", type=float, nargs="+", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    plot_direction_ratio_sector(
        load_csv_rows(args.summary),
        args.output,
        total_directions=args.directions,
        budget_ratios=args.budget_ratios,
    )
    output_stem = args.output.with_suffix("")
    print(
        "[plot] saved "
        f"{output_stem.with_suffix('.pdf')}, "
        f"{output_stem.with_suffix('.png')}, and "
        f"{output_stem.with_suffix('.svg')}"
    )


if __name__ == "__main__":
    main()
