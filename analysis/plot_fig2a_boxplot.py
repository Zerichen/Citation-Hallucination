#!/usr/bin/env python3
"""
Generate Figure 2a: Per-claim verification fraction boxplots.

Reads the tidy CSV exported by export_fig2a_data.py and produces a 1×5
horizontal strip of boxplots (one per condition, four models each),
matching the visual style of Figure 1.

Usage:
    python analysis/plot_fig2a_boxplot.py \
        --in_csv  out/analysis/fig2a_frac_existing.csv \
        --out_png out/analysis/fig2a_frac_existing_boxplot.png
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


# ── Display order ─────────────────────────────────────────────────────
CONDITION_ORDER = ["Baseline", "Temporal", "Survey", "Non-Disc.", "Combo"]
MODEL_ORDER = ["Claude Sonnet", "GPT-4o", "LLaMA 3.1-8B", "Qwen 2.5-14B"]
MODEL_SHORT = ["Claude", "GPT-4o", "LLaMA", "Qwen"]

# Colors matching the stacked-bar figure (same palette)
MODEL_COLORS = {
    "Claude Sonnet": "#4a8c6f",   # muted teal-green
    "GPT-4o":        "#e8956a",   # salmon/coral
    "LLaMA 3.1-8B":  "#9b8ec4",  # muted purple
    "Qwen 2.5-14B":  "#6baed6",  # muted blue
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 2a boxplots from per-claim verification fractions."
    )
    parser.add_argument(
        "--in_csv",
        default="out/analysis/fig2a_frac_existing.csv",
        help="Path to per-claim verification fraction CSV (from export_fig2a_data.py)",
    )
    parser.add_argument(
        "--out_png",
        default="out/analysis/fig2a_frac_existing_boxplot.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    # ── Figure setup: 1×5 horizontal strip ────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
    fig.subplots_adjust(wspace=0.15, left=0.05, right=0.98, top=0.88, bottom=0.18)

    for ax_idx, cond in enumerate(CONDITION_ORDER):
        ax = axes[ax_idx]
        cond_df = df[df["condition_display"] == cond]

        box_data = []
        colors = []
        for m in MODEL_ORDER:
            vals = cond_df[cond_df["model_display"] == m]["frac_existing"].dropna().values
            box_data.append(vals)
            colors.append(MODEL_COLORS[m])

        bp = ax.boxplot(
            box_data,
            positions=range(len(MODEL_ORDER)),
            widths=0.55,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(
                marker="o", markersize=3.5, markerfacecolor="grey",
                markeredgecolor="grey", alpha=0.6,
            ),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=0.8),
            capprops=dict(color="black", linewidth=0.8),
            boxprops=dict(linewidth=0.8),
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_alpha(0.85)
            patch.set_clip_on(False)
        for element in ["whiskers", "caps", "medians", "fliers"]:
            for item in bp[element]:
                item.set_clip_on(False)

        ax.set_title(cond, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_SHORT, fontsize=8.5, rotation=0)
        ax.set_ylim(-0.05, 1.08)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

        if ax_idx == 0:
            ax.set_ylabel("Frac. Existing", fontsize=10)
        else:
            ax.tick_params(axis="y", length=0)

        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#888888")

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Saved → {args.out_png}")
    plt.close()


if __name__ == "__main__":
    main()
