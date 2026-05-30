"""
Per-direction flow-weight heatmap (Zang et al. style).

Usage:
    python experiments/plot_weight_heatmap.py                          # baseline (default)
    python experiments/plot_weight_heatmap.py --solution results/surrogate_v3/best_solution.npy --title "Surrogate V3 + CMA-ES" --out fig_weight_heatmap_v3
    python experiments/plot_weight_heatmap.py --solution results/gradient_refinement/refined_solution.npy --title "Gradient Refinement" --out fig_weight_heatmap_refined
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.utils.reshape import SolutionReshaper

CH = {"Right": 0, "Up": 1, "Left": 2, "Down": 3, "Wait": 4}
DIRECTIONS = ["Left", "Right", "Up", "Down", "Wait"]
OUT_DIR = "Thesis_Final"


def plot_weight_heatmap(solution_path, title, out_stem):
    solution = np.load(solution_path)
    reshaper  = SolutionReshaper.get()
    tensor    = reshaper.flat_to_tensor(solution, add_obstacle_mask=True)

    obstacle_mask = tensor[:, :, 5].astype(bool)

    grids = {}
    for name in DIRECTIONS:
        g = tensor[:, :, CH[name]].copy().astype(float)
        g[obstacle_mask] = np.nan
        grids[name] = g

    all_valid = np.concatenate([g[~np.isnan(g)] for g in grids.values()])
    raw_min, raw_max = all_valid.min(), all_valid.max()
    for name in DIRECTIONS:
        grids[name] = (grids[name] - raw_min) / (raw_max - raw_min) * 100.0

    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2),
                             gridspec_kw={"wspace": 0.04})

    for ax, name in zip(axes, DIRECTIONS):
        im = ax.imshow(grids[name], cmap=cmap, vmin=0, vmax=100,
                       aspect="auto", interpolation="nearest")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    cbar = fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(["0", "25", "50", "75", "100"])
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(f"(b) Warehouse-33-36: {title} (150 agents)",
                 y=0.02, fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{out_stem}.{ext}"),
                    dpi=300, bbox_inches="tight")
    print(f"Saved to {OUT_DIR}/{out_stem}.{{pdf,png}}")
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--solution", default="results/baseline/best_solution.npy")
    p.add_argument("--title",    default="Vanilla CMA-ES")
    p.add_argument("--out",      default="fig_weight_heatmap")
    args = p.parse_args()
    plot_weight_heatmap(args.solution, args.title, args.out)
