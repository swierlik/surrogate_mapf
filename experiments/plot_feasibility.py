"""Generate figures for surrogate feasibility study.

Produces 3 publication-quality figures saved to results/figures/:
  fig1_convergence.png         - Baseline CMA-ES convergence curve
  fig2_sliding_window_rho.png  - Spearman rho: XGBoost vs MLP vs CNN
  fig3_training_time.png       - Training time comparison (bar chart)

Usage:
    python -m experiments.plot_feasibility
    python -m experiments.plot_feasibility --output-dir results/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Results from sliding-window CLI runs (Sessions 2 & 3, findings.md)
# ---------------------------------------------------------------------------
GENS = [10, 20, 50, 100, 150, 200, 250, 299]

SLIDING_RHO = {
    "XGBoost": [0.4230, 0.6262, 0.5124, 0.8164, 0.7884, 0.8746, 0.8556, 0.9115],
    "MLP":     [0.4966, 0.6272, 0.5160, 0.8781, 0.6405, 0.8915, 0.8742, 0.9409],
    "CNN":     [0.0510, 0.5835, 0.3342, 0.7825, 0.4682, 0.8257, 0.8837, 0.8842],
}

COLORS  = {"XGBoost": "#2196F3", "MLP": "#4CAF50", "CNN": "#FF9800"}
MARKERS = {"XGBoost": "o",       "MLP": "s",        "CNN": "^"}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Figure 1 — Baseline CMA-ES convergence
# ---------------------------------------------------------------------------
def fig1_convergence(log_path: Path, out_path: Path):
    df = pd.read_csv(log_path)
    gen_mean = df.groupby("generation")["mean_throughput"].mean()
    best_so_far = df.groupby("generation")["mean_throughput"].max().cummax()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gen_mean.index, gen_mean.values,
            color="#2196F3", linewidth=1.5, alpha=0.8, label="Population mean")
    ax.plot(best_so_far.index, best_so_far.values,
            color="#F44336", linewidth=2.0, label="Best so far")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Throughput (agents/timestep)", fontsize=12)
    ax.set_title("Baseline CMA-ES Convergence (300 gens, 30 000 simulations)", fontsize=12)
    ax.legend(fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 2 — Sliding window Spearman rho
# ---------------------------------------------------------------------------
def fig2_sliding_rho(out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))

    for name, rhos in SLIDING_RHO.items():
        ax.plot(GENS, rhos,
                color=COLORS[name], marker=MARKERS[name],
                linewidth=2.0, markersize=6, label=name)

    # Warmup boundary and decision threshold
    ax.axvline(20, color="grey", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(21, 0.05, "warmup ends", fontsize=8, color="grey")
    ax.axhline(0.4, color="grey", linestyle="--", linewidth=1.0,
               label="Decision threshold (ρ = 0.4)")

    ax.set_xlabel("Training cutoff generation", fontsize=12)
    ax.set_ylabel("Spearman ρ  (predict next generation)", fontsize=12)
    ax.set_title("Surrogate Temporal Generalisation: Sliding Window Test", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 3 — Training time comparison
# ---------------------------------------------------------------------------
def fig3_training_time(out_path: Path):
    labels = ["XGBoost\n(full retrain)", "MLP\n(full retrain)",
              "MLP\n(fine-tune)", "CNN\n(full retrain)"]
    times  = [241, 8, 9, 17]
    colors = ["#2196F3", "#4CAF50", "#81C784", "#FF9800"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, times, color=colors, width=0.5, edgecolor="white")

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 4,
                f"{t}s", ha="center", va="bottom", fontsize=10)

    ax.axhline(120, color="#F44336", linestyle="--", linewidth=1.5,
               label="1 simulation generation (~120 s)")

    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("Surrogate Training Time at 20 000 Samples", fontsize=12)
    ax.legend(fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate surrogate feasibility figures")
    parser.add_argument("--data-dir",    default="results/baseline")
    parser.add_argument("--output-dir",  default="results/figures")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    fig1_convergence(Path(args.data_dir) / "cmaes_log.csv",
                     out_dir / "fig1_convergence.png")
    fig2_sliding_rho(out_dir / "fig2_sliding_window_rho.png")
    fig3_training_time(out_dir / "fig3_training_time.png")
    print(f"\nDone — figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
