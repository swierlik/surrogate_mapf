"""Experiment 2: Surrogate-assisted vs vanilla CMA-ES comparison.

Runs both optimizers with the same seed and 300 generations, then
produces a comparison report and figures:
  - Convergence curves (throughput vs generation)
  - Cumulative simulations used
  - Surrogate rho over time (evolution control gens only)

Results are saved to results/surrogate/ (surrogate run) and
results/baseline/ (vanilla run, already exists).

Usage:
    # Run surrogate (assumes vanilla baseline already in results/baseline)
    python -m experiments.02_comparison --run-surrogate

    # Just plot from existing results
    python -m experiments.02_comparison --plot-only

    # Resume a partial surrogate run
    python -m experiments.02_comparison --run-surrogate --resume
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.optimizer.surrogate_cmaes import run_surrogate_cmaes


VANILLA_DIR      = "results/baseline"
SURROGATE_DIR    = "results/surrogate"
SURROGATE_V2_DIR = "results/surrogate_v2"
FIGURES_DIR      = "results/figures"


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_comparison(vanilla_dir, surrogate_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v_best = pd.read_csv(Path(vanilla_dir) / "cmaes_best.csv")
    s_best = pd.read_csv(Path(surrogate_dir) / "cmaes_best.csv")
    s_surr = pd.read_csv(Path(surrogate_dir) / "surrogate_log.csv")

    N_EVALS = 5
    s_cum_sims = np.cumsum(s_surr["n_simulated"].values * N_EVALS)
    v_cum_sims = np.arange(1, len(v_best) + 1) * 100 * N_EVALS

    gens = np.arange(len(v_best))
    s_gens = np.arange(len(s_best))

    # --- Figure 5a: Convergence curves (gen axis) ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gens, v_best["best_throughput"],
            color="#2196F3", linewidth=2.0, label="Vanilla CMA-ES")
    ax.plot(s_gens, s_best["best_throughput"],
            color="#4CAF50", linewidth=2.0, label="Surrogate-assisted CMA-ES")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best throughput (agents/timestep)", fontsize=12)
    ax.set_title("Convergence: Surrogate-Assisted vs Vanilla CMA-ES", fontsize=12)
    ax.legend(fontsize=10)
    _style(ax)
    fig.tight_layout()
    p = out_dir / "fig5a_convergence_comparison.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"  Saved {p.name}")

    # --- Figure 5b: Sample efficiency (simulation axis) ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(v_cum_sims, v_best["best_throughput"].values,
            color="#2196F3", linewidth=2.0, label="Vanilla CMA-ES")
    ax.plot(s_cum_sims[:len(s_best)], s_best["best_throughput"].values,
            color="#4CAF50", linewidth=2.0, label="Surrogate-assisted CMA-ES")
    ax.set_xlabel("Cumulative simulations", fontsize=12)
    ax.set_ylabel("Best throughput (agents/timestep)", fontsize=12)
    ax.set_title("Sample Efficiency: Throughput vs Simulations Used", fontsize=12)
    ax.legend(fontsize=10)
    _style(ax)
    fig.tight_layout()
    p = out_dir / "fig5b_sample_efficiency.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"  Saved {p.name}")

    # --- Figure 5c: Surrogate rho over time ---
    rho_col = s_surr["surrogate_rho"].replace("", np.nan)
    control_mask = rho_col.notna()
    if control_mask.any():
        ctrl_gens = s_surr.loc[control_mask, "generation"].values
        ctrl_rho  = rho_col[control_mask].astype(float).values

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(ctrl_gens, ctrl_rho, color="#FF9800", marker="o",
                linewidth=2.0, markersize=5, label="Surrogate ρ (control gens)")
        ax.axhline(0.4, color="grey", linestyle="--", linewidth=1.0,
                   label="Decision threshold")
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Spearman ρ", fontsize=12)
        ax.set_title("Surrogate Accuracy During Optimization", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        _style(ax)
        fig.tight_layout()
        p = out_dir / "fig5c_surrogate_rho.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        print(f"  Saved {p.name}")

    # --- Figure 5a/5b/5c annotations: print mean rho ---
    # (summary follows below)

    # --- Summary stats ---
    n = min(len(s_best), len(s_cum_sims))
    speedup = v_cum_sims[n - 1] / s_cum_sims[n - 1]
    print("\n=== Comparison Summary ===")
    print(f"{'':25s} {'Vanilla':>12s} {'Surrogate':>12s}")
    print(f"{'Best throughput':25s} {v_best['best_throughput'].max():>12.4f} "
          f"{s_best['best_throughput'].max():>12.4f}")
    print(f"{'Total simulations':25s} {int(v_cum_sims[n-1]):>12,} "
          f"{int(s_cum_sims[n-1]):>12,}")
    print(f"{'Simulation speedup':25s} {'1.0x':>12s} {speedup:>11.1f}x")
    if control_mask.any():
        print(f"{'Mean surrogate rho':25s} {'N/A':>12s} {ctrl_rho.mean():>12.4f}")


def plot_v2_comparison(vanilla_dir, surrogate_v1_dir, surrogate_v2_dir, out_dir):
    """Generate fig6a/b/c: V1 vs V2 surrogate side-by-side (vanilla shown for reference)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v_best  = pd.read_csv(Path(vanilla_dir)       / "cmaes_best.csv")
    s1_best = pd.read_csv(Path(surrogate_v1_dir)  / "cmaes_best.csv")
    s2_best = pd.read_csv(Path(surrogate_v2_dir)  / "cmaes_best.csv")
    s1_surr = pd.read_csv(Path(surrogate_v1_dir)  / "surrogate_log.csv")
    s2_surr = pd.read_csv(Path(surrogate_v2_dir)  / "surrogate_log.csv")

    N_EVALS = 5
    v_cum_sims  = np.arange(1, len(v_best)  + 1) * 100 * N_EVALS
    s1_cum_sims = np.cumsum(s1_surr["n_simulated"].values * N_EVALS)
    s2_cum_sims = np.cumsum(s2_surr["n_simulated"].values * N_EVALS)

    gens    = np.arange(len(v_best))
    s1_gens = np.arange(len(s1_best))
    s2_gens = np.arange(len(s2_best))

    VANILLA_C = "#2196F3"
    V1_C      = "#4CAF50"
    V2_C      = "#F44336"

    # --- Figure 6a: Convergence curves (generation axis) ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gens,    v_best["best_throughput"],  color=VANILLA_C, linewidth=1.5, linestyle="--", alpha=0.6, label="Vanilla CMA-ES (reference)")
    ax.plot(s1_gens, s1_best["best_throughput"], color=V1_C,      linewidth=2.0, label="Surrogate V1 (original)")
    ax.plot(s2_gens, s2_best["best_throughput"], color=V2_C,      linewidth=2.0, label="Surrogate V2 (restart-retrain fix)")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best throughput (agents/timestep)", fontsize=12)
    ax.set_title("Surrogate V1 vs V2: Convergence Comparison", fontsize=12)
    ax.legend(fontsize=10)
    _style(ax)
    fig.tight_layout()
    p = out_dir / "fig6a_v2_convergence.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"  Saved {p.name}")

    # --- Figure 6b: Sample efficiency (simulation axis) ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(v_cum_sims,                        v_best["best_throughput"].values,  color=VANILLA_C, linewidth=1.5, linestyle="--", alpha=0.6, label="Vanilla CMA-ES (reference)")
    ax.plot(s1_cum_sims[:len(s1_best)],        s1_best["best_throughput"].values, color=V1_C,      linewidth=2.0, label="Surrogate V1 (original)")
    ax.plot(s2_cum_sims[:len(s2_best)],        s2_best["best_throughput"].values, color=V2_C,      linewidth=2.0, label="Surrogate V2 (restart-retrain fix)")
    ax.set_xlabel("Cumulative simulations", fontsize=12)
    ax.set_ylabel("Best throughput (agents/timestep)", fontsize=12)
    ax.set_title("Surrogate V1 vs V2: Sample Efficiency", fontsize=12)
    ax.legend(fontsize=10)
    _style(ax)
    fig.tight_layout()
    p = out_dir / "fig6b_v2_sample_efficiency.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"  Saved {p.name}")

    # --- Figure 6c: Surrogate rho over time (both versions) ---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for s_surr, color, label in [
        (s1_surr, V1_C, "Surrogate V1"),
        (s2_surr, V2_C, "Surrogate V2"),
    ]:
        rho_col = s_surr["surrogate_rho"].replace("", np.nan)
        mask = rho_col.notna()
        if mask.any():
            ax.plot(s_surr.loc[mask, "generation"].values,
                    rho_col[mask].astype(float).values,
                    color=color, marker="o", linewidth=1.8, markersize=4, label=label)
    ax.axhline(0.4, color="grey", linestyle="--", linewidth=1.0, label="Decision threshold")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_title("Surrogate Accuracy: V1 vs V2", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    _style(ax)
    fig.tight_layout()
    p = out_dir / "fig6c_v2_surrogate_rho.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"  Saved {p.name}")

    # --- Summary ---
    n1, n2 = len(s1_best), len(s2_best)
    print("\n=== V1 vs V2 Summary ===")
    print(f"{'':25s} {'Vanilla':>12s} {'Surr V1':>12s} {'Surr V2':>12s}")
    print(f"{'Best throughput':25s} {v_best['best_throughput'].max():>12.4f} "
          f"{s1_best['best_throughput'].max():>12.4f} "
          f"{s2_best['best_throughput'].max():>12.4f}")
    print(f"{'Total simulations':25s} {int(v_cum_sims[-1]):>12,} "
          f"{int(s1_cum_sims[n1-1]):>12,} "
          f"{int(s2_cum_sims[n2-1]):>12,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: Surrogate vs vanilla CMA-ES comparison")
    parser.add_argument("--vanilla-dir",      default=VANILLA_DIR)
    parser.add_argument("--surrogate-dir",    default=SURROGATE_DIR)
    parser.add_argument("--surrogate-v2-dir", default=SURROGATE_V2_DIR)
    parser.add_argument("--figures-dir",      default=FIGURES_DIR)
    parser.add_argument("--v2", action="store_true",
                        help="Generate fig6a/b/c comparing V1 vs V2 surrogate runs")
    parser.add_argument("--run-surrogate", action="store_true",
                        help="Run the surrogate optimizer (vanilla must already exist)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a partial surrogate run")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip running, just generate figures from existing results")
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the vanilla baseline seed")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--warmup-gens", type=int, default=20)
    parser.add_argument("--screen-k", type=int, default=20)
    parser.add_argument("--evolution-control-interval", type=int, default=10)
    args = parser.parse_args()

    if not args.plot_only and args.run_surrogate:
        print("Running surrogate-assisted CMA-ES...")
        run_surrogate_cmaes(
            generations=args.generations,
            seed=args.seed,
            output_dir=args.surrogate_dir,
            resume=args.resume,
            n_workers=args.n_workers,
            warmup_gens=args.warmup_gens,
            screen_k=args.screen_k,
            evolution_control_interval=args.evolution_control_interval,
        )

    if args.v2:
        print("\nGenerating V1 vs V2 comparison figures (fig6a/b/c)...")
        plot_v2_comparison(args.vanilla_dir, args.surrogate_dir,
                           args.surrogate_v2_dir, args.figures_dir)
    else:
        print("\nGenerating comparison figures...")
        plot_comparison(args.vanilla_dir, args.surrogate_dir, args.figures_dir)


if __name__ == "__main__":
    main()
