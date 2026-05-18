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
SURROGATE_V3_DIR = "results/surrogate_v3"
SURROGATE_LAM2_DIR = "results/surrogate_lam2"
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


def plot_v3_comparison(vanilla_dir, surrogate_v1_dir, surrogate_v3_dir, out_dir):
    """Generate fig7a/b/c/d: V1 (point-estimate) vs V3 (ensemble UCB) comparison."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v_best  = pd.read_csv(Path(vanilla_dir)      / "cmaes_best.csv")
    s1_best = pd.read_csv(Path(surrogate_v1_dir) / "cmaes_best.csv")
    s3_best = pd.read_csv(Path(surrogate_v3_dir) / "cmaes_best.csv")
    s1_surr = pd.read_csv(Path(surrogate_v1_dir) / "surrogate_log.csv")
    s3_surr = pd.read_csv(Path(surrogate_v3_dir) / "surrogate_log.csv")

    N_EVALS = 5
    v_cum_sims  = np.arange(1, len(v_best)  + 1) * 100 * N_EVALS
    s1_cum_sims = np.cumsum(s1_surr["n_simulated"].values * N_EVALS)
    s3_cum_sims = np.cumsum(s3_surr["n_simulated"].values * N_EVALS)

    gens    = np.arange(len(v_best))
    s1_gens = np.arange(len(s1_best))
    s3_gens = np.arange(len(s3_best))

    VANILLA_C = "#2196F3"
    V1_C      = "#4CAF50"
    V3_C      = "#9C27B0"

    # --- Figure 7a: Convergence ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gens,    v_best["best_throughput"],  color=VANILLA_C, linewidth=1.5, linestyle="--", alpha=0.6, label="Vanilla CMA-ES (reference)")
    ax.plot(s1_gens, s1_best["best_throughput"], color=V1_C,      linewidth=2.0, label="Surrogate V1 (point-estimate)")
    ax.plot(s3_gens, s3_best["best_throughput"], color=V3_C,      linewidth=2.0, label="Surrogate V3 (ensemble UCB)")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best throughput (agents/timestep)", fontsize=12)
    ax.set_title("Surrogate V1 vs V3: Convergence Comparison", fontsize=12)
    ax.legend(fontsize=10)
    _style(ax)
    fig.tight_layout()
    p = out_dir / "fig7a_v3_convergence.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"  Saved {p.name}")

    # --- Figure 7b: Sample efficiency ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(v_cum_sims,                   v_best["best_throughput"].values,  color=VANILLA_C, linewidth=1.5, linestyle="--", alpha=0.6, label="Vanilla CMA-ES (reference)")
    ax.plot(s1_cum_sims[:len(s1_best)],   s1_best["best_throughput"].values, color=V1_C,      linewidth=2.0, label="Surrogate V1 (point-estimate)")
    ax.plot(s3_cum_sims[:len(s3_best)],   s3_best["best_throughput"].values, color=V3_C,      linewidth=2.0, label="Surrogate V3 (ensemble UCB)")
    ax.set_xlabel("Cumulative simulations", fontsize=12)
    ax.set_ylabel("Best throughput (agents/timestep)", fontsize=12)
    ax.set_title("Surrogate V1 vs V3: Sample Efficiency", fontsize=12)
    ax.legend(fontsize=10)
    _style(ax)
    fig.tight_layout()
    p = out_dir / "fig7b_v3_sample_efficiency.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"  Saved {p.name}")

    # --- Figure 7c: Surrogate rho V1 vs V3 ---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for s_surr, color, label in [
        (s1_surr, V1_C, "Surrogate V1"),
        (s3_surr, V3_C, "Surrogate V3 (ensemble)"),
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
    ax.set_title("Surrogate Accuracy: V1 vs V3", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    _style(ax)
    fig.tight_layout()
    p = out_dir / "fig7c_v3_surrogate_rho.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"  Saved {p.name}")

    # --- Figure 7d: V3 ensemble uncertainty over time ---
    if "mean_std" in s3_surr.columns:
        std_col = s3_surr["mean_std"].replace("", np.nan)
        mask = std_col.notna()
        if mask.any():
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(s3_surr.loc[mask, "generation"].values,
                    std_col[mask].astype(float).values,
                    color=V3_C, linewidth=1.8, alpha=0.9, label="Mean ensemble std (all 100)")
            if "selected_std" in s3_surr.columns:
                sel_col = s3_surr["selected_std"].replace("", np.nan)
                sel_mask = sel_col.notna()
                if sel_mask.any():
                    ax.plot(s3_surr.loc[sel_mask, "generation"].values,
                            sel_col[sel_mask].astype(float).values,
                            color="#E91E63", linewidth=1.8, linestyle="--",
                            label="Mean std of selected top-20")
            ax.set_xlabel("Generation", fontsize=12)
            ax.set_ylabel("Ensemble std (throughput units)", fontsize=12)
            ax.set_title("V3 Surrogate Uncertainty Over Time", fontsize=12)
            ax.legend(fontsize=10)
            _style(ax)
            fig.tight_layout()
            p = out_dir / "fig7d_v3_uncertainty.png"
            fig.savefig(p, dpi=300)
            plt.close(fig)
            print(f"  Saved {p.name}")

    # --- Summary ---
    n1, n3 = len(s1_best), len(s3_best)
    print("\n=== V1 vs V3 Summary ===")
    print(f"{'':25s} {'Vanilla':>12s} {'Surr V1':>12s} {'Surr V3':>12s}")
    print(f"{'Best throughput':25s} {v_best['best_throughput'].max():>12.4f} "
          f"{s1_best['best_throughput'].max():>12.4f} "
          f"{s3_best['best_throughput'].max():>12.4f}")
    print(f"{'Total simulations':25s} {int(v_cum_sims[-1]):>12,} "
          f"{int(s1_cum_sims[n1-1]):>12,} "
          f"{int(s3_cum_sims[n3-1]):>12,}")


def plot_rho_vs_throughput(surrogate_dir, out_dir, jump_threshold=0.03):
    """Generate fig9: dual-axis — surrogate rho + best throughput over generations.

    Marks generations where best throughput makes a significant jump (>jump_threshold)
    to test whether rho dips correlate with improvement events.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_df = pd.read_csv(Path(surrogate_dir) / "cmaes_best.csv")
    surr_df = pd.read_csv(Path(surrogate_dir) / "surrogate_log.csv")

    gens       = best_df["generation"].values
    throughput = best_df["best_throughput"].values

    # Surrogate rho — only logged at control/warmup gens
    rho_col  = surr_df["surrogate_rho"].replace("", np.nan)
    rho_mask = rho_col.notna()
    rho_gens = surr_df.loc[rho_mask, "generation"].values
    rho_vals = rho_col[rho_mask].astype(float).values

    # Improvement events — gens where best throughput jumped significantly
    deltas = np.diff(throughput, prepend=throughput[0])
    jump_gens = gens[deltas > jump_threshold]

    fig, ax1 = plt.subplots(figsize=(10, 4.5))

    # Left axis: best throughput
    ax1.plot(gens, throughput, color="#9C27B0", linewidth=2.0, label="Best throughput (V3)")
    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Best throughput (agents/timestep)", fontsize=12, color="#9C27B0")
    ax1.tick_params(axis="y", labelcolor="#9C27B0")

    # Mark improvement events
    for g in jump_gens:
        ax1.axvline(g, color="#9C27B0", linewidth=0.8, linestyle=":", alpha=0.5)

    # Right axis: surrogate rho
    ax2 = ax1.twinx()
    ax2.scatter(rho_gens, rho_vals, color="#FF9800", s=25, zorder=5,
                label="Surrogate rho (control gens)")
    ax2.plot(rho_gens, rho_vals, color="#FF9800", linewidth=1.2, alpha=0.6)
    ax2.axhline(0.4, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.set_ylabel("Spearman rho", fontsize=12, color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")
    ax2.set_ylim(0, 1.05)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")

    ax1.set_title(
        f"Surrogate Accuracy vs Convergence (V3)\n"
        f"Dotted verticals = throughput jumps > {jump_threshold}",
        fontsize=11,
    )
    _style(ax1)
    fig.tight_layout()
    p = out_dir / "fig9_rho_vs_throughput.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"  Saved {p.name}")

    # Print correlation summary
    print(f"\n  Improvement events (jump > {jump_threshold}): {list(jump_gens)}")
    # For each jump, report nearest rho measurement
    print(f"  Rho at/around jump generations:")
    for g in jump_gens:
        dists = np.abs(rho_gens - g)
        nearest_idx = dists.argmin()
        if dists[nearest_idx] <= 15:
            print(f"    gen {g:3d}: nearest rho = {rho_vals[nearest_idx]:.3f} "
                  f"(gen {rho_gens[nearest_idx]})")


def plot_speedup_crossover(vanilla_dir, surrogate_v1_dir, surrogate_v3_dir, out_dir):
    """Generate fig10: throughput vs cumulative simulations for all three methods.

    Annotates two crossover points per surrogate variant:
      - V1: when V1 reaches its own peak vs when vanilla reaches that same level
      - V3 (300g): when V3 (first 300 gens) reaches its 300-gen peak vs vanilla's equivalent
      - V3 (400g): when V3 matches vanilla's all-time peak

    Speedup = vanilla_sims_to_quality / surrogate_sims_to_quality
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v_best  = pd.read_csv(Path(vanilla_dir)      / "cmaes_best.csv")
    s1_best = pd.read_csv(Path(surrogate_v1_dir) / "cmaes_best.csv")
    s3_best = pd.read_csv(Path(surrogate_v3_dir) / "cmaes_best.csv")
    s1_surr = pd.read_csv(Path(surrogate_v1_dir) / "surrogate_log.csv")
    s3_surr = pd.read_csv(Path(surrogate_v3_dir) / "surrogate_log.csv")

    N_EVALS = 5
    v_cum_sims  = np.arange(1, len(v_best)  + 1) * 100 * N_EVALS
    s1_cum_sims = np.cumsum(s1_surr["n_simulated"].values * N_EVALS)
    s3_cum_sims = np.cumsum(s3_surr["n_simulated"].values * N_EVALS)

    v_tp  = v_best["best_throughput"].values
    s1_tp = s1_best["best_throughput"].values
    s3_tp = s3_best["best_throughput"].values

    def first_passage(tp_arr, cum_sims, threshold):
        """Return (gen, sims) at which running-max first meets threshold."""
        running_max = np.maximum.accumulate(tp_arr)
        idx = np.argmax(running_max >= threshold)
        if running_max[idx] < threshold:
            return None, None
        return idx, int(cum_sims[idx])

    # --- Crossover targets ---
    v1_peak   = float(np.max(s1_tp))
    v3_300_tp = float(np.max(s3_tp[:300]))       # V3 quality at gen 300
    v_peak    = float(np.max(v_tp))              # vanilla all-time peak

    # V1: sims to reach v1_peak for both methods
    _, v1_s_sims  = first_passage(s1_tp, s1_cum_sims, v1_peak)   # V1 self
    _, v1_v_sims  = first_passage(v_tp,  v_cum_sims,  v1_peak)   # vanilla matching V1

    # V3 300g crossover: V3 reaches v3_300_tp, vanilla reaches same level
    _, v3_300_s_sims = first_passage(s3_tp, s3_cum_sims, v3_300_tp)
    _, v3_300_v_sims = first_passage(v_tp,  v_cum_sims,  v3_300_tp)

    # V3 400g crossover: V3 reaches vanilla peak
    _, v3_400_s_sims = first_passage(s3_tp, s3_cum_sims, v_peak)

    v1_speedup   = v1_v_sims   / v1_s_sims   if (v1_s_sims   and v1_v_sims)   else None
    v3_300_speedup = v3_300_v_sims / v3_300_s_sims if (v3_300_s_sims and v3_300_v_sims) else None
    _, v3_400_v_sims = first_passage(v_tp, v_cum_sims, v_peak)
    v3_400_speedup = v3_400_v_sims / v3_400_s_sims if (v3_400_s_sims and v3_400_v_sims) else None

    VANILLA_C = "#2196F3"
    V1_C      = "#4CAF50"
    V3_C      = "#9C27B0"

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(v_cum_sims, v_tp,
            color=VANILLA_C, linewidth=2.0, linestyle="--", alpha=0.8,
            label="Vanilla CMA-ES (300 gen)")
    ax.plot(s1_cum_sims[:len(s1_tp)], s1_tp,
            color=V1_C, linewidth=2.0,
            label="Surrogate V1 — point-estimate (300 gen)")
    ax.plot(s3_cum_sims[:len(s3_tp)], s3_tp,
            color=V3_C, linewidth=2.0,
            label="Surrogate V3 — ensemble UCB (400 gen)")

    # Horizontal reference at vanilla peak — label at far right to stay clear
    ax.axhline(v_peak, color=VANILLA_C, linewidth=0.8, linestyle=":", alpha=0.5)
    ax.text(v_cum_sims[-1] * 0.98, v_peak + 0.012,
            f"Vanilla peak\n{v_peak:.4f}", color=VANILLA_C, fontsize=7.5,
            alpha=0.85, ha="right", va="bottom")

    def _annotate_crossover(ax, s_sims, s_tp_val, v_sims, color, label, text_xy,
                            arc_rad=0.25):
        """Draw crossover dot + arrow label + horizontal budget span."""
        if s_sims is None:
            return
        # Dot on surrogate line
        ax.scatter([s_sims], [s_tp_val], color=color, s=70, zorder=7)
        # Faint horizontal span showing vanilla's equivalent budget
        if v_sims is not None:
            ax.annotate("", xy=(v_sims, s_tp_val), xytext=(s_sims, s_tp_val),
                        arrowprops=dict(arrowstyle="<->", color=color, alpha=0.35,
                                        lw=1.1, shrinkA=3, shrinkB=3))
        # Offset label with pointer arrow
        ax.annotate(label,
                    xy=(s_sims, s_tp_val), xytext=text_xy,
                    fontsize=8.5, color="black",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color,
                              alpha=0.92, lw=1.0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.1,
                                    connectionstyle=f"arc3,rad={arc_rad}"),
                    zorder=8)

    # Place labels well away from the crowded 48-60k region:
    #   V1       → bottom-left corner
    #   V3 300g  → right-centre
    #   V3 400g  → top-right (above vanilla peak line)
    if v1_speedup:
        _annotate_crossover(
            ax, v1_s_sims, v1_peak, v1_v_sims,
            V1_C,
            f"V1: {v1_speedup:.2f}x speedup\n({v1_s_sims:,} vs {v1_v_sims:,} sims)",
            text_xy=(8_000, 6.9),
            arc_rad=-0.25,
        )

    if v3_300_speedup:
        _annotate_crossover(
            ax, v3_300_s_sims, v3_300_tp, v3_300_v_sims,
            V3_C,
            f"V3 (300g): {v3_300_speedup:.2f}x speedup\n({v3_300_s_sims:,} vs {v3_300_v_sims:,} sims)",
            text_xy=(78_000, 7.65),
            arc_rad=0.30,
        )

    if v3_400_speedup:
        _annotate_crossover(
            ax, v3_400_s_sims, v_peak, v3_400_v_sims,
            V3_C,
            f"V3 (400g): {v3_400_speedup:.2f}x speedup\n({v3_400_s_sims:,} vs {v3_400_v_sims:,} sims)",
            text_xy=(105_000, 7.90),
            arc_rad=0.20,
        )

    ax.set_xlabel("Cumulative simulations", fontsize=12)
    ax.set_ylabel("Best throughput (agents/timestep)", fontsize=12)
    ax.set_title("Sample Efficiency: Crossover Analysis\n"
                 "Speedup = vanilla simulations / surrogate simulations to reach same quality",
                 fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    _style(ax)
    fig.tight_layout()
    p = out_dir / "fig10_speedup_crossover.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"  Saved {p.name}")

    print("\n=== Speedup Crossover Summary ===")
    print(f"  Vanilla peak: {v_peak:.4f}  at {int(v_cum_sims[np.argmax(v_tp)]):,} sims")
    if v1_speedup:
        print(f"  V1 speedup:      {v1_speedup:.2f}x  "
              f"({v1_s_sims:,} sims vs {v1_v_sims:,} sims to reach {v1_peak:.4f})")
    if v3_300_speedup:
        print(f"  V3 (300g) speedup: {v3_300_speedup:.2f}x  "
              f"({v3_300_s_sims:,} sims vs {v3_300_v_sims:,} sims to reach {v3_300_tp:.4f})")
    if v3_400_speedup:
        print(f"  V3 (400g) speedup: {v3_400_speedup:.2f}x  "
              f"({v3_400_s_sims:,} sims vs {v3_400_v_sims:,} sims to reach {v_peak:.4f})")


def plot_lambda_ablation(vanilla_dir, surrogate_v1_dir, surrogate_v3_dir,
                         surrogate_lam2_dir, out_dir):
    """Generate fig8: UCB lambda ablation — throughput vs lambda value.

    Three data points: λ=0 (V1, point-estimate), λ=1.0 (V3), λ=2.0 (lam2).
    Vanilla shown as horizontal reference line.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vanilla_tp = pd.read_csv(Path(vanilla_dir)      / "cmaes_best.csv")["best_throughput"].max()
    v1_tp      = pd.read_csv(Path(surrogate_v1_dir) / "cmaes_best.csv")["best_throughput"].max()
    v3_tp      = pd.read_csv(Path(surrogate_v3_dir) / "cmaes_best.csv")["best_throughput"].max()
    lam2_tp    = pd.read_csv(Path(surrogate_lam2_dir) / "cmaes_best.csv")["best_throughput"].max()

    lambdas = [0.0, 1.0, 2.0]
    tps     = [v1_tp, v3_tp, lam2_tp]
    colors  = ["#4CAF50", "#9C27B0", "#FF5722"]
    labels  = ["λ=0\n(V1, point-estimate)", "λ=1.0\n(V3, ensemble UCB)", "λ=2.0\n(over-exploration)"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(labels, tps, color=colors, width=0.5, edgecolor="white", zorder=3)
    for bar, tp in zip(bars, tps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{tp:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(vanilla_tp, color="#2196F3", linestyle="--", linewidth=1.8,
               label=f"Vanilla CMA-ES ({vanilla_tp:.4f})", zorder=2)

    # Zoom y-axis to make differences visible
    y_min = min(tps) - 0.15
    ax.set_ylim(y_min, vanilla_tp + 0.08)
    ax.set_ylabel("Best throughput (agents/timestep)", fontsize=12)
    ax.set_title("UCB Exploration Weight Ablation\n(ensemble surrogate, 49 200 simulations each)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    _style(ax)
    fig.tight_layout()
    p = out_dir / "fig8_lambda_ablation.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"  Saved {p.name}")

    print("\n=== Lambda Ablation Summary ===")
    print(f"{'Vanilla':20s} {vanilla_tp:.4f}  (150,000 sims)")
    print(f"{'lam=0  (V1)':20s} {v1_tp:.4f}  (49,200 sims)  gap={vanilla_tp-v1_tp:+.4f}")
    print(f"{'lam=1.0 (V3)':20s} {v3_tp:.4f}  (49,200 sims)  gap={vanilla_tp-v3_tp:+.4f}")
    print(f"{'lam=2.0':20s} {lam2_tp:.4f}  (49,200 sims)  gap={vanilla_tp-lam2_tp:+.4f}")


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
    parser.add_argument("--surrogate-v3-dir",   default=SURROGATE_V3_DIR)
    parser.add_argument("--surrogate-lam2-dir", default=SURROGATE_LAM2_DIR)
    parser.add_argument("--v3", action="store_true",
                        help="Generate fig7a/b/c/d comparing V1 vs V3 (ensemble UCB) runs")
    parser.add_argument("--lambda-ablation", action="store_true",
                        help="Generate fig8: UCB lambda ablation bar chart")
    parser.add_argument("--rho-analysis", action="store_true",
                        help="Generate fig9: dual-axis rho vs throughput (V3)")
    parser.add_argument("--speedup", action="store_true",
                        help="Generate fig10: crossover speedup plot (vanilla vs V1 vs V3)")
    parser.add_argument("--jump-threshold", type=float, default=0.03,
                        help="Throughput jump size to mark as improvement event")
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

    if args.speedup:
        print("\nGenerating speedup crossover figure (fig10)...")
        plot_speedup_crossover(args.vanilla_dir, args.surrogate_dir,
                               args.surrogate_v3_dir, args.figures_dir)
    elif args.rho_analysis:
        print("\nGenerating rho vs throughput dual-axis figure (fig9)...")
        plot_rho_vs_throughput(args.surrogate_v3_dir, args.figures_dir,
                               jump_threshold=args.jump_threshold)
    elif args.lambda_ablation:
        print("\nGenerating lambda ablation figure (fig8)...")
        plot_lambda_ablation(args.vanilla_dir, args.surrogate_dir,
                             args.surrogate_v3_dir, args.surrogate_lam2_dir,
                             args.figures_dir)
    elif args.v3:
        print("\nGenerating V1 vs V3 comparison figures (fig7a/b/c/d)...")
        plot_v3_comparison(args.vanilla_dir, args.surrogate_dir,
                           args.surrogate_v3_dir, args.figures_dir)
    elif args.v2:
        print("\nGenerating V1 vs V2 comparison figures (fig6a/b/c)...")
        plot_v2_comparison(args.vanilla_dir, args.surrogate_dir,
                           args.surrogate_v2_dir, args.figures_dir)
    else:
        print("\nGenerating comparison figures...")
        plot_comparison(args.vanilla_dir, args.surrogate_dir, args.figures_dir)


if __name__ == "__main__":
    main()
