"""Thesis figure generation script.

Usage:
    python -m experiments.plot_thesis_figures          # all figures
    python -m experiments.plot_thesis_figures --fig 1 3 7  # selected figures
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(".")
RES     = ROOT / "results"
OUT     = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
C_VAN  = "#1a6faf"   # dark blue  — vanilla / main result
C_SURR = "#e06c00"   # orange     — surrogate / V3
C_V1   = "#5ba5d4"   # light blue — V1 / secondary surrogate
C_GRAY = "#888888"   # gray       — ablations / CNN / references
C_GRID = "#e0e0e0"
C_BAND = "#fff3e0"   # very light orange for stable-rho band
C_WARM = "#f0f0f0"   # very light gray for warmup band

LEGEND_KW = dict(frameon=False, fontsize=8.5)


def _style(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=C_GRID, linestyle="--",
                linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ── Figure 1 — Surrogate architecture comparison ─────────────────────────────

def fig1_surrogate_comparison():
    test_gens = [10, 20, 50, 100, 150, 200, 250, 299]
    mlp = [0.497, 0.627, 0.516, 0.878, 0.641, 0.892, 0.874, 0.941]
    xgb = [0.423, 0.626, 0.512, 0.816, 0.788, 0.875, 0.856, 0.912]
    cnn = [0.051, 0.584, 0.334, 0.783, 0.468, 0.826, 0.884, 0.884]

    fig, ax = plt.subplots(figsize=(7, 4))
    _style(ax)
    ax.tick_params(labelsize=12)

    ax.plot(test_gens, mlp, color=C_VAN,  linewidth=1.8, marker="o",
            markersize=4, label="MLP (ensemble)")
    ax.plot(test_gens, xgb, color=C_SURR, linewidth=1.8, marker="o",
            markersize=4, linestyle="--", label="XGBoost")
    ax.plot(test_gens, cnn, color=C_GRAY, linewidth=1.5,
            linestyle=":", label="CNN")

    ax.set_xticks(test_gens)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Test generation", fontsize=12)
    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_title("Surrogate Model Comparison: Rank Correlation vs. Training Generation",
                 fontsize=12, pad=8)
    ax.legend(loc="lower right", **{**LEGEND_KW, "fontsize": 10.5})

    # Training time annotation
    time_lines = ["Training time (gen 299):\n  MLP: 13 s   XGBoost: 183 s   CNN: 33 s"]
    ax.text(0.02, 0.05, time_lines[0],
            transform=ax.transAxes, fontsize=9.5,
            va="bottom", ha="left", color="#444444",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=C_GRID, linewidth=0.8))

    ax.set_xlim(left=0, right=310)
    fig.tight_layout()
    _save(fig, "fig1_surrogate_comparison.png")


# ── Figure 2 — System architecture flowchart ─────────────────────────────────

def fig2_flowchart():
    fig, ax = plt.subplots(figsize=(10, 9.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(1.2, 10.2)
    ax.axis("off")
    ax.set_title("Surrogate-Assisted CMA-ES: Optimisation Loop", fontsize=10, pad=6)

    BOX_FC = "#dce8f5"; BOX_EC = "#7aabcf"
    DIA_FC = "#e8f4ea"; DIA_EC = "#6aac78"
    LW = 0.8; FS = 8.5; AC = "#555555"
    AKW = dict(arrowstyle="-|>", color=AC, lw=0.9, mutation_scale=10)
    BH  = 0.52   # standard box height
    BH2 = 0.68   # tall box (2-line text)
    BW  = 3.0    # centre column width
    BW2 = 2.85   # branch column width
    CX  = 5.0    # centre x (warmup + diamond)
    CL  = 2.0    # left branch centre x
    CR  = 8.0    # right branch centre x
    DW, DH = 2.8, 0.72

    def mkbox(cx, cy, w, h, txt, fc=BOX_FC, ec=BOX_EC):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.12", facecolor=fc, edgecolor=ec,
            linewidth=LW, zorder=3))
        ax.text(cx, cy, txt, ha="center", va="center",
                fontsize=FS, zorder=4, multialignment="center")
        return cy - h/2   # bottom y

    def arr(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=AKW, zorder=2)

    def mkdia(cx, cy, text):
        xs = [cx, cx+DW/2, cx, cx-DW/2, cx]
        ys = [cy+DH/2, cy, cy-DH/2, cy, cy+DH/2]
        ax.fill(xs, ys, facecolor=DIA_FC, edgecolor=DIA_EC, lw=LW, zorder=3)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=FS, zorder=4,
                multialignment="center")

    # ── Warmup section (top, linear) ─────────────────────────────────────────
    ax.text(CX, 9.75, "Warmup Phase  (gens 0–19)", ha="center", va="center",
            fontsize=9, style="italic", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f4f8",
                      edgecolor=BOX_EC, lw=0.7))

    w1 = mkbox(CX, 9.1,  BW, BH, "Evaluate all 100 candidates")
    arr(CX, 9.57, CX, w1 + BH)
    w2 = mkbox(CX, 8.3,  BW, BH, "Append results to dataset")
    arr(CX, w1, CX, w2 + BH)
    w3 = mkbox(CX, 7.5,  BW, BH, "Train surrogate ensemble")
    arr(CX, w2, CX, w3 + BH)

    # ── Decision diamond ──────────────────────────────────────────────────────
    DIAM_Y = 6.45
    mkdia(CX, DIAM_Y, "Evolution\ncontrol gen?")
    arr(CX, w3, CX, DIAM_Y + DH/2)

    # Yes / No labels with branch names
    ax.text(CX - DW/2 - 0.1, DIAM_Y + 0.08, "Yes\n(Evo. Control)",
            ha="right", va="center", fontsize=7.5, color=AC, multialignment="right")
    ax.text(CX + DW/2 + 0.1, DIAM_Y + 0.08, "No\n(Surrogate Phase)",
            ha="left",  va="center", fontsize=7.5, color=AC)

    # ── Branch arrows from diamond sides ─────────────────────────────────────
    BR_CY1 = 5.05    # centre y of first branch box
    arr(CX - DW/2, DIAM_Y, CL, BR_CY1 + BH2/2)
    arr(CX + DW/2, DIAM_Y, CR, BR_CY1 + BH2/2)

    # ── Left branch: Evolution Control ───────────────────────────────────────
    lb1 = mkbox(CL, BR_CY1,        BW2, BH,  "Evaluate all 100 candidates")
    lb2 = mkbox(CL, BR_CY1 - 0.88, BW2, BH,  "Retrain ensemble from scratch")
    lb3 = mkbox(CL, BR_CY1 - 1.76, BW2, BH,  "CMA-ES update")
    arr(CL, lb1, CL, lb2 + BH)
    arr(CL, lb2, CL, lb3 + BH)

    # ── Right branch: Surrogate Phase ─────────────────────────────────────────
    rb1 = mkbox(CR, BR_CY1,        BW2, BH2, "UCB: Score 100 candidates\n→ Simulate top 20")
    rb2 = mkbox(CR, BR_CY1 - 1.05, BW2, BH2, "Fine-tune ensemble\n(10 epochs)")
    rb3 = mkbox(CR, BR_CY1 - 2.10, BW2, BH,  "CMA-ES update")
    arr(CR, rb1, CR, rb2 + BH2)
    arr(CR, rb2, CR, rb3 + BH)

    # ── Convergence lines ────────────────────────────────────────────────────
    CONV_Y = min(lb3, rb3) - 0.45
    ax.plot([CL, CL, CX], [lb3, CONV_Y, CONV_Y], color=AC, lw=0.9, zorder=2)
    ax.plot([CR, CR, CX], [rb3, CONV_Y, CONV_Y], color=AC, lw=0.9, zorder=2)
    ax.plot(CX, CONV_Y, "o", color=AC, markersize=3.5, zorder=3)

    # ── Loop arrow: convergence → right rail → up → diamond right tip ─────────
    LOOP_X = 9.55
    ax.plot([CX, LOOP_X], [CONV_Y, CONV_Y], color=AC, lw=0.9, zorder=2)
    ax.plot([LOOP_X, LOOP_X], [CONV_Y, DIAM_Y], color=AC, lw=0.9, zorder=2)
    ax.annotate("", xy=(CX + DW/2, DIAM_Y), xytext=(LOOP_X, DIAM_Y),
                arrowprops=AKW, zorder=2)
    ax.text(LOOP_X + 0.1, (CONV_Y + DIAM_Y) / 2, "Next generation",
            ha="left", va="center", fontsize=8, color="#444444", rotation=90)

    fig.tight_layout(pad=0.4)
    _save(fig, "fig2_flowchart.png")


# ── Figure 3 — Baseline convergence ─────────────────────────────────────────

def _draw_restart_markers(ax, df, color="#bbbbbb", label="Emitter restart"):
    """Draw thin vertical lines at generations where any emitter restarted."""
    if "restarted_emitters" not in df.columns:
        return
    restart_gens = df.loc[
        df["restarted_emitters"].notna() & (df["restarted_emitters"] != ""),
        "generation"
    ].values
    first = True
    for rg in restart_gens:
        ax.axvline(rg, color=color, linewidth=0.7, linestyle="--", zorder=0,
                   label=label if first else "_nolegend_")
        first = False


def fig3_baseline_convergence():
    df = pd.read_csv(RES / "baseline" / "cmaes_best.csv")
    gens = df["generation"].values
    tp   = df["best_throughput"].values

    fig, ax = plt.subplots(figsize=(7, 4))
    _style(ax)
    ax.tick_params(labelsize=12)

    _draw_restart_markers(ax, df)
    ax.plot(gens, tp, color=C_VAN, linewidth=1.8, label="Vanilla CMA-ES (5 emitters)")

    tau = 7.64
    ax.axhline(tau, color=C_GRAY, linestyle="--", linewidth=1.0, zorder=1,
               label="Zhang et al. (100 gens)")

    ax.set_xlim(0, gens[-1] + 5)
    ax.set_ylim(3.0, 8.7)
    ax.set_yticks([3, 4, 5, 6, 7, 8])
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best throughput (tasks/step)", fontsize=12)
    ax.set_title("Offline Vanilla CMA-ES Convergence (5 Emitters, 300 Generations)", fontsize=12, pad=8)
    ax.legend(loc="lower right", **{**LEGEND_KW, "fontsize": 10.5})

    fig.tight_layout()
    _save(fig, "fig3_baseline_convergence.png")


# ── Figure 4 — Sample efficiency (offline) ──────────────────────────────────

def fig4_sample_efficiency():
    vb = pd.read_csv(RES / "baseline" / "cmaes_best.csv")
    v_tp  = np.maximum.accumulate(vb["best_throughput"].values)
    v_cum = np.arange(1, len(v_tp) + 1) * 500   # 100 cands × 5 evals

    sb = pd.read_csv(RES / "surrogate_v3" / "cmaes_best.csv")
    sl = pd.read_csv(RES / "surrogate_v3" / "surrogate_log.csv")
    s_tp  = np.maximum.accumulate(sb["best_throughput"].values)
    s_cum = np.cumsum(sl["n_simulated"].values * 5)

    tau = 8.23
    v_cross_idx  = int(np.argmax(v_tp >= tau))
    s_cross_idx  = int(np.argmax(s_tp >= tau))
    v_cross_sims = int(v_cum[v_cross_idx])   # 137 500
    s_cross_sims = int(s_cum[s_cross_idx])   # ~49 000
    speedup      = v_cross_sims / s_cross_sims

    # Show surrogate's full run; vanilla is truncated at the same x limit
    X_MAX = int(s_cum[-1]) + 2000   # ~65 200

    fig, ax = plt.subplots(figsize=(7, 4))
    _style(ax)
    ax.tick_params(labelsize=12)

    # Clip vanilla to X_MAX for display
    v_mask = v_cum <= X_MAX
    ax.plot(v_cum[v_mask] / 1000, v_tp[v_mask],
            color=C_VAN,  linewidth=1.8, label="Vanilla CMA-ES")
    ax.plot(s_cum / 1000, s_tp,
            color=C_SURR, linewidth=1.8, label="Surrogate V3")

    # Threshold line
    ax.axhline(tau, color=C_GRAY, linestyle="--", linewidth=0.9, zorder=1,
               label=f"τ = {tau}")

    # Surrogate crossover (in-view)
    ax.axvline(s_cross_sims / 1000, color=C_SURR, linestyle=":", linewidth=0.9, zorder=1)
    ax.text(s_cross_sims / 1000 - 0.5, 8.59,
            f"Surrogate: {s_cross_sims // 1000}k sims",
            color=C_SURR, fontsize=9.5, ha="right", va="top")

    # Vanilla crossover is off-screen — annotate at right edge
    ax.annotate(f"Vanilla: {v_cross_sims // 1000}k sims →",
                xy=(X_MAX / 1000, ax.get_ylim()[0] if False else tau),
                xytext=(X_MAX / 1000 * 0.97, tau - 0.38),
                color=C_VAN, fontsize=9.5, ha="right", va="top",
                arrowprops=dict(arrowstyle="-|>", color=C_VAN,
                                lw=0.7, mutation_scale=8))

    # Speedup box — top left
    ax.text(0.03, 0.97, f"{speedup:.2f}× fewer simulations at equal quality",
            transform=ax.transAxes, fontsize=10, ha="left", va="top",
            color="#444444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_GRID, linewidth=0.8))

    ax.set_xlim(0, X_MAX / 1000)
    ax.set_ylim(3.0, 8.7)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_yticks([3, 4, 5, 6, 7, 8])
    ax.set_xlabel("Simulations (thousands)", fontsize=12)
    ax.set_ylabel("Best throughput (tasks/step)", fontsize=12)
    ax.set_title("Offline Sample Efficiency: Surrogate vs. Vanilla CMA-ES", fontsize=12, pad=8)
    ax.legend(loc="lower right", **{**LEGEND_KW, "fontsize": 10.5})

    fig.tight_layout()
    _save(fig, "fig4_sample_efficiency.png")


# ── Figure 5 — Surrogate accuracy in live loop ───────────────────────────────

def fig5_surrogate_accuracy():
    sl = pd.read_csv(RES / "surrogate_v3" / "surrogate_log.csv")

    rho_mask = sl["mode"].isin(["warmup", "control"])
    rho_gens = sl.loc[rho_mask, "generation"].values
    rho_vals = sl.loc[rho_mask, "surrogate_rho"].values

    n_gens = sl["generation"].max() + 1

    fig, ax = plt.subplots(figsize=(8, 4))
    _style(ax)

    ax.axvspan(0, 19, color=C_WARM, alpha=0.8, zorder=0)

    band = ax.axhspan(0.68, 0.89, color=C_BAND, alpha=0.7, zorder=0,
                      label="Stable range (0.68–0.89)")

    sb = pd.read_csv(RES / "surrogate_v3" / "cmaes_best.csv")
    _draw_restart_markers(ax, sb)

    line1, = ax.plot(rho_gens, rho_vals, color=C_VAN, linewidth=1.2,
                     marker="o", markersize=3, label="Spearman ρ (control gens)")
    warm_patch = mpatches.Patch(color=C_WARM, label="Warmup phase")
    handles = [line1, band, warm_patch]
    if "restarted_emitters" in sb.columns:
        handles.append(plt.Line2D([0], [0], color="#bbbbbb", linewidth=0.7,
                                  linestyle="--", label="Emitter restart"))
    ax.legend(handles=handles, loc="lower right", **LEGEND_KW)

    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Spearman ρ")
    ax.set_xlabel("Generation")
    ax.set_xlim(-5, n_gens + 5)
    ax.set_title("Surrogate Rank Correlation over Training (offline, 300 gens)",
                 fontsize=10, pad=8)

    fig.tight_layout()
    _save(fig, "fig5_surrogate_accuracy.png")


# ── Figure 6 — Lambda ablation ───────────────────────────────────────────────

def fig6_lambda_ablation():
    configs = [
        ("Vanilla CMA-ES",     8.2273, C_GRAY),
        ("V3  (λ = 1.0)",      8.1523, C_VAN),
        ("V1  (λ = 0)",        8.0852, C_V1),
        ("Ablation (λ = 2.0)", 7.9024, C_SURR),
    ]
    labels = [c[0] for c in configs]
    values = [c[1] for c in configs]
    colors = [c[2] for c in configs]

    fig, ax = plt.subplots(figsize=(7, 3.0))
    _style(ax, grid_axis=None)
    ax.spines["left"].set_visible(False)

    y_pos = np.arange(len(configs))
    bars = ax.barh(y_pos, values, color=colors, height=0.55, zorder=3)

    # Value labels at bar ends (outside, since bars go to ~8.x on a 0-8.5 axis)
    for bar, val in zip(bars, values):
        ax.text(val + 0.04, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8.5, color="#444444")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(values) + 0.35)
    ax.set_xlabel("Best throughput after 300 gens (tasks/step)")
    ax.set_title("UCB Exploration Weight (λ) Ablation", fontsize=10, pad=8)
    ax.tick_params(left=False)
    ax.invert_yaxis()

    ax.grid(axis="x", color=C_GRID, linestyle="--", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, "fig6_lambda_ablation.png")


# ── Figure 7 — Online GGO extension ─────────────────────────────────────────

def fig7_online_ggo():
    vb  = pd.read_csv(RES / "online_baseline"  / "cmaes_best.csv")
    v_tp  = np.maximum.accumulate(vb["best_throughput"].values)
    v_cum = np.arange(1, len(v_tp) + 1) * 100

    sb  = pd.read_csv(RES / "online_surrogate"  / "cmaes_best.csv")
    sl  = pd.read_csv(RES / "online_surrogate"  / "surrogate_log.csv")
    s_tp  = np.maximum.accumulate(sb["best_throughput"].values)
    s_cum = np.cumsum(sl["n_simulated"].values)

    tau  = 7.332
    v_cross_sims = int(v_cum[np.argmax(np.maximum.accumulate(vb["best_throughput"].values) >= tau)])
    s_cross_sims = int(s_cum[np.argmax(np.maximum.accumulate(sb["best_throughput"].values) >= tau)])

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 3.8))
    _style(ax1)
    ax1.tick_params(labelsize=12)

    ax1.plot(v_cum, v_tp, color=C_VAN,  linewidth=1.8, label="Vanilla CMA-ES")
    ax1.plot(s_cum, s_tp, color=C_SURR, linewidth=1.8, label="Surrogate (UCB, k=20)")

    # Flat dashed extension for surrogate
    ax1.plot([s_cum[-1], v_cum[-1]], [s_tp[-1], s_tp[-1]],
             color=C_SURR, linewidth=1.2, linestyle="--")

    # Crossover vertical lines
    TOP_Y = 7.78
    for x_cross, lbl, color in [
            (s_cross_sims, f"Surrogate\n{s_cross_sims:,} sims", C_SURR),
            (v_cross_sims, f"Vanilla\n{v_cross_sims:,} sims",   C_VAN)]:
        ax1.axvline(x_cross, color=color, linestyle=":", linewidth=0.9, zorder=1)
        ax1.text(x_cross, TOP_Y, lbl, color=color, fontsize=9.5,
                 ha="center", va="top")

    speedup = v_cross_sims / s_cross_sims
    ax1.text(0.03, 0.97, f"{speedup:.1f}× crossover speedup",
             transform=ax1.transAxes, fontsize=10, ha="left", va="top",
             color="#444444",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor=C_GRID, linewidth=0.8))

    ax1.set_xlim(0, 10500)
    ax1.set_ylim(5.0, 7.95)
    ax1.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
    ax1.set_xticklabels(["0", "2k", "4k", "6k", "8k", "10k"], fontsize=12)
    ax1.set_yticks([5.0, 5.5, 6.0, 6.5, 7.0, 7.5])
    ax1.set_xlabel("Cumulative simulations", fontsize=12)
    ax1.set_ylabel("Best throughput (tasks/step)", fontsize=12)
    ax1.set_title("Online GGO: Surrogate vs. Vanilla CMA-ES on CNN Policy Optimisation",
                  fontsize=11, pad=6)
    ax1.legend(loc="lower right", **{**LEGEND_KW, "fontsize": 10.5})

    fig.tight_layout()
    _save(fig, "fig7_online_ggo.png")


# ── Figure 8 — Speedup comparison across all conditions ─────────────────────

def fig8_speedup_comparison():
    conditions = [
        ("Warehouse\nseed 42",  2.81, "offline", C_VAN),
        ("Warehouse\nseed 123", 2.92, "offline", C_VAN),
        ("Online GGO\nwarehouse", 2.41, "online", C_SURR),
        ("Random-32x32\nseed 42", 1.00, "offline", C_GRAY),
    ]
    labels   = [c[0] for c in conditions]
    speedups = [c[1] for c in conditions]
    colors   = [c[3] for c in conditions]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    _style(ax, grid_axis="y")

    x = np.arange(len(conditions))
    bars = ax.bar(x, speedups, color=colors, width=0.55, zorder=3)

    # Value labels above bars; dagger on the 1.00x bar (converged in warmup)
    for i, (bar, val) in enumerate(zip(bars, speedups)):
        label = f"1.00x†" if val == 1.00 else f"{val:.2f}x"
        offset = 0.06 if val == 1.00 else 0.04
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                label, ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#333333")

    # Reference line at 1× (no speedup)
    ax.axhline(1.0, color=C_GRAY, linewidth=0.9, linestyle="--",
               zorder=1, label="No speedup (1×)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlim(-0.5, len(conditions) - 0.5)
    ax.set_ylim(0, 3.6)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("Simulation speedup (×)", fontsize=10)
    ax.set_title("Crossover-Point Speedup: Surrogate V3 vs. Vanilla CMA-ES",
                 fontsize=11, pad=8)


    # Legend patches for the two colours used
    handles = [
        mpatches.Patch(color=C_VAN,  label="Offline — warehouse"),
        mpatches.Patch(color=C_SURR, label="Online GGO — warehouse"),
        mpatches.Patch(color=C_GRAY, label="Offline — random-32x32"),
    ]
    ax.legend(handles=handles, loc="upper right", **LEGEND_KW)

    fig.tight_layout()
    _save(fig, "fig8_speedup_comparison.png")


# ── Figure 9 — Surrogate accuracy + uncertainty (s123) ──────────────────────

def fig9_surrogate_accuracy_s123():
    sl = pd.read_csv(RES / "surrogate_v3_s123" / "surrogate_log.csv")
    cb = pd.read_csv(RES / "surrogate_v3_s123" / "cmaes_best.csv")

    n_gens   = sl["generation"].max() + 1
    xlim     = (-5, n_gens + 5)
    WARMUP_END = 19.5

    # Rho data — only where surrogate was ready (gen >= 1)
    rho_df   = sl[sl["surrogate_rho"].notna()].copy()
    warm_rho = rho_df[rho_df["mode"] == "warmup"]
    ctrl_rho = rho_df[rho_df["mode"] == "control"]

    # Uncertainty data
    std_df   = sl[sl["mean_std"].notna()].copy()
    sel_df   = sl[sl["selected_std"].notna()].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6),
                                   gridspec_kw={"hspace": 0.52})
    for ax in (ax1, ax2):
        _style(ax)
        ax.axvspan(0, WARMUP_END, color=C_WARM, alpha=0.8, zorder=0)
        _draw_restart_markers(ax, cb)

    # ── Top: Spearman ρ ───────────────────────────────────────────────────────
    # Join warmup + control with a connecting line
    joined = rho_df[rho_df["mode"].isin(["warmup", "control"])].sort_values("generation")
    ax1.plot(joined["generation"].values, joined["surrogate_rho"].values,
             color=C_VAN, linewidth=1.2, zorder=2)
    ax1.scatter(warm_rho["generation"].values, warm_rho["surrogate_rho"].values,
                color=C_VAN, s=18, zorder=3, label="Warmup (full eval)")
    ax1.scatter(ctrl_rho["generation"].values, ctrl_rho["surrogate_rho"].values,
                color=C_SURR, s=22, marker="D", zorder=3, label="Control (full eval)")

    ax1.set_ylim(0.0, 1.05)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_ylabel("Spearman ρ")
    ax1.set_xlabel("Generation")
    ax1.set_xlim(*xlim)
    ax1.set_title("Surrogate Rank Accuracy (Spearman ρ, seed 123)", fontsize=10, pad=8)

    warm_patch = mpatches.Patch(color=C_WARM, label="Warmup phase")
    legend_handles = [
        plt.Line2D([0], [0], color=C_VAN,  marker="o", markersize=4,
                   linewidth=1.2, label="Warmup (full eval)"),
        plt.Line2D([0], [0], color=C_SURR, marker="D", markersize=4,
                   linewidth=0, label="Control (full eval)"),
        warm_patch,
    ]
    ax1.legend(handles=legend_handles, loc="lower right", **LEGEND_KW)

    # ── Bottom: Ensemble uncertainty ──────────────────────────────────────────
    ax2.plot(std_df["generation"].values, std_df["mean_std"].values,
             color=C_GRAY, linewidth=1.4, label="Mean ensemble std (all gens)")
    if len(sel_df) > 0:
        ax2.plot(sel_df["generation"].values, sel_df["selected_std"].values,
                 color=C_SURR, linewidth=1.0, linestyle="--", alpha=0.85,
                 label="Top-k selected std")

    ax2.set_ylabel("Prediction std (normalized units)")
    ax2.set_xlabel("Generation")
    ax2.set_xlim(*xlim)
    ax2.set_title("Surrogate Prediction Uncertainty over Training", fontsize=10, pad=8)
    ax2.legend(loc="upper right", **LEGEND_KW)

    fig.suptitle("Surrogate Model Accuracy and Uncertainty — Seed 123 (300 gens)",
                 fontsize=10.5, y=1.01)
    fig.tight_layout()
    _save(fig, "fig9_surrogate_accuracy_s123.png")


# ── Figure 9b — same plot for surrogate_v3 (seed 42) ────────────────────────

def fig9b_surrogate_accuracy_s42():
    sl  = pd.read_csv(RES / "surrogate_v3" / "surrogate_log.csv")
    cb  = pd.read_csv(RES / "surrogate_v3" / "cmaes_best.csv")
    rmse_df = pd.read_csv(RES / "surrogate_v3" / "surrogate_rmse.csv")

    n_gens   = sl["generation"].max() + 1
    xlim     = (-5, n_gens + 5)
    WARMUP_END = 19.5

    rho_df   = sl[sl["surrogate_rho"].notna()].copy()
    warm_rho = rho_df[rho_df["mode"] == "warmup"]
    ctrl_rho = rho_df[rho_df["mode"] == "control"]

    std_df = sl[sl["mean_std"].notna()].copy()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 7),
                                        gridspec_kw={"hspace": 0.55})
    for ax in (ax1, ax2, ax3):
        _style(ax)
        ax.axvspan(0, WARMUP_END, color=C_WARM, alpha=0.8, zorder=0)

    # ── Panel 1: Spearman ρ ───────────────────────────────────────────────────
    joined = rho_df[rho_df["mode"].isin(["warmup", "control"])].sort_values("generation")
    ax1.plot(joined["generation"].values, joined["surrogate_rho"].values,
             color=C_VAN, linewidth=1.2, zorder=2)
    ax1.scatter(warm_rho["generation"].values, warm_rho["surrogate_rho"].values,
                color=C_VAN, s=14, zorder=3)
    ax1.scatter(ctrl_rho["generation"].values, ctrl_rho["surrogate_rho"].values,
                color=C_SURR, s=16, marker="D", zorder=3)

    ax1.set_ylim(0.0, 1.05)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_ylabel("Spearman ρ")
    ax1.set_xlim(*xlim)
    ax1.set_title("Rank Accuracy (Spearman ρ)", fontsize=11, pad=6)

    warm_patch = mpatches.Patch(color=C_WARM, label="Warmup")
    ax1.legend(handles=[
        plt.Line2D([0], [0], color=C_VAN,  marker="o", markersize=3,
                   linewidth=1.2, label="Warmup"),
        plt.Line2D([0], [0], color=C_SURR, marker="D", markersize=3,
                   linewidth=0, label="Control"),
        warm_patch,
    ], loc="lower right", **LEGEND_KW)

    # ── Panel 2: Reconstructed nRMSE at control gens ─────────────────────────
    ax2.plot(rmse_df["generation"].values, rmse_df["nrmse"].values,
             color=C_SURR, linewidth=1.4, marker="D", markersize=3, label="nRMSE (control gens)")
    ax2.axhline(1.0, color=C_GRAY, linewidth=0.8, linestyle="--", zorder=1,
                label="nRMSE = 1 (random)")
    ax2.axvline(100, color="#cccccc", linewidth=0.7, linestyle=":", zorder=1)
    ax2.text(102, ax2.get_ylim()[1] if False else 0.95, "plateau", fontsize=8,
             color="#888888", va="top")

    ax2.set_ylim(0.0, 1.25)
    ax2.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax2.set_ylabel("nRMSE  (RMSE / σ_pop)")
    ax2.set_xlim(*xlim)
    ax2.set_title("Prediction Error (reconstructed)", fontsize=11, pad=6)
    ax2.legend(loc="upper right", **LEGEND_KW)

    # ── Panel 3: Per-emitter mean throughput ─────────────────────────────────
    cl = pd.read_csv(RES / "surrogate_v3" / "cmaes_log.csv")
    ctrl_gens = sl.loc[sl["mode"] == "control", "generation"].sort_values().tolist()
    EM_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
    for em_id in range(5):
        em_means = []
        for g in ctrl_gens:
            sub = cl[(cl["generation"] == g) & (cl["emitter_id"] == em_id)]
            em_means.append(sub["mean_throughput"].mean() if len(sub) > 0 else np.nan)
        lw = 1.8 if em_id == 2 else 1.0
        ax3.plot(ctrl_gens, em_means, color=EM_COLORS[em_id], linewidth=lw,
                 marker="o", markersize=2.5,
                 label=f"Emitter {em_id}" + (" ★" if em_id == 2 else ""))

    ax3.set_ylabel("Mean throughput (tasks/step)")
    ax3.set_xlabel("Generation")
    ax3.set_xlim(*xlim)
    ax3.set_title("Per-Emitter Mean Throughput (control gens)", fontsize=11, pad=6)
    ax3.legend(loc="upper left", **LEGEND_KW)

    fig.suptitle("Surrogate Accuracy, Error & Emitter Structure\n— Seed 42 (400 gens)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, "fig9b_surrogate_accuracy_s42.png")


# ── Figure 11 — emitter starvation + ICC (combined) ─────────────────────────

def fig11_rho_emitter_structure():
    sl   = pd.read_csv(RES / "surrogate_v3" / "surrogate_log.csv")
    cl   = pd.read_csv(RES / "surrogate_v3" / "cmaes_log.csv")   # emitter starvation
    cl5  = pd.read_csv(RES / "baseline"     / "cmaes_log.csv")   # ICC (vanilla, gradual divergence)
    cl1  = pd.read_csv(RES / "baseline_1em" / "cmaes_log.csv")

    n_gens    = sl["generation"].max() + 1
    xlim      = (-5, n_gens + 5)
    WARMUP_END = 19.5
    ctrl_gens  = sl.loc[sl["mode"] == "control", "generation"].sort_values().tolist()

    EM_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

    # ── Compute ICC from vanilla baseline at surrogate control gen indices ────
    eval_cols   = [c for c in cl1.columns if c.startswith("eval_")]
    noise_floor = (cl1[eval_cols].std(axis=1).mean() / np.sqrt(len(eval_cols))
                   if eval_cols else 0.04)

    ctrl_df = sl[sl["mode"] == "control"].dropna(subset=["surrogate_rho"])
    icc_vals, within_vals, icc_gens = [], [], []
    for _, row in ctrl_df.iterrows():
        g   = int(row["generation"])
        sub = cl5[cl5["generation"] == g]   # vanilla baseline data at gen g
        if len(sub) < 5:
            continue
        em_means   = sub.groupby("emitter_id")["mean_throughput"].mean().values
        between    = float(em_means.var())
        within_var = sub.groupby("emitter_id")["mean_throughput"].var().dropna()
        if len(within_var) == 0:
            continue
        within = float(within_var.mean())
        total  = between + within
        if total < 1e-9:
            continue
        icc_vals.append(between / total)
        within_vals.append(np.sqrt(within))
        icc_gens.append(g)

    icc_vals    = np.array(icc_vals)
    within_vals = np.array(within_vals)
    icc_gens    = np.array(icc_gens)

    # ── Compute ICC from surrogate run at the same control gen indices ─────────
    surr_icc_vals, surr_icc_gens = [], []
    for _, row in ctrl_df.iterrows():
        g   = int(row["generation"])
        sub = cl[cl["generation"] == g]   # surrogate run (starvation already active)
        if len(sub) < 5:
            continue
        em_means   = sub.groupby("emitter_id")["mean_throughput"].mean().values
        between    = float(em_means.var())
        within_var = sub.groupby("emitter_id")["mean_throughput"].var().dropna()
        if len(within_var) == 0:
            continue
        within = float(within_var.mean())
        total  = between + within
        if total < 1e-9:
            continue
        surr_icc_vals.append(between / total)
        surr_icc_gens.append(g)

    surr_icc_vals = np.array(surr_icc_vals)
    surr_icc_gens = np.array(surr_icc_gens)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6),
                                   gridspec_kw={"hspace": 0.50})
    for ax in (ax1, ax2):
        _style(ax)
        ax.axvspan(0, WARMUP_END, color=C_WARM, alpha=0.8, zorder=0)

    # ── Top: per-emitter mean throughput ─────────────────────────────────────
    for em_id in range(5):
        em_means = []
        for g in ctrl_gens:
            sub = cl[(cl["generation"] == g) & (cl["emitter_id"] == em_id)]
            em_means.append(sub["mean_throughput"].mean() if len(sub) > 0 else np.nan)
        lw  = 2.0 if em_id == 2 else 1.1
        ms  = 3   if em_id == 2 else 2
        lbl = f"Emitter {em_id} ★" if em_id == 2 else f"Emitter {em_id}"
        ax1.plot(ctrl_gens, em_means, color=EM_COLORS[em_id],
                 linewidth=lw, marker="o", markersize=ms, label=lbl)

    ax1.set_ylabel("Mean throughput (tasks/step)")
    ax1.set_xlim(*xlim)
    ax1.set_title("Per-Emitter Mean Throughput at Control Gens", fontsize=10, pad=6)
    ax1.legend(loc="upper left", **LEGEND_KW)

    # ── Bottom: ICC + within-emitter std ─────────────────────────────────────
    ax2b = ax2.twinx()
    ax2b.spines["top"].set_visible(False)

    ax2.plot(icc_gens, icc_vals, color=C_SURR, linewidth=1.6,
             marker="o", markersize=3, label="ICC (vanilla run)")
    ax2.plot(surr_icc_gens, surr_icc_vals, color="#c44e52", linewidth=1.4,
             marker="^", markersize=2.5, linestyle="--", label="ICC (surrogate run)")
    ax2b.plot(icc_gens, within_vals, color=C_VAN, linewidth=1.4,
              marker="s", markersize=2.5, linestyle="--",
              label="Within-emitter σ")
    ax2b.axhline(noise_floor, color=C_GRAY, linewidth=0.9, linestyle=":",
                 label=f"Noise floor ({noise_floor:.2f})")

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("ICC", color=C_SURR)
    ax2b.set_ylabel("Within-emitter std", color=C_VAN)
    ax2.set_ylim(0, 1.05)
    ax2b.set_ylim(0, 0.35)
    ax2.tick_params(axis="y", labelcolor=C_SURR)
    ax2b.tick_params(axis="y", labelcolor=C_VAN)
    ax2.set_xlim(*xlim)
    ax2.set_title("Population Structure: ICC and Within-Emitter Variance",
                  fontsize=10, pad=6)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="center right", **LEGEND_KW)

    fig.suptitle("Emitter Divergence and Population Structure — Seed 42 (400 gens)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    _save(fig, "fig11_rho_emitter_structure.png")


# ── Figure 12 — ICC: why surrogate needs multi-emitter structure ─────────────

def fig12_icc_population_structure():
    cl5  = pd.read_csv(RES / "baseline"      / "cmaes_log.csv")
    sl5  = pd.read_csv(RES / "surrogate_v3"  / "surrogate_log.csv")
    cl1  = pd.read_csv(RES / "baseline_1em"  / "cmaes_log.csv")

    eval_cols = [c for c in cl1.columns if c.startswith("eval_")]
    sigma_noise_single = cl1[eval_cols].std(axis=1).mean()
    noise_floor = sigma_noise_single / np.sqrt(len(eval_cols))

    ctrl = sl5[sl5["mode"] == "control"].dropna(subset=["surrogate_rho"])

    icc_vals, within_vals, rho_vals, ctrl_gens = [], [], [], []
    for _, row in ctrl.iterrows():
        g   = int(row["generation"])
        sub = cl5[cl5["generation"] == g]
        if len(sub) == 0:
            continue
        em_means = sub.groupby("emitter_id")["mean_throughput"].mean().values
        between  = float(em_means.var())
        within   = float(sub.groupby("emitter_id")["mean_throughput"].var().mean())
        total    = between + within
        if total < 1e-9:
            continue
        icc_vals.append(between / total)
        within_vals.append(np.sqrt(within))
        rho_vals.append(float(row["surrogate_rho"]))
        ctrl_gens.append(g)

    icc_vals    = np.array(icc_vals)
    within_vals = np.array(within_vals)
    ctrl_gens   = np.array(ctrl_gens)

    # 1-emitter rho results (from 01b run)
    rho_1em = [0.1219, 0.0671, 0.0038, 0.0350, 0.0099, 0.1223, 0.1019, 0.0191]   # xgboost
    rho_1em_mlp = [0.2407, 0.2274, 0.0797, -0.0057, -0.1418, 0.0215, -0.0134, -0.0868]
    rho_1em_cnn = [0.2518, 0.3296, 0.2123, 0.2603, -0.0370, -0.0804, -0.0005, 0.0638]
    all_1em = rho_1em + rho_1em_mlp + rho_1em_cnn

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5),
                                   gridspec_kw={"wspace": 0.45, "width_ratios": [2.5, 1]})
    for ax in (ax1, ax2):
        _style(ax)

    # ── Left: ICC + within-emitter std over gens ─────────────────────────────
    ax1b = ax1.twinx()
    ax1b.spines["top"].set_visible(False)

    ax1.plot(ctrl_gens, icc_vals, color=C_SURR, linewidth=1.6,
             marker="o", markersize=3, label="ICC (between / total var)")
    ax1b.plot(ctrl_gens, within_vals, color=C_VAN, linewidth=1.4,
              marker="s", markersize=2.5, linestyle="--",
              label="Within-emitter σ")
    ax1b.axhline(noise_floor, color=C_GRAY, linewidth=0.9, linestyle=":",
                 label=f"Noise floor (σ/√5 ≈ {noise_floor:.2f})")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("ICC", color=C_SURR)
    ax1b.set_ylabel("Within-emitter std", color=C_VAN)
    ax1.set_ylim(0, 1.05)
    ax1b.set_ylim(0, 0.35)
    ax1.tick_params(axis="y", labelcolor=C_SURR)
    ax1b.tick_params(axis="y", labelcolor=C_VAN)
    ax1.set_title("Population Structure (5-emitter run)", fontsize=10, pad=6)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", **LEGEND_KW)

    # ── Right: ρ distribution 5-emitter vs 1-emitter ─────────────────────────
    ax2.boxplot([rho_vals, all_1em],
                labels=["5-emitter\n(control gens)", "1-emitter\n(all models)"],
                patch_artist=True,
                boxprops=dict(facecolor="#dce8f5", color=C_VAN),
                medianprops=dict(color=C_SURR, linewidth=2),
                whiskerprops=dict(color=C_VAN),
                capprops=dict(color=C_VAN),
                flierprops=dict(marker="o", markersize=3, color=C_GRAY))
    ax2.axhline(0, color=C_GRAY, linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Spearman ρ")
    ax2.set_ylim(-0.3, 1.05)
    ax2.set_title("Surrogate ρ: 5-emitter vs 1-emitter", fontsize=10, pad=6)

    fig.suptitle("ICC Analysis: Why Surrogate Requires Multi-Emitter Population Structure",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    _save(fig, "fig12_icc_population_structure.png")


# ── CLI ──────────────────────────────────────────────────────────────────────

FIGURES = {
    1:  ("Surrogate architecture comparison",        fig1_surrogate_comparison),
    2:  ("System architecture flowchart",            fig2_flowchart),
    3:  ("Baseline convergence",                     fig3_baseline_convergence),
    4:  ("Sample efficiency (offline)",              fig4_sample_efficiency),
    6:  ("Lambda ablation",                          fig6_lambda_ablation),
    7:  ("Online GGO extension",                     fig7_online_ggo),
    8:  ("Speedup comparison across conditions",     fig8_speedup_comparison),
    9:  ("Surrogate accuracy + uncertainty (s123)",  fig9_surrogate_accuracy_s123),
    10: ("Surrogate accuracy + uncertainty (s42)",   fig9b_surrogate_accuracy_s42),
    11: ("Emitter starvation + ICC (combined)",      fig11_rho_emitter_structure),
}


def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures")
    parser.add_argument("--fig", type=int, nargs="+",
                        help="Which figures to generate (default: all)")
    args = parser.parse_args()

    targets = args.fig if args.fig else list(FIGURES.keys())

    for n in targets:
        if n not in FIGURES:
            print(f"  Figure {n} not defined — skipping")
            continue
        title, fn = FIGURES[n]
        print(f"Figure {n}: {title}")
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
