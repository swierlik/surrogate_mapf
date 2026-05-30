"""Evaluate all best solutions with 100 simulations for robust throughput estimates.

Covers: warehouse seed 42, warehouse seed 123, random-32x32 seed 42.
For each pair (vanilla, surrogate) computes the crossover-point speedup:
  tau = surrogate's 100-sim mean
  speedup = vanilla_sims_at_tau / surrogate_sims_at_tau

Usage:
    python -m experiments.evaluate_best_solutions
"""

import sys
import numpy as np
import pandas as pd
import scipy.stats as st
from pathlib import Path

sys.path.insert(0, ".")

WAREHOUSE_MAP = "maps/competition/human/pibt_warehouse_33x36_w_mode.json"
RANDOM32_MAP  = "maps/competition/human/pibt_random_unweight_32x32.json"
N_EVALS   = 100
BASE_SEED = 9999


# ── helpers ───────────────────────────────────────────────────────────────────

def ci95(vals):
    m  = float(vals.mean())
    s  = float(vals.std())
    lo, hi = st.t.interval(0.95, len(vals) - 1, loc=m, scale=st.sem(vals))
    return m, s, float(lo), float(hi)


def switch_map(map_rel_path):
    import src.simulator.evaluate as ev
    ev.MAP_REL_PATH      = map_rel_path
    ev.MapInfo._instance = None


def eval_solution(sol_path, num_agents=400, n_evals=N_EVALS):
    from src.simulator.evaluate import evaluate_batch
    sol = np.load(sol_path)
    _, all_tp = evaluate_batch(
        sol.reshape(1, -1),
        num_agents=num_agents,
        n_evals=n_evals,
        base_seed=BASE_SEED,
        n_workers=8,
    )
    return all_tp[0]          # shape (n_evals,)


def crossover_speedup(van_dir, surr_dir, tau, n_evals_run=5):
    """Return (vanilla_sims, surrogate_sims, speedup) at first crossing of tau.

    tau = vanilla's 100-sim true mean (the quality target).
    Measures the cumulative simulation count at which each method's running
    5-sim best first exceeded tau, then returns vanilla_sims / surrogate_sims.
    """
    vb = pd.read_csv(Path(van_dir)  / "cmaes_best.csv")
    sb = pd.read_csv(Path(surr_dir) / "cmaes_best.csv")
    sl = pd.read_csv(Path(surr_dir) / "surrogate_log.csv")

    v_tp  = np.maximum.accumulate(vb["best_throughput"].values)
    s_tp  = np.maximum.accumulate(sb["best_throughput"].values)
    v_cum = np.arange(1, len(v_tp) + 1) * 100 * n_evals_run
    s_cum = np.cumsum(sl["n_simulated"].values * n_evals_run)

    v_idx = int(np.argmax(v_tp >= tau))
    s_idx = int(np.argmax(s_tp >= tau))

    if v_tp[v_idx] < tau:
        return None, None, None   # vanilla never reaches tau in its run
    if s_tp[s_idx] < tau:
        return None, None, None   # surrogate never reaches tau in its run

    v_sims  = int(v_cum[v_idx])
    s_sims  = int(s_cum[s_idx])
    speedup = v_sims / s_sims
    return v_sims, s_sims, speedup


def section(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ── warehouse seed 42 & seed 123 ──────────────────────────────────────────────

switch_map(WAREHOUSE_MAP)

section("WAREHOUSE MAP — seed 42")
print(f"Evaluating each best solution with {N_EVALS} simulations...\n")

results = {}

for label, sol_dir in [("Vanilla s42",      "results/baseline"),
                        ("Surrogate V3 s42", "results/surrogate_v3")]:
    print(f"  {label}...")
    vals = eval_solution(Path(sol_dir) / "best_solution.npy", num_agents=400)
    m, s, lo, hi = ci95(vals)
    results[label] = dict(mean=m, std=s, ci_lo=lo, ci_hi=hi)
    print(f"    {m:.4f} ± {s:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

tau_s42 = results["Vanilla s42"]["mean"]   # target = vanilla's true quality
v_sims, s_sims, sp = crossover_speedup(
    "results/baseline", "results/surrogate_v3", tau_s42)
results["speedup_s42"] = dict(tau=tau_s42, v_sims=v_sims, s_sims=s_sims, speedup=sp)
if sp:
    print(f"\n  Speedup at tau={tau_s42:.4f}: vanilla={v_sims:,} sims, "
          f"surrogate={s_sims:,} sims -> {sp:.2f}x")

section("WAREHOUSE MAP — seed 123")
print(f"Evaluating each best solution with {N_EVALS} simulations...\n")

for label, sol_dir in [("Vanilla s123",      "results/baseline_s123"),
                        ("Surrogate V3 s123", "results/surrogate_v3_s123")]:
    print(f"  {label}...")
    vals = eval_solution(Path(sol_dir) / "best_solution.npy", num_agents=400)
    m, s, lo, hi = ci95(vals)
    results[label] = dict(mean=m, std=s, ci_lo=lo, ci_hi=hi)
    print(f"    {m:.4f} ± {s:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

tau_s123 = results["Vanilla s123"]["mean"]
v_sims123, s_sims123, sp123 = crossover_speedup(
    "results/baseline_s123", "results/surrogate_v3_s123", tau_s123)
results["speedup_s123"] = dict(tau=tau_s123, v_sims=v_sims123, s_sims=s_sims123, speedup=sp123)
if sp123:
    print(f"\n  Speedup at tau={tau_s123:.4f}: vanilla={v_sims123:,} sims, "
          f"surrogate={s_sims123:,} sims -> {sp123:.2f}x")

# ── random32 map ──────────────────────────────────────────────────────────────

switch_map(RANDOM32_MAP)

section("RANDOM-32x32 MAP — seed 42")
print(f"Evaluating each best solution with {N_EVALS} simulations...\n")

for label, sol_dir in [("Vanilla r32",      "results/baseline_random32"),
                        ("Surrogate V3 r32", "results/surrogate_v3_random32")]:
    print(f"  {label}...")
    vals = eval_solution(Path(sol_dir) / "best_solution.npy", num_agents=300)
    m, s, lo, hi = ci95(vals)
    results[label] = dict(mean=m, std=s, ci_lo=lo, ci_hi=hi)
    print(f"    {m:.4f} ± {s:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

tau_r32 = results["Vanilla r32"]["mean"]
v_sims_r, s_sims_r, sp_r = crossover_speedup(
    "results/baseline_random32", "results/surrogate_v3_random32", tau_r32)
results["speedup_r32"] = dict(tau=tau_r32, v_sims=v_sims_r, s_sims=s_sims_r, speedup=sp_r)
if sp_r:
    print(f"\n  Speedup at tau={tau_r32:.4f}: vanilla={v_sims_r:,} sims, "
          f"surrogate={s_sims_r:,} sims -> {sp_r:.2f}x")

# ── summary table ─────────────────────────────────────────────────────────────

section("SUMMARY")
fmt = "{:<24} {:>8} {:>8} {:>16}"
print(fmt.format("Run", "Mean", "Std", "95% CI"))
print("-" * 60)
for k in ["Vanilla s42", "Surrogate V3 s42",
          "Vanilla s123", "Surrogate V3 s123",
          "Vanilla r32", "Surrogate V3 r32"]:
    r = results[k]
    print(fmt.format(k, f"{r['mean']:.4f}", f"{r['std']:.4f}",
                     f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]"))

print("\nCrossover-point speedups:")
for key, tag in [("speedup_s42",  "Warehouse seed 42"),
                 ("speedup_s123", "Warehouse seed 123"),
                 ("speedup_r32",  "Random-32x32 seed 42")]:
    sp_d = results[key]
    if sp_d["speedup"]:
        print(f"  {tag:25s}: {sp_d['speedup']:.2f}x  "
              f"(vanilla {sp_d['v_sims']:,} vs surrogate {sp_d['s_sims']:,} sims "
              f"at tau={sp_d['tau']:.4f})")
    else:
        print(f"  {tag:25s}: tau not crossed by one method")

# ── save results for journal ──────────────────────────────────────────────────

import json
out_path = Path("results/best_solution_evals.json")
json_results = {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                    for kk, vv in v.items()}
                for k, v in results.items()}
out_path.write_text(json.dumps(json_results, indent=2))
print(f"\nResults saved to {out_path}")
