"""Bootstrap the online surrogate checkpoint from vanilla baseline data.

The first 10 warmup gens of the surrogate run are identical to gens 0-9 of the
vanilla baseline (same seed, sigma0, initial_mean → same solutions asked).
This script replays those 10 gens to advance the CMA-ES emitters and train the
surrogate, then saves a checkpoint at gen 9 so the surrogate run resumes from
gen 10, skipping ~1h17m of redundant warmup evaluation.

Usage:
    python bootstrap_online_surrogate.py
"""

import gc
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.optimizer.vanilla_cmaes import CMAEmitter
from src.optimizer.surrogate_cmaes import save_checkpoint, CHECKPOINT_FILE
from src.surrogate.mlp_model import EnsembleSurrogate
from src.utils.data import RunLogger
from src.simulator.evaluate_online import get_n_params

# ── settings (must match 06_online_surrogate defaults) ──────────────────────
SEED             = 42
N_EMITTERS       = 5
POPSIZE          = 20
SIGMA0           = 0.1
INITIAL_MEAN     = 0.0
N_EVALS          = 1
WARMUP_GENS      = 10
N_ENSEMBLE       = 5
BOOTSTRAP_FRAC   = 0.8

BASELINE_DIR     = Path("results/online_baseline")
OUTPUT_DIR       = Path("results/online_surrogate")

# ── load baseline data ───────────────────────────────────────────────────────
print("Loading baseline data...")
sols_all = np.load(BASELINE_DIR / "cmaes_solutions.npy")          # (10000, 4271)
log_all  = pd.read_csv(BASELINE_DIR / "cmaes_log.csv")
best_all = pd.read_csv(BASELINE_DIR / "cmaes_best.csv")

# First WARMUP_GENS gens only
warmup_mask = log_all["generation"] < WARMUP_GENS
log_w = log_all[warmup_mask].reset_index(drop=True)
sols_w = sols_all[: WARMUP_GENS * 100]                            # (1000, 4271)
tp_w   = log_w["mean_throughput"].values                           # (1000,)

# Actual wall clock for warmup from baseline
warmup_wallclock = best_all["cumulative_wallclock_s"].iloc[WARMUP_GENS - 1]
best_tp_warmup   = best_all["best_throughput"].iloc[WARMUP_GENS - 1]
print(f"Warmup: {WARMUP_GENS} gens, {len(sols_w)} solutions, "
      f"best={best_tp_warmup:.4f}, wallclock={warmup_wallclock:.0f}s "
      f"({warmup_wallclock/3600:.2f}h)")

# ── replay CMA-ES emitters through warmup ────────────────────────────────────
print("Replaying emitter state through warmup gens...")
sol_size = get_n_params()
rng = np.random.default_rng(SEED)
x0  = np.full(sol_size, INITIAL_MEAN)
emitter_seeds = rng.integers(0, 2**31, size=N_EMITTERS)
emitters = [
    CMAEmitter(i, x0, SIGMA0, POPSIZE, int(emitter_seeds[i]))
    for i in range(N_EMITTERS)
]

best_solution   = None
best_throughput = -np.inf

for gen in range(WARMUP_GENS):
    # Ask (advances internal CMA-ES state, must match original ask order)
    all_sols = np.concatenate([e.ask() for e in emitters], axis=0)

    # Retrieve true throughputs from baseline for this gen
    gen_tp = tp_w[gen * 100 : (gen + 1) * 100]

    # Tell each emitter its slice (CMA-ES minimises → negate)
    pos = 0
    for emitter in emitters:
        end = pos + POPSIZE
        emitter.tell(all_sols[pos:end], -gen_tp[pos:end])
        pos = end

    # Track best
    idx = np.argmax(gen_tp)
    if gen_tp[idx] > best_throughput:
        best_throughput = gen_tp[idx]
        best_solution   = all_sols[idx].copy()

    print(f"  Gen {gen+1:2d}/{WARMUP_GENS} replayed | best={best_throughput:.4f}")

# ── train surrogate on warmup data ───────────────────────────────────────────
print("\nTraining ensemble surrogate on warmup data...")
surrogate = EnsembleSurrogate(
    n_models=N_ENSEMBLE,
    bootstrap_frac=BOOTSTRAP_FRAC,
    max_epochs=100,
    patience=10,
)
surrogate.fit(sols_w, tp_w)
print(f"Surrogate fitted on {len(sols_w)} samples.")

# ── write CSV logs so RunLogger / SurrogateLogger have correct history ────────
print("\nWriting log files...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = RunLogger(OUTPUT_DIR, prefix="cmaes", n_evals=N_EVALS)
from src.optimizer.surrogate_cmaes import SurrogateLogger
surr_logger = SurrogateLogger(OUTPUT_DIR, resume=False)

for gen in range(WARMUP_GENS):
    gen_mask  = log_all["generation"] == gen
    gen_log   = log_all[gen_mask]
    gen_sols  = sols_all[gen * 100 : (gen + 1) * 100]
    gen_tp    = gen_log["mean_throughput"].values
    gen_all_tp = gen_tp.reshape(-1, 1)           # (100, 1) for n_evals=1
    emitter_ids = gen_log["emitter_id"].values

    gen_best_tp = float(gen_tp.max())
    best_so_far = float(best_all["best_throughput"].iloc[gen])
    wallclock   = float(best_all["gen_wallclock_s"].iloc[gen]) if "gen_wallclock_s" in best_all.columns else 460.0
    cum_wc      = float(best_all["cumulative_wallclock_s"].iloc[gen])

    logger.log_generation(gen, gen_sols, gen_tp, gen_all_tp, emitter_ids)
    logger.log_best(gen, best_so_far, gen_best_tp,
                    gen_wallclock_s=wallclock,
                    cumulative_wallclock_s=cum_wc,
                    restarted_emitters=[])
    surr_logger.log(gen, "warmup", 100, None, None, None, 0.0, 0)

logger.flush_solutions()
logger.close()
surr_logger.close()

# ── save checkpoint at gen 9 ─────────────────────────────────────────────────
print("\nSaving checkpoint at gen 9...")
gc.collect()
save_checkpoint(
    OUTPUT_DIR / CHECKPOINT_FILE,
    generation=WARMUP_GENS - 1,        # gen index 9 (0-based)
    emitters=emitters,
    best_solution=best_solution,
    best_throughput=best_throughput,
    rng_state=rng.bit_generator.state,
    total_sims=WARMUP_GENS * 100 * N_EVALS,
    cumulative_wallclock_s=warmup_wallclock,
    surrogate=surrogate,
)

print(f"\nBootstrap complete.")
print(f"  Checkpoint: gen=9, best={best_throughput:.4f}, "
      f"sims={WARMUP_GENS*100}, wallclock={warmup_wallclock:.0f}s")
print(f"\nNow run:")
print(f"  python -m experiments.06_online_surrogate --generations 100 "
      f"--n-evals 1 --n-workers 4 --chunk-size 20 --resume")
