"""Smoke test: 2 generations of multi-emitter CMA-ES."""
from src.optimizer.vanilla_cmaes import run_vanilla_cmaes

best_sol, best_tp = run_vanilla_cmaes(
    generations=2,
    n_emitters=2,          # 2 emitters for smoke test
    popsize_per_emitter=3, # tiny pop for speed
    n_evals=1,             # 1 eval per solution for speed
    sigma0=5.0,
    initial_mean=5.0,
    seed=42,
    output_dir="results/smoke_test_v2",
)
print(f"\nSmoke test done. Best throughput: {best_tp:.4f}")
