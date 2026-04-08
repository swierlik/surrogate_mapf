"""Quick smoke test of the evaluate wrapper."""
import numpy as np
from src.simulator.evaluate import evaluate, evaluate_batch, get_sol_size

sol_size = get_sol_size()
print(f"Solution size: {sol_size}")

# Test batch eval: 3 different solutions, 1 eval each
print("Running batch eval (3 solutions, 1 seed each)...")
rng = np.random.default_rng(42)
solutions = np.array([
    np.ones(sol_size),                          # uniform
    rng.uniform(1.0, 10.0, sol_size),           # random
    np.full(sol_size, 5.0),                     # all-5s (CMA-ES initial mean)
])

tps = evaluate_batch(solutions, n_evals=1, base_seed=42)
for i, tp in enumerate(tps):
    print(f"  Solution {i}: throughput = {tp:.4f}")
print("Batch wrapper works!")
