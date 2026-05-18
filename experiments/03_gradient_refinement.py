"""Post-hoc gradient refinement of the best solution using the V3 surrogate.

Loads the trained EnsembleSurrogate from a checkpoint, then runs gradient
ascent (Adam) from the best solution found during optimisation, maximising
the uncertainty-penalised surrogate score:

    score = mean_pred - penalty_lambda * std_pred

The penalty keeps the gradient from wandering into regions of high uncertainty
where the surrogate extrapolates badly.  Values are clamped to [0, 10] after
every step (projected gradient descent).

The refined candidate is evaluated in the real simulator and compared against
the original best.

Usage:
    python -m experiments.03_gradient_refinement
    python -m experiments.03_gradient_refinement --n-steps 2000 --lr 0.005
    python -m experiments.03_gradient_refinement --no-simulate   # dry-run, surrogate only
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from src.optimizer.surrogate_cmaes import _load_surrogate_state
from src.simulator.evaluate import evaluate_batch


# ---------------------------------------------------------------------------
# Differentiable forward pass through the ensemble
# ---------------------------------------------------------------------------

def ensemble_forward(surrogate, x_torch):
    """PyTorch forward through all ensemble models — gradients flow to x_torch.

    Args:
        surrogate: fitted EnsembleSurrogate
        x_torch:   tensor of shape (1, sol_size), requires_grad=True

    Returns:
        mean (1,), std (1,) — both differentiable w.r.t. x_torch
    """
    preds = []
    for mlp in surrogate.models:
        device = next(mlp.model.parameters()).device

        X_mean = torch.from_numpy(mlp._X_mean).float().to(device)  # (1, 4074)
        X_std  = torch.from_numpy(mlp._X_std).float().to(device)   # (1, 4074)

        x_std = (x_torch.to(device) - X_mean) / X_std

        mlp.model.eval()
        pred_norm = mlp.model(x_std)                      # (1,) standardised
        pred = pred_norm * mlp._y_std + mlp._y_mean       # back to throughput units
        preds.append(pred)

    stack = torch.stack(preds, dim=0)   # (n_models, 1)
    mean  = stack.mean(dim=0)           # (1,)
    std   = stack.std(dim=0)            # (1,)
    return mean, std


# ---------------------------------------------------------------------------
# Gradient ascent
# ---------------------------------------------------------------------------

def run_gradient_ascent(surrogate, start_solution, n_steps, lr, penalty_lambda,
                        reg_lambda, log_every=200):
    """Gradient ascent with L2 regularisation to stay near the starting point.

    No hard clamp — reg_lambda penalises large deviations from start_solution,
    keeping the gradient in the region where the surrogate is well-calibrated.

    loss = -(mean - penalty_lambda * std) + reg_lambda * ||x - x_start||^2

    Returns:
        refined_solution (np.ndarray, shape sol_size)
        surr_mean_before (float)
        surr_mean_after  (float)
        surr_std_after   (float)
    """
    x_start = torch.from_numpy(start_solution.copy()).float().unsqueeze(0)  # (1, 4074)
    x = x_start.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=lr)

    # Score before any steps
    with torch.no_grad():
        m0, s0 = ensemble_forward(surrogate, x)
    surr_mean_before = m0.item()
    print(f"  Initial surrogate  mean={m0.item():.4f}  std={s0.item():.4f}  "
          f"score={m0.item() - penalty_lambda * s0.item():.4f}")

    t0 = time.time()
    for step in range(n_steps):
        optimizer.zero_grad()
        mean, std = ensemble_forward(surrogate, x)
        reg  = reg_lambda * ((x - x_start.to(x.device)) ** 2).sum()
        loss = -(mean - penalty_lambda * std) + reg
        loss.backward()
        optimizer.step()

        if (step + 1) % log_every == 0:
            l2 = float(((x - x_start) ** 2).sum().sqrt().item())
            print(f"  step {step+1:4d}/{n_steps} | "
                  f"mean={mean.item():.4f}  std={std.item():.4f}  "
                  f"L2={l2:.2f}  score={mean.item() - penalty_lambda * std.item():.4f}")

    elapsed = time.time() - t0

    with torch.no_grad():
        mf, sf = ensemble_forward(surrogate, x)

    refined  = x.detach().squeeze(0).cpu().numpy()
    l2_final = float(np.linalg.norm(refined - start_solution))

    print(f"\n  Gradient ascent done in {elapsed:.1f}s")
    print(f"  L2 distance from start: {l2_final:.3f}")
    print(f"  Final surrogate  mean={mf.item():.4f}  std={sf.item():.4f}  "
          f"score={mf.item() - penalty_lambda * sf.item():.4f}")

    return refined, surr_mean_before, mf.item(), sf.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc gradient refinement using the V3 EnsembleSurrogate")
    parser.add_argument("--checkpoint",       default="results/surrogate_v3/checkpoint.pkl",
                        help="Path to V3 checkpoint (contains trained surrogate)")
    parser.add_argument("--best-solution",    default="results/surrogate_v3/best_solution.npy",
                        help="Starting point for gradient ascent")
    parser.add_argument("--v3-best-csv",      default="results/surrogate_v3/cmaes_best.csv",
                        help="CSV with V3 best throughputs (for comparison)")
    parser.add_argument("--n-steps",          type=int,   default=1000,
                        help="Gradient ascent steps")
    parser.add_argument("--lr",               type=float, default=0.01,
                        help="Adam learning rate")
    parser.add_argument("--penalty-lambda",   type=float, default=0.5,
                        help="Uncertainty penalty coefficient (lower = more conservative)")
    parser.add_argument("--reg-lambda",       type=float, default=1e-6,
                        help="L2 regularisation — penalises deviation from start solution")
    parser.add_argument("--n-evals",          type=int,   default=5)
    parser.add_argument("--n-workers",        type=int,   default=8)
    parser.add_argument("--seed",             type=int,   default=9999,
                        help="Evaluation seed (distinct from any training gen)")
    parser.add_argument("--num-agents",       type=int,   default=400)
    parser.add_argument("--simulation-steps", type=int,   default=1000)
    parser.add_argument("--no-simulate",      action="store_true",
                        help="Skip real simulator — show surrogate predictions only")
    parser.add_argument("--output-dir",       default="results/gradient_refinement")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load surrogate ---
    print(f"Loading surrogate from {args.checkpoint} ...")
    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    surrogate_state = ckpt.get("surrogate_state")
    if surrogate_state is None or surrogate_state.get("type") != "ensemble":
        raise ValueError("Checkpoint does not contain a fitted EnsembleSurrogate. "
                         "Make sure you're pointing at a V3 checkpoint.")
    surrogate = _load_surrogate_state(surrogate_state, input_dim=4074)
    print(f"  Loaded {surrogate.n_models}-model ensemble  (is_fitted={surrogate.is_fitted})")

    # --- Load best solution ---
    best_solution = np.load(args.best_solution)
    print(f"  Starting solution: shape={best_solution.shape}  "
          f"range=[{best_solution.min():.3f}, {best_solution.max():.3f}]")

    # --- Load V3 best throughput for comparison ---
    import pandas as pd
    v3_best_tp = pd.read_csv(args.v3_best_csv)["best_throughput"].max()
    print(f"  V3 best throughput (to beat): {v3_best_tp:.4f}\n")

    # --- Gradient ascent ---
    print(f"Running gradient ascent: {args.n_steps} steps | lr={args.lr} | "
          f"penalty_lambda={args.penalty_lambda}")
    refined, surr_before, surr_after, surr_std_after = run_gradient_ascent(
        surrogate=surrogate,
        start_solution=best_solution,
        n_steps=args.n_steps,
        lr=args.lr,
        penalty_lambda=args.penalty_lambda,
        reg_lambda=args.reg_lambda,
    )

    # Save refined solution regardless of simulation
    refined_path = out_dir / "refined_solution.npy"
    np.save(refined_path, refined)
    print(f"\n  Refined solution saved to {refined_path}")

    if args.no_simulate:
        print("\n--no-simulate: skipping real evaluation. Done.")
        return

    # --- Real evaluation ---
    print(f"\nEvaluating refined solution in simulator "
          f"(n_evals={args.n_evals}, n_workers={args.n_workers}) ...")
    t0 = time.time()
    mean_tp, all_tp = evaluate_batch(
        refined[np.newaxis, :],
        num_agents=args.num_agents,
        simulation_steps=args.simulation_steps,
        n_evals=args.n_evals,
        base_seed=args.seed,
        normalize=True,
        n_workers=args.n_workers,
    )
    elapsed_sim = time.time() - t0
    real_tp = float(mean_tp[0])

    # --- Results ---
    print(f"\n{'='*52}")
    print(f"  GRADIENT REFINEMENT RESULTS")
    print(f"{'='*52}")
    print(f"  Surrogate (start)  : {surr_before:.4f}")
    print(f"  Surrogate (refined): {surr_after:.4f}  ({surr_after - surr_before:+.4f})")
    print(f"  Surrogate std      : {surr_std_after:.4f}  (lower = more confident)")
    print(f"  Real throughput    : {real_tp:.4f}")
    print(f"  V3 best            : {v3_best_tp:.4f}")
    print(f"  Delta vs V3        : {real_tp - v3_best_tp:+.4f}")
    print(f"  Simulation time    : {elapsed_sim:.1f}s")
    print(f"{'='*52}")

    # Save a small result summary
    summary_path = out_dir / "results.txt"
    with open(summary_path, "w") as f:
        f.write(f"surrogate_before={surr_before:.4f}\n")
        f.write(f"surrogate_after={surr_after:.4f}\n")
        f.write(f"surrogate_std={surr_std_after:.4f}\n")
        f.write(f"real_throughput={real_tp:.4f}\n")
        f.write(f"v3_best={v3_best_tp:.4f}\n")
        f.write(f"delta={real_tp - v3_best_tp:+.4f}\n")
    print(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
