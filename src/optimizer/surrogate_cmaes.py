"""Surrogate-assisted CMA-ES optimizer.

Extends the vanilla multi-emitter CMA-ES with an ensemble MLP surrogate that
pre-screens candidates before expensive simulation using UCB acquisition.

Three modes per generation:
  - Warmup  (gen < warmup_gens):       Full eval all 100, build training data
  - Control (gen % interval == 0):     Full eval all 100, full surrogate retrain
  - Surrogate (all other gens):        Eval top screen_k by UCB score, fine-tune

UCB screening: score = mean_pred + ucb_lambda * std_pred
  High uncertainty (e.g. post-restart regions) automatically gets evaluated
  regardless of predicted rank, correcting for surrogate blind spots.

Unevaluated candidates receive surrogate mean predictions as placeholder
fitnesses when telling emitters, so CMA-ES always gets a complete update.

Usage:
    python -m src.optimizer.surrogate_cmaes --generations 300 --output results/surrogate_v3
    python -m src.optimizer.surrogate_cmaes --resume results/surrogate_v3
"""

import argparse
import csv
import pickle
import time
from pathlib import Path

import numpy as np

from src.optimizer.vanilla_cmaes import CMAEmitter
from src.simulator.evaluate import evaluate_batch, get_sol_size
from src.surrogate.mlp_model import EnsembleSurrogate, MLPSurrogate, MLPThroughputModel
from src.utils.data import RunLogger, load_run_data
from src.utils.metrics import spearman_rho


CHECKPOINT_FILE = "checkpoint.pkl"


# ---------------------------------------------------------------------------
# Surrogate serialization
# ---------------------------------------------------------------------------

def _save_surrogate_state(surrogate):
    """Serialize surrogate to a picklable dict. Handles both MLP and Ensemble."""
    if not surrogate.is_fitted:
        return None

    if isinstance(surrogate, EnsembleSurrogate):
        return {
            "type": "ensemble",
            "n_models": surrogate.n_models,
            "bootstrap_frac": surrogate.bootstrap_frac,
            "model_states": [
                {
                    "state_dict": {k: v.cpu().clone()
                                   for k, v in m.model.state_dict().items()},
                    "X_mean": m._X_mean,
                    "X_std":  m._X_std,
                    "y_mean": m._y_mean,
                    "y_std":  m._y_std,
                }
                for m in surrogate.models
            ],
        }
    else:  # MLPSurrogate (backward compat for V1/V2 checkpoints)
        return {
            "type": "mlp",
            "state_dict": {k: v.cpu().clone()
                           for k, v in surrogate.model.state_dict().items()},
            "X_mean": surrogate._X_mean,
            "X_std":  surrogate._X_std,
            "y_mean": surrogate._y_mean,
            "y_std":  surrogate._y_std,
            "is_fitted": True,
        }


def _load_surrogate_state(state, input_dim=4074, **kwargs):
    """Restore surrogate from serialized dict."""
    stype = state.get("type", "mlp")

    if stype == "ensemble":
        s = EnsembleSurrogate(
            n_models=state["n_models"],
            bootstrap_frac=state["bootstrap_frac"],
            **kwargs,
        )
        for model, ms in zip(s.models, state["model_states"]):
            model.model = MLPThroughputModel(input_dim=input_dim).to(model.device)
            model.model.load_state_dict(ms["state_dict"])
            model.model.to(model.device)
            model.model.eval()
            model._X_mean = ms["X_mean"]
            model._X_std  = ms["X_std"]
            model._y_mean = ms["y_mean"]
            model._y_std  = ms["y_std"]
            model.is_fitted = True
        s.is_fitted = True
        return s
    else:  # "mlp"
        s = MLPSurrogate(**kwargs)
        s.model = MLPThroughputModel(input_dim=input_dim).to(s.device)
        s.model.load_state_dict(state["state_dict"])
        s.model.to(s.device)
        s.model.eval()
        s._X_mean = state["X_mean"]
        s._X_std  = state["X_std"]
        s._y_mean = state["y_mean"]
        s._y_std  = state["y_std"]
        s.is_fitted = True
        return s


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(path, generation, emitters, best_solution, best_throughput,
                    rng_state, total_sims, cumulative_wallclock_s, surrogate):
    state = {
        "generation": generation,
        "emitter_states": [e.get_state() for e in emitters],
        "best_solution": best_solution,
        "best_throughput": best_throughput,
        "rng_state": rng_state,
        "total_sims": total_sims,
        "cumulative_wallclock_s": cumulative_wallclock_s,
        "surrogate_state": _save_surrogate_state(surrogate),
    }
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    tmp.replace(path)


def load_checkpoint(path):
    with open(path, "rb") as f:
        state = pickle.load(f)
    emitters = [CMAEmitter.from_state(s) for s in state["emitter_states"]]
    return (
        state["generation"],
        emitters,
        state["best_solution"],
        state["best_throughput"],
        state["rng_state"],
        state["total_sims"],
        state.get("cumulative_wallclock_s", 0.0),
        state.get("surrogate_state"),
    )


# ---------------------------------------------------------------------------
# Surrogate logger (per-generation metadata)
# ---------------------------------------------------------------------------

class SurrogateLogger:
    """Writes per-generation surrogate metadata to surrogate_log.csv."""

    HEADER = ["generation", "mode", "n_simulated", "surrogate_rho",
              "mean_std", "selected_std", "train_time_s", "total_sims_saved"]

    def __init__(self, log_dir, resume=False):
        self.path = Path(log_dir) / "surrogate_log.csv"
        if resume and self.path.exists():
            self._f = open(self.path, "a", newline="")
            self._w = csv.writer(self._f)
        else:
            self._f = open(self.path, "w", newline="")
            self._w = csv.writer(self._f)
            self._w.writerow(self.HEADER)

    def log(self, generation, mode, n_simulated, surrogate_rho,
            mean_std, selected_std, train_time_s, total_sims_saved):
        rho_str  = f"{surrogate_rho:.4f}" if surrogate_rho  is not None else ""
        mstd_str = f"{mean_std:.4f}"      if mean_std       is not None else ""
        sstd_str = f"{selected_std:.4f}"  if selected_std   is not None else ""
        self._w.writerow([generation, mode, n_simulated, rho_str,
                          mstd_str, sstd_str,
                          f"{train_time_s:.2f}", total_sims_saved])
        self._f.flush()

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_surrogate_cmaes(
    generations=300,
    n_emitters=5,
    popsize_per_emitter=20,
    n_evals=5,
    num_agents=400,
    simulation_steps=1000,
    sigma0=5.0,
    initial_mean=5.0,
    seed=42,
    output_dir="results/surrogate_v3",
    resume=False,
    n_workers=8,
    warmup_gens=20,
    screen_k=20,
    evolution_control_interval=10,
    n_ensemble=5,
    bootstrap_frac=0.8,
    ucb_lambda=1.0,
):
    """Run surrogate-assisted multi-emitter CMA-ES with ensemble UCB screening.

    Args:
        warmup_gens: Generations of full evaluation before activating surrogate.
        screen_k: Candidates to simulate per generation (after warmup).
        evolution_control_interval: Every N gens after warmup, do a full eval
            and retrain the surrogate from scratch to correct drift.
        n_ensemble: Number of MLP models in the ensemble.
        bootstrap_frac: Fraction of data each ensemble model trains on.
        ucb_lambda: Exploration weight in UCB = mean + ucb_lambda * std.
            Higher values favour uncertain (unexplored) candidates.

    Returns:
        best_solution, best_throughput
    """
    try:
        import cma
    except ImportError:
        raise ImportError("pycma is required: pip install cma")

    sol_size = get_sol_size()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_FILE
    total_pop = n_emitters * popsize_per_emitter

    surrogate = EnsembleSurrogate(
        n_models=n_ensemble,
        bootstrap_frac=bootstrap_frac,
        max_epochs=100,
        patience=10,
    )
    surrogate_ready = False
    sims_saved = 0

    acc_X_list = []
    acc_y_list = []

    # --- Initialize or resume ---
    if resume and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        (start_gen, emitters, best_solution, best_throughput, rng_state,
         total_sims, cumulative_wallclock, surrogate_state) = load_checkpoint(checkpoint_path)
        start_gen += 1
        rng = np.random.default_rng()
        rng.bit_generator.state = rng_state

        if surrogate_state is not None:
            surrogate = _load_surrogate_state(surrogate_state, input_dim=sol_size)
            surrogate_ready = True

        try:
            acc_sols, acc_tp, _ = load_run_data(output_dir)
            acc_X_list = [acc_sols]
            acc_y_list = [acc_tp]
            print(f"  Loaded {len(acc_sols)} accumulated training samples from disk")
        except FileNotFoundError:
            pass

        print(f"  Resuming at generation {start_gen}, best={best_throughput:.4f}, "
              f"total_sims={total_sims}")
        logger = RunLogger(output_dir, prefix="cmaes", n_evals=n_evals, resume=True)
        surr_logger = SurrogateLogger(output_dir, resume=True)
    else:
        start_gen = 0
        best_throughput = -np.inf
        best_solution = None
        total_sims = 0
        cumulative_wallclock = 0.0
        rng = np.random.default_rng(seed)

        x0 = np.full(sol_size, initial_mean)
        emitter_seeds = rng.integers(0, 2**31, size=n_emitters)
        emitters = [
            CMAEmitter(i, x0, sigma0, popsize_per_emitter, int(emitter_seeds[i]))
            for i in range(n_emitters)
        ]
        logger = RunLogger(output_dir, prefix="cmaes", n_evals=n_evals)
        surr_logger = SurrogateLogger(output_dir, resume=False)

    print(f"Solution dimensionality: {sol_size}")
    print(f"Config: {generations} gens | warmup={warmup_gens} | "
          f"screen_k={screen_k}/{total_pop} | control_every={evolution_control_interval} | "
          f"ensemble={n_ensemble} | ucb_lambda={ucb_lambda}")
    print(f"Vanilla budget: {generations * total_pop * n_evals} sims | "
          f"Expected surrogate budget: ~{_estimate_sims(generations, warmup_gens, total_pop, n_evals, screen_k, evolution_control_interval)} sims")

    # --- Main loop ---
    try:
        for gen in range(start_gen, generations):
            t0 = time.time()

            # 1. Ask all emitters
            all_solutions = []
            emitter_ids = []
            for emitter in emitters:
                sols = emitter.ask()
                all_solutions.append(sols)
                emitter_ids.extend([emitter.emitter_id] * len(sols))

            all_solutions = np.concatenate(all_solutions, axis=0)
            emitter_ids = np.array(emitter_ids, dtype=int)

            # 2. Decide mode
            is_warmup = gen < warmup_gens
            is_control = (not is_warmup) and (gen % evolution_control_interval == 0)
            use_surrogate = surrogate_ready and not is_warmup and not is_control

            # 3. Get ensemble predictions (mean + uncertainty) for all candidates
            surr_mean = None
            surr_std  = None
            mean_std_val = None

            if surrogate_ready:
                surr_mean, surr_std = surrogate.predict_with_uncertainty(all_solutions)
                mean_std_val = float(surr_std.mean())

            eval_seed = seed + gen * n_evals
            train_time = 0.0
            selected_std_val = None

            if is_warmup or is_control:
                # --- Full evaluation ---
                mode = "warmup" if is_warmup else "control"
                mean_tp, all_tp = evaluate_batch(
                    all_solutions,
                    num_agents=num_agents,
                    simulation_steps=simulation_steps,
                    n_evals=n_evals,
                    base_seed=eval_seed,
                    normalize=True,
                    n_workers=n_workers,
                )
                total_sims += total_pop * n_evals
                n_simulated = total_pop
                full_mean_tp = mean_tp

                # Rho BEFORE retraining = honest accuracy estimate
                rho = None
                if surr_mean is not None:
                    rho = spearman_rho(mean_tp, surr_mean)

                # Accumulate and retrain ensemble on all real data
                acc_X_list.append(all_solutions)
                acc_y_list.append(mean_tp)
                t_train = time.time()
                X_acc = np.concatenate(acc_X_list, axis=0)
                y_acc = np.concatenate(acc_y_list, axis=0)
                surrogate.fit(X_acc, y_acc)
                train_time = time.time() - t_train
                surrogate_ready = True

                logger.log_generation(gen, all_solutions, mean_tp, all_tp, emitter_ids)

            else:
                # --- Surrogate-assisted with UCB screening ---
                mode = "surrogate"
                ucb_scores = surr_mean + ucb_lambda * surr_std
                top_idx = np.argsort(-ucb_scores)[:screen_k]
                selected_std_val = float(surr_std[top_idx].mean())

                eval_solutions   = all_solutions[top_idx]
                eval_emitter_ids = emitter_ids[top_idx]

                mean_tp, all_tp = evaluate_batch(
                    eval_solutions,
                    num_agents=num_agents,
                    simulation_steps=simulation_steps,
                    n_evals=n_evals,
                    base_seed=eval_seed,
                    normalize=True,
                    n_workers=n_workers,
                )
                total_sims += screen_k * n_evals
                sims_saved += (total_pop - screen_k) * n_evals
                n_simulated = screen_k

                # Placeholder fitnesses: real for top-k, surrogate mean for rest
                full_mean_tp = surr_mean.copy()
                full_mean_tp[top_idx] = mean_tp

                acc_X_list.append(eval_solutions)
                acc_y_list.append(mean_tp)
                t_train = time.time()
                surrogate.fine_tune(eval_solutions, mean_tp, epochs=10)
                train_time = time.time() - t_train

                logger.log_generation(gen, eval_solutions, mean_tp, all_tp, eval_emitter_ids)

                rho = None

            # 4. Tell each emitter (always full batch)
            pos = 0
            for emitter in emitters:
                end = pos + emitter.popsize
                emitter.tell(
                    all_solutions[pos:end],
                    -full_mean_tp[pos:end],
                )
                pos = end

            # 5. Track best (only from real evaluations)
            if use_surrogate:
                real_mean_tp  = mean_tp
                real_solutions = eval_solutions
            else:
                real_mean_tp  = full_mean_tp
                real_solutions = all_solutions

            gen_best_idx = np.argmax(real_mean_tp)
            gen_best_tp  = real_mean_tp[gen_best_idx]
            if gen_best_tp > best_throughput:
                best_throughput = gen_best_tp
                best_solution   = real_solutions[gen_best_idx].copy()

            elapsed = time.time() - t0
            cumulative_wallclock += elapsed

            # 6. Log
            logger.log_best(gen, best_throughput, gen_best_tp,
                            gen_wallclock_s=elapsed,
                            cumulative_wallclock_s=cumulative_wallclock)
            surr_logger.log(gen, mode, n_simulated, rho,
                            mean_std_val, selected_std_val,
                            train_time, sims_saved)

            if (gen + 1) % 5 == 0:
                logger.flush_solutions()

            # 7. Emitter restarts
            restart_info = []
            for emitter in emitters:
                if emitter.should_restart():
                    restart_x0 = best_solution if best_solution is not None else \
                        np.full(sol_size, initial_mean)
                    noise = rng.normal(0, sigma0 * 0.5, sol_size)
                    new_seed = int(rng.integers(0, 2**31))
                    emitter.restart(restart_x0 + noise, sigma0, new_seed)
                    restart_info.append(emitter.emitter_id)

            # 8. Checkpoint
            save_checkpoint(
                checkpoint_path, gen, emitters, best_solution, best_throughput,
                rng.bit_generator.state, total_sims, cumulative_wallclock, surrogate,
            )

            # 9. Print progress
            cum_h, cum_m = divmod(int(cumulative_wallclock), 3600)
            cum_m = cum_m // 60
            restart_str = f" | restarts={restart_info}" if restart_info else ""
            rho_str  = f" | rho={rho:.3f}"               if rho          is not None else ""
            std_str  = f" | std={mean_std_val:.3f}"       if mean_std_val is not None else ""
            print(
                f"Gen {gen+1:3d}/{generations} [{mode:9s}] | "
                f"best={best_throughput:.4f} | "
                f"gen_best={gen_best_tp:.4f} | "
                f"sims={total_sims} (saved={sims_saved}) | "
                f"time={elapsed:.1f}s | "
                f"total={cum_h}h{cum_m:02d}m"
                f"{rho_str}{std_str}{restart_str}"
            )

    finally:
        logger.close()
        surr_logger.close()

    h, m = divmod(int(cumulative_wallclock), 3600)
    m = m // 60
    print(f"\nOptimization complete.")
    print(f"Best throughput: {best_throughput:.4f}")
    print(f"Total simulations: {total_sims} (saved {sims_saved} vs vanilla)")
    print(f"Total wallclock: {h}h{m:02d}m ({cumulative_wallclock:.0f}s)")

    if best_solution is not None:
        np.save(output_dir / "best_solution.npy", best_solution)

    return best_solution, best_throughput


def _estimate_sims(generations, warmup_gens, total_pop, n_evals, screen_k, ctrl_interval):
    warmup = warmup_gens * total_pop * n_evals
    remaining = generations - warmup_gens
    control_gens = remaining // ctrl_interval
    surrogate_gens = remaining - control_gens
    return warmup + control_gens * total_pop * n_evals + surrogate_gens * screen_k * n_evals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Surrogate-assisted CMA-ES (ensemble MLP + UCB screening)")
    parser.add_argument("--generations",                type=int,   default=300)
    parser.add_argument("--n-emitters",                 type=int,   default=5)
    parser.add_argument("--popsize",                    type=int,   default=20)
    parser.add_argument("--n-evals",                    type=int,   default=5)
    parser.add_argument("--num-agents",                 type=int,   default=400)
    parser.add_argument("--simulation-steps",           type=int,   default=1000)
    parser.add_argument("--sigma0",                     type=float, default=5.0)
    parser.add_argument("--initial-mean",               type=float, default=5.0)
    parser.add_argument("--seed",                       type=int,   default=42)
    parser.add_argument("--output",                     type=str,   default="results/surrogate_v3")
    parser.add_argument("--resume",                     action="store_true")
    parser.add_argument("--n-workers",                  type=int,   default=8)
    parser.add_argument("--warmup-gens",                type=int,   default=20)
    parser.add_argument("--screen-k",                   type=int,   default=20)
    parser.add_argument("--evolution-control-interval", type=int,   default=10)
    parser.add_argument("--n-ensemble",                 type=int,   default=5)
    parser.add_argument("--bootstrap-frac",             type=float, default=0.8)
    parser.add_argument("--ucb-lambda",                 type=float, default=1.0)
    args = parser.parse_args()

    run_surrogate_cmaes(
        generations=args.generations,
        n_emitters=args.n_emitters,
        popsize_per_emitter=args.popsize,
        n_evals=args.n_evals,
        num_agents=args.num_agents,
        simulation_steps=args.simulation_steps,
        sigma0=args.sigma0,
        initial_mean=args.initial_mean,
        seed=args.seed,
        output_dir=args.output,
        resume=args.resume,
        n_workers=args.n_workers,
        warmup_gens=args.warmup_gens,
        screen_k=args.screen_k,
        evolution_control_interval=args.evolution_control_interval,
        n_ensemble=args.n_ensemble,
        bootstrap_frac=args.bootstrap_frac,
        ucb_lambda=args.ucb_lambda,
    )


if __name__ == "__main__":
    main()
