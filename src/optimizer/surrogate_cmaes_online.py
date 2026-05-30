"""Surrogate-assisted CMA-ES for online GGO (CNN policy optimization).

Identical logic to surrogate_cmaes.py, adapted for the online GGO evaluator:
    - sol_size = 4271  (CNN policy params)
    - sigma0 = 0.1, initial_mean = 0.0
    - n_evals = 2  (online simulation is slower)
    - Uses evaluate_online_batch instead of evaluate_batch (no normalize flag)
    - chunk_size controls Docker batch memory

Imports all infrastructure (emitters, surrogate, logging) from the offline modules.

Usage:
    python -m src.optimizer.surrogate_cmaes_online --generations 100 --output results/online_surrogate
    python -m src.optimizer.surrogate_cmaes_online --resume results/online_surrogate
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from src.optimizer.vanilla_cmaes import CMAEmitter
from src.optimizer.surrogate_cmaes import (
    CHECKPOINT_FILE,
    SurrogateLogger,
    _save_surrogate_state,
    _load_surrogate_state,
    save_checkpoint,
    load_checkpoint,
    _estimate_sims,
)
from src.simulator.evaluate_online import evaluate_online_batch, get_n_params
from src.surrogate.mlp_model import EnsembleSurrogate
from src.utils.data import RunLogger, load_run_data
from src.utils.metrics import spearman_rho


def run_surrogate_cmaes_online(
    generations=100,
    n_emitters=5,
    popsize_per_emitter=20,
    n_evals=2,
    num_agents=400,
    simulation_steps=1000,
    update_interval=20,
    sigma0=0.1,
    initial_mean=0.0,
    seed=42,
    output_dir="results/online_surrogate",
    resume=False,
    n_workers=4,
    chunk_size=20,
    warmup_gens=10,
    screen_k=20,
    evolution_control_interval=10,
    n_ensemble=5,
    bootstrap_frac=0.8,
    ucb_lambda=1.0,
):
    """Run surrogate-assisted multi-emitter CMA-ES on the online GGO CNN policy.

    Args:
        warmup_gens: Generations of full evaluation before activating surrogate.
        screen_k: Candidates to simulate per surrogate generation.
        evolution_control_interval: Every N gens, do full eval and retrain surrogate.
        n_ensemble: Ensemble MLP models.
        bootstrap_frac: Bootstrap fraction per model.
        ucb_lambda: UCB exploration weight (score = mean + lambda * std).
        chunk_size: Candidates per Docker call (limits peak memory).

    Returns:
        best_solution, best_throughput
    """
    try:
        import cma
    except ImportError:
        raise ImportError("pycma is required: pip install cma")

    sol_size = get_n_params()
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

    print(f"Online GGO surrogate CMA-ES | sol_size={sol_size}")
    print(f"Config: {generations} gens | warmup={warmup_gens} | "
          f"screen_k={screen_k}/{total_pop} | control_every={evolution_control_interval} | "
          f"ensemble={n_ensemble} | ucb_lambda={ucb_lambda}")
    print(f"Vanilla budget: {generations * total_pop * n_evals} sims | "
          f"Expected surrogate budget: ~{_estimate_sims(generations, warmup_gens, total_pop, n_evals, screen_k, evolution_control_interval)} sims")

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

            # 3. Surrogate predictions
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
                mode = "warmup" if is_warmup else "control"
                mean_tp, all_tp = evaluate_online_batch(
                    all_solutions,
                    num_agents=num_agents,
                    simulation_steps=simulation_steps,
                    update_interval=update_interval,
                    n_evals=n_evals,
                    base_seed=eval_seed,
                    n_workers=n_workers,
                    chunk_size=chunk_size,
                )
                total_sims += total_pop * n_evals
                n_simulated = total_pop
                full_mean_tp = mean_tp

                rho = None
                if surr_mean is not None:
                    rho = spearman_rho(mean_tp, surr_mean)

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
                mode = "surrogate"
                ucb_scores = surr_mean + ucb_lambda * surr_std
                top_idx = np.argsort(-ucb_scores)[:screen_k]
                selected_std_val = float(surr_std[top_idx].mean())

                eval_solutions   = all_solutions[top_idx]
                eval_emitter_ids = emitter_ids[top_idx]

                mean_tp, all_tp = evaluate_online_batch(
                    eval_solutions,
                    num_agents=num_agents,
                    simulation_steps=simulation_steps,
                    update_interval=update_interval,
                    n_evals=n_evals,
                    base_seed=eval_seed,
                    n_workers=n_workers,
                    chunk_size=chunk_size,
                )
                total_sims += screen_k * n_evals
                sims_saved += (total_pop - screen_k) * n_evals
                n_simulated = screen_k

                full_mean_tp = surr_mean.copy()
                full_mean_tp[top_idx] = mean_tp

                acc_X_list.append(eval_solutions)
                acc_y_list.append(mean_tp)
                t_train = time.time()
                surrogate.fine_tune(eval_solutions, mean_tp, epochs=10)
                train_time = time.time() - t_train

                logger.log_generation(gen, eval_solutions, mean_tp, all_tp, eval_emitter_ids)

                rho = None

            # 4. Tell emitters (full batch)
            pos = 0
            for emitter in emitters:
                end = pos + emitter.popsize
                emitter.tell(all_solutions[pos:end], -full_mean_tp[pos:end])
                pos = end

            # 5. Track best (real evaluations only)
            if use_surrogate:
                real_mean_tp   = mean_tp
                real_solutions = eval_solutions
            else:
                real_mean_tp   = full_mean_tp
                real_solutions = all_solutions

            gen_best_idx = np.argmax(real_mean_tp)
            gen_best_tp  = real_mean_tp[gen_best_idx]
            if gen_best_tp > best_throughput:
                best_throughput = gen_best_tp
                best_solution   = real_solutions[gen_best_idx].copy()

            elapsed = time.time() - t0
            cumulative_wallclock += elapsed

            # 6. Emitter restarts (before logging so it lands in the same row)
            restart_info = []
            for emitter in emitters:
                if emitter.should_restart():
                    restart_x0 = best_solution if best_solution is not None else \
                        np.full(sol_size, initial_mean)
                    noise = rng.normal(0, sigma0 * 0.5, sol_size)
                    new_seed = int(rng.integers(0, 2**31))
                    emitter.restart(restart_x0 + noise, sigma0, new_seed)
                    restart_info.append(emitter.emitter_id)

            # 7. Log
            logger.log_best(gen, best_throughput, gen_best_tp,
                            gen_wallclock_s=elapsed,
                            cumulative_wallclock_s=cumulative_wallclock,
                            restarted_emitters=restart_info)
            surr_logger.log(gen, mode, n_simulated, rho,
                            mean_std_val, selected_std_val,
                            train_time, sims_saved)

            if (gen + 1) % 5 == 0:
                logger.flush_solutions()

            # 8. Checkpoint (gc first to free training arrays before pickling emitters)
            gc.collect()
            save_checkpoint(
                checkpoint_path, gen, emitters, best_solution, best_throughput,
                rng.bit_generator.state, total_sims, cumulative_wallclock, surrogate,
            )

            # 9. Print
            cum_h, cum_rem = divmod(int(cumulative_wallclock), 3600)
            cum_m = cum_rem // 60
            restart_str = f" | restarts={restart_info}" if restart_info else ""
            rho_str     = f" | rho={rho:.3f}"           if rho is not None else ""
            std_str     = f" | std={mean_std_val:.3f}"   if mean_std_val is not None else ""
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

    h, rem = divmod(int(cumulative_wallclock), 3600)
    m = rem // 60
    print(f"\nOptimization complete.")
    print(f"Best throughput: {best_throughput:.4f}")
    print(f"Total simulations: {total_sims} (saved {sims_saved} vs vanilla)")
    print(f"Total wallclock: {h}h{m:02d}m ({cumulative_wallclock:.0f}s)")

    if best_solution is not None:
        np.save(output_dir / "best_solution.npy", best_solution)

    return best_solution, best_throughput


def main():
    parser = argparse.ArgumentParser(
        description="Surrogate-assisted CMA-ES for online GGO (CNN policy)"
    )
    parser.add_argument("--generations",                type=int,   default=100)
    parser.add_argument("--n-emitters",                 type=int,   default=5)
    parser.add_argument("--popsize",                    type=int,   default=20)
    parser.add_argument("--n-evals",                    type=int,   default=2)
    parser.add_argument("--num-agents",                 type=int,   default=400)
    parser.add_argument("--simulation-steps",           type=int,   default=1000)
    parser.add_argument("--update-interval",            type=int,   default=20)
    parser.add_argument("--sigma0",                     type=float, default=0.1)
    parser.add_argument("--initial-mean",               type=float, default=0.0)
    parser.add_argument("--seed",                       type=int,   default=42)
    parser.add_argument("--output",                     type=str,   default="results/online_surrogate")
    parser.add_argument("--resume",                     action="store_true")
    parser.add_argument("--n-workers",                  type=int,   default=4)
    parser.add_argument("--chunk-size",                 type=int,   default=20)
    parser.add_argument("--warmup-gens",                type=int,   default=10)
    parser.add_argument("--screen-k",                   type=int,   default=20)
    parser.add_argument("--evolution-control-interval", type=int,   default=10)
    parser.add_argument("--n-ensemble",                 type=int,   default=5)
    parser.add_argument("--bootstrap-frac",             type=float, default=0.8)
    parser.add_argument("--ucb-lambda",                 type=float, default=1.0)
    args = parser.parse_args()

    run_surrogate_cmaes_online(
        generations=args.generations,
        n_emitters=args.n_emitters,
        popsize_per_emitter=args.popsize,
        n_evals=args.n_evals,
        num_agents=args.num_agents,
        simulation_steps=args.simulation_steps,
        update_interval=args.update_interval,
        sigma0=args.sigma0,
        initial_mean=args.initial_mean,
        seed=args.seed,
        output_dir=args.output,
        resume=args.resume,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        warmup_gens=args.warmup_gens,
        screen_k=args.screen_k,
        evolution_control_interval=args.evolution_control_interval,
        n_ensemble=args.n_ensemble,
        bootstrap_frac=args.bootstrap_frac,
        ucb_lambda=args.ucb_lambda,
    )


if __name__ == "__main__":
    main()
