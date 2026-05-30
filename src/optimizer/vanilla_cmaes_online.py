"""Vanilla CMA-ES for online GGO (CNN policy optimization baseline).

Optimizes a 4271-dim CNN policy parameter vector θ using multi-emitter CMA-ES.
The CNN generates edge weights every `update_interval` timesteps from observed
traffic (Zang et al., AAAI 2025 — online GGO).

Key differences from vanilla_cmaes.py (offline GGO):
    - sol_size = 4271  (CNN params) vs 4074 (edge/wait weights)
    - sigma0 = 0.1, initial_mean = 0.0  (centred small weights for CNN)
    - n_evals = 2  (online sim is slower; 2 repeats is sufficient)
    - evaluate_online_batch instead of evaluate_batch

Usage:
    python -m src.optimizer.vanilla_cmaes_online --generations 100 --output results/online_baseline
    python -m src.optimizer.vanilla_cmaes_online --resume results/online_baseline
"""

import argparse
import pickle
import time
import numpy as np

try:
    import cma
except ImportError:
    cma = None

from pathlib import Path
from src.simulator.evaluate_online import evaluate_online_batch, get_n_params
from src.utils.data import RunLogger


CHECKPOINT_FILE = "checkpoint.pkl"


class CMAEmitter:
    """Single pycma CMAEvolutionStrategy wrapper (identical to offline version)."""

    def __init__(self, emitter_id, x0, sigma0, popsize, seed):
        self.emitter_id = emitter_id
        self.popsize = popsize
        self.x0 = x0.copy()
        self.sigma0 = sigma0
        self._seed = seed
        self.restarts = 0
        self.es = cma.CMAEvolutionStrategy(
            x0.tolist(), sigma0,
            {"popsize": popsize, "seed": seed, "verbose": -1},
        )

    def ask(self):
        return np.array(self.es.ask())

    def tell(self, solutions, fitnesses):
        self.es.tell(solutions.tolist(), fitnesses.tolist())

    def should_restart(self):
        return bool(self.es.stop())

    def restart(self, x0, sigma0, seed):
        self.restarts += 1
        self._seed = seed
        self.es = cma.CMAEvolutionStrategy(
            x0.tolist(), sigma0,
            {"popsize": self.popsize, "seed": seed, "verbose": -1},
        )

    def get_state(self):
        return {
            "emitter_id": self.emitter_id,
            "popsize": self.popsize,
            "x0": self.x0,
            "sigma0": self.sigma0,
            "seed": self._seed,
            "restarts": self.restarts,
            "es_pickle": self.es.pickle_dumps(),
        }

    @classmethod
    def from_state(cls, state):
        emitter = cls.__new__(cls)
        emitter.emitter_id = state["emitter_id"]
        emitter.popsize = state["popsize"]
        emitter.x0 = state["x0"]
        emitter.sigma0 = state["sigma0"]
        emitter._seed = state["seed"]
        emitter.restarts = state["restarts"]
        emitter.es = pickle.loads(state["es_pickle"])
        return emitter


def save_checkpoint(path, generation, emitters, best_solution, best_throughput,
                    rng_state, total_sims, cumulative_wallclock_s):
    state = {
        "generation": generation,
        "emitter_states": [e.get_state() for e in emitters],
        "best_solution": best_solution,
        "best_throughput": best_throughput,
        "rng_state": rng_state,
        "total_sims": total_sims,
        "cumulative_wallclock_s": cumulative_wallclock_s,
    }
    path = Path(path)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f)
    tmp_path.replace(path)


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
    )


def run_vanilla_cmaes_online(
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
    output_dir="results/online_baseline",
    resume=False,
    n_workers=4,
    chunk_size=20,
):
    """Run multi-emitter CMA-ES on the online GGO CNN policy.

    Args:
        generations: Total generations to run.
        n_emitters: Number of independent CMA-ES emitters.
        popsize_per_emitter: Population per emitter.
        n_evals: Stochastic simulation repeats per candidate.
        num_agents: Agents in simulation.
        simulation_steps: Total timesteps per evaluation.
        update_interval: Steps between CNN policy calls.
        sigma0: Initial CMA-ES step size.
        initial_mean: Initial mean (0.0 = centred CNN weights).
        seed: Random seed.
        output_dir: Directory for logs and checkpoints.
        resume: If True, resume from checkpoint.
        n_workers: Parallel workers inside each Docker call.
        chunk_size: Candidates per Docker call (limits peak memory).

    Returns:
        best_solution: np.ndarray (N_PARAMS,)
        best_throughput: float
    """
    if cma is None:
        raise ImportError("pycma is required: pip install cma")

    sol_size = get_n_params()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_FILE
    total_pop = n_emitters * popsize_per_emitter

    if resume and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_gen, emitters, best_solution, best_throughput, rng_state, \
            total_sims, cumulative_wallclock = load_checkpoint(checkpoint_path)
        start_gen += 1
        rng = np.random.default_rng()
        rng.bit_generator.state = rng_state
        print(f"  Resuming at generation {start_gen}, best={best_throughput:.4f}, "
              f"total_sims={total_sims}, wallclock={cumulative_wallclock:.0f}s")
        logger = RunLogger(output_dir, prefix="cmaes", n_evals=n_evals, resume=True)
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

    print(f"Online GGO vanilla CMA-ES | sol_size={sol_size}")
    print(f"Config: {generations} gens × {n_emitters} emitters × "
          f"{popsize_per_emitter} pop × {n_evals} evals")
    print(f"Total pop/gen: {total_pop} | budget: {generations * total_pop * n_evals} sims")

    try:
        for gen in range(start_gen, generations):
            t0 = time.time()

            all_solutions = []
            emitter_ids = []
            for emitter in emitters:
                sols = emitter.ask()
                all_solutions.append(sols)
                emitter_ids.extend([emitter.emitter_id] * len(sols))

            all_solutions = np.concatenate(all_solutions, axis=0)
            emitter_ids = np.array(emitter_ids, dtype=int)

            eval_seed = seed + gen * n_evals
            mean_throughputs, all_throughputs = evaluate_online_batch(
                all_solutions,
                num_agents=num_agents,
                simulation_steps=simulation_steps,
                update_interval=update_interval,
                n_evals=n_evals,
                base_seed=eval_seed,
                n_workers=n_workers,
                chunk_size=chunk_size,
            )
            total_sims += len(all_solutions) * n_evals

            pos = 0
            for emitter in emitters:
                end = pos + emitter.popsize
                emitter.tell(all_solutions[pos:end], -mean_throughputs[pos:end])
                pos = end

            gen_best_idx = np.argmax(mean_throughputs)
            gen_best_tp = mean_throughputs[gen_best_idx]
            if gen_best_tp > best_throughput:
                best_throughput = gen_best_tp
                best_solution = all_solutions[gen_best_idx].copy()

            elapsed = time.time() - t0
            cumulative_wallclock += elapsed

            restart_info = []
            for emitter in emitters:
                if emitter.should_restart():
                    restart_x0 = best_solution if best_solution is not None else \
                        np.full(sol_size, initial_mean)
                    noise = rng.normal(0, sigma0 * 0.5, sol_size)
                    new_seed = int(rng.integers(0, 2**31))
                    emitter.restart(restart_x0 + noise, sigma0, new_seed)
                    restart_info.append(emitter.emitter_id)

            logger.log_generation(
                gen, all_solutions, mean_throughputs, all_throughputs, emitter_ids
            )
            logger.log_best(gen, best_throughput, gen_best_tp,
                            gen_wallclock_s=elapsed,
                            cumulative_wallclock_s=cumulative_wallclock,
                            restarted_emitters=restart_info)

            if (gen + 1) % 5 == 0:
                logger.flush_solutions()

            save_checkpoint(
                checkpoint_path, gen, emitters, best_solution, best_throughput,
                rng.bit_generator.state, total_sims, cumulative_wallclock,
            )

            cum_h, cum_rem = divmod(int(cumulative_wallclock), 3600)
            cum_m = cum_rem // 60
            print(
                f"Gen {gen+1:3d}/{generations} | "
                f"best={best_throughput:.4f} | "
                f"gen_mean={mean_throughputs.mean():.4f} | "
                f"gen_best={gen_best_tp:.4f} | "
                f"sims={total_sims} | "
                f"time={elapsed:.1f}s | "
                f"total={cum_h}h{cum_m:02d}m"
            )

    finally:
        logger.close()

    h, rem = divmod(int(cumulative_wallclock), 3600)
    m = rem // 60
    print(f"\nOptimization complete.")
    print(f"Best throughput: {best_throughput:.4f}")
    print(f"Total simulations: {total_sims}")
    print(f"Total wallclock: {h}h{m:02d}m ({cumulative_wallclock:.0f}s)")

    if best_solution is not None:
        np.save(output_dir / "best_solution.npy", best_solution)

    return best_solution, best_throughput


def main():
    parser = argparse.ArgumentParser(
        description="Vanilla CMA-ES baseline for online GGO (CNN policy)"
    )
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--n-emitters", type=int, default=5)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--n-evals", type=int, default=2)
    parser.add_argument("--num-agents", type=int, default=400)
    parser.add_argument("--simulation-steps", type=int, default=1000)
    parser.add_argument("--update-interval", type=int, default=20)
    parser.add_argument("--sigma0", type=float, default=0.1)
    parser.add_argument("--initial-mean", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/online_baseline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=20)
    args = parser.parse_args()

    run_vanilla_cmaes_online(
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
    )


if __name__ == "__main__":
    main()
