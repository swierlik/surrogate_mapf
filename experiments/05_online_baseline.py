"""Experiment 5: Vanilla CMA-ES baseline for online GGO.

Optimizes a 4271-dim CNN policy (OnlineCNNPolicy) via multi-emitter CMA-ES.
The CNN produces edge weights from traffic observations every 20 timesteps
(online GGO, Zang et al. AAAI 2025).

Results saved to results/online_baseline/

Usage:
    # Fresh run (100 generations, ~4-5 hrs)
    python -m experiments.05_online_baseline

    # Resume from checkpoint
    python -m experiments.05_online_baseline --resume

    # Shorter smoke test
    python -m experiments.05_online_baseline --generations 5
"""

import argparse
from src.optimizer.vanilla_cmaes_online import run_vanilla_cmaes_online


OUTPUT_DIR = "results/online_baseline"


def main():
    parser = argparse.ArgumentParser(
        description="Vanilla CMA-ES baseline for online GGO"
    )
    parser.add_argument("--generations", type=int, default=100,
                        help="Number of CMA-ES generations (default: 100)")
    parser.add_argument("--n-emitters", type=int, default=5)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--n-evals", type=int, default=2,
                        help="Stochastic repeats per candidate (default: 2)")
    parser.add_argument("--num-agents", type=int, default=400)
    parser.add_argument("--simulation-steps", type=int, default=1000)
    parser.add_argument("--update-interval", type=int, default=20)
    parser.add_argument("--sigma0", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint in output dir")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=20,
                        help="Candidates per Docker call (limits peak memory)")
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
        output_dir=args.output,
        resume=args.resume,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
