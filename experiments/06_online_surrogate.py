"""Experiment 6: Surrogate-assisted CMA-ES for online GGO.

Uses ensemble MLP surrogate + UCB screening to reduce simulation calls when
optimising the 4271-dim CNN policy for online LMAPF (Zang et al. AAAI 2025).

Results saved to results/online_surrogate/

Usage:
    # Fresh run (100 generations)
    python -m experiments.06_online_surrogate

    # Resume from checkpoint
    python -m experiments.06_online_surrogate --resume

    # Smoke test
    python -m experiments.06_online_surrogate --generations 5
"""

import argparse
from src.optimizer.surrogate_cmaes_online import run_surrogate_cmaes_online


OUTPUT_DIR = "results/online_surrogate"


def main():
    parser = argparse.ArgumentParser(
        description="Surrogate-assisted CMA-ES for online GGO"
    )
    parser.add_argument("--generations",                type=int,   default=100)
    parser.add_argument("--n-emitters",                 type=int,   default=5)
    parser.add_argument("--popsize",                    type=int,   default=20)
    parser.add_argument("--n-evals",                    type=int,   default=2)
    parser.add_argument("--num-agents",                 type=int,   default=400)
    parser.add_argument("--simulation-steps",           type=int,   default=1000)
    parser.add_argument("--update-interval",            type=int,   default=20)
    parser.add_argument("--sigma0",                     type=float, default=0.1)
    parser.add_argument("--seed",                       type=int,   default=42)
    parser.add_argument("--output",                     type=str,   default=OUTPUT_DIR)
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
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
