"""Experiment 1: Static surrogate feasibility evaluation.

Loads 30k (solution, throughput) pairs from CMA-ES baseline run.
Trains XGBoost and CNN surrogates and reports:
  - 5-fold CV Spearman rho, MSE, top-20% precision
  - Learning curves at sizes 500, 1000, 2000, 5000, 10000, 20000
  - Temporal split: train on gens 0-199, test on gens 200-299

Usage:
    python -m experiments.01_feasibility --data-dir results/baseline
    python -m experiments.01_feasibility --data-dir results/baseline --skip-cnn
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.data import load_run_data
from src.surrogate.xgboost_model import XGBoostSurrogate
from src.surrogate.cnn_model import CNNSurrogate
from src.surrogate.training import (
    cross_validate, learning_curve, temporal_split_eval,
)


def load_generations(log_dir, prefix="cmaes"):
    """Extract per-sample generation numbers from the CSV log."""
    csv_path = Path(log_dir) / f"{prefix}_log.csv"
    df = pd.read_csv(csv_path)
    return df["generation"].values


def _convert_for_json(obj):
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_for_json(v) for v in obj]
    return obj


def run_experiment(data_dir, output_dir, skip_cnn=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("Loading data...")
    solutions, mean_tp, all_tp = load_run_data(data_dir)
    generations = load_generations(data_dir)
    print(f"  Solutions: {solutions.shape}")
    print(f"  Throughput range: [{mean_tp.min():.4f}, {mean_tp.max():.4f}]")
    print(f"  Generations: {generations.min()} to {generations.max()}")

    # Sanity checks
    assert not np.any(np.isnan(solutions)), "NaN in solutions!"
    assert not np.any(np.isnan(mean_tp)), "NaN in throughputs!"
    assert len(solutions) == len(mean_tp) == len(generations)

    all_results = {}

    # --- Model definitions ---
    models = {"xgboost": lambda: XGBoostSurrogate()}
    if not skip_cnn:
        models["cnn"] = lambda: CNNSurrogate(max_epochs=100, patience=10)

    for model_name, factory in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*60}")

        model_results = {}

        # 1. Cross-validation
        print(f"\n--- 5-fold Cross-Validation ---")
        cv = cross_validate(factory, solutions, mean_tp, n_folds=5)
        model_results["cv"] = {
            "mean": cv["mean_metrics"],
            "std": cv["std_metrics"],
            "folds": cv["fold_metrics"],
        }
        print(f"  MEAN: rho={cv['mean_metrics']['spearman_rho']:.4f} "
              f"(+/- {cv['std_metrics']['spearman_rho']:.4f})")

        # 2. Learning curve
        print(f"\n--- Learning Curve ---")
        lc = learning_curve(
            factory, solutions, mean_tp,
            train_sizes=[500, 1000, 2000, 5000, 10000, 20000],
        )
        model_results["learning_curve"] = lc

        # 3. Temporal split
        print(f"\n--- Temporal Split (gens 0-199 → 200-299) ---")
        temp = temporal_split_eval(
            factory, solutions, mean_tp, generations,
            train_gen_range=(0, 200), test_gen_range=(200, 300),
        )
        model_results["temporal_split"] = temp

        all_results[model_name] = model_results

    # --- Save results ---
    results_path = output_dir / "feasibility_results.json"
    with open(results_path, "w") as f:
        json.dump(_convert_for_json(all_results), f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Decision gate ---
    print(f"\n{'='*60}")
    print("DECISION GATE")
    print(f"{'='*60}")
    for model_name, results in all_results.items():
        rho_cv = results["cv"]["mean"]["spearman_rho"]
        lc = results["learning_curve"]
        rho_500 = next(
            (r["spearman_rho"] for r in lc if r["train_size"] == 500), None)
        rho_temporal = results["temporal_split"]["spearman_rho"]

        rho_500_str = f"{rho_500:.4f}" if rho_500 is not None else "N/A"
        print(f"  {model_name:>10s}: CV rho={rho_cv:.4f}, "
              f"500-sample rho={rho_500_str}, "
              f"temporal rho={rho_temporal:.4f}")

    best_rho = max(
        r["cv"]["mean"]["spearman_rho"] for r in all_results.values())
    if best_rho > 0.4:
        print(f"\n  PASS: Best Spearman rho = {best_rho:.4f} > 0.4")
        print(f"  --> Proceed to surrogate-assisted CMA-ES (Phase 2)")
    else:
        print(f"\n  FAIL: Best Spearman rho = {best_rho:.4f} <= 0.4")
        print(f"  --> Revisit feature engineering or architecture")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: Static surrogate feasibility")
    parser.add_argument("--data-dir", type=str, default="results/baseline")
    parser.add_argument("--output-dir", type=str,
                        default="results/feasibility")
    parser.add_argument("--skip-cnn", action="store_true",
                        help="Skip CNN (faster, for debugging)")
    args = parser.parse_args()

    run_experiment(args.data_dir, args.output_dir, skip_cnn=args.skip_cnn)


if __name__ == "__main__":
    main()
