"""Experiment 1c: Top-K ranking analysis.

Trains XGBoost on gens 0-199, predicts gen 200 (temporal, no data leakage).
For each k in [1, 2, 3, 5, 10, 20]:
  - How many of the true top-k are found in the predicted top-k?
  - Where does the worst true top-k solution land in the surrogate ranking?

Also reports where the true #1 solution ends up in the surrogate ranking.

Key question: even if Spearman rho isn't perfect, does the surrogate
reliably identify the very best candidates? This determines whether
pre-screening (keeping top-20 of 100) is safe.

Results recorded in findings.md Session 2.

Usage:
    python -m experiments.01c_topk_analysis
    python -m experiments.01c_topk_analysis --train-cutoff 150 --test-gen 150
"""

import argparse

import numpy as np
import pandas as pd

from src.utils.data import load_run_data
from src.surrogate.xgboost_model import XGBoostSurrogate


def run_topk_analysis(solutions, mean_tp, generations, train_cutoff, test_gen):
    train_mask = generations < train_cutoff
    test_mask = generations == test_gen

    print(f"\nTrain: gens [0, {train_cutoff}) — {train_mask.sum()} samples")
    print(f"Test:  gen {test_gen} — {test_mask.sum()} samples")

    model = XGBoostSurrogate()
    model.fit(solutions[train_mask], mean_tp[train_mask])
    y_pred = model.predict(solutions[test_mask])
    y_true = mean_tp[test_mask]

    true_rank = np.argsort(np.argsort(-y_true))   # rank 0 = best
    pred_rank = np.argsort(np.argsort(-y_pred))

    print(f"\n=== Top-K Analysis: train gens 0-{train_cutoff-1}, test gen {test_gen} (N={len(y_true)}) ===")
    for k in [1, 2, 3, 5, 10, 20]:
        true_top_k = set(np.argsort(-y_true)[:k])
        pred_top_k = set(np.argsort(-y_pred)[:k])
        overlap = len(true_top_k & pred_top_k)
        worst = max(pred_rank[i] for i in true_top_k)
        median = int(np.median([pred_rank[i] for i in true_top_k]))
        print(f"Top-{k:>2d}: {overlap}/{k} in predicted top-{k} | "
              f"worst pred rank: {worst:>3d} | median pred rank: {median:>3d}")

    best_idx = np.argmax(y_true)
    print(f"\nTrue #1: tp={y_true[best_idx]:.4f}, "
          f"predicted={y_pred[best_idx]:.4f}, "
          f"surrogate rank={pred_rank[best_idx]+1}/{len(y_true)}")

    pred_best_idx = np.argmax(y_pred)
    print(f"Surrogate #1: predicted={y_pred[pred_best_idx]:.4f}, "
          f"actual={y_true[pred_best_idx]:.4f}, "
          f"true rank={true_rank[pred_best_idx]+1}/{len(y_true)}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1c: Top-K ranking analysis")
    parser.add_argument("--data-dir", default="results/baseline")
    parser.add_argument("--train-cutoff", type=int, default=200,
                        help="Train on gens [0, train_cutoff)")
    parser.add_argument("--test-gen", type=int, default=200,
                        help="Test on this generation")
    args = parser.parse_args()

    solutions, mean_tp, _ = load_run_data(args.data_dir)
    generations = pd.read_csv(f"{args.data_dir}/cmaes_log.csv")["generation"].values

    run_topk_analysis(solutions, mean_tp, generations,
                      args.train_cutoff, args.test_gen)


if __name__ == "__main__":
    main()


""" 
Train: gens [0, 200) — 20000 samples
Test:  gen 200 — 100 samples

=== Top-K Analysis: train gens 0-199, test gen 200 (N=100) ===
Top- 1: 0/1 in predicted top-1 | worst pred rank:   7 | median pred rank:   7
Top- 2: 0/2 in predicted top-2 | worst pred rank:  13 | median pred rank:  10
Top- 3: 0/3 in predicted top-3 | worst pred rank:  13 | median pred rank:  12
Top- 5: 2/5 in predicted top-5 | worst pred rank:  13 | median pred rank:   7
Top-10: 5/10 in predicted top-10 | worst pred rank:  19 | median pred rank:   8
Top-20: 20/20 in predicted top-20 | worst pred rank:  19 | median pred rank:   9

True #1: tp=8.0192, predicted=7.9101, surrogate rank=8/100
Surrogate #1: predicted=7.9617, actual=8.0040, true rank=4/100
"""