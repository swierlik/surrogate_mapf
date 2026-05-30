"""Experiment 1b: Sliding-window temporal generalization test.

For each test generation g in [10, 20, 50, 100, 150, 200, 250, 299]:
  - Train surrogate on all data from generations [0, g)
  - Predict throughput for generation g (never seen during training)
  - Report Spearman rho, top-20% precision, training time

Also runs MLP fine-tune speed test:
  - Full train on gens [0, 200)
  - Fine-tune on gen 200 (10 epochs, warm-start from previous weights)
  - Predict gen 201

This is the realistic benchmark for surrogate-assisted CMA-ES:
the model only ever sees past data, simulating the actual deployment setting.

Results recorded in findings.md Sessions 2 & 3.

Usage:
    python -m experiments.01b_sliding_window
    python -m experiments.01b_sliding_window --models xgboost,mlp
    python -m experiments.01b_sliding_window --skip-finetune-test
"""

import argparse
import time

import numpy as np
import pandas as pd

from src.utils.data import load_run_data
from src.utils.metrics import spearman_rho, top_k_precision

TEST_GENS = [10, 20, 50, 100, 150, 200, 250, 299]  # default (5-emitter 300-gen run)


def make_model(name):
    if name == "xgboost":
        from src.surrogate.xgboost_model import XGBoostSurrogate
        return XGBoostSurrogate()
    elif name == "mlp":
        from src.surrogate.mlp_model import MLPSurrogate
        return MLPSurrogate(max_epochs=100, patience=10)
    elif name == "cnn":
        from src.surrogate.cnn_model import CNNSurrogate
        return CNNSurrogate(max_epochs=100, patience=10)
    raise ValueError(f"Unknown model: {name}")


def run_sliding_window(model_name, solutions, mean_tp, generations, test_gens=None):
    if test_gens is None:
        test_gens = TEST_GENS
    print(f"\n=== {model_name.upper()}: Train on [0,g), predict gen g ===")
    for g in test_gens:
        tr = generations < g
        te = generations == g
        t0 = time.time()
        m = make_model(model_name)
        m.fit(solutions[tr], mean_tp[tr])
        p = m.predict(solutions[te])
        rho = spearman_rho(mean_tp[te], p)
        k = max(1, te.sum() // 5)
        tp20 = top_k_precision(mean_tp[te], p, k=k)
        print(f"Gen {g:>3d} | train={tr.sum():>5d} | rho={rho:.4f} | top20%={tp20:.3f} | {time.time()-t0:.0f}s")


def run_oracle(model_name, solutions, mean_tp, generations, test_gens=None,
               full_leakage=False):
    """Oracle test: train and predict on candidates from the same generation.

    full_leakage=False (default): 80/20 split within the generation.
      Tests local learnability — if rho~0, the landscape is locally rough.
    full_leakage=True: train on all 100, predict the same 100 (train=test).
      Tests whether the model can memorize its own training data at all.
      If rho~1.0 here but ~0 in 80/20, roughness is confirmed.
      If rho<1.0 even here, the model cannot even fit the data.
    """
    if test_gens is None:
        test_gens = TEST_GENS
    rng = np.random.default_rng(42)
    mode_str = "FULL LEAKAGE (train=test 100/100)" if full_leakage else "ORACLE (80/20 same gen)"
    print(f"\n=== {model_name.upper()} {mode_str} ===")
    for g in test_gens:
        mask = np.where(generations == g)[0]
        if len(mask) < 10:
            print(f"Gen {g:>3d} | too few samples ({len(mask)}), skipping")
            continue
        if full_leakage:
            tr_idx = te_idx = mask
        else:
            idx = rng.permutation(len(mask))
            n_train = int(len(mask) * 0.8)
            tr_idx = mask[idx[:n_train]]
            te_idx = mask[idx[n_train:]]
        t0 = time.time()
        m = make_model(model_name)
        m.fit(solutions[tr_idx], mean_tp[tr_idx])
        p = m.predict(solutions[te_idx])
        rho = spearman_rho(mean_tp[te_idx], p)
        print(f"Gen {g:>3d} | train={len(tr_idx):>3d} test={len(te_idx):>3d} | rho={rho:.4f} | {time.time()-t0:.0f}s")


def run_mlp_finetune_test(solutions, mean_tp, generations):
    from src.surrogate.mlp_model import MLPSurrogate

    print("\n=== MLP: Fine-tune test (train on [0,200), fine-tune on 200, predict 201) ===")
    tr200 = generations < 200
    m = MLPSurrogate(max_epochs=100, patience=10)

    t0 = time.time()
    m.fit(solutions[tr200], mean_tp[tr200])
    print(f"Full train on {tr200.sum()} samples: {time.time()-t0:.1f}s")

    ft_mask = generations == 200
    t0 = time.time()
    m.fine_tune(solutions[ft_mask], mean_tp[ft_mask], epochs=10)
    ft_time = time.time() - t0

    te201 = generations == 201
    p201 = m.predict(solutions[te201])
    rho201 = spearman_rho(mean_tp[te201], p201)
    print(f"Fine-tune on {ft_mask.sum()} samples: {ft_time:.1f}s | predict gen 201: rho={rho201:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1b: Sliding-window temporal generalization")
    parser.add_argument("--data-dir", default="results/baseline")
    parser.add_argument("--models", default="xgboost,mlp,cnn",
                        help="Comma-separated: xgboost,mlp,cnn")
    parser.add_argument("--skip-finetune-test", action="store_true")
    parser.add_argument("--test-gens", default=None,
                        help="Comma-separated test generations, e.g. 5,10,20,50")
    parser.add_argument("--oracle", action="store_true",
                        help="Oracle mode: train/test on same generation (80/20 split)")
    parser.add_argument("--full-leakage", action="store_true",
                        help="Full leakage: train and predict on identical 100 samples (train=test)")
    args = parser.parse_args()

    test_gens = ([int(g) for g in args.test_gens.split(",")]
                 if args.test_gens else TEST_GENS)

    solutions, mean_tp, _ = load_run_data(args.data_dir)
    generations = pd.read_csv(f"{args.data_dir}/cmaes_log.csv")["generation"].values

    for model_name in [m.strip() for m in args.models.split(",")]:
        if args.full_leakage:
            run_oracle(model_name, solutions, mean_tp, generations, test_gens,
                       full_leakage=True)
        elif args.oracle:
            run_oracle(model_name, solutions, mean_tp, generations, test_gens)
        else:
            run_sliding_window(model_name, solutions, mean_tp, generations, test_gens)

    max_gen = generations.max()
    if not args.oracle and not args.skip_finetune_test and "mlp" in args.models and max_gen >= 201:
        run_mlp_finetune_test(solutions, mean_tp, generations)


if __name__ == "__main__":
    main()
