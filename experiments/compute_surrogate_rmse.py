"""Reconstruct RMSE of surrogate predictions vs true throughput at every control gen.

For each control gen t (gens 20, 30, 40, ...):
  - Train EnsembleSurrogate on all data evaluated BEFORE gen t
  - Predict throughput for gen t's 100 candidates
  - Compute RMSE, normalised RMSE, and population std

Saves results/surrogate_v3/surrogate_rmse.csv.

Usage:
    python -m experiments.compute_surrogate_rmse
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")

from src.optimizer.surrogate_cmaes import EnsembleSurrogate
from src.optimizer.surrogate_cmaes import MLPSurrogate  # noqa — imported via EnsembleSurrogate

RUN_DIR = Path("results/surrogate_v3")


def main():
    sols = np.load(RUN_DIR / "cmaes_solutions.npy")   # (12640, 4074)
    cl   = pd.read_csv(RUN_DIR / "cmaes_log.csv")
    sl   = pd.read_csv(RUN_DIR / "surrogate_log.csv")

    # Add a global row index matching cmaes_solutions.npy row order
    cl = cl.reset_index(drop=True)
    cl["row_idx"] = cl.index

    # Control gens (where we evaluate RMSE)
    ctrl_gens = sl.loc[sl["mode"] == "control", "generation"].sort_values().tolist()
    print(f"Control gens to evaluate: {len(ctrl_gens)}  ({ctrl_gens[:5]}...{ctrl_gens[-3:]})")

    records = []
    for t in ctrl_gens:
        t0 = time.time()

        # Training data: all rows evaluated in gens BEFORE t
        train_mask = cl["generation"] < t
        X_train = sols[cl.loc[train_mask, "row_idx"].values]
        y_train = cl.loc[train_mask, "mean_throughput"].values

        # Test data: gen t's 100 candidates
        test_mask = cl["generation"] == t
        X_test  = sols[cl.loc[test_mask, "row_idx"].values]
        y_test  = cl.loc[test_mask, "mean_throughput"].values

        if len(X_train) < 50 or len(X_test) == 0:
            continue

        # Train surrogate (same config as live run)
        surr = EnsembleSurrogate(n_models=5, lr=1e-3, weight_decay=1e-4,
                                 batch_size=64, max_epochs=100, patience=10)
        surr.fit(X_train, y_train)

        # Predict
        pred_mean, pred_std = surr.predict_with_uncertainty(X_test)

        rmse        = float(np.sqrt(np.mean((pred_mean - y_test) ** 2)))
        pop_std     = float(y_test.std())
        nrmse       = rmse / pop_std if pop_std > 1e-6 else np.nan
        pct95_std   = float(np.percentile(pred_std, 95))
        elapsed     = time.time() - t0

        # Logged rho for comparison
        rho_row = sl.loc[sl["generation"] == t, "surrogate_rho"]
        rho_val = float(rho_row.values[0]) if len(rho_row) > 0 else np.nan

        records.append(dict(
            generation   = t,
            rmse         = rmse,
            nrmse        = nrmse,
            pop_std      = pop_std,
            mean_pred_std= float(pred_std.mean()),
            pct95_pred_std= pct95_std,
            surrogate_rho= rho_val,
            n_train      = len(X_train),
            train_time_s = elapsed,
        ))
        print(f"  gen {t:3d} | n_train={len(X_train):5d} | RMSE={rmse:.4f} "
              f"| nRMSE={nrmse:.3f} | rho={rho_val:.3f} | {elapsed:.1f}s")

    out = pd.DataFrame(records)
    out.to_csv(RUN_DIR / "surrogate_rmse.csv", index=False)
    print(f"\nSaved {RUN_DIR / 'surrogate_rmse.csv'}")

    # Quick correlation summary
    from scipy import stats
    for col in ["rmse", "nrmse", "pop_std", "mean_pred_std"]:
        r, p = stats.pearsonr(out[col], out["surrogate_rho"])
        print(f"  rho vs {col:16s}: Pearson r={r:+.3f}  p={p:.4f}")


if __name__ == "__main__":
    main()
