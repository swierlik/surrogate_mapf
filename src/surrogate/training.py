"""Surrogate training and retraining logic.

Provides cross-validation, learning curve, and temporal split evaluation
utilities that work with any model exposing fit(X, y) / predict(X).
"""

import time
import numpy as np
from sklearn.model_selection import KFold

from src.utils.metrics import compute_all_metrics


def cross_validate(model_factory, X, y, n_folds=5, seed=42):
    """K-fold cross-validation with full metric reporting.

    Args:
        model_factory: Callable returning a fresh model with
                       fit(X, y) and predict(X) methods.
        X: (N, D) feature matrix.
        y: (N,) target values.
        n_folds: Number of CV folds.
        seed: Random seed for fold splitting.

    Returns:
        dict with fold_metrics, mean_metrics, std_metrics,
        and fold_predictions (for later analysis).
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []
    fold_predictions = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        t0 = time.time()
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        elapsed = time.time() - t0

        metrics = compute_all_metrics(y[test_idx], y_pred)
        metrics["train_time_s"] = elapsed
        metrics["fold"] = fold_idx
        metrics["train_size"] = len(train_idx)
        metrics["test_size"] = len(test_idx)
        fold_metrics.append(metrics)
        fold_predictions.append((test_idx, y[test_idx], y_pred))

        print(f"  Fold {fold_idx+1}/{n_folds}: "
              f"rho={metrics['spearman_rho']:.4f}, "
              f"mse={metrics['mse']:.6f}, "
              f"top20_prec={metrics['top_20pct_precision']:.3f}, "
              f"time={elapsed:.1f}s")

    metric_keys = [k for k in fold_metrics[0]
                   if k not in ("fold", "train_size", "test_size")]
    mean_metrics = {k: np.mean([m[k] for m in fold_metrics])
                    for k in metric_keys}
    std_metrics = {k: np.std([m[k] for m in fold_metrics])
                   for k in metric_keys}

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "fold_predictions": fold_predictions,
    }


def learning_curve(model_factory, X, y,
                   train_sizes=(500, 1000, 2000, 5000, 10000, 20000),
                   test_fraction=0.2, seed=42):
    """Evaluate model at increasing training set sizes.

    Uses a fixed held-out test set; training subsets are drawn from the
    remaining pool.

    Returns:
        list of dicts, one per train_size.
    """
    rng = np.random.default_rng(seed)
    n = len(X)

    idx = rng.permutation(n)
    n_test = int(n * test_fraction)
    test_idx = idx[:n_test]
    pool_idx = idx[n_test:]

    X_test, y_test = X[test_idx], y[test_idx]
    results = []

    for size in train_sizes:
        if size > len(pool_idx):
            print(f"  Skipping size {size} (only {len(pool_idx)} available)")
            continue

        sub_idx = rng.choice(len(pool_idx), size=size, replace=False)
        X_train = X[pool_idx[sub_idx]]
        y_train = y[pool_idx[sub_idx]]

        t0 = time.time()
        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.time() - t0

        metrics = compute_all_metrics(y_test, y_pred)
        metrics["train_size"] = size
        metrics["test_size"] = n_test
        metrics["train_time_s"] = elapsed
        results.append(metrics)

        print(f"  Size {size:>6d}: "
              f"rho={metrics['spearman_rho']:.4f}, "
              f"mse={metrics['mse']:.6f}, "
              f"top20_prec={metrics['top_20pct_precision']:.3f}, "
              f"time={elapsed:.1f}s")

    return results


def temporal_split_eval(model_factory, X, y, generations,
                        train_gen_range=(0, 200), test_gen_range=(200, 300)):
    """Train on early generations, test on later generations.

    This mimics the real surrogate-assisted CMA-ES scenario where the
    surrogate must predict on solutions from the evolving distribution.

    Args:
        model_factory: Callable returning a fresh model.
        X: (N, D) feature matrix.
        y: (N,) target values.
        generations: (N,) int array of generation numbers.
        train_gen_range: [start, end) range for training.
        test_gen_range: [start, end) range for testing.

    Returns:
        dict with metrics and metadata.
    """
    train_mask = ((generations >= train_gen_range[0])
                  & (generations < train_gen_range[1]))
    test_mask = ((generations >= test_gen_range[0])
                 & (generations < test_gen_range[1]))

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Temporal split: train={train_mask.sum()} "
          f"(gens {train_gen_range}), "
          f"test={test_mask.sum()} (gens {test_gen_range})")

    t0 = time.time()
    model = model_factory()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = time.time() - t0

    metrics = compute_all_metrics(y_test, y_pred)
    metrics["train_size"] = int(train_mask.sum())
    metrics["test_size"] = int(test_mask.sum())
    metrics["train_time_s"] = elapsed
    metrics["train_gen_range"] = list(train_gen_range)
    metrics["test_gen_range"] = list(test_gen_range)

    print(f"  Temporal: rho={metrics['spearman_rho']:.4f}, "
          f"mse={metrics['mse']:.6f}, "
          f"top20_prec={metrics['top_20pct_precision']:.3f}")

    return metrics
