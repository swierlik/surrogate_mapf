"""Evaluation metrics: Spearman rho, MSE, ranking utilities."""

import warnings
import numpy as np
from scipy.stats import spearmanr


def spearman_rho(y_true, y_pred):
    """Spearman rank correlation between true and predicted values.

    Returns float in [-1, 1]. Returns 0.0 if predictions are constant.
    """
    if np.std(y_pred) < 1e-12:
        warnings.warn("Predictions are nearly constant; returning rho=0.0")
        return 0.0
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


def mse(y_true, y_pred):
    """Mean squared error."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true, y_pred):
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def top_k_precision(y_true, y_pred, k=20):
    """Fraction of surrogate's predicted top-k that are actually in the real top-k.

    Args:
        y_true: True throughput values, shape (N,).
        y_pred: Predicted throughput values, shape (N,).
        k: Number of top candidates to consider.

    Returns:
        Precision in [0.0, 1.0].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    k = min(k, len(y_true))
    true_top_k = set(np.argsort(y_true)[-k:])
    pred_top_k = set(np.argsort(y_pred)[-k:])
    return len(true_top_k & pred_top_k) / k


def top_k_precision_pct(y_true, y_pred, pct=0.2):
    """Top-k precision where k is a percentage of N."""
    k = max(1, int(len(y_true) * pct))
    return top_k_precision(y_true, y_pred, k=k)


def compute_all_metrics(y_true, y_pred, k=20):
    """Compute all metrics and return as a dict."""
    return {
        "spearman_rho": spearman_rho(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "top_k_precision": top_k_precision(y_true, y_pred, k=k),
        "top_20pct_precision": top_k_precision_pct(y_true, y_pred, pct=0.2),
    }
