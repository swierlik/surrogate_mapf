"""XGBoost surrogate baseline (flattened weight vector input)."""

import numpy as np
import xgboost as xgb


class XGBoostSurrogate:
    """XGBoost regressor for throughput prediction from flat solution vectors.

    Takes raw flat (N, 4074) vectors as input — no spatial reshape needed.
    XGBoost is invariant to monotone transforms, so raw vs normalized
    solutions give equivalent rankings.
    """

    def __init__(self, params=None):
        default_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": 0,
        }
        if params:
            default_params.update(params)
        self.model = xgb.XGBRegressor(**default_params)
        self.is_fitted = False

    def fit(self, X, y, X_val=None, y_val=None):
        """Train on (N, 4074) solutions and (N,) throughputs."""
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True

    def predict(self, X):
        """Predict throughput. Returns (N,) array."""
        return self.model.predict(X)

    @property
    def name(self):
        return "XGBoost"
