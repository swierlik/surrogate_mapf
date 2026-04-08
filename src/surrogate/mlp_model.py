"""Simple MLP surrogate (flat vector input, supports fine-tuning)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLPThroughputModel(nn.Module):
    """Simple feedforward network: 4074 → 256 → 128 → 64 → 1."""

    def __init__(self, input_dim=4074):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPSurrogate:
    """MLP surrogate with fit/predict interface and fine-tuning support.

    Key advantage over XGBoost: can fine-tune on new data in seconds
    by continuing training from previous weights (warm start).
    """

    def __init__(self, lr=1e-3, weight_decay=1e-4, batch_size=64,
                 max_epochs=100, patience=10, device=None):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.is_fitted = False

        self._X_mean = None
        self._X_std = None
        self._y_mean = None
        self._y_std = None

    def _standardize_X(self, X, fit=False):
        if fit:
            self._X_mean = X.mean(axis=0, keepdims=True)
            self._X_std = X.std(axis=0, keepdims=True) + 1e-8
        return (X - self._X_mean) / self._X_std

    def _standardize_y(self, y, fit=False):
        if fit:
            self._y_mean = float(y.mean())
            self._y_std = float(y.std()) + 1e-8
        return (y - self._y_mean) / self._y_std

    def _unstandardize_y(self, y):
        return y * self._y_std + self._y_mean

    def fit(self, X, y, X_val=None, y_val=None):
        """Full training from scratch."""
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        # Split validation if needed
        if X_val is None:
            n = len(X)
            idx = np.random.permutation(n)
            split = int(0.8 * n)
            X_train, X_val_local = X[idx[:split]], X[idx[split:]]
            y_train, y_val_local = y[idx[:split]], y[idx[split:]]
        else:
            X_train, y_train = X, y
            X_val_local = X_val.astype(np.float32)
            y_val_local = y_val.astype(np.float32)

        # Standardize
        X_train = self._standardize_X(X_train, fit=True)
        X_val_local = self._standardize_X(X_val_local)
        y_train = self._standardize_y(y_train, fit=True)
        y_val_local = self._standardize_y(y_val_local)

        self._train_loop(X_train, y_train, X_val_local, y_val_local,
                         init_model=True)

    def fine_tune(self, X_new, y_new, epochs=10):
        """Continue training on new data from previous weights.

        Much faster than full retrain — typically 2-5s for 100 samples.
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before fine_tune()")

        X_new = self._standardize_X(X_new.astype(np.float32))
        y_new = self._standardize_y(y_new.astype(np.float32))

        # Short training with lower LR to avoid catastrophic forgetting
        old_max_epochs = self.max_epochs
        old_patience = self.patience
        old_lr = self.lr
        self.max_epochs = epochs
        self.patience = epochs  # no early stopping for fine-tuning
        self.lr = old_lr * 0.1  # lower LR for fine-tuning

        self._train_loop(X_new, y_new, X_new, y_new, init_model=False)

        self.max_epochs = old_max_epochs
        self.patience = old_patience
        self.lr = old_lr

    def _train_loop(self, X_train, y_train, X_val, y_val, init_model=True):
        X_train_th = torch.from_numpy(X_train)
        y_train_th = torch.from_numpy(y_train)
        X_val_th = torch.from_numpy(X_val).to(self.device)
        y_val_th = torch.from_numpy(y_val).to(self.device)

        train_loader = DataLoader(
            TensorDataset(X_train_th, y_train_th),
            batch_size=self.batch_size, shuffle=True,
        )

        if init_model:
            self.model = MLPThroughputModel(
                input_dim=X_train.shape[1]).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_th)
                val_loss = criterion(val_pred, y_val_th).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.is_fitted = True

    def predict(self, X):
        """Predict throughput from flat (N, 4074) vectors."""
        X = self._standardize_X(X.astype(np.float32))
        X_th = torch.from_numpy(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_th).cpu().numpy()
        return self._unstandardize_y(pred)

    @property
    def name(self):
        return "MLP"
