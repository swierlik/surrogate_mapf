"""Lightweight CNN surrogate (spatial tensor input)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.reshape import SolutionReshaper


class CNNThroughputModel(nn.Module):
    """3-layer CNN for throughput prediction from (6, 33, 36) tensor.

    Architecture (from thesis plan):
        Conv2d(6→16, 3×3) → BN → ReLU → Dropout(0.3)
        Conv2d(16→32, 3×3) → BN → ReLU → Dropout(0.3)
        Conv2d(32→64, 3×3) → BN → ReLU
        GlobalAvgPool → FC(64→32) → ReLU → Dropout(0.5) → FC(32→1)

    ~15-20k parameters.
    """

    def __init__(self, in_channels=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x.squeeze(-1)


class CNNSurrogate:
    """Wrapper with the same fit/predict interface as XGBoostSurrogate.

    Handles flat→tensor conversion, per-channel z-score standardization,
    training loop with early stopping, and inference.
    """

    def __init__(self, lr=1e-3, weight_decay=1e-4, batch_size=64,
                 max_epochs=100, patience=10, device=None):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.reshaper = SolutionReshaper.get()
        self.model = None
        self.is_fitted = False

        # Standardization stats (fit on training data)
        self._X_mean = None
        self._X_std = None
        self._y_mean = None
        self._y_std = None

    # ------------------------------------------------------------------
    # Standardization helpers
    # ------------------------------------------------------------------

    def _standardize_X(self, X_tensor, fit=False):
        """Per-channel z-score. X_tensor shape: (N, H, W, C)."""
        if fit:
            self._X_mean = X_tensor.mean(axis=(0, 1, 2), keepdims=True)
            self._X_std = X_tensor.std(axis=(0, 1, 2), keepdims=True) + 1e-8
        return (X_tensor - self._X_mean) / self._X_std

    def _standardize_y(self, y, fit=False):
        if fit:
            self._y_mean = float(y.mean())
            self._y_std = float(y.std()) + 1e-8
        return (y - self._y_mean) / self._y_std

    def _unstandardize_y(self, y):
        return y * self._y_std + self._y_mean

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y, X_val=None, y_val=None):
        """Train CNN from flat (N, 4074) solutions and (N,) throughputs.

        If no validation set is given, 20 % of training data is held out.
        """
        # Flat → spatial tensor (N, H, W, 6)
        X_tensor = self.reshaper.flat_to_tensor_batch(X, add_obstacle_mask=True)

        # Split validation if needed
        if X_val is None:
            n = len(X)
            idx = np.random.permutation(n)
            split = int(0.8 * n)
            X_train_t, X_val_t = X_tensor[idx[:split]], X_tensor[idx[split:]]
            y_train, y_val_local = y[idx[:split]], y[idx[split:]]
        else:
            X_train_t = X_tensor
            y_train = y
            X_val_t = self.reshaper.flat_to_tensor_batch(
                X_val, add_obstacle_mask=True)
            y_val_local = y_val

        # Standardize
        X_train_t = self._standardize_X(X_train_t, fit=True)
        X_val_t = self._standardize_X(X_val_t)
        y_train = self._standardize_y(y_train.astype(np.float32), fit=True)
        y_val_local = self._standardize_y(y_val_local.astype(np.float32))

        # To torch (NHWC → NCHW)
        X_train_th = torch.from_numpy(X_train_t.transpose(0, 3, 1, 2))
        y_train_th = torch.from_numpy(y_train)
        X_val_th = torch.from_numpy(X_val_t.transpose(0, 3, 1, 2))
        y_val_th = torch.from_numpy(y_val_local)

        train_loader = DataLoader(
            TensorDataset(X_train_th, y_train_th),
            batch_size=self.batch_size, shuffle=True,
        )

        # Init model
        self.model = CNNThroughputModel(in_channels=6).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        # Training with early stopping
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

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_th.to(self.device))
                val_loss = criterion(val_pred, y_val_th.to(self.device)).item()

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
        """Predict throughput from flat (N, 4074) vectors. Returns (N,)."""
        X_tensor = self.reshaper.flat_to_tensor_batch(
            X, add_obstacle_mask=True)
        X_tensor = self._standardize_X(X_tensor)
        X_th = torch.from_numpy(
            X_tensor.transpose(0, 3, 1, 2)).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_th).cpu().numpy()
        return self._unstandardize_y(pred)

    @property
    def name(self):
        return "CNN"
