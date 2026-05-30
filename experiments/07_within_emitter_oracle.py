"""Experiment 7: Within-emitter oracle test with richer architectures.

Tests whether a LargeCNN, CellTransformer, or GNN can find within-emitter
rank signal that XGBoost and MLP cannot (median rho ~0 in the thesis).

Test design (per-emitter sliding window):
  For each test generation g in TEST_GENS:
    For each emitter e in 0..4:
      - Train on ALL past data from emitter e (generations < g)
      - Predict the 20 candidates from emitter e at generation g
      - Compute Spearman rho (within-emitter ranking)
  Report: mean rho per emitter, mean across emitters

This gives models a realistic amount of historical data (up to 6000 samples
per emitter) while isolating whether they can rank within a single emitter
at a single generation.

Usage:
    python -m experiments.07_within_emitter_oracle
    python -m experiments.07_within_emitter_oracle --models mlp,large_cnn,transformer,gnn
    python -m experiments.07_within_emitter_oracle --data-dir results/baseline
"""

import argparse
import contextlib
import math
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GATConv, global_mean_pool

TEST_GENS = [50, 100, 150, 200, 250, 299]
N_EMITTERS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _spearman(y_true, y_pred):
    if len(y_true) < 3:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = spearmanr(y_true, y_pred)
    return float(r) if not math.isnan(r) else 0.0


def _metrics(y_true, y_pred):
    """Return dict with rho, rmse, mae, q2 (R²)."""
    if len(y_true) < 3:
        nan = float("nan")
        return dict(rho=nan, rmse=nan, mae=nan, q2=nan)
    rho = _spearman(y_true, y_pred)
    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae  = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    q2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(rho=rho, rmse=rmse, mae=mae, q2=q2)


def _standardize(X_tr, X_te, y_tr, y_te):
    mu_x = X_tr.mean(0, keepdims=True)
    sd_x = X_tr.std(0, keepdims=True) + 1e-8
    mu_y = float(y_tr.mean())
    sd_y = float(y_tr.std()) + 1e-8
    return ((X_tr - mu_x) / sd_x, (X_te - mu_x) / sd_x,
            (y_tr - mu_y) / sd_y, mu_y, sd_y)


def _train_loop(model, loader, X_val_th, y_val_th, max_epochs, patience, lr, wd):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    best_loss, patience_count, best_state = float("inf"), 0, None
    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_val_th.to(DEVICE)), y_val_th.to(DEVICE)).item()
        if vl < best_loss:
            best_loss, patience_count = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()


# ---------------------------------------------------------------------------
# MLP baseline (same as production surrogate)
# ---------------------------------------------------------------------------

class _MLPNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


class MLPModel:
    name = "MLP"
    def __init__(self): self._stats = None

    def fit(self, X, y):
        X, y = X.astype(np.float32), y.astype(np.float32)
        n = len(X); idx = np.random.permutation(n); sp = int(0.85 * n)
        Xtr, Xv = X[idx[:sp]], X[idx[sp:]]
        ytr, yv = y[idx[:sp]], y[idx[sp:]]
        mu_x = Xtr.mean(0, keepdims=True); sd_x = Xtr.std(0, keepdims=True) + 1e-8
        mu_y = float(ytr.mean()); sd_y = float(ytr.std()) + 1e-8
        self._stats = (mu_x, sd_x, mu_y, sd_y)
        Xtr = (Xtr - mu_x) / sd_x; Xv = (Xv - mu_x) / sd_x
        ytr = (ytr - mu_y) / sd_y; yv = (yv - mu_y) / sd_y
        self.model = _MLPNet(X.shape[1]).to(DEVICE)
        loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                            batch_size=128, shuffle=True)
        _train_loop(self.model, loader,
                    torch.from_numpy(Xv), torch.from_numpy(yv),
                    max_epochs=150, patience=15, lr=1e-3, wd=1e-4)

    def predict(self, X):
        mu_x, sd_x, mu_y, sd_y = self._stats
        X = ((X.astype(np.float32) - mu_x) / sd_x)
        with torch.no_grad():
            p = self.model(torch.from_numpy(X).to(DEVICE)).cpu().numpy()
        return p * sd_y + mu_y


# ---------------------------------------------------------------------------
# LargeCNN  (deeper 2-D CNN on spatial warehouse tensor)
# ---------------------------------------------------------------------------

class _LargeCNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ci, co, k=3):
            return nn.Sequential(
                nn.Conv2d(ci, co, k, padding=k//2),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True), nn.Dropout2d(0.2))
        self.features = nn.Sequential(
            block(6, 32), block(32, 64), block(64, 128),
            block(128, 128), block(128, 64))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.4), nn.Linear(32, 1))
    def forward(self, x): return self.head(self.features(x)).squeeze(-1)


class LargeCNNModel:
    name = "LargeCNN"
    def __init__(self): self._stats = None

    def _to_tensor(self, X):
        from src.utils.reshape import SolutionReshaper
        t = SolutionReshaper.get().flat_to_tensor_batch(X.astype(np.float32),
                                                        add_obstacle_mask=True)
        return t.transpose(0, 3, 1, 2)  # NHWC → NCHW

    def fit(self, X, y):
        y = y.astype(np.float32)
        T = self._to_tensor(X)
        mu_x = T.mean(axis=(0, 2, 3), keepdims=True)
        sd_x = T.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        mu_y = float(y.mean()); sd_y = float(y.std()) + 1e-8
        self._stats = (mu_x, sd_x, mu_y, sd_y)
        T = (T - mu_x) / sd_x; y_s = (y - mu_y) / sd_y
        n = len(T); idx = np.random.permutation(n); sp = int(0.85 * n)
        Ttr, Tv = T[idx[:sp]], T[idx[sp:]]
        ytr, yv = y_s[idx[:sp]], y_s[idx[sp:]]
        self.model = _LargeCNNNet().to(DEVICE)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(Ttr), torch.from_numpy(ytr)),
            batch_size=64, shuffle=True)
        _train_loop(self.model, loader,
                    torch.from_numpy(Tv), torch.from_numpy(yv),
                    max_epochs=150, patience=15, lr=5e-4, wd=1e-4)

    def predict(self, X):
        mu_x, sd_x, mu_y, sd_y = self._stats
        T = (self._to_tensor(X) - mu_x) / sd_x
        with torch.no_grad():
            p = self.model(torch.from_numpy(T).to(DEVICE)).cpu().numpy()
        return p * sd_y + mu_y


# ---------------------------------------------------------------------------
# CellTransformer  (948 per-cell tokens, each = [wait, R, U, L, D])
# Speedups: Flash Attention (auto in PyTorch 2.x on CUDA), mixed-precision
# training (autocast fp16), and torch.compile() JIT compilation.
# ---------------------------------------------------------------------------

class _CellTransformerNet(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(5, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=256,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.encoder(self.input_proj(x))
        return self.head(x.mean(dim=1)).squeeze(-1)


class CellTransformerModel:
    """Treats each warehouse cell as a token with [wait, R, U, L, D] features.
    Uses Flash Attention (PyTorch 2.x auto-enable on CUDA), mixed-precision
    autocast, and torch.compile() for maximum GPU throughput.
    """
    name = "CellTransformer"

    def __init__(self):
        self._cell_edge_idx = None
        self._n_valid = None
        self._stats = None

    def _build_cell_map(self):
        from src.utils.reshape import SolutionReshaper
        r = SolutionReshaper.get()
        n_cells = r.n_valid
        cell_to_idx = np.full((r.h, r.w), -1, dtype=int)
        for i, (row, col) in enumerate(zip(r.valid_rows, r.valid_cols)):
            cell_to_idx[row, col] = i
        cell_edge_idx = np.full((n_cells, 4), -1, dtype=int)
        for ei, (row, col, d) in enumerate(zip(r.edge_rows, r.edge_cols, r.edge_chans)):
            cell_edge_idx[cell_to_idx[row, col], d] = ei
        self._cell_edge_idx = cell_edge_idx
        self._n_valid = n_cells

    def _to_tokens(self, X):
        if self._cell_edge_idx is None:
            self._build_cell_map()
        N = len(X)
        tokens = np.zeros((N, self._n_valid, 5), dtype=np.float32)
        tokens[:, :, 0] = X[:, :self._n_valid]
        edges = X[:, self._n_valid:]
        for d in range(4):
            idx = self._cell_edge_idx[:, d]
            valid = idx >= 0
            tokens[:, valid, d + 1] = edges[:, idx[valid]]
        return tokens

    def fit(self, X, y):
        y = y.astype(np.float32)
        T = self._to_tokens(X)
        mu = T.mean(axis=(0, 1), keepdims=True)
        sd = T.std(axis=(0, 1), keepdims=True) + 1e-8
        mu_y = float(y.mean()); sd_y = float(y.std()) + 1e-8
        self._stats = (mu, sd, mu_y, sd_y)
        T = (T - mu) / sd; y_s = (y - mu_y) / sd_y
        n = len(T); idx = np.random.permutation(n); sp = int(0.85 * n)
        Ttr, Tv = T[idx[:sp]], T[idx[sp:]]
        ytr, yv = y_s[idx[:sp]], y_s[idx[sp:]]

        self.model = _CellTransformerNet().to(DEVICE)

        opt = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        crit = nn.MSELoss()

        Ttr_th = torch.from_numpy(Ttr).to(DEVICE)
        ytr_th = torch.from_numpy(ytr).to(DEVICE)
        Tv_th  = torch.from_numpy(Tv).to(DEVICE)
        yv_th  = torch.from_numpy(yv).to(DEVICE)

        loader = DataLoader(TensorDataset(Ttr_th, ytr_th), batch_size=64, shuffle=True)
        best_loss, patience_count, best_state = float("inf"), 0, None

        for _ in range(120):
            self.model.train()
            for xb, yb in loader:
                opt.zero_grad()
                crit(self.model(xb), yb).backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                vl = crit(self.model(Tv_th), yv_th).item()
            if vl < best_loss:
                best_loss, patience_count = vl, 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= 12:
                    break
        if best_state:
            self.model.load_state_dict(best_state)
        self.model.eval()

    def predict(self, X):
        mu, sd, mu_y, sd_y = self._stats
        T = torch.from_numpy((self._to_tokens(X) - mu) / sd).to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            p = self.model(T).cpu().numpy()
        return p * sd_y + mu_y


# ---------------------------------------------------------------------------
# GNN  (PyTorch Geometric, GAT on warehouse graph)
# ---------------------------------------------------------------------------

class _GNNNet(nn.Module):
    def __init__(self, hidden=64, n_layers=3, heads=4):
        super().__init__()
        self.node_enc = nn.Linear(1, hidden)
        self.convs = nn.ModuleList([
            GATConv(hidden, hidden // heads, heads=heads, edge_dim=1, concat=True)
            for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.node_enc(x))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr=edge_attr))
        x = global_mean_pool(x, batch)  # type: ignore[attr-defined]
        return self.head(x).squeeze(-1)


class GNNModel:
    name = "GNN"

    def __init__(self):
        self._graph_topo = None
        self._stats = None

    def _build_topology(self):
        from src.utils.reshape import SolutionReshaper
        r = SolutionReshaper.get()
        dirs = np.array([(0, 1), (-1, 0), (0, -1), (1, 0)])

        cell_to_idx = np.full((r.h, r.w), -1, dtype=int)
        for i, (row, col) in enumerate(zip(r.valid_rows, r.valid_cols)):
            cell_to_idx[row, col] = i

        tgt_rows = r.edge_rows + dirs[r.edge_chans, 0]
        tgt_cols = r.edge_cols + dirs[r.edge_chans, 1]
        src_idx = cell_to_idx[r.edge_rows, r.edge_cols]
        tgt_idx = cell_to_idx[tgt_rows, tgt_cols]
        edge_index = torch.tensor(np.stack([src_idx, tgt_idx], axis=0), dtype=torch.long)
        self._graph_topo = edge_index
        self._n_nodes = r.n_valid    # 948
        self._n_edges = r.n_edges    # 3126

    def _build_dataset(self, X, y=None):
        """Precompute all graph Data objects at once — avoids per-batch construction."""
        from torch_geometric.data import Data
        if self._graph_topo is None:
            self._build_topology()
        mu_x, sd_x, mu_y, sd_y = self._stats
        X_s = (X.astype(np.float32) - mu_x) / sd_x
        data_list = []
        # Precompute all node/edge tensors in one vectorised step
        nodes = torch.from_numpy(X_s[:, :self._n_nodes]).unsqueeze(-1)   # (N, 948, 1)
        edges = torch.from_numpy(X_s[:, self._n_nodes:]).unsqueeze(-1)   # (N, 3126, 1)
        ys = None
        if y is not None:
            ys = torch.tensor((y - mu_y) / sd_y, dtype=torch.float32)
        for i in range(len(X_s)):
            d = Data(x=nodes[i], edge_index=self._graph_topo,
                     edge_attr=edges[i],
                     y=ys[i:i+1] if ys is not None else None)
            data_list.append(d)
        return data_list

    def fit(self, X, y):
        from torch_geometric.loader import DataLoader as PyGLoader
        y = y.astype(np.float32)
        mu_x = X.mean(0, keepdims=True).astype(np.float32)
        sd_x = X.std(0, keepdims=True).astype(np.float32) + 1e-8
        mu_y = float(y.mean()); sd_y = float(y.std()) + 1e-8
        self._stats = (mu_x, sd_x, mu_y, sd_y)

        n = len(X); idx = np.random.permutation(n); sp = int(0.85 * n)
        tr_idx, val_idx = idx[:sp], idx[sp:]

        # Build all graphs once upfront
        all_data = self._build_dataset(X, y)
        tr_data  = [all_data[i] for i in tr_idx]
        val_data = [all_data[i] for i in val_idx]
        tr_loader  = PyGLoader(tr_data,  batch_size=64, shuffle=True)
        val_loader = PyGLoader(val_data, batch_size=256, shuffle=False)

        self.model = _GNNNet().to(DEVICE)
        opt  = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        crit = nn.MSELoss()
        best_loss, patience_count, best_state = float("inf"), 0, None

        for epoch in range(120):
            self.model.train()
            for batch in tr_loader:
                batch = batch.to(DEVICE)
                opt.zero_grad()
                pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                crit(pred, batch.y.squeeze(-1)).backward()
                opt.step()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(DEVICE)
                    pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    val_loss += crit(pred, batch.y.squeeze(-1)).item() * len(batch)
            val_loss /= len(val_data)

            if val_loss < best_loss:
                best_loss, patience_count = val_loss, 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= 12:
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        self.model.eval()

    def predict(self, X):
        from torch_geometric.loader import DataLoader as PyGLoader
        if self._graph_topo is None:
            self._build_topology()
        data_list = self._build_dataset(X)
        loader = PyGLoader(data_list, batch_size=256, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                p = self.model(batch.x, batch.edge_index,
                               batch.edge_attr, batch.batch).cpu().numpy()
                preds.append(p)
        mu_x, sd_x, mu_y, sd_y = self._stats
        return np.concatenate(preds) * sd_y + mu_y


# ---------------------------------------------------------------------------
# Single-emitter test
# Two modes:
#   sliding: train on ALL past gens (gens < g), test on ALL 100 candidates
#            of gen g — no holdout needed since train/test are different gens
#   oracle:  train on 80% of same gen, test on remaining 20% (same gen)
# ---------------------------------------------------------------------------

def run_single_emitter(model_cls, solutions, mean_tp, generations,
                       test_gens, mode="sliding", rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    name = model_cls.name
    tag = f"[{name}/{mode}]"

    gen_records = []
    for g in test_gens:
        gen_mask = generations == g
        if gen_mask.sum() < 5:
            continue

        all_te_idx = np.where(gen_mask)[0]

        if mode == "sliding":
            tr_mask = generations < g
            train_idx = np.where(tr_mask)[0]
            if len(train_idx) < 10:
                continue
            test_idx = all_te_idx          # ALL candidates from gen g

        else:  # oracle: 80% of same gen as train, 20% as test
            perm = rng.permutation(len(all_te_idx))
            n_train = max(int(len(all_te_idx) * 0.8), 5)
            train_idx = all_te_idx[perm[:n_train]]
            test_idx  = all_te_idx[perm[n_train:]]

        t0 = time.time()
        m = model_cls()
        m.fit(solutions[train_idx], mean_tp[train_idx])
        preds = m.predict(solutions[test_idx])
        met = _metrics(mean_tp[test_idx], preds)
        elapsed = time.time() - t0

        rec = dict(gen=g, mode=mode, model=name,
                   n_train=len(train_idx), n_test=len(test_idx),
                   elapsed_s=round(elapsed, 1), **met)
        gen_records.append(rec)

        print(f"{tag}/g{g:02d}  rho={met['rho']:+.4f}  rmse={met['rmse']:.4f}  "
              f"mae={met['mae']:.4f}  q2={met['q2']:+.4f}  "
              f"n_tr={len(train_idx)}  n_te={len(test_idx)}  {elapsed:.0f}s",
              flush=True)

    if gen_records:
        mean_rho  = float(np.mean([r["rho"]  for r in gen_records]))
        mean_rmse = float(np.mean([r["rmse"] for r in gen_records]))
        mean_mae  = float(np.mean([r["mae"]  for r in gen_records]))
        mean_q2   = float(np.mean([r["q2"]   for r in gen_records]))
        print(f"{tag}/MEAN  rho={mean_rho:+.4f}  rmse={mean_rmse:.4f}  "
              f"mae={mean_mae:.4f}  q2={mean_q2:+.4f}", flush=True)

    return gen_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "mlp": MLPModel,
    "large_cnn": LargeCNNModel,
    "transformer": CellTransformerModel,
    "gnn": GNNModel,
}

DEFAULT_TEST_GENS_1EM = [20, 40, 60, 80, 99]


def _save_results(all_records, out_path):
    import json
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"  [saved -> {out_path}]")


def main():
    parser = argparse.ArgumentParser(
        description="Exp 7: Within-emitter oracle with richer architectures")
    parser.add_argument("--data-dir", default="results/baseline_1em",
                        help="Should point to single-emitter run data")
    parser.add_argument("--models", default="mlp,large_cnn,transformer,gnn",
                        help="Comma-separated model names")
    parser.add_argument("--test-gens", default=None,
                        help="Override test generations, e.g. 20,50,80,99")
    parser.add_argument("--mode", default="both",
                        choices=["sliding", "oracle", "both"],
                        help="Test mode: sliding window, same-gen oracle, or both")
    parser.add_argument("--out", default="results/exp7_within_emitter.json",
                        help="Path to save results JSON")
    parser.add_argument("--parallel", action="store_true",
                        help="Run each model in a separate subprocess (one per model)")
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir} ...")
    solutions = np.load(f"{args.data_dir}/cmaes_solutions.npy")
    log = pd.read_csv(f"{args.data_dir}/cmaes_log.csv")
    mean_tp = log["mean_throughput"].values
    generations = log["generation"].values

    max_gen = int(generations.max())
    default_gens = [g for g in DEFAULT_TEST_GENS_1EM if g <= max_gen]
    test_gens = ([int(g) for g in args.test_gens.split(",")]
                 if args.test_gens else default_gens)

    print(f"  {len(solutions)} candidates, {solutions.shape[1]}-dim, "
          f"{log['generation'].nunique()} generations")
    print(f"  Using device: {DEVICE}")
    print(f"  Test generations: {test_gens}")
    print(f"  Models: {args.models}")
    print(f"  Mode: {args.mode}")
    print(f"  Output: {args.out}")
    print()
    print("  Context: thesis showed XGBoost and MLP give rho~0 in the")
    print("  single-emitter oracle (ICC=0 by construction). Testing whether")
    print("  richer architectures (LargeCNN, Transformer, GNN) can do better.")

    modes = ["sliding", "oracle"] if args.mode == "both" else [args.mode]
    model_names = [m.strip() for m in args.models.split(",")]

    # ------------------------------------------------------------------
    # Parallel mode: spawn one subprocess per model, merge at the end
    # ------------------------------------------------------------------
    if args.parallel and len(model_names) > 1:
        import subprocess, tempfile, json as _json
        print(f"  Launching {len(model_names)} subprocesses in parallel ...")
        procs, tmp_paths = [], []
        for name in model_names:
            tmp = tempfile.mktemp(suffix=f"_{name}.json")
            tmp_paths.append(tmp)
            cmd = [sys.executable, __file__,
                   "--data-dir", args.data_dir,
                   "--models", name,
                   "--mode", args.mode,
                   "--test-gens", ",".join(map(str, test_gens)),
                   "--out", tmp]
            procs.append(subprocess.Popen(cmd))
            print(f"    started PID {procs[-1].pid} for {name}")
        for p in procs:
            p.wait()
        all_records = []
        for tmp in tmp_paths:
            try:
                with open(tmp) as f:
                    all_records.extend(_json.load(f))
            except Exception:
                pass
        _save_results(all_records, args.out)
        # Fall through to print summary from merged records
        summary = {}
        for mode in modes:
            summary[mode] = {}
            for name in model_names:
                recs = [r for r in all_records if r["model"] == name and r["mode"] == mode]
                if recs:
                    summary[mode][name] = {
                        k: float(np.mean([r[k] for r in recs]))
                        for k in ("rho", "rmse", "mae", "q2")
                    }
    else:
        # ------------------------------------------------------------------
        # Sequential mode (default)
        # ------------------------------------------------------------------
        all_records = []
        summary = {}
        for mode in modes:
            summary[mode] = {}
            for name in model_names:
                cls = MODEL_REGISTRY.get(name)
                if cls is None:
                    print(f"Unknown model '{name}', skipping.")
                    continue
                records = run_single_emitter(cls, solutions, mean_tp,
                                             generations, test_gens, mode=mode)
                all_records.extend(records)
                if records:
                    summary[mode][name] = {
                        k: float(np.mean([r[k] for r in records]))
                        for k in ("rho", "rmse", "mae", "q2")
                    }
                _save_results(all_records, args.out)

    print("\n" + "="*60)
    print("  FINAL SUMMARY  (means across test generations)")
    print("="*60)
    for mode in modes:
        print(f"\n  Mode: {mode}")
        print(f"  {'Model':<20s}  {'rho':>8}  {'rmse':>8}  {'mae':>8}  {'q2':>8}")
        for name, met in sorted(summary[mode].items(),
                                 key=lambda kv: -kv[1]["rho"]):
            print(f"  {name:<20s}  {met['rho']:>+8.4f}  {met['rmse']:>8.4f}  "
                  f"{met['mae']:>8.4f}  {met['q2']:>+8.4f}")
    print()
    print("  Thesis baseline (XGBoost/MLP same-gen oracle, single emitter):  rho~0.0")
    print("  If any model scores consistently above rho~0.1, spatial structure")
    print("  is partially learnable and the ICC claim should be model-scoped.")
    print(f"\n  Full results saved to: {args.out}")


if __name__ == "__main__":
    main()
