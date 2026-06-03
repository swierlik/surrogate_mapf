"""Experiment 8: Power-iteration Markov-chain features for within-emitter oracle.

Hypothesis: replacing raw edge weights with a per-cell stationary distribution
(derived from treating the edge-weight graph as a Markov chain) gives the
surrogate a richer, globally-consistent signal and improves within-emitter ranking.

Feature construction (per solution):
  1. Normalize edge weights to [0.1, 100] (same transform as the simulator).
  2. Convert to softmax transition probs: p(i→j) = softmax(-w_ij) over outgoing
     edges from cell i  (lower cost edge → higher probability).
  3. Power-iterate a uniform initial distribution for N_STEPS steps.
  4. Output features = [stationary_dist(948) | normalized_wait_costs(948)] = 1896-dim.
     For CNN: map each 948-dim part back to a (H, W) spatial image → 3-channel input.

Test design mirrors Experiment 7 (per-emitter sliding window and/or same-gen oracle).

Usage:
    python -m experiments.08_power_iter_features
    python -m experiments.08_power_iter_features --data-dir results/baseline_1em
    python -m experiments.08_power_iter_features --models mlp,cnn
    python -m experiments.08_power_iter_features --models mlp --mode sliding --n-steps 50
"""

import argparse
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
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_TEST_GENS = [20, 40, 60, 80, 99]
N_STEPS_DEFAULT  = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Markov-chain topology (precomputed once, then cached)
# ---------------------------------------------------------------------------

_TOPO_CACHE = None


def _get_topo():
    """Return (reshaper, src_idx, tgt_idx, gather) cached after first call.

    gather: scipy csr_matrix (n_valid, n_edges) — scatter edge contributions
            to target cells.  pi_new = gather.dot(contrib.T).T
    """
    global _TOPO_CACHE
    if _TOPO_CACHE is not None:
        return _TOPO_CACHE

    from src.utils.reshape import SolutionReshaper
    r = SolutionReshaper.get()

    dirs = np.array([(0, 1), (-1, 0), (0, -1), (1, 0)])

    cell_to_idx = np.full((r.h, r.w), -1, dtype=np.int32)
    for i, (row, col) in enumerate(zip(r.valid_rows, r.valid_cols)):
        cell_to_idx[row, col] = i

    tgt_rows = r.edge_rows + dirs[r.edge_chans, 0]
    tgt_cols = r.edge_cols + dirs[r.edge_chans, 1]
    src_idx  = cell_to_idx[r.edge_rows, r.edge_cols].astype(np.int32)  # (n_edges,)
    tgt_idx  = cell_to_idx[tgt_rows, tgt_cols].astype(np.int32)        # (n_edges,)

    # gather[j, k] = 1  iff  edge k points to cell j
    # pi_new = gather.dot(contrib.T).T  gives  pi_new[n, j] = Σ_k contrib[n,k] * gather[j,k]
    gather = csr_matrix(
        (np.ones(r.n_edges, dtype=np.float32),
         (tgt_idx, np.arange(r.n_edges, dtype=np.int32))),
        shape=(r.n_valid, r.n_edges),
    )

    _TOPO_CACHE = (r, src_idx, tgt_idx, gather)
    return _TOPO_CACHE


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _batch_normalize(X_raw, n_v, lb=0.1, ub=100.0):
    """Vectorized per-sample, per-part min-max normalization to [lb, ub].

    Matches normalize_solution() from src.simulator.evaluate but processes
    the entire batch at once without Python loops over samples.
    """
    X = X_raw.astype(np.float32).copy()
    for s, e in [(0, n_v), (n_v, X.shape[1])]:
        part = X[:, s:e]
        mn   = part.min(axis=1, keepdims=True)
        mx   = part.max(axis=1, keepdims=True)
        rng  = mx - mn
        X[:, s:e] = np.where(
            rng < 1e-3,
            np.clip(part, lb, ub),
            lb + (part - mn) * (ub - lb) / (rng + 1e-8),
        )
    return X


def power_iter_features(X_raw, n_steps=N_STEPS_DEFAULT):
    """Convert raw (N, 4074) solutions to (N, 1896) power-iteration features.

    Returns:
        feats: np.float32 (N, 1896) = [stationary_dist(948) | wait_01(948)]
    """
    r, src_idx, tgt_idx, gather = _get_topo()
    n_valid = r.n_valid   # 948
    N       = len(X_raw)

    # 1. Normalize to [0.1, 100] (same as simulator)
    X_norm   = _batch_normalize(X_raw, n_valid)
    wait_norm = X_norm[:, :n_valid]   # (N, 948) in [0.1, 100]
    edge_w    = X_norm[:, n_valid:]   # (N, 3126)

    # 2. Transition probabilities: softmax(-w) per source cell over directions
    #    slot[n, cell, d] = -edge_weight of the edge leaving `cell` in direction d,
    #    or -inf if that edge doesn't exist.
    slot = np.full((N, n_valid, 4), -np.inf, dtype=np.float32)
    slot[:, src_idx, r.edge_chans] = -edge_w  # lower cost → larger logit → higher prob

    slot_max = slot.max(axis=2, keepdims=True)
    exp_s    = np.exp(slot - slot_max)
    exp_s[slot == -np.inf] = 0.0              # zero out missing edges after exp
    denom    = exp_s.sum(axis=2, keepdims=True) + 1e-8
    probs    = exp_s / denom                  # (N, 948, 4)

    # Flatten to per-edge probabilities (N, 3126)
    prob = probs[:, src_idx, r.edge_chans]    # (N, n_edges)

    # 3. Power iteration: π_{t+1}[j] = Σ_k π_t[src_k] * prob[src_k→j]
    pi = np.full((N, n_valid), 1.0 / n_valid, dtype=np.float32)
    for _ in range(n_steps):
        contrib = pi[:, src_idx] * prob                        # (N, n_edges)
        pi      = np.asarray(gather.dot(contrib.T)).T          # (N, n_valid)
        pi     /= pi.sum(axis=1, keepdims=True) + 1e-8         # re-normalize

    # 4. Normalize wait costs to [0, 1] per sample
    wmin   = wait_norm.min(axis=1, keepdims=True)
    wmax   = wait_norm.max(axis=1, keepdims=True)
    wait01 = (wait_norm - wmin) / (wmax - wmin + 1e-8)

    return np.concatenate([pi, wait01], axis=1).astype(np.float32)   # (N, 1896)


def _feats_to_spatial(feats):
    """Map (N, 1896) feature array to (N, 3, H, W) CNN tensor.

    Channels:
      0 — stationary distribution (per-cell Markov occupancy)
      1 — normalized wait costs
      2 — obstacle mask (1 = obstacle, same for all samples)
    """
    r = _get_topo()[0]
    N  = len(feats)
    pi    = feats[:, :r.n_valid]   # (N, 948)
    wait  = feats[:, r.n_valid:]   # (N, 948)

    stat_sp = np.zeros((N, r.h, r.w), dtype=np.float32)
    wait_sp = np.zeros((N, r.h, r.w), dtype=np.float32)
    stat_sp[:, r.valid_rows, r.valid_cols] = pi
    wait_sp[:, r.valid_rows, r.valid_cols] = wait

    obs = np.broadcast_to(r.obstacle_mask[None], (N, r.h, r.w)).copy()

    return np.stack([stat_sp, wait_sp, obs], axis=1)  # (N, 3, H, W)


# ---------------------------------------------------------------------------
# Shared training utilities (mirrors Experiment 7)
# ---------------------------------------------------------------------------

def _spearman(y_true, y_pred):
    if len(y_true) < 3:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = spearmanr(y_true, y_pred)
    return float(r) if not math.isnan(r) else 0.0


def _metrics(y_true, y_pred):
    if len(y_true) < 3:
        nan = float("nan")
        return dict(rho=nan, rmse=nan, mae=nan, q2=nan)
    rho      = _spearman(y_true, y_pred)
    residuals = y_true - y_pred
    rmse     = float(np.sqrt(np.mean(residuals ** 2)))
    mae      = float(np.mean(np.abs(residuals)))
    ss_res   = float(np.sum(residuals ** 2))
    ss_tot   = float(np.sum((y_true - y_true.mean()) ** 2))
    q2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(rho=rho, rmse=rmse, mae=mae, q2=q2)


def _train_loop(model, loader, X_val_th, y_val_th, max_epochs, patience, lr, wd):
    opt       = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
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
# Neural network architectures
# ---------------------------------------------------------------------------

class _MLPNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


class _CNNNet(nn.Module):
    """Same depth as LargeCNN in Experiment 7 but with configurable in_channels."""
    def __init__(self, in_channels=3):
        super().__init__()
        def block(ci, co, k=3):
            return nn.Sequential(
                nn.Conv2d(ci, co, k, padding=k // 2),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True), nn.Dropout2d(0.2))
        self.features = nn.Sequential(
            block(in_channels, 32), block(32, 64), block(64, 128),
            block(128, 128), block(128, 64))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.4), nn.Linear(32, 1))
    def forward(self, x): return self.head(self.features(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class PowerIterMLPModel:
    """MLP on [stationary_dist | wait_costs] (1896-dim flat input)."""
    name = "PowerIterMLP"

    def __init__(self, n_steps=N_STEPS_DEFAULT):
        self.n_steps = n_steps
        self._stats  = None

    def fit(self, X, y):
        feats = power_iter_features(X, self.n_steps).astype(np.float32)
        y     = y.astype(np.float32)

        n  = len(feats)
        idx = np.random.permutation(n)
        sp  = int(0.85 * n)
        Xtr, Xv = feats[idx[:sp]], feats[idx[sp:]]
        ytr, yv = y[idx[:sp]], y[idx[sp:]]

        mu_x = Xtr.mean(0, keepdims=True)
        sd_x = Xtr.std(0, keepdims=True) + 1e-8
        mu_y = float(ytr.mean()); sd_y = float(ytr.std()) + 1e-8
        self._stats = (mu_x, sd_x, mu_y, sd_y)

        Xtr = (Xtr - mu_x) / sd_x;  Xv = (Xv - mu_x) / sd_x
        ytr = (ytr - mu_y) / sd_y;  yv = (yv - mu_y) / sd_y

        self.model = _MLPNet(feats.shape[1]).to(DEVICE)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
            batch_size=128, shuffle=True)
        _train_loop(self.model, loader,
                    torch.from_numpy(Xv), torch.from_numpy(yv),
                    max_epochs=150, patience=15, lr=1e-3, wd=1e-4)

    def predict(self, X):
        mu_x, sd_x, mu_y, sd_y = self._stats
        feats = (power_iter_features(X, self.n_steps).astype(np.float32) - mu_x) / sd_x
        with torch.no_grad():
            p = self.model(torch.from_numpy(feats).to(DEVICE)).cpu().numpy()
        return p * sd_y + mu_y


class PowerIterCNNModel:
    """CNN on spatial [stationary_dist | wait_costs | obstacle_mask] (3 channels)."""
    name = "PowerIterCNN"

    def __init__(self, n_steps=N_STEPS_DEFAULT):
        self.n_steps = n_steps
        self._stats  = None

    def _to_tensor(self, X):
        feats = power_iter_features(X, self.n_steps)
        return _feats_to_spatial(feats)   # (N, 3, H, W)

    def fit(self, X, y):
        y = y.astype(np.float32)
        T = self._to_tensor(X)            # (N, 3, H, W)

        mu_x = T.mean(axis=(0, 2, 3), keepdims=True)
        sd_x = T.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        mu_y = float(y.mean()); sd_y = float(y.std()) + 1e-8
        self._stats = (mu_x, sd_x, mu_y, sd_y)

        T  = (T - mu_x) / sd_x
        ys = (y - mu_y) / sd_y

        n   = len(T)
        idx = np.random.permutation(n)
        sp  = int(0.85 * n)
        Ttr, Tv = T[idx[:sp]], T[idx[sp:]]
        ytr, yv = ys[idx[:sp]], ys[idx[sp:]]

        self.model = _CNNNet(in_channels=3).to(DEVICE)
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
# Per-emitter evaluation loop (identical logic to Experiment 7)
# ---------------------------------------------------------------------------

def run_single_emitter(model_cls, solutions, mean_tp, generations,
                       test_gens, mode="sliding", rng_seed=42, **model_kwargs):
    rng  = np.random.default_rng(rng_seed)
    name = model_cls.name
    tag  = f"[{name}/{mode}]"

    gen_records = []
    for g in test_gens:
        gen_mask = generations == g
        if gen_mask.sum() < 5:
            continue

        all_te_idx = np.where(gen_mask)[0]

        if mode == "sliding":
            tr_mask   = generations < g
            train_idx = np.where(tr_mask)[0]
            if len(train_idx) < 10:
                continue
            test_idx = all_te_idx

        else:  # oracle: 80% same-gen train, 20% test
            perm    = rng.permutation(len(all_te_idx))
            n_train = max(int(len(all_te_idx) * 0.8), 5)
            train_idx = all_te_idx[perm[:n_train]]
            test_idx  = all_te_idx[perm[n_train:]]

        t0 = time.time()
        m  = model_cls(**model_kwargs)
        m.fit(solutions[train_idx], mean_tp[train_idx])
        preds = m.predict(solutions[test_idx])
        met   = _metrics(mean_tp[test_idx], preds)
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
        mean_rho  = float(np.nanmean([r["rho"]  for r in gen_records]))
        mean_rmse = float(np.nanmean([r["rmse"] for r in gen_records]))
        mean_mae  = float(np.nanmean([r["mae"]  for r in gen_records]))
        mean_q2   = float(np.nanmean([r["q2"]   for r in gen_records]))
        print(f"{tag}/MEAN  rho={mean_rho:+.4f}  rmse={mean_rmse:.4f}  "
              f"mae={mean_mae:.4f}  q2={mean_q2:+.4f}", flush=True)

    return gen_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "mlp": PowerIterMLPModel,
    "cnn": PowerIterCNNModel,
}


def _save_results(records, out_path):
    import json
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  [saved → {out_path}]")


def main():
    parser = argparse.ArgumentParser(
        description="Exp 8: Power-iteration Markov features for within-emitter oracle")
    parser.add_argument("--data-dir",  default="results/baseline_1em",
                        help="Single-emitter run directory")
    parser.add_argument("--models",    default="mlp,cnn",
                        help="Comma-separated: mlp, cnn")
    parser.add_argument("--test-gens", default=None,
                        help="Override test generations, e.g. 20,50,80,99")
    parser.add_argument("--mode",      default="both",
                        choices=["sliding", "oracle", "both"])
    parser.add_argument("--n-steps",   default=N_STEPS_DEFAULT, type=int,
                        help="Power-iteration steps (default 30)")
    parser.add_argument("--out",       default="results/exp8_power_iter.json")
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir} ...")
    solutions  = np.load(f"{args.data_dir}/cmaes_solutions.npy")
    log        = pd.read_csv(f"{args.data_dir}/cmaes_log.csv")
    mean_tp    = log["mean_throughput"].values
    generations = log["generation"].values

    max_gen     = int(generations.max())
    default_gens = [g for g in DEFAULT_TEST_GENS if g <= max_gen]
    test_gens   = ([int(g) for g in args.test_gens.split(",")]
                   if args.test_gens else default_gens)

    print(f"  {len(solutions)} candidates, {solutions.shape[1]}-dim, "
          f"{log['generation'].nunique()} generations")
    print(f"  Device:          {DEVICE}")
    print(f"  Test generations:{test_gens}")
    print(f"  Models:          {args.models}")
    print(f"  Mode:            {args.mode}")
    print(f"  Power-iter steps:{args.n_steps}")
    print(f"  Output:          {args.out}")
    print()

    modes       = ["sliding", "oracle"] if args.mode == "both" else [args.mode]
    model_names = [m.strip() for m in args.models.split(",")]

    all_records = []
    summary     = {}

    for mode in modes:
        summary[mode] = {}
        for name in model_names:
            cls = MODEL_REGISTRY.get(name)
            if cls is None:
                print(f"Unknown model '{name}', skipping.")
                continue
            records = run_single_emitter(
                cls, solutions, mean_tp, generations, test_gens,
                mode=mode, n_steps=args.n_steps)
            all_records.extend(records)
            if records:
                summary[mode][name] = {
                    k: float(np.nanmean([r[k] for r in records]))
                    for k in ("rho", "rmse", "mae", "q2")
                }
            _save_results(all_records, args.out)

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY  (means across test generations)")
    print("=" * 60)
    for mode in modes:
        print(f"\n  Mode: {mode}")
        print(f"  {'Model':<22}  {'rho':>8}  {'rmse':>8}  {'mae':>8}  {'q2':>8}")
        for name, met in sorted(summary[mode].items(), key=lambda kv: -kv[1]["rho"]):
            print(f"  {name:<22}  {met['rho']:>+8.4f}  {met['rmse']:>8.4f}  "
                  f"{met['mae']:>8.4f}  {met['q2']:>+8.4f}")
    print()
    print("  Exp-7 baseline (raw MLP/CNN, same-gen oracle, single emitter): rho~0.0")
    print("  If power-iter features improve rho consistently above ~0.1,")
    print("  global Markov structure is learnable and the ICC claim needs nuance.")
    print(f"\n  Full results saved to: {args.out}")


if __name__ == "__main__":
    main()
