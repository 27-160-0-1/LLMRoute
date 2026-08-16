# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""P1/P4: direct promotion-gain (delta) prediction heads (LightGBM).

Motivation: allocation needs the promotion gain Delta_s and its ranking, not
absolute scores (RouteLLM win-rate framing adapted to this challenge).

Variants:
  delta           3 heads: s_light, D_ax31 = s_ax31 - s_light,
                  D_k1 = s_k1 - s_light. Reconstruct
                  score = [s_light, s_light + D_ax31, s_light + D_k1], clip 0..1.
  delta-costaware P4: delta targets adjusted to
                  D_j - LAMBDA * log1p(cost_j / cost_light)   (LAMBDA = 0.002,
                  true train costs). The adjustment is NOT undone after
                  prediction; the reconstructed "score" is a cost-aware ranking
                  signal (ties broken toward cheaper models).

Score/delta heads: LightGBM regression, num_leaves=31, n_estimators=400,
lr=0.05, min_child_samples=15, seed=42, deterministic.

Cost head (shared by both variants): LGBM regression on log(in_tok) (3 heads)
and log(out_tok) (3 heads), same hyperparameters;
cost = exp(lin)*rin/1e6 + exp(lout)*rout/1e6, then monotone correction
cost[:,1] >= cost[:,0]*(1+1e-12), cost[:,2] >= cost[:,1]*(1+1e-12).

Features: dense(81) + TruncatedSVD(word,128) + TruncatedSVD(char,128),
SVD fit on train only (reuses build/lgbm-work cache when present).

OOF protocol: 5 folds, fold_id = np.arange(N) % 5; dev preds from full-train
models.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import lightgbm as lgb
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[2]
FEATS = ROOT / "build" / "feats"
PREDS = ROOT / "build" / "preds"
SVD_CACHE = ROOT / "build" / "lgbm-work"

SEED = 42
N_FOLDS = 5
MODELS = ["ax31-light", "ax31", "axk1-think"]
RATES = np.array([[1.0, 4.0], [2.127, 8.509], [6.565, 26.260]])  # (rin, rout)
LAMBDA = 0.002  # cost-aware delta adjustment coefficient (P4)

HP = dict(num_leaves=31, n_estimators=400, learning_rate=0.05,
          min_child_samples=15)

LGB_PARAMS = dict(
    objective="regression",
    learning_rate=HP["learning_rate"],
    num_leaves=HP["num_leaves"],
    min_data_in_leaf=HP["min_child_samples"],
    seed=SEED,
    verbosity=-1,
    force_col_wise=True,
    deterministic=True,
    num_threads=4,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_features():
    """dense81 + SVD128(word) + SVD128(char); reuse lgbm-work cache if present."""
    ctr, cde = SVD_CACHE / "svd-train.npy", SVD_CACHE / "svd-dev.npy"
    if ctr.exists() and cde.exists():
        log("features: using cached SVD features from build/lgbm-work")
        return np.load(ctr), np.load(cde)
    log("features: fitting TruncatedSVD(128) on train word/char...")
    dense = {s: np.load(FEATS / f"dense-{s}.npy") for s in ("train", "dev")}
    word = {s: sp.load_npz(FEATS / f"word-{s}.npz").tocsr() for s in ("train", "dev")}
    char = {s: sp.load_npz(FEATS / f"char-{s}.npz").tocsr() for s in ("train", "dev")}
    svd_w = TruncatedSVD(n_components=128, random_state=SEED).fit(word["train"])
    svd_c = TruncatedSVD(n_components=128, random_state=SEED).fit(char["train"])
    out = {}
    for s in ("train", "dev"):
        out[s] = np.hstack(
            [dense[s], svd_w.transform(word[s]), svd_c.transform(char[s])])
    SVD_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(ctr, out["train"])
    np.save(cde, out["dev"])
    return out["train"], out["dev"]


def run_head(X, X_dev, y, fids):
    """One regression head: OOF preds (5-fold) + dev preds (full-train fit)."""
    oof = np.zeros(len(y))
    for f in range(N_FOLDS):
        tr = fids != f
        ds = lgb.Dataset(X[tr], label=y[tr],
                         params=dict(min_data_in_leaf=HP["min_child_samples"],
                                     verbosity=-1))
        bst = lgb.train(LGB_PARAMS, ds, num_boost_round=HP["n_estimators"])
        oof[~tr] = bst.predict(X[~tr])
    ds = lgb.Dataset(X, label=y,
                     params=dict(min_data_in_leaf=HP["min_child_samples"],
                                 verbosity=-1))
    bst = lgb.train(LGB_PARAMS, ds, num_boost_round=HP["n_estimators"])
    return oof, bst.predict(X_dev)


def monotone_cost(cost: np.ndarray) -> np.ndarray:
    cost = cost.copy()
    cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1 + 1e-12))
    cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1 + 1e-12))
    return cost


def cost_from_tokens(log_in, log_out):
    cost = (np.exp(log_in) * RATES[None, :, 0] / 1e6
            + np.exp(log_out) * RATES[None, :, 1] / 1e6)
    return np.maximum(cost, 1e-9)


def reconstruct(base, d1, d2):
    """score matrix from light head + two delta heads, clipped to [0,1]."""
    return np.clip(np.stack([base, base + d1, base + d2], axis=1), 0.0, 1.0)


def metrics(score_pred, score_true, cost_pred, cost_true):
    out = {"score_mse": {}, "score_mae": {}, "logcost_mse": {}}
    for j, m in enumerate(MODELS):
        out["score_mse"][m] = float(np.mean((score_pred[:, j] - score_true[:, j]) ** 2))
        out["score_mae"][m] = float(np.mean(np.abs(score_pred[:, j] - score_true[:, j])))
        out["logcost_mse"][m] = float(
            np.mean((np.log(cost_pred[:, j]) - np.log(cost_true[:, j])) ** 2))
    out["score_mse_mean"] = float(np.mean(list(out["score_mse"].values())))
    out["logcost_mse_mean"] = float(np.mean(list(out["logcost_mse"].values())))
    return out


def main() -> None:
    PREDS.mkdir(parents=True, exist_ok=True)
    X, X_dev = get_features()
    targ = dict(np.load(FEATS / "targets-train.npz"))
    scores = targ["scores"].astype(float)
    costs_true = targ["costs"].astype(float)
    n, m = X.shape[0], X_dev.shape[0]
    fids = np.arange(n) % N_FOLDS

    # ---- cost heads (shared): log token regressions -------------------------
    log_in = np.log(np.maximum(targ["in_tokens"].astype(float), 1.0))
    log_out = np.log(np.maximum(targ["out_tokens"].astype(float), 1.0))
    lin_oof, lin_dev = np.zeros((n, 3)), np.zeros((m, 3))
    lout_oof, lout_dev = np.zeros((n, 3)), np.zeros((m, 3))
    for j in range(3):
        t0 = time.time()
        lin_oof[:, j], lin_dev[:, j] = run_head(X, X_dev, log_in[:, j], fids)
        lout_oof[:, j], lout_dev[:, j] = run_head(X, X_dev, log_out[:, j], fids)
        log(f"cost heads {MODELS[j]} ({time.time() - t0:.0f}s)")
    cost_oof = monotone_cost(cost_from_tokens(lin_oof, lout_oof))
    cost_dev = monotone_cost(cost_from_tokens(lin_dev, lout_dev))

    # ---- score/delta targets ------------------------------------------------
    s_light = scores[:, 0]
    d_ax31 = scores[:, 1] - s_light
    d_k1 = scores[:, 2] - s_light
    # P4 cost-aware adjustment (true train costs, light as reference)
    adj_ax31 = LAMBDA * np.log1p(costs_true[:, 1] / costs_true[:, 0])
    adj_k1 = LAMBDA * np.log1p(costs_true[:, 2] / costs_true[:, 0])

    # base head is shared by both variants
    t0 = time.time()
    base_oof, base_dev = run_head(X, X_dev, s_light, fids)
    log(f"score head s_light ({time.time() - t0:.0f}s)")

    variants = {
        "delta": (d_ax31, d_k1),
        "delta-costaware": (d_ax31 - adj_ax31, d_k1 - adj_k1),
    }
    results = {}
    for tag, (y1, y2) in variants.items():
        t0 = time.time()
        d1_oof, d1_dev = run_head(X, X_dev, y1, fids)
        d2_oof, d2_dev = run_head(X, X_dev, y2, fids)
        log(f"{tag}: delta heads ({time.time() - t0:.0f}s)")
        score_oof = reconstruct(base_oof, d1_oof, d2_oof)
        score_dev = reconstruct(base_dev, d1_dev, d2_dev)
        np.savez(PREDS / f"{tag}-train.npz", score=score_oof, cost=cost_oof)
        np.savez(PREDS / f"{tag}-dev.npz", score=score_dev, cost=cost_dev)
        oof_metrics = metrics(score_oof, scores, cost_oof, costs_true)
        summary = dict(
            variant=tag,
            targets=("s_light, D_ax31=s_ax31-s_light, D_k1=s_k1-s_light"
                     + ("" if tag == "delta" else
                        f" with D_j := D_j - {LAMBDA}*log1p(cost_j/cost_light) "
                        "(true train costs); adjustment NOT undone at predict "
                        "time -- score is a cost-aware ranking signal")),
            hyperparams=dict(HP, seed=SEED, deterministic=True,
                             force_col_wise=True, num_threads=4,
                             features="dense81+SVD128(word)+SVD128(char)",
                             folds=N_FOLDS),
            cost_head=("LGBM regression log(in_tok) + log(out_tok), same HP; "
                       "cost=exp(lin)*rin/1e6+exp(lout)*rout/1e6, monotone"),
            cost_aware_lambda=(None if tag == "delta" else LAMBDA),
            oof=oof_metrics,
            notes=("base s_light head and cost heads shared across delta / "
                   "delta-costaware; only delta-head targets differ. "
                   "score reconstructed as [s, s+D1, s+D2] then clipped 0..1."),
        )
        with open(PREDS / f"{tag}-summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        results[tag] = oof_metrics
        log(f"{tag}: oof score_mse={oof_metrics['score_mse_mean']:.5f} "
            f"logcost_mse={oof_metrics['logcost_mse_mean']:.5f}")
    return results


if __name__ == "__main__":
    main()
