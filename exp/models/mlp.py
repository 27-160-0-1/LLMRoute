# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""M3 — MLP variants.

variant "mlp":
    dense 81 (StandardScaler) + TruncatedSVD(word,192) + TruncatedSVD(char,192)
    sklearn MLPRegressor multi-output 6-head (score 3 + log-cost 3).
    Grid: hidden {(256,128),(512,256)} x alpha {1e-4,1e-3}; selected by OOF score MSE.
variant "mlp-ens":
    same architecture (best hp from "mlp"), seed {42,43,44} mean ensemble.

Outputs per variant:
    build/preds/{variant}-train.npz  score (1760,3), cost (1760,3)  -- 5-fold OOF
    build/preds/{variant}-dev.npz    score (880,3),  cost (880,3)   -- fit on full train
    build/preds/{variant}-summary.json

Reproducible: seed 42 fixed (ensemble adds 43, 44). fold_id = np.arange(N) % 5.
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATS = ROOT / "build" / "feats"
PREDS = ROOT / "build" / "preds"

SEED = 42
N_FOLDS = 5
SVD_DIM = 192
HIDDEN_GRID = [(256, 128), (512, 256)]
ALPHA_GRID = [1e-4, 1e-3]
ENS_SEEDS = [42, 43, 44]

MODEL_NAMES = ["ax31-light", "ax31", "axk1-think"]


def load_features():
    """StandardScaler(dense) + SVD(word,192) + SVD(char,192); all fit on train."""
    dense_tr = np.load(FEATS / "dense-train.npy")
    dense_dev = np.load(FEATS / "dense-dev.npy")
    word_tr = sp.load_npz(FEATS / "word-train.npz")
    word_dev = sp.load_npz(FEATS / "word-dev.npz")
    char_tr = sp.load_npz(FEATS / "char-train.npz")
    char_dev = sp.load_npz(FEATS / "char-dev.npz")

    scaler = StandardScaler().fit(dense_tr)
    svd_w = TruncatedSVD(n_components=SVD_DIM, random_state=SEED).fit(word_tr)
    svd_c = TruncatedSVD(n_components=SVD_DIM, random_state=SEED).fit(char_tr)

    x_tr = np.hstack(
        [scaler.transform(dense_tr), svd_w.transform(word_tr), svd_c.transform(char_tr)]
    )
    x_dev = np.hstack(
        [scaler.transform(dense_dev), svd_w.transform(word_dev), svd_c.transform(char_dev)]
    )
    return x_tr, x_dev


def load_targets(split: str):
    t = np.load(FEATS / f"targets-{split}.npz")
    return np.asarray(t["scores"], dtype=np.float64), np.asarray(t["costs"], dtype=np.float64)


def make_mlp(hidden, alpha, seed):
    return MLPRegressor(
        hidden_layer_sizes=hidden,
        alpha=alpha,
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )


def fit_predict(x_fit, y_fit, x_eval_list, hidden, alpha, seeds):
    """Mean over seeds of MLP predictions; returns one array per x_eval."""
    outs = [np.zeros((x.shape[0], y_fit.shape[1])) for x in x_eval_list]
    for seed in seeds:
        model = make_mlp(hidden, alpha, seed).fit(x_fit, y_fit)
        for out, x in zip(outs, x_eval_list):
            out += model.predict(x)
    return [out / len(seeds) for out in outs]


def run_oof(x, y, fold_id, hidden, alpha, seeds):
    oof = np.zeros_like(y)
    for k in range(N_FOLDS):
        tr = fold_id != k
        te = ~tr
        (oof[te],) = fit_predict(x[tr], y[tr], [x[te]], hidden, alpha, seeds)
    return oof


def postprocess(raw):
    """raw (N,6) -> score (N,3) clipped [0,1], cost (N,3) positive + monotonic."""
    score = np.clip(raw[:, :3], 0.0, 1.0)
    cost = np.exp(raw[:, 3:])
    cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1 + 1e-12))
    cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1 + 1e-12))
    return score, cost


def metrics(raw_oof, scores, log_costs):
    score_pred, cost_pred = postprocess(raw_oof)
    log_cost_pred = np.log(cost_pred)
    return {
        "score_mse_per_model": {
            name: float(np.mean((score_pred[:, j] - scores[:, j]) ** 2))
            for j, name in enumerate(MODEL_NAMES)
        },
        "score_mae_per_model": {
            name: float(np.mean(np.abs(score_pred[:, j] - scores[:, j])))
            for j, name in enumerate(MODEL_NAMES)
        },
        "logcost_mse_per_model": {
            name: float(np.mean((log_cost_pred[:, j] - log_costs[:, j]) ** 2))
            for j, name in enumerate(MODEL_NAMES)
        },
        "score_mse": float(np.mean((score_pred - scores) ** 2)),
        "score_mae": float(np.mean(np.abs(score_pred - scores))),
        "logcost_mse": float(np.mean((log_cost_pred - log_costs) ** 2)),
    }


def save_variant(variant, raw_train, raw_dev, summary):
    PREDS.mkdir(parents=True, exist_ok=True)
    s_tr, c_tr = postprocess(raw_train)
    s_dev, c_dev = postprocess(raw_dev)
    np.savez(PREDS / f"{variant}-train.npz", score=s_tr, cost=c_tr)
    np.savez(PREDS / f"{variant}-dev.npz", score=s_dev, cost=c_dev)
    with open(PREDS / f"{variant}-summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[saved] {variant}: train {s_tr.shape}, dev {s_dev.shape}", flush=True)


def main() -> None:
    t0 = time.time()
    x_tr, x_dev = load_features()
    scores_tr, costs_tr = load_targets("train")
    log_costs_tr = np.log(costs_tr)
    y = np.hstack([scores_tr, log_costs_tr])  # (N,6)
    n = len(y)
    fold_id = np.arange(n) % N_FOLDS
    print(f"[feat] X_train {x_tr.shape}, X_dev {x_dev.shape} ({time.time()-t0:.1f}s)", flush=True)

    # ---- grid search on OOF score MSE (single seed 42) ----
    grid_results = []
    oof_cache = {}
    for hidden, alpha in product(HIDDEN_GRID, ALPHA_GRID):
        t1 = time.time()
        oof = run_oof(x_tr, y, fold_id, hidden, alpha, [SEED])
        score_pred, _ = postprocess(oof)
        score_mse = float(np.mean((score_pred - scores_tr) ** 2))
        oof_cache[(hidden, alpha)] = oof
        grid_results.append(
            {"hidden": list(hidden), "alpha": alpha, "oof_score_mse": score_mse,
             "fit_seconds": round(time.time() - t1, 1)}
        )
        print(f"[grid] hidden={hidden} alpha={alpha} oof_score_mse={score_mse:.5f} "
              f"({time.time()-t1:.1f}s)", flush=True)

    best = min(grid_results, key=lambda r: r["oof_score_mse"])
    best_hidden = tuple(best["hidden"])
    best_alpha = best["alpha"]
    print(f"[best] hidden={best_hidden} alpha={best_alpha}", flush=True)

    # ---- variant "mlp" ----
    raw_train = oof_cache[(best_hidden, best_alpha)]
    (raw_dev,) = fit_predict(x_tr, y, [x_dev], best_hidden, best_alpha, [SEED])
    summary = {
        "variant": "mlp",
        "features": f"dense81(StandardScaler)+SVD{SVD_DIM}(word)+SVD{SVD_DIM}(char), train-fit",
        "model": "MLPRegressor 6-head (score3 + logcost3)",
        "hyperparameters": {
            "hidden_layer_sizes": list(best_hidden), "alpha": best_alpha,
            "max_iter": 400, "early_stopping": True, "validation_fraction": 0.15,
            "random_state": SEED,
        },
        "grid": grid_results,
        "oof_metrics": metrics(raw_train, scores_tr, log_costs_tr),
    }
    save_variant("mlp", raw_train, raw_dev, summary)

    # ---- variant "mlp-ens" ----
    t2 = time.time()
    raw_train_ens = run_oof(x_tr, y, fold_id, best_hidden, best_alpha, ENS_SEEDS)
    (raw_dev_ens,) = fit_predict(x_tr, y, [x_dev], best_hidden, best_alpha, ENS_SEEDS)
    summary_ens = {
        "variant": "mlp-ens",
        "features": summary["features"],
        "model": f"mean of {len(ENS_SEEDS)} MLPRegressor (seeds {ENS_SEEDS})",
        "hyperparameters": dict(summary["hyperparameters"], random_state=ENS_SEEDS),
        "oof_metrics": metrics(raw_train_ens, scores_tr, log_costs_tr),
        "fit_seconds": round(time.time() - t2, 1),
    }
    save_variant("mlp-ens", raw_train_ens, raw_dev_ens, summary_ens)

    print(f"[done] total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
