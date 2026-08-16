# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""M1 linear model family.

Variants
  ridge-full : Ridge regression, 6 heads (score x3, log-cost x3) on
               dense(81, standardized) + word(65536) + char(65536) hstack.
               Per-target alpha selected from ALPHAS by 5-fold OOF MSE.
               Solved exactly in the dual (Gram + eigendecomposition), which
               is identical to primal Ridge(fit_intercept=True) but far
               faster when n_samples << n_features.
  ridge-iso  : ridge-full scores passed through per-model isotonic
               calibration fit on train OOF (applied to OOF itself -> mild
               leakage, noted in summary). Costs copied from ridge-full.
  logit-cls  : Per-model logistic regression treating score as probability
               (row duplication with sample weights s / 1-s). C selected
               from CS by 5-fold OOF Brier/MSE. Costs from ridge-full.
               liblinear dual solver (fast for n << d; intercept lightly
               regularized by liblinear -- noted in summary).

Contract
  build/preds/{variant}-train.npz  score (N,3) OOF, cost (N,3) OOF
  build/preds/{variant}-dev.npz    score (880,3), cost (880,3), full-train fit
  build/preds/{variant}-summary.json

Reproducible: seed 42, fold_id = np.arange(N) % 5.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATS = ROOT / "build" / "feats"
PREDS = ROOT / "build" / "preds"

SEED = 42
N_FOLDS = 5
ALPHAS = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
CS = [0.1, 1.0, 10.0]
MODELS = ["ax31-light", "ax31", "axk1-think"]
EPS_MONO = 1e-12
EPS_LOG = 1e-9


def load_split(split: str):
    dense = np.load(FEATS / f"dense-{split}.npy")
    word = sp.load_npz(FEATS / f"word-{split}.npz").tocsr()
    char = sp.load_npz(FEATS / f"char-{split}.npz").tocsr()
    return dense, sp.hstack([word, char], format="csr")


def monotone_costs(cost: np.ndarray) -> np.ndarray:
    cost = np.maximum(cost, EPS_LOG)
    cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1.0 + EPS_MONO))
    cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1.0 + EPS_MONO))
    return cost


def metrics(score_pred, cost_pred, y_score, y_cost):
    lc_pred = np.log(np.maximum(cost_pred, EPS_LOG))
    lc_true = np.log(np.maximum(y_cost, EPS_LOG))
    return {
        "score_mse_per_model": [
            float(np.mean((score_pred[:, j] - y_score[:, j]) ** 2)) for j in range(3)
        ],
        "score_mse_mean": float(np.mean((score_pred - y_score) ** 2)),
        "score_mae_per_model": [
            float(np.mean(np.abs(score_pred[:, j] - y_score[:, j]))) for j in range(3)
        ],
        "score_mae_mean": float(np.mean(np.abs(score_pred - y_score))),
        "logcost_mse_per_model": [
            float(np.mean((lc_pred[:, j] - lc_true[:, j]) ** 2)) for j in range(3)
        ],
        "logcost_mse_mean": float(np.mean((lc_pred - lc_true) ** 2)),
    }


def save_variant(name, score_tr, cost_tr, score_dev, cost_dev, summary):
    PREDS.mkdir(parents=True, exist_ok=True)
    np.savez(
        PREDS / f"{name}-train.npz",
        score=np.clip(score_tr, 0.0, 1.0),
        cost=monotone_costs(cost_tr.copy()),
    )
    np.savez(
        PREDS / f"{name}-dev.npz",
        score=np.clip(score_dev, 0.0, 1.0),
        cost=monotone_costs(cost_dev.copy()),
    )
    (PREDS / f"{name}-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )


def dual_ridge_predict(K_tr, K_te_tr, Y_tr, alphas):
    """Exact multi-alpha ridge-with-intercept via kernel centering + eigh.

    K_tr    (m,m)  Gram of training rows
    K_te_tr (t,m)  cross Gram test x train
    Returns {alpha: (t, n_targets) predictions}.
    """

    m = K_tr.shape[0]
    col_mean = K_tr.mean(axis=0)  # mu . x_j
    grand = K_tr.mean()  # mu . mu
    Kc = K_tr - col_mean[None, :] - col_mean[:, None] + grand
    row_mean_te = K_te_tr.mean(axis=1)  # mu . x_test_i
    Kc_te = K_te_tr - row_mean_te[:, None] - col_mean[None, :] + grand

    y_mean = Y_tr.mean(axis=0)
    Yc = Y_tr - y_mean

    lam, V = np.linalg.eigh(Kc)
    lam = np.maximum(lam, 0.0)
    VtY = V.T @ Yc
    out = {}
    for a in alphas:
        dual = V @ (VtY / (lam + a)[:, None])
        out[a] = Kc_te @ dual + y_mean
    return out


def main() -> None:
    t0 = time.time()

    dense_tr, sparse_tr = load_split("train")
    dense_dev, sparse_dev = load_split("dev")
    tgt = np.load(FEATS / "targets-train.npz")
    y_score = tgt["scores"].astype(np.float64)
    y_cost = tgt["costs"].astype(np.float64)
    y_logcost = np.log(np.maximum(y_cost, EPS_LOG))
    Y = np.hstack([y_score, y_logcost])  # (N,6)

    tgt_dev = np.load(FEATS / "targets-dev.npz")
    yd_score = tgt_dev["scores"].astype(np.float64)
    yd_cost = tgt_dev["costs"].astype(np.float64)

    n = dense_tr.shape[0]
    n_dev = dense_dev.shape[0]
    fold_id = np.arange(n) % N_FOLDS

    # ---- Gram of the (fixed) sparse block, computed once ----
    G_tr = np.asarray((sparse_tr @ sparse_tr.T).todense())  # (N,N)
    G_dev_tr = np.asarray((sparse_dev @ sparse_tr.T).todense())  # (880,N)
    print("gram done", round(time.time() - t0, 1), flush=True)

    # =================== ridge-full (dual, exact) ===================
    oof = {a: np.zeros((n, 6)) for a in ALPHAS}
    for k in range(N_FOLDS):
        tr = fold_id != k
        va = fold_id == k
        scaler = StandardScaler().fit(dense_tr[tr])
        Dtr = scaler.transform(dense_tr[tr])
        Dva = scaler.transform(dense_tr[va])
        K_tr = G_tr[np.ix_(tr, tr)] + Dtr @ Dtr.T
        K_va = G_tr[np.ix_(va, tr)] + Dva @ Dtr.T
        preds = dual_ridge_predict(K_tr, K_va, Y[tr], ALPHAS)
        for a in ALPHAS:
            oof[a][va] = preds[a]
    print("ridge oof done", round(time.time() - t0, 1), flush=True)

    best_alpha = []
    for j in range(6):
        mses = {a: float(np.mean((oof[a][:, j] - Y[:, j]) ** 2)) for a in ALPHAS}
        best_alpha.append(min(mses, key=mses.get))

    oof_ridge = np.column_stack([oof[best_alpha[j]][:, j] for j in range(6)])

    # full-train fit -> dev predictions
    scaler_full = StandardScaler().fit(dense_tr)
    D_full = scaler_full.transform(dense_tr)
    D_dev = scaler_full.transform(dense_dev)
    K_full = G_tr + D_full @ D_full.T
    K_dev = G_dev_tr + D_dev @ D_full.T
    preds_dev = dual_ridge_predict(K_full, K_dev, Y, sorted(set(best_alpha)))
    dev_pred = np.zeros((n_dev, 6))
    for j in range(6):
        dev_pred[:, j] = preds_dev[best_alpha[j]][:, j]
    print("ridge dev done", round(time.time() - t0, 1), flush=True)

    rf_score_tr = np.clip(oof_ridge[:, :3], 0.0, 1.0)
    rf_cost_tr = np.exp(oof_ridge[:, 3:])
    rf_score_dev = np.clip(dev_pred[:, :3], 0.0, 1.0)
    rf_cost_dev = np.exp(dev_pred[:, 3:])

    rf_cost_tr_m = monotone_costs(rf_cost_tr.copy())
    rf_cost_dev_m = monotone_costs(rf_cost_dev.copy())

    summary_rf = {
        "variant": "ridge-full",
        "hyperparams": {
            "alphas_grid": ALPHAS,
            "alpha_score": {MODELS[j]: best_alpha[j] for j in range(3)},
            "alpha_logcost": {MODELS[j]: best_alpha[3 + j] for j in range(3)},
            "features": "dense81(std) + word65536 + char65536",
            "solver": "exact dual ridge (kernel centering + eigh), == sklearn Ridge",
            "folds": N_FOLDS,
            "seed": SEED,
        },
        "oof": metrics(rf_score_tr, rf_cost_tr_m, y_score, y_cost),
        "dev": metrics(rf_score_dev, rf_cost_dev_m, yd_score, yd_cost),
        "alpha_oof_mse_grid": {
            str(a): [float(np.mean((oof[a][:, j] - Y[:, j]) ** 2)) for j in range(6)]
            for a in ALPHAS
        },
    }
    save_variant("ridge-full", rf_score_tr, rf_cost_tr, rf_score_dev, rf_cost_dev, summary_rf)
    print("ridge-full done", round(time.time() - t0, 1), flush=True)

    # =================== ridge-iso ===================
    iso_score_tr = np.zeros_like(rf_score_tr)
    iso_score_dev = np.zeros_like(rf_score_dev)
    for j in range(3):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(rf_score_tr[:, j], y_score[:, j])
        iso_score_tr[:, j] = iso.predict(rf_score_tr[:, j])
        iso_score_dev[:, j] = iso.predict(rf_score_dev[:, j])

    summary_iso = {
        "variant": "ridge-iso",
        "base": "ridge-full",
        "hyperparams": summary_rf["hyperparams"],
        "calibration": (
            "per-model IsotonicRegression fit on full train OOF and applied to "
            "the same OOF (mild leakage accepted; dev application is clean)"
        ),
        "oof": metrics(iso_score_tr, rf_cost_tr_m, y_score, y_cost),
        "dev": metrics(iso_score_dev, rf_cost_dev_m, yd_score, yd_cost),
    }
    save_variant(
        "ridge-iso", iso_score_tr, rf_cost_tr, iso_score_dev, rf_cost_dev, summary_iso
    )
    print("ridge-iso done", round(time.time() - t0, 1), flush=True)

    # =================== logit-cls ===================
    def build_matrix(rows_mask=None, split="train"):
        if split == "train":
            dense, sparse = dense_tr, sparse_tr
        else:
            dense, sparse = dense_dev, sparse_dev
        if rows_mask is not None:
            dense, sparse = dense[rows_mask], sparse[rows_mask]
        return dense, sparse

    def fit_logit(dense_scaled, sparse_part, s, C):
        X = sp.hstack([sp.csr_matrix(dense_scaled), sparse_part], format="csr")
        w = np.concatenate([s, 1.0 - s])
        y = np.concatenate([np.ones(len(s)), np.zeros(len(s))])
        Xd = sp.vstack([X, X], format="csr")
        keep = w > 0
        clf = LogisticRegression(
            C=C,
            solver="liblinear",
            dual=True,
            max_iter=2000,
            random_state=SEED,
            tol=1e-5,
        )
        clf.fit(Xd[keep], y[keep], sample_weight=w[keep])
        return clf

    def predict_logit(clf, dense_scaled, sparse_part):
        X = sp.hstack([sp.csr_matrix(dense_scaled), sparse_part], format="csr")
        return clf.predict_proba(X)[:, 1]

    oof_logit = {C: np.zeros((n, 3)) for C in CS}
    for k in range(N_FOLDS):
        tr = fold_id != k
        va = fold_id == k
        scaler = StandardScaler().fit(dense_tr[tr])
        Dtr = scaler.transform(dense_tr[tr])
        Dva = scaler.transform(dense_tr[va])
        for C in CS:
            for j in range(3):
                clf = fit_logit(Dtr, sparse_tr[tr], y_score[tr, j], C)
                oof_logit[C][va, j] = predict_logit(clf, Dva, sparse_tr[va])
        print(f"logit fold {k} done", round(time.time() - t0, 1), flush=True)

    best_C = []
    for j in range(3):
        mses = {
            C: float(np.mean((oof_logit[C][:, j] - y_score[:, j]) ** 2)) for C in CS
        }
        best_C.append(min(mses, key=mses.get))

    lc_score_tr = np.column_stack([oof_logit[best_C[j]][:, j] for j in range(3)])
    lc_score_dev = np.zeros((n_dev, 3))
    for j in range(3):
        clf = fit_logit(D_full, sparse_tr, y_score[:, j], best_C[j])
        lc_score_dev[:, j] = predict_logit(clf, D_dev, sparse_dev)

    summary_lc = {
        "variant": "logit-cls",
        "hyperparams": {
            "C_grid": CS,
            "C_per_model": {MODELS[j]: best_C[j] for j in range(3)},
            "solver": "liblinear dual=True (intercept lightly regularized)",
            "max_iter": 2000,
            "sample_weight_trick": "rows duplicated y=1 w=s / y=0 w=1-s, zero-weight rows dropped",
            "cost_head": "reused from ridge-full",
            "folds": N_FOLDS,
            "seed": SEED,
        },
        "oof": metrics(lc_score_tr, rf_cost_tr_m, y_score, y_cost),
        "dev": metrics(lc_score_dev, rf_cost_dev_m, yd_score, yd_cost),
        "C_oof_mse_grid": {
            str(C): [
                float(np.mean((oof_logit[C][:, j] - y_score[:, j]) ** 2))
                for j in range(3)
            ]
            for C in CS
        },
    }
    save_variant(
        "logit-cls", lc_score_tr, rf_cost_tr, lc_score_dev, rf_cost_dev, summary_lc
    )
    print("logit-cls done", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
