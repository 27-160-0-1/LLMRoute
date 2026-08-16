# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""P2 — EmbedLLM/IRT latent-ability model family.

Framing: score_ij ~ sigmoid(a_j . theta_i + b_j), where theta_i is a latent
item-difficulty vector predicted from prompt features and (a_j, b_j) are
per-LLM discrimination/offset parameters.

Variants
  irt1d   : theta in R^1, theta_i = W x_i (linear head on dense81(std) +
            SVD128(word) + SVD128(char)). Joint soft-label BCE loss over all
            (item, model) cells, minimized with L-BFGS (analytic gradient).
            L2 on W selected from a grid by 5-fold OOF score MSE.
  irt2d   : same with theta in R^2, a_j in R^2 (low-rank interaction test).
  irt-gbm : two-stage. Stage 1 estimates FREE per-item theta* (1-D) by
            alternating Newton optimization (per-model logistic (a_j,b_j)
            refit <-> per-item theta Newton update, theta standardized each
            round). Stage 2 fits LightGBM regression x -> theta*, scores are
            reconstructed as sigmoid(a_j theta_hat + b_j). Both stages use
            only fold-train rows for each OOF fold.

Cost head: copied from build/preds/mlp-ens-{split}.npz (noted in summary).

Contract per variant:
  build/preds/{variant}-train.npz  score (1760,3), cost (1760,3)  -- 5-fold OOF
  build/preds/{variant}-dev.npz    score (880,3),  cost (880,3)   -- full-train fit
  build/preds/{variant}-summary.json

Reproducible: seed 42, fold_id = np.arange(N) % 5.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATS = ROOT / "build" / "feats"
PREDS = ROOT / "build" / "preds"

SEED = 42
N_FOLDS = 5
SVD_DIM = 128
MODEL_NAMES = ["ax31-light", "ax31", "axk1-think"]

LAM_W_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
LAM_A = 1e-4
MAXITER = 1000

# free-theta alternating optimization (stage 1 of irt-gbm)
FREE_ROUNDS = 40
LAM_THETA = 1e-2
# L2 prior on discrimination a_j (added to the summed BCE): without it the
# free-theta fit drives |a| -> ~100 (memorization) and OOF collapses.
LAM_AB_GRID = [10.0, 30.0, 100.0]

LGBM_PARAMS = dict(
    objective="regression",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    random_state=SEED,
    n_jobs=4,
    verbose=-1,
)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_features():
    """dense81(StandardScaler) + SVD128(word) + SVD128(char); all fit on train."""
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
    return np.asarray(t["scores"], dtype=np.float64)


# --------------------------------------------------------------------------
# linear-theta IRT (irt1d / irt2d)
# --------------------------------------------------------------------------

def init_params(X: np.ndarray, S: np.ndarray, d: int):
    """theta init = standardized -mean_j(score) (difficulty); W by ridge; a<0."""
    n, p = X.shape
    t0 = -S.mean(axis=1)
    t0 = (t0 - t0.mean()) / (t0.std() + 1e-12)
    cols = [t0]
    if d == 2:
        c = S[:, 2] - S[:, 0]  # k1-think vs light contrast
        c = (c - c.mean()) / (c.std() + 1e-12)
        cols.append(c)
    T_init = np.column_stack(cols)  # (n, d)

    # ridge solve X W^T = T_init
    A = X.T @ X + 1.0 * np.eye(p)
    W = np.linalg.solve(A, X.T @ T_init).T  # (d, p)

    a = np.full((3, d), 0.0)
    a[:, 0] = -1.0  # higher difficulty -> lower score
    if d == 2:
        a[:, 1] = [-0.5, 0.0, 0.5]
    b = logit(np.clip(S.mean(axis=0), 0.02, 0.98))
    return W, a, b


def pack(W, a, b):
    return np.concatenate([W.ravel(), a.ravel(), b])


def make_objective(X, S, d, lam_w, lam_a):
    n, p = X.shape
    m = S.shape[1]

    def unpack(v):
        W = v[: d * p].reshape(d, p)
        a = v[d * p : d * p + m * d].reshape(m, d)
        b = v[d * p + m * d :]
        return W, a, b

    def fun(v):
        W, a, b = unpack(v)
        theta = X @ W.T  # (n, d)
        Z = theta @ a.T + b  # (n, m)
        bce = float(np.mean(np.logaddexp(0.0, Z) - S * Z))
        loss = bce + lam_w * float(np.sum(W * W)) + lam_a * float(np.sum(a * a))
        Gz = (expit(Z) - S) / (n * m)
        gW = (Gz @ a).T @ X + 2.0 * lam_w * W
        ga = Gz.T @ theta + 2.0 * lam_a * a
        gb = Gz.sum(axis=0)
        return loss, np.concatenate([gW.ravel(), ga.ravel(), gb])

    return fun, unpack


def fit_irt_linear(X, S, d, lam_w):
    W0, a0, b0 = init_params(X, S, d)
    fun, unpack = make_objective(X, S, d, lam_w, LAM_A)
    res = minimize(
        fun,
        pack(W0, a0, b0),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": MAXITER, "maxfun": 2 * MAXITER},
    )
    W, a, b = unpack(res.x)
    return W, a, b, float(res.fun), int(res.nit)


def predict_irt(X, W, a, b):
    return expit((X @ W.T) @ a.T + b)


# --------------------------------------------------------------------------
# free-theta alternating optimization (stage 1 of irt-gbm)
# --------------------------------------------------------------------------

def fit_ab_1d(theta, s, a0, b0, lam_ab, iters=25):
    """Per-model 2-param logistic Newton with soft labels; L2 prior on a."""
    a, b = a0, b0
    for _ in range(iters):
        z = a * theta + b
        p = expit(z)
        g = p - s
        ga = float(g @ theta) + lam_ab * a
        gb = float(g.sum())
        w = p * (1.0 - p)
        haa = float((w * theta * theta).sum()) + lam_ab + 1e-6
        hab = float((w * theta).sum())
        hbb = float(w.sum()) + 1e-6
        det = haa * hbb - hab * hab
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        da = np.clip(da, -1.0, 1.0)
        db = np.clip(db, -1.0, 1.0)
        a -= da
        b -= db
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
    return a, b


def fit_free_theta(S, lam_ab, n_rounds=FREE_ROUNDS, lam_theta=LAM_THETA):
    """Alternating: (a_j, b_j) logistic refit <-> per-item theta Newton.

    theta is re-standardized every round for identifiability. Returns the
    final (theta, a, b) with (a, b) refit after the last standardization.
    """
    n = S.shape[0]
    theta = -S.mean(axis=1)
    theta = (theta - theta.mean()) / (theta.std() + 1e-12)
    a = np.full(3, -1.0)
    b = logit(np.clip(S.mean(axis=0), 0.02, 0.98))

    def bce():
        Z = theta[:, None] * a[None, :] + b[None, :]
        return float(np.mean(np.logaddexp(0.0, Z) - S * Z))

    history = [bce()]
    for _ in range(n_rounds):
        for j in range(3):
            a[j], b[j] = fit_ab_1d(theta, S[:, j], a[j], b[j], lam_ab)
        for _ in range(20):
            Z = theta[:, None] * a[None, :] + b[None, :]
            P = expit(Z)
            G = ((P - S) * a[None, :]).sum(axis=1) + lam_theta * theta
            H = (P * (1.0 - P) * (a**2)[None, :]).sum(axis=1) + lam_theta
            step = np.clip(G / H, -2.0, 2.0)
            theta -= step
            if np.max(np.abs(step)) < 1e-9:
                break
        theta = (theta - theta.mean()) / (theta.std() + 1e-12)
        history.append(bce())
    for j in range(3):
        a[j], b[j] = fit_ab_1d(theta, S[:, j], a[j], b[j], lam_ab)
    history.append(bce())
    return theta, a, b, history


# --------------------------------------------------------------------------
# shared output helpers
# --------------------------------------------------------------------------

def monotone_costs(cost: np.ndarray) -> np.ndarray:
    cost = np.maximum(cost, 1e-9)
    cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1.0 + 1e-12))
    cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1.0 + 1e-12))
    return cost


def score_metrics(pred, true):
    return {
        "score_mse_per_model": {
            name: float(np.mean((pred[:, j] - true[:, j]) ** 2))
            for j, name in enumerate(MODEL_NAMES)
        },
        "score_mae_per_model": {
            name: float(np.mean(np.abs(pred[:, j] - true[:, j])))
            for j, name in enumerate(MODEL_NAMES)
        },
        "score_mse": float(np.mean((pred - true) ** 2)),
        "score_mae": float(np.mean(np.abs(pred - true))),
    }


def save_variant(variant, score_tr, cost_tr, score_dev, cost_dev, summary):
    PREDS.mkdir(parents=True, exist_ok=True)
    np.savez(
        PREDS / f"{variant}-train.npz",
        score=np.clip(score_tr, 0.0, 1.0),
        cost=monotone_costs(np.asarray(cost_tr, dtype=np.float64).copy()),
    )
    np.savez(
        PREDS / f"{variant}-dev.npz",
        score=np.clip(score_dev, 0.0, 1.0),
        cost=monotone_costs(np.asarray(cost_dev, dtype=np.float64).copy()),
    )
    (PREDS / f"{variant}-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[saved] {variant}", flush=True)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants", default="irt1d,irt2d,irt-gbm",
        help="comma list among irt1d,irt2d,irt-gbm",
    )
    args = parser.parse_args()
    wanted = set(args.variants.split(","))

    t0 = time.time()
    x_tr, x_dev = load_features()
    S = load_targets("train")
    S_dev = load_targets("dev")
    n = len(S)
    fold_id = np.arange(n) % N_FOLDS
    print(f"[feat] X_train {x_tr.shape}, X_dev {x_dev.shape} ({time.time()-t0:.1f}s)", flush=True)

    cost_tr = np.load(PREDS / "mlp-ens-train.npz")["cost"]
    cost_dev = np.load(PREDS / "mlp-ens-dev.npz")["cost"]

    # =================== irt1d / irt2d ===================
    for d, variant in ((1, "irt1d"), (2, "irt2d")):
        if variant not in wanted:
            continue
        oof = {lam: np.zeros((n, 3)) for lam in LAM_W_GRID}
        for k in range(N_FOLDS):
            tr = fold_id != k
            va = ~tr
            for lam in LAM_W_GRID:
                W, a, b, _, _ = fit_irt_linear(x_tr[tr], S[tr], d, lam)
                oof[lam][va] = predict_irt(x_tr[va], W, a, b)
        lam_mse = {
            lam: float(np.mean((oof[lam] - S) ** 2)) for lam in LAM_W_GRID
        }
        best_lam = min(lam_mse, key=lam_mse.get)
        score_tr_oof = oof[best_lam]

        W, a, b, final_loss, nit = fit_irt_linear(x_tr, S, d, best_lam)
        score_dev_pred = predict_irt(x_dev, W, a, b)

        summary = {
            "variant": variant,
            "framing": "score_ij ~ sigmoid(a_j . theta_i + b_j), theta_i = W x_i",
            "features": f"dense81(StandardScaler)+SVD{SVD_DIM}(word)+SVD{SVD_DIM}(char), train-fit",
            "hyperparameters": {
                "latent_dim": d,
                "lam_w_grid": LAM_W_GRID,
                "lam_w_best": best_lam,
                "lam_a": LAM_A,
                "optimizer": f"L-BFGS-B joint (W,a,b), maxiter={MAXITER}, analytic grad",
                "init": "theta = standardized -mean_j(score) -> ridge W; a[:,0]=-1; b=logit(mean)",
                "folds": N_FOLDS,
                "seed": SEED,
            },
            "lam_oof_score_mse": {str(l): lam_mse[l] for l in LAM_W_GRID},
            "full_train_fit": {
                "a": a.tolist(),
                "b": b.tolist(),
                "final_loss": final_loss,
                "lbfgs_iters": nit,
            },
            "oof_metrics": score_metrics(np.clip(score_tr_oof, 0, 1), S),
            "dev_metrics": score_metrics(np.clip(score_dev_pred, 0, 1), S_dev),
            "cost_head": "copied from mlp-ens predictions (build/preds/mlp-ens-*.npz)",
        }
        save_variant(variant, score_tr_oof, cost_tr, score_dev_pred, cost_dev, summary)
        print(
            f"[{variant}] best lam_w={best_lam} oof_score_mse={lam_mse[best_lam]:.5f} "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )

    # =================== irt-gbm ===================
    if "irt-gbm" in wanted:
        from lightgbm import LGBMRegressor

        oof_by_lam = {}
        fold_info_by_lam = {}
        for lam_ab in LAM_AB_GRID:
            score_tr_oof = np.zeros((n, 3))
            fold_info = []
            for k in range(N_FOLDS):
                tr = fold_id != k
                va = ~tr
                theta_star, a, b, hist = fit_free_theta(S[tr], lam_ab)
                recon = expit(theta_star[:, None] * a[None, :] + b[None, :])
                recon_mse = float(np.mean((recon - S[tr]) ** 2))
                reg = LGBMRegressor(**LGBM_PARAMS)
                reg.fit(x_tr[tr], theta_star)
                theta_hat = reg.predict(x_tr[va])
                score_tr_oof[va] = expit(theta_hat[:, None] * a[None, :] + b[None, :])
                fold_info.append(
                    {
                        "fold": k,
                        "a": a.tolist(),
                        "b": b.tolist(),
                        "bce_first_last": [hist[0], hist[-1]],
                        "stage1_insample_recon_score_mse": recon_mse,
                    }
                )
            oof_by_lam[lam_ab] = score_tr_oof
            fold_info_by_lam[lam_ab] = fold_info
            mse = float(np.mean((np.clip(score_tr_oof, 0, 1) - S) ** 2))
            print(f"[irt-gbm] lam_ab={lam_ab} oof_score_mse={mse:.5f} ({time.time()-t0:.1f}s)", flush=True)

        lam_mse = {
            lam: float(np.mean((np.clip(oof_by_lam[lam], 0, 1) - S) ** 2))
            for lam in LAM_AB_GRID
        }
        best_lam_ab = min(lam_mse, key=lam_mse.get)
        score_tr_oof = oof_by_lam[best_lam_ab]

        theta_star_full, a_full, b_full, hist_full = fit_free_theta(S, best_lam_ab)
        recon_full = expit(theta_star_full[:, None] * a_full[None, :] + b_full[None, :])
        recon_full_mse = float(np.mean((recon_full - S) ** 2))
        reg_full = LGBMRegressor(**LGBM_PARAMS)
        reg_full.fit(x_tr, theta_star_full)
        theta_hat_dev = reg_full.predict(x_dev)
        score_dev_pred = expit(theta_hat_dev[:, None] * a_full[None, :] + b_full[None, :])

        oof_mse = lam_mse[best_lam_ab]
        summary = {
            "variant": "irt-gbm",
            "framing": (
                "stage1: free per-item theta* (1-D) by alternating Newton "
                "(per-model logistic (a_j,b_j) with L2 prior on a <-> per-item theta), "
                "theta standardized each round; stage2: LightGBM x -> theta*; "
                "score = sigmoid(a_j theta_hat + b_j). "
                "Both stages fit only on fold-train rows for OOF."
            ),
            "features": f"dense81(StandardScaler)+SVD{SVD_DIM}(word)+SVD{SVD_DIM}(char), train-fit",
            "hyperparameters": {
                "latent_dim": 1,
                "stage1": {
                    "n_rounds": FREE_ROUNDS,
                    "lam_theta": LAM_THETA,
                    "lam_ab_grid": LAM_AB_GRID,
                    "lam_ab_best": best_lam_ab,
                    "ab_newton_iters": 25,
                },
                "stage2_lgbm": {k: v for k, v in LGBM_PARAMS.items()},
                "folds": N_FOLDS,
                "seed": SEED,
            },
            "lam_ab_oof_score_mse": {str(l): lam_mse[l] for l in LAM_AB_GRID},
            "fold_info": fold_info_by_lam[best_lam_ab],
            "full_train_fit": {
                "a": a_full.tolist(),
                "b": b_full.tolist(),
                "stage1_bce_first_last": [hist_full[0], hist_full[-1]],
                "stage1_insample_recon_score_mse": recon_full_mse,
            },
            "oof_metrics": score_metrics(np.clip(score_tr_oof, 0, 1), S),
            "dev_metrics": score_metrics(np.clip(score_dev_pred, 0, 1), S_dev),
            "cost_head": "copied from mlp-ens predictions (build/preds/mlp-ens-*.npz)",
        }
        save_variant("irt-gbm", score_tr_oof, cost_tr, score_dev_pred, cost_dev, summary)
        print(f"[irt-gbm] best lam_ab={best_lam_ab} oof_score_mse={oof_mse:.5f} ({time.time()-t0:.1f}s)", flush=True)

    print(f"[done] total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
