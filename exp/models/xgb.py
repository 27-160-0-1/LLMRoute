# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""M2b XGBoost predictor (hist).

Features: hstack(dense 81, word CSR 65536, char CSR 65536) -> CSR float32,
restricted to columns with at least one nonzero in train (prediction-equivalent
to the full matrix: constant-zero columns in the fitting folds can never be
split on, and columns unseen at fit time are never used at predict time).

Heads (all reg:squarederror, tree_method=hist):
  score_j              j in {light, ax31, k1}
  log(out_tokens_j)
  log(in_tokens_j)
cost_j = exp(log_in_j) * rin_j / 1e6 + exp(log_out_j) * rout_j / 1e6
(same combination rule as the lgbm task), then monotone correction
cost[:,1] >= cost[:,0], cost[:,2] >= cost[:,1].

Grid: max_depth in {5, 7}; fixed n_estimators=400, lr=0.06, min_child_weight=5,
subsample=0.9, colsample_bytree=0.5, max_bin=63.
Depth for score heads picked by mean OOF score MSE; depth for token heads
picked by mean OOF log-cost MSE.

Trained on GPU (device=cuda, RTX 4090; CPU hist was ~250s/fit vs ~30s on GPU).
GPU hist verified bit-reproducible on rerun with fixed seed. Depth-wise growth
OOMs for the multi-output tree (level histogram = 117k feats x 63 bins x 3
targets), so multi-output heads use grow_policy=lossguide with
max_leaves=2**max_depth (same capacity, one-node-at-a-time growth).

Variants:
  xgb       per-head models as above.
  xgb-mono  experimental: score predicted by ONE multi-output model
            (multi_strategy=multi_output_tree, same grid) vs the per-head
            version; whichever has the lower mean OOF score MSE is saved.
            Costs identical to xgb.

train npz: 5-fold OOF (fold_id = arange(N) % 5).  dev npz: fit on full train.
Run:  .venv/Scripts/python.exe exp/models/xgb.py
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import xgboost as xgb

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
FEATS = ROOT / "build" / "feats"
PREDS = ROOT / "build" / "preds"
PREDS.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
DEPTHS = [5, 7]
N_ESTIMATORS = 400
MAX_BIN = 63
MODELS = ["ax31-light", "ax31", "axk1-think"]
RATES = [(1.0, 4.0), (2.127, 8.509), (6.565, 26.260)]  # (rin, rout) per model

BASE_PARAMS = dict(
    tree_method="hist",
    device="cuda",
    learning_rate=0.06,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.5,
    max_bin=MAX_BIN,
    objective="reg:squarederror",
    seed=SEED,
    verbosity=0,
)
def multi_extra(depth: int) -> dict:
    return {
        "multi_strategy": "multi_output_tree",
        "grow_policy": "lossguide",
        "max_leaves": 2 ** depth,
        "max_cached_hist_node": 8,
    }


def load_features():
    def stack(split):
        dense = np.load(FEATS / f"dense-{split}.npy")
        word = sp.load_npz(FEATS / f"word-{split}.npz").tocsr()
        char = sp.load_npz(FEATS / f"char-{split}.npz").tocsr()
        return sp.hstack([sp.csr_matrix(dense), word, char], format="csr").astype(
            np.float32
        )

    xtr = stack("train")
    xdev = stack("dev")
    col_nnz = np.diff(xtr.tocsc().indptr)
    keep = np.where(col_nnz > 0)[0]
    return xtr[:, keep].tocsr(), xdev[:, keep].tocsr(), int(len(keep))


def load_targets(split: str):
    data = np.load(FEATS / f"targets-{split}.npz")
    return (
        np.asarray(data["scores"], dtype=np.float64),
        np.asarray(data["costs"], dtype=np.float64),
        np.asarray(data["in_tokens"], dtype=np.float64),
        np.asarray(data["out_tokens"], dtype=np.float64),
    )


def train_predict(dfit, labels, dpred_list, depth, extra=None):
    """Train one booster per label column set on a shared QuantileDMatrix.

    labels: (n_fit,) or (n_fit, k). Returns list of predictions per dpred.
    """
    params = dict(BASE_PARAMS, max_depth=depth)
    if extra:
        params.update(extra)
    dfit.set_label(labels)
    bst = xgb.train(params, dfit, num_boost_round=N_ESTIMATORS)
    preds = [bst.predict(d) for d in dpred_list]
    del bst
    return preds


def combine_cost(log_in, log_out):
    """(N,3) log token predictions -> absolute credit costs."""
    cost = np.empty_like(log_in)
    for j, (rin, rout) in enumerate(RATES):
        cost[:, j] = np.exp(log_in[:, j]) * rin / 1e6 + np.exp(log_out[:, j]) * rout / 1e6
    return cost


def finalize(score, cost):
    score = np.clip(score, 0.0, 1.0)
    cost = cost.copy()
    cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1 + 1e-12))
    cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1 + 1e-12))
    return score, cost


def metrics(score_pred, cost_pred, scores_true, costs_true):
    lc_pred = np.log(np.maximum(cost_pred, 1e-12))
    lc_true = np.log(costs_true)
    return {
        "score_mse": ((score_pred - scores_true) ** 2).mean(axis=0).tolist(),
        "score_mae": np.abs(score_pred - scores_true).mean(axis=0).tolist(),
        "logcost_mse": ((lc_pred - lc_true) ** 2).mean(axis=0).tolist(),
        "score_mse_mean": float(((score_pred - scores_true) ** 2).mean()),
        "logcost_mse_mean": float(((lc_pred - lc_true) ** 2).mean()),
    }


def main() -> None:
    t_start = time.time()
    xtr, xdev, n_cols = load_features()
    scores_tr, costs_tr, in_tr, out_tr = load_targets("train")
    scores_dev, costs_dev, _, _ = load_targets("dev")
    log_in_tr = np.log(in_tr)
    log_out_tr = np.log(out_tr)

    n_tr = xtr.shape[0]
    fold = np.arange(n_tr) % N_FOLDS
    print(f"features: {xtr.shape} (kept {n_cols} nonzero cols)", flush=True)

    # OOF containers: [depth][head_kind] -> (N,3)
    oof_score = {d: np.full((n_tr, 3), np.nan) for d in DEPTHS}
    oof_lin = {d: np.full((n_tr, 3), np.nan) for d in DEPTHS}
    oof_lout = {d: np.full((n_tr, 3), np.nan) for d in DEPTHS}
    oof_score_multi = {d: np.full((n_tr, 3), np.nan) for d in DEPTHS}

    for f in range(N_FOLDS):
        val = fold == f
        fit = ~val
        dfit = xgb.QuantileDMatrix(xtr[fit], max_bin=MAX_BIN)
        dval = xgb.DMatrix(xtr[val])
        for d in DEPTHS:
            for j in range(3):
                oof_score[d][val, j] = train_predict(
                    dfit, scores_tr[fit, j], [dval], d
                )[0]
                oof_lout[d][val, j] = train_predict(
                    dfit, log_out_tr[fit, j], [dval], d
                )[0]
                oof_lin[d][val, j] = train_predict(
                    dfit, log_in_tr[fit, j], [dval], d
                )[0]
            oof_score_multi[d][val] = train_predict(
                dfit, scores_tr[fit], [dval], d, extra=multi_extra(d)
            )[0]
        del dfit, dval
        gc.collect()
        print(f"fold {f} done ({time.time() - t_start:.0f}s)", flush=True)

    # --- depth selection ---
    score_mse_by_depth = {
        d: float(((np.clip(oof_score[d], 0, 1) - scores_tr) ** 2).mean()) for d in DEPTHS
    }
    logcost_mse_by_depth = {
        d: float(
            (
                (np.log(combine_cost(oof_lin[d], oof_lout[d])) - np.log(costs_tr)) ** 2
            ).mean()
        )
        for d in DEPTHS
    }
    multi_mse_by_depth = {
        d: float(((np.clip(oof_score_multi[d], 0, 1) - scores_tr) ** 2).mean())
        for d in DEPTHS
    }
    depth_score = min(DEPTHS, key=lambda d: score_mse_by_depth[d])
    depth_cost = min(DEPTHS, key=lambda d: logcost_mse_by_depth[d])
    depth_multi = min(DEPTHS, key=lambda d: multi_mse_by_depth[d])
    print("score MSE by depth", score_mse_by_depth, "->", depth_score, flush=True)
    print("logcost MSE by depth", logcost_mse_by_depth, "->", depth_cost, flush=True)
    print("multi score MSE by depth", multi_mse_by_depth, "->", depth_multi, flush=True)

    # --- full-train fits for dev predictions (chosen depths only) ---
    dfull = xgb.QuantileDMatrix(xtr, max_bin=MAX_BIN)
    ddev = xgb.DMatrix(xdev)
    dev_score = np.empty((xdev.shape[0], 3))
    dev_lin = np.empty((xdev.shape[0], 3))
    dev_lout = np.empty((xdev.shape[0], 3))
    for j in range(3):
        dev_score[:, j] = train_predict(dfull, scores_tr[:, j], [ddev], depth_score)[0]
        dev_lout[:, j] = train_predict(dfull, log_out_tr[:, j], [ddev], depth_cost)[0]
        dev_lin[:, j] = train_predict(dfull, log_in_tr[:, j], [ddev], depth_cost)[0]
    dev_score_multi = train_predict(dfull, scores_tr, [ddev], depth_multi,
                                    extra=multi_extra(depth_multi))[0]
    print(f"full-train fits done ({time.time() - t_start:.0f}s)", flush=True)

    # --- variant xgb ---
    tr_cost = combine_cost(oof_lin[depth_cost], oof_lout[depth_cost])
    dv_cost = combine_cost(dev_lin, dev_lout)
    tr_s, tr_c = finalize(oof_score[depth_score], tr_cost)
    dv_s, dv_c = finalize(dev_score, dv_cost)
    np.savez(PREDS / "xgb-train.npz", score=tr_s, cost=tr_c)
    np.savez(PREDS / "xgb-dev.npz", score=dv_s, cost=dv_c)
    oof = metrics(tr_s, tr_c, scores_tr, costs_tr)
    hyper_common = {
        "n_estimators": N_ESTIMATORS,
        "learning_rate": 0.06,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.5,
        "max_bin": MAX_BIN,
        "tree_method": "hist",
        "device": "cuda",
        "features": f"dense+word+char hstack, {n_cols} train-nonzero cols",
        "cost_space": "log-token heads -> exp -> rate combine",
        "seed": SEED,
    }
    summary = {
        "variant": "xgb",
        "hyperparams": dict(
            hyper_common, max_depth_score=depth_score, max_depth_tokens=depth_cost
        ),
        "oof": oof,
        "depth_selection": {
            "score_mse": {str(d): score_mse_by_depth[d] for d in DEPTHS},
            "logcost_mse": {str(d): logcost_mse_by_depth[d] for d in DEPTHS},
        },
    }
    (PREDS / "xgb-summary.json").write_text(json.dumps(summary, indent=2))
    print("xgb oof score_mse", round(oof["score_mse_mean"], 5),
          "logcost_mse", round(oof["logcost_mse_mean"], 5), flush=True)

    # --- variant xgb-mono: per-head vs multi-output score, better OOF wins ---
    use_multi = multi_mse_by_depth[depth_multi] < score_mse_by_depth[depth_score]
    if use_multi:
        mono_tr_score, mono_dev_score = oof_score_multi[depth_multi], dev_score_multi
        chosen = {"score_head": "multi_output_tree", "max_depth_score": depth_multi,
                  "multi_growth": f"lossguide max_leaves={2 ** depth_multi}"}
    else:
        mono_tr_score, mono_dev_score = oof_score[depth_score], dev_score
        chosen = {"score_head": "per-head (same as xgb)", "max_depth_score": depth_score}
    m_tr_s, m_tr_c = finalize(mono_tr_score, tr_cost)
    m_dv_s, m_dv_c = finalize(mono_dev_score, dv_cost)
    np.savez(PREDS / "xgb-mono-train.npz", score=m_tr_s, cost=m_tr_c)
    np.savez(PREDS / "xgb-mono-dev.npz", score=m_dv_s, cost=m_dv_c)
    oof_m = metrics(m_tr_s, m_tr_c, scores_tr, costs_tr)
    summary_m = {
        "variant": "xgb-mono",
        "hyperparams": dict(hyper_common, **chosen, max_depth_tokens=depth_cost),
        "oof": oof_m,
        "comparison": {
            "per_head_score_mse": score_mse_by_depth[depth_score],
            "multi_output_score_mse": {str(d): multi_mse_by_depth[d] for d in DEPTHS},
            "picked": "multi_output_tree" if use_multi else "per-head",
        },
    }
    (PREDS / "xgb-mono-summary.json").write_text(json.dumps(summary_m, indent=2))
    print("xgb-mono oof score_mse", round(oof_m["score_mse_mean"], 5),
          "logcost_mse", round(oof_m["logcost_mse_mean"], 5),
          "picked", chosen["score_head"], flush=True)
    print(f"total {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
