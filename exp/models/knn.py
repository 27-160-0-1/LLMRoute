# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""M4 kNN / prototype predictor.

Vectors: hstack(word CSR, char CSR), row L2-normalized after hstack.
Similarity: cosine (dot product of L2-normalized rows).
Neighbor weights: w = max(sim, 0)^3.
score pred = weighted mean of neighbor scores (per model).
cost pred  = exp(weighted mean of neighbor log-costs) (per model).

Variants:
  knn-k{K} for K in {5,10,20,40}
    train: 5-fold OOF (fold_id = arange(N) % 5), neighbors only from other folds
    dev:   neighbors from full train
  knn-best
    best K by OOF score MSE; rows with sim_max < 0.15 are shrunk toward the
    global (fitting-set) mean with lam = sim_max / 0.15.

Run:  .venv/Scripts/python.exe exp/models/knn.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
FEATS = ROOT / "build" / "feats"
PREDS = ROOT / "build" / "preds"
PREDS.mkdir(parents=True, exist_ok=True)

KS = [5, 10, 20, 40]
KMAX = max(KS)
N_FOLDS = 5
POWER = 3.0
SIM_FLOOR = 0.15  # knn-best shrink threshold
EPS = 1e-12


def load_vectors(split: str) -> sp.csr_matrix:
    word = sp.load_npz(FEATS / f"word-{split}.npz").tocsr()
    char = sp.load_npz(FEATS / f"char-{split}.npz").tocsr()
    x = sp.hstack([word, char], format="csr")
    return normalize(x, norm="l2", axis=1, copy=False)


def load_targets(split: str):
    data = np.load(FEATS / f"targets-{split}.npz")
    return np.asarray(data["scores"], dtype=np.float64), np.asarray(
        data["costs"], dtype=np.float64
    )


def top_neighbors(sim: np.ndarray, kmax: int):
    """Per row: indices+sims of the kmax most similar allowed columns.

    Disallowed entries must already be set to -inf. Returns (idx, s) each
    (N, kmax), sorted by similarity descending within each row.
    """
    part = np.argpartition(sim, -kmax, axis=1)[:, -kmax:]
    rows = np.arange(sim.shape[0])[:, None]
    vals = sim[rows, part]
    order = np.argsort(-vals, axis=1)
    idx = part[rows, order]
    s = vals[rows, order]
    return idx, s


def knn_predict(idx, sims, k, scores_fit, logcost_fit, fb_score, fb_logcost):
    """Weighted-average prediction from precomputed sorted neighbors.

    idx, sims: (N, kmax) sorted desc. fb_*: (N,3) per-row fallback means used
    when all weights vanish (sim <= 0 for every neighbor).
    """
    idx_k = idx[:, :k]
    w = np.clip(sims[:, :k], 0.0, None) ** POWER  # (N,k)
    wsum = w.sum(axis=1)  # (N,)
    ns = scores_fit[idx_k]  # (N,k,3)
    nl = logcost_fit[idx_k]  # (N,k,3)
    num_s = np.einsum("nk,nkm->nm", w, ns)
    num_l = np.einsum("nk,nkm->nm", w, nl)
    ok = wsum > EPS
    score = np.where(ok[:, None], num_s / np.maximum(wsum, EPS)[:, None], fb_score)
    logc = np.where(ok[:, None], num_l / np.maximum(wsum, EPS)[:, None], fb_logcost)
    return score, logc


def finalize(score, logcost):
    score = np.clip(score, 0.0, 1.0)
    cost = np.exp(logcost)
    cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1 + 1e-12))
    cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1 + 1e-12))
    return score, cost


def metrics(score_pred, cost_pred, scores_true, costs_true):
    out = {}
    lc_pred = np.log(cost_pred)
    lc_true = np.log(costs_true)
    out["score_mse"] = ((score_pred - scores_true) ** 2).mean(axis=0).tolist()
    out["score_mae"] = np.abs(score_pred - scores_true).mean(axis=0).tolist()
    out["logcost_mse"] = ((lc_pred - lc_true) ** 2).mean(axis=0).tolist()
    out["score_mse_mean"] = float(((score_pred - scores_true) ** 2).mean())
    out["logcost_mse_mean"] = float(((lc_pred - lc_true) ** 2).mean())
    return out


def main() -> None:
    xtr = load_vectors("train")
    xdev = load_vectors("dev")
    scores_tr, costs_tr = load_targets("train")
    scores_dev, costs_dev = load_targets("dev")
    logcost_tr = np.log(costs_tr)

    n_tr = xtr.shape[0]
    n_dev = xdev.shape[0]
    fold = np.arange(n_tr) % N_FOLDS

    # --- similarity matrices ---
    sim_tt = np.asarray((xtr @ xtr.T).todense(), dtype=np.float64)
    sim_dt = np.asarray((xdev @ xtr.T).todense(), dtype=np.float64)

    # OOF mask: row i may only use columns from other folds (self excluded too).
    sim_tt_masked = sim_tt.copy()
    for f in range(N_FOLDS):
        r = np.where(fold == f)[0]
        sim_tt_masked[np.ix_(r, r)] = -np.inf

    idx_tr, sims_tr = top_neighbors(sim_tt_masked, KMAX)
    idx_dev, sims_dev = top_neighbors(sim_dt, KMAX)

    # Fallback / shrink means. OOF rows use out-of-fold means; dev uses full train.
    oof_mean_score = np.empty((n_tr, 3))
    oof_mean_logc = np.empty((n_tr, 3))
    for f in range(N_FOLDS):
        r = fold == f
        oof_mean_score[r] = scores_tr[~r].mean(axis=0)
        oof_mean_logc[r] = logcost_tr[~r].mean(axis=0)
    full_mean_score = np.broadcast_to(scores_tr.mean(axis=0), (n_dev, 3)).copy()
    full_mean_logc = np.broadcast_to(logcost_tr.mean(axis=0), (n_dev, 3)).copy()

    # dev nearest-neighbor similarity distribution (private-set proxy).
    nn_dev = sims_dev[:, 0]
    nn_stats = {
        "p10": float(np.percentile(nn_dev, 10)),
        "p50": float(np.percentile(nn_dev, 50)),
        "p90": float(np.percentile(nn_dev, 90)),
        "mean": float(nn_dev.mean()),
        "frac_below_0.15": float((nn_dev < SIM_FLOOR).mean()),
    }
    nn_tr = sims_tr[:, 0]
    nn_stats_train_oof = {
        "p10": float(np.percentile(nn_tr, 10)),
        "p50": float(np.percentile(nn_tr, 50)),
        "p90": float(np.percentile(nn_tr, 90)),
        "frac_below_0.15": float((nn_tr < SIM_FLOOR).mean()),
    }

    results = {}
    raw = {}
    for k in KS:
        s_tr, l_tr = knn_predict(
            idx_tr, sims_tr, k, scores_tr, logcost_tr, oof_mean_score, oof_mean_logc
        )
        s_dev, l_dev = knn_predict(
            idx_dev, sims_dev, k, scores_tr, logcost_tr, full_mean_score, full_mean_logc
        )
        raw[k] = (s_tr.copy(), l_tr.copy(), s_dev.copy(), l_dev.copy())
        score_tr_f, cost_tr_f = finalize(s_tr, l_tr)
        score_dev_f, cost_dev_f = finalize(s_dev, l_dev)
        name = f"knn-k{k}"
        np.savez(PREDS / f"{name}-train.npz", score=score_tr_f, cost=cost_tr_f)
        np.savez(PREDS / f"{name}-dev.npz", score=score_dev_f, cost=cost_dev_f)
        oof = metrics(score_tr_f, cost_tr_f, scores_tr, costs_tr)
        results[k] = oof
        summary = {
            "variant": name,
            "hyperparams": {
                "k": k,
                "weight": "max(sim,0)^3",
                "vectors": "hstack(word,char) row-L2",
                "cost_space": "log-mean -> exp",
                "seed": SEED,
            },
            "oof": oof,
            "dev_nn_sim": nn_stats,
            "train_oof_nn_sim": nn_stats_train_oof,
        }
        (PREDS / f"{name}-summary.json").write_text(json.dumps(summary, indent=2))
        print(name, "oof score_mse", round(oof["score_mse_mean"], 5),
              "logcost_mse", round(oof["logcost_mse_mean"], 5))

    # --- knn-best: best K by OOF score MSE + low-similarity shrink ---
    best_k = min(KS, key=lambda k: results[k]["score_mse_mean"])
    s_tr, l_tr, s_dev, l_dev = (a.copy() for a in raw[best_k])

    lam_tr = np.clip(np.clip(nn_tr, 0.0, None) / SIM_FLOOR, 0.0, 1.0)[:, None]
    lam_dev = np.clip(np.clip(nn_dev, 0.0, None) / SIM_FLOOR, 0.0, 1.0)[:, None]
    s_tr = lam_tr * s_tr + (1 - lam_tr) * oof_mean_score
    l_tr = lam_tr * l_tr + (1 - lam_tr) * oof_mean_logc
    s_dev = lam_dev * s_dev + (1 - lam_dev) * full_mean_score
    l_dev = lam_dev * l_dev + (1 - lam_dev) * full_mean_logc

    score_tr_f, cost_tr_f = finalize(s_tr, l_tr)
    score_dev_f, cost_dev_f = finalize(s_dev, l_dev)
    np.savez(PREDS / "knn-best-train.npz", score=score_tr_f, cost=cost_tr_f)
    np.savez(PREDS / "knn-best-dev.npz", score=score_dev_f, cost=cost_dev_f)
    oof = metrics(score_tr_f, cost_tr_f, scores_tr, costs_tr)
    summary = {
        "variant": "knn-best",
        "hyperparams": {
            "base_k": best_k,
            "weight": "max(sim,0)^3",
            "vectors": "hstack(word,char) row-L2",
            "cost_space": "log-mean -> exp",
            "shrink": f"rows with sim_max < {SIM_FLOOR} blended toward fitting-set "
                      f"mean with lam = sim_max/{SIM_FLOOR}",
            "seed": SEED,
        },
        "oof": oof,
        "k_selection": {str(k): results[k]["score_mse_mean"] for k in KS},
        "n_shrunk_train": int((nn_tr < SIM_FLOOR).sum()),
        "n_shrunk_dev": int((nn_dev < SIM_FLOOR).sum()),
        "dev_nn_sim": nn_stats,
        "train_oof_nn_sim": nn_stats_train_oof,
    }
    (PREDS / "knn-best-summary.json").write_text(json.dumps(summary, indent=2))
    print("knn-best (k=%d) oof score_mse" % best_k, round(oof["score_mse_mean"], 5),
          "logcost_mse", round(oof["logcost_mse_mean"], 5))
    print("dev nn sim:", json.dumps(nn_stats))


if __name__ == "__main__":
    main()
