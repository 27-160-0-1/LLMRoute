# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""A7 overfitting / statistical falsification harness.

PROXY: the shipped router is a 4-member blend (xgb-mono + irt1d + knn-k40 +
lgbm) with a 2-member geometric-mean cost.  This audit uses ONLY the LightGBM
member (12 heads: score x3, log_in x3, log_out x3, q90_out x3) with the exact
same hyperparameters as exp/build_final.py, then runs the SAME frozen
allocation policy and the SAME official Decimal scorer.  Lookup is never used
(offline pipeline has no lookup path), so every number here is lookup-OFF.

All artifacts go to build/a7/.  Nothing is ever written to models/final-v1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp"))

import lightgbm as lgb  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402

import alloc_lib  # noqa: E402
from harness import TIERS, evaluate  # noqa: E402

FEATS = ROOT / "build" / "feats"
OUT = ROOT / "build" / "a7"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
RATES = np.array([[1.0, 4.0], [2.127, 8.509], [6.565, 26.260]])
MODEL_IDS = ("ax31-light", "ax31", "axk1-think")
N_FOLDS = 5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ features


def load_raw(split: str):
    dense = np.load(FEATS / f"dense-{split}.npy")
    word = sp.load_npz(FEATS / f"word-{split}.npz").tocsr()
    char = sp.load_npz(FEATS / f"char-{split}.npz").tocsr()
    t = np.load(FEATS / f"targets-{split}.npz")
    tgt = {k: np.asarray(t[k], dtype=np.float64)
           for k in ("scores", "costs", "in_tokens", "out_tokens")}
    return dense, word, char, tgt


def svd_transformer(dense_fit, word_fit, char_fit):
    """Fit TruncatedSVD(128) on the FIT set only; return a transform fn."""
    ncomp = min(128, min(word_fit.shape[0], char_fit.shape[0]) - 1)
    svd_w = TruncatedSVD(n_components=ncomp, random_state=SEED).fit(word_fit)
    svd_c = TruncatedSVD(n_components=ncomp, random_state=SEED).fit(char_fit)

    def tf(dense, word, char):
        return np.hstack([dense, svd_w.transform(word), svd_c.transform(char)])

    return tf


# ------------------------------------------------------------------ training


def lgb_params(objective="regression", alpha=None, threads=4):
    p = dict(objective=objective, learning_rate=0.05, num_leaves=31,
             min_data_in_leaf=20, seed=SEED, verbosity=-1,
             force_col_wise=True, deterministic=True, num_threads=threads)
    if alpha is not None:
        p["alpha"] = alpha
    return p


def fit_heads(X, scores, in_tok, out_tok, threads=4):
    def fit(y, objective="regression", alpha=None):
        ds = lgb.Dataset(X, label=y, params={"min_data_in_leaf": 20})
        return lgb.train(lgb_params(objective, alpha, threads), ds,
                         num_boost_round=300)

    b = {}
    for j in range(3):
        b[f"score-{j}"] = fit(scores[:, j])
        b[f"lin-{j}"] = fit(np.log(np.maximum(in_tok[:, j], 1.0)))
        b[f"lout-{j}"] = fit(np.log(np.maximum(out_tok[:, j], 1.0)))
        b[f"q90-{j}"] = fit(np.log(np.maximum(out_tok[:, j], 1.0)),
                            "quantile", 0.90)
    return b


def monotone(cost):
    cost = cost.copy()
    cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1 + 1e-12))
    cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1 + 1e-12))
    return cost


def predict(boosters, X):
    col = lambda tag: np.column_stack(
        [boosters[f"{tag}-{j}"].predict(X) for j in range(3)])
    score = np.clip(col("score"), 0.0, 1.0)
    lin, lout, q90 = col("lin"), col("lout"), col("q90")
    cost = np.exp(lin) * RATES[:, 0] / 1e6 + np.exp(lout) * RATES[:, 1] / 1e6
    costq = np.exp(lin) * RATES[:, 0] / 1e6 + np.exp(q90) * RATES[:, 1] / 1e6
    return score, monotone(cost), monotone(costq)


# ---------------------------------------------------------------- allocation


def allocate(score, cost, cost_q90):
    """The frozen final policy (exp/freeze_final.py POLICY_SPEC)."""
    q = cost.copy()
    q[:, 2] = cost_q90[:, 2]
    q = monotone(q)
    return {
        "fast": alloc_lib.greedy_allocate(score, cost, multiplier=1.25,
                                          utilization=0.93, allow_k1=False),
        "balanced": alloc_lib.greedy_allocate(score, cost, multiplier=2.0,
                                              utilization=0.88, allow_k1=False),
        "premium": alloc_lib.two_stage_premium(score, q, multiplier=4.0,
                                               k1_utilization=0.65,
                                               fill_utilization=0.70,
                                               k1_cost_cap=0.1),
    }


def official(split, picks):
    assignment = {t: [MODEL_IDS[j] for j in picks[t]] for t in TIERS}
    rep = evaluate(split, assignment)
    out = {"final": float(rep["final_score"])}
    for t in TIERS:
        out[t] = {
            "score": float(rep["tiers"][t]["quality_score"]),
            "ratio": float(rep["tiers"][t]["budget_ratio"]),
            "passed": bool(rep["tiers"][t]["budget_passed"]),
            "counts": {k: int(v) for k, v in
                       rep["tiers"][t]["model_counts"].items()},
        }
    return out


def run_and_score(score, cost, costq, split, tag):
    picks = allocate(score, cost, costq)
    res = official(split, picks)
    res["tag"] = tag
    res["eval_split"] = split
    log(f"{tag} [{split}] final={res['final']:.6f} "
        f"F={res['fast']['score']:.4f} B={res['balanced']['score']:.4f} "
        f"P={res['premium']['score']:.4f}")
    return res


# --------------------------------------------------------------- experiments


def fit_predict(fit_split, eval_split, *, rows=None, shuffle_scores=None,
                shuffle_tokens=False, random_features=None, threads=4):
    """Fit the 12-head LGBM on fit_split (optionally a row subset) and predict
    on eval_split.  Returns (score, cost, cost_q90)."""
    d_f, w_f, c_f, t_f = load_raw(fit_split)
    d_e, w_e, c_e, _ = load_raw(eval_split)
    if rows is not None:
        d_f, w_f, c_f = d_f[rows], w_f[rows], c_f[rows]
        t_f = {k: v[rows] for k, v in t_f.items()}

    if random_features is not None:
        rng = np.random.default_rng(random_features)
        ncomp = min(128, d_f.shape[0] - 1)
        X_f = rng.standard_normal((d_f.shape[0], d_f.shape[1] + 2 * ncomp))
        X_e = rng.standard_normal((d_e.shape[0], d_f.shape[1] + 2 * ncomp))
    else:
        tf = svd_transformer(d_f, w_f, c_f)
        X_f = tf(d_f, w_f, c_f)
        X_e = tf(d_e, w_e, c_e)

    scores = t_f["scores"]
    in_tok, out_tok = t_f["in_tokens"], t_f["out_tokens"]
    if shuffle_scores is not None:
        rng = np.random.default_rng(shuffle_scores)
        perm = rng.permutation(scores.shape[0])
        scores = scores[perm]
        if shuffle_tokens:
            in_tok = in_tok[perm]
            out_tok = out_tok[perm]

    b = fit_heads(X_f, scores, in_tok, out_tok, threads)
    return predict(b, X_e)


def oof_predict(split, threads=4):
    """5-fold OOF predictions on `split` (fold_id = arange(N) % 5)."""
    d, w, c, t = load_raw(split)
    n = d.shape[0]
    fids = np.arange(n) % N_FOLDS
    tf = svd_transformer(d, w, c)   # unsupervised, fit on the whole split
    X = tf(d, w, c)
    score = np.zeros((n, 3))
    cost = np.zeros((n, 3))
    costq = np.zeros((n, 3))
    for f in range(N_FOLDS):
        tr = fids != f
        b = fit_heads(X[tr], t["scores"][tr], t["in_tokens"][tr],
                      t["out_tokens"][tr], threads)
        s, co, cq = predict(b, X[~tr])
        score[~tr], cost[~tr], costq[~tr] = s, co, cq
        log(f"oof {split} fold {f} done")
    return score, cost, costq


def const_predict(fit_split, eval_split):
    """Mean-predictor control: per-model constants from fit_split labels."""
    _, _, _, t_f = load_raw(fit_split)
    _, _, _, t_e = load_raw(eval_split)
    n = t_e["scores"].shape[0]
    score = np.repeat(t_f["scores"].mean(axis=0)[None, :], n, axis=0)
    cost = np.repeat(np.exp(np.log(t_f["costs"]).mean(axis=0))[None, :],
                     n, axis=0)
    return score, monotone(cost), monotone(cost.copy())


# --------------------------------------------------------------------- main


def save(name, obj):
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1), encoding="utf-8")
    log(f"wrote build/a7/{name}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baseline", "shuffle", "randfeat", "curve",
                                    "gap", "reverse", "const", "alllight"])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    th = a.threads

    if a.cmd == "alllight":
        n = load_raw("dev")[0].shape[0]
        z = {t: np.zeros(n, dtype=int) for t in TIERS}
        m = load_raw("train")[0].shape[0]
        zt = {t: np.zeros(m, dtype=int) for t in TIERS}
        save("alllight", {"dev": official("dev", z), "train": official("train", zt)})

    elif a.cmd == "baseline":
        s, c, q = fit_predict("train", "dev", threads=th)
        np.savez(OUT / "pred-baseline-dev.npz", score=s, cost=c, costq=q)
        save("baseline", run_and_score(s, c, q, "dev", "lgbm-proxy train->dev"))

    elif a.cmd == "const":
        s, c, q = const_predict("train", "dev")
        save("const", run_and_score(s, c, q, "dev", "mean-predictor train->dev"))

    elif a.cmd == "shuffle":
        res = []
        for seed in (101, 202, 303):
            s, c, q = fit_predict("train", "dev", shuffle_scores=seed, threads=th)
            res.append(run_and_score(s, c, q, "dev", f"shuffle-scores seed={seed}"))
        for seed in (101,):
            s, c, q = fit_predict("train", "dev", shuffle_scores=seed,
                                  shuffle_tokens=True, threads=th)
            res.append(run_and_score(s, c, q, "dev",
                                     f"shuffle-scores+tokens seed={seed}"))
        save("shuffle", res)

    elif a.cmd == "randfeat":
        res = []
        for seed in (7, 8, 9):
            s, c, q = fit_predict("train", "dev", random_features=seed, threads=th)
            res.append(run_and_score(s, c, q, "dev", f"random-features seed={seed}"))
        save("randfeat", res)

    elif a.cmd == "curve":
        n = load_raw("train")[0].shape[0]
        rng = np.random.default_rng(2026)
        order = rng.permutation(n)
        res = []
        for frac in (0.10, 0.25, 0.50, 1.00):
            k = int(round(frac * n))
            rows = np.sort(order[:k])
            s, c, q = fit_predict("train", "dev", rows=rows, threads=th)
            r = run_and_score(s, c, q, "dev", f"curve frac={frac:.2f} n={k}")
            r["frac"] = frac
            r["n_train"] = k
            res.append(r)
        save("curve", res)

    elif a.cmd == "gap":
        s, c, q = oof_predict("train", threads=th)
        np.savez(OUT / "pred-oof-train.npz", score=s, cost=c, costq=q)
        save("gap", run_and_score(s, c, q, "train", "lgbm-proxy OOF on train"))

    elif a.cmd == "reverse":
        s, c, q = fit_predict("dev", "train", threads=th)
        save("reverse", run_and_score(s, c, q, "train",
                                      "lgbm-proxy dev(880)->train(1760)"))


if __name__ == "__main__":
    main()
