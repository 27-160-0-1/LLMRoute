# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Build the final runtime model bundle (build/final-model/) and verify that
every rebuilt member reproduces its snapshotted dev predictions, and that the
frozen policy reproduces the E049 dev assignment exactly.

Members (train-only fits, per frozen spec):
  lgbm      : svd features; score x3, log_in x3, log_out x3, q90 out x3
  xgb-mono  : raw sparse (train-nonzero cols); multi-output score + token heads
  irt1d     : scaler+svd features; linear W, a, b
  knn-k40   : train index (word+char CSR, L2 after hstack) + train outcomes
Extras:
  lookup    : sha256(text) -> realized scores/costs for train+dev (rule-allowed)
  policy    : frozen decision parameters
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
FEATS = ROOT / "build" / "feats"
SNAP = ROOT / "build" / "final-preds"
OUT = ROOT / "build" / "final-model"

SEED = 42
RATES = np.array([[1.0, 4.0], [2.127, 8.509], [6.565, 26.260]])


def log(msg):
    print(msg, flush=True)


def load_feats(split):
    dense = np.load(FEATS / f"dense-{split}.npy")
    word = sp.load_npz(FEATS / f"word-{split}.npz").tocsr()
    char = sp.load_npz(FEATS / f"char-{split}.npz").tocsr()
    return dense, word, char


def monotone(cost):
    cost = cost.copy()
    cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1 + 1e-12))
    cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1 + 1e-12))
    return cost


def check(name, got, snap_file, key, tol):
    want = np.load(SNAP / snap_file)[key]
    err = float(np.abs(got - want).max())
    status = "OK" if err <= tol else "FAIL"
    log(f"  verify {name} vs {snap_file}[{key}]: max abs err {err:.3e} [{status}]")
    if err > tol:
        raise AssertionError(f"{name} 재현 실패: {err}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lgbm").mkdir(exist_ok=True)
    (OUT / "xgb").mkdir(exist_ok=True)
    (OUT / "knn").mkdir(exist_ok=True)

    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler, normalize
    import lightgbm as lgb
    import xgboost as xgb

    dense_tr, word_tr, char_tr = load_feats("train")
    dense_dev, word_dev, char_dev = load_feats("dev")
    tgt = np.load(FEATS / "targets-train.npz")
    scores_tr = np.asarray(tgt["scores"], dtype=np.float64)
    in_tr = np.asarray(tgt["in_tokens"], dtype=np.float64)
    out_tr = np.asarray(tgt["out_tokens"], dtype=np.float64)
    costs_tr = np.asarray(tgt["costs"], dtype=np.float64)

    # ---------------- SVD (shared: lgbm / irt) ----------------
    log("[1/6] SVD fit")
    svd_w = TruncatedSVD(n_components=128, random_state=SEED).fit(word_tr)
    svd_c = TruncatedSVD(n_components=128, random_state=SEED).fit(char_tr)
    np.savez_compressed(OUT / "svd-word.npz", components=svd_w.components_.astype(np.float64))
    np.savez_compressed(OUT / "svd-char.npz", components=svd_c.components_.astype(np.float64))
    X_svd_tr = np.hstack([dense_tr, svd_w.transform(word_tr), svd_c.transform(char_tr)])
    X_svd_dev = np.hstack([dense_dev, svd_w.transform(word_dev), svd_c.transform(char_dev)])

    # ---------------- LGBM member ----------------
    log("[2/6] LGBM heads")
    def lgb_params(objective="regression", alpha=None):
        p = dict(
            objective=objective, learning_rate=0.05, num_leaves=31,
            min_data_in_leaf=20, seed=SEED, verbosity=-1,
            force_col_wise=True, deterministic=True, num_threads=4,
        )
        if alpha is not None:
            p["alpha"] = alpha
        return p

    def fit_head(y, objective="regression", alpha=None):
        ds = lgb.Dataset(X_svd_tr, label=y, params={"min_data_in_leaf": 20})
        booster = lgb.train(lgb_params(objective, alpha), ds, num_boost_round=300)
        return booster

    lgbm_boosters = {}
    for j in range(3):
        lgbm_boosters[f"score-{j}"] = fit_head(scores_tr[:, j])
        lgbm_boosters[f"lin-{j}"] = fit_head(np.log(in_tr[:, j]))
        lgbm_boosters[f"lout-{j}"] = fit_head(np.log(out_tr[:, j]))
        lgbm_boosters[f"q90-{j}"] = fit_head(
            np.log(out_tr[:, j]), objective="quantile", alpha=0.90
        )
    for name, booster in lgbm_boosters.items():
        booster.save_model(str(OUT / "lgbm" / f"{name}.txt"))

    def lgbm_predict(X):
        score = np.column_stack([lgbm_boosters[f"score-{j}"].predict(X) for j in range(3)])
        lin = np.column_stack([lgbm_boosters[f"lin-{j}"].predict(X) for j in range(3)])
        lout = np.column_stack([lgbm_boosters[f"lout-{j}"].predict(X) for j in range(3)])
        q90 = np.column_stack([lgbm_boosters[f"q90-{j}"].predict(X) for j in range(3)])
        cost = np.exp(lin) * RATES[:, 0] / 1e6 + np.exp(lout) * RATES[:, 1] / 1e6
        cost_q90 = np.exp(lin) * RATES[:, 0] / 1e6 + np.exp(q90) * RATES[:, 1] / 1e6
        return np.clip(score, 0, 1), monotone(cost), monotone(cost_q90)

    lgbm_score_dev, lgbm_cost_dev, lgbm_q90_dev = lgbm_predict(X_svd_dev)
    check("lgbm score", lgbm_score_dev, "lgbm-dev.npz", "score", 1e-9)
    check("lgbm cost", lgbm_cost_dev, "lgbm-dev.npz", "cost", 1e-9)
    check("lgbm q90 k1 cost", lgbm_q90_dev[:, 2], "lgbm-q90-dev.npz", "cost", 1e-9) if False else None
    # q90 snapshot cost includes q90 out for ALL models
    check("lgbm-q90 cost", lgbm_q90_dev, "lgbm-q90-dev.npz", "cost", 1e-9)

    # ---------------- XGB member ----------------
    log("[3/6] XGB-mono heads (cuda)")
    X_sp_tr = sp.hstack(
        [sp.csr_matrix(dense_tr), word_tr, char_tr], format="csr"
    ).astype(np.float32)
    X_sp_dev = sp.hstack(
        [sp.csr_matrix(dense_dev), word_dev, char_dev], format="csr"
    ).astype(np.float32)
    keep = np.where(np.diff(X_sp_tr.tocsc().indptr) > 0)[0]
    np.save(OUT / "xgb" / "keep-cols.npy", keep)
    X_sp_tr = X_sp_tr[:, keep].tocsr()
    X_sp_dev = X_sp_dev[:, keep].tocsr()

    base = dict(
        tree_method="hist", device="cuda", learning_rate=0.06,
        min_child_weight=5, subsample=0.9, colsample_bytree=0.5,
        max_bin=63, objective="reg:squarederror", seed=SEED, verbosity=0,
    )
    dfull = xgb.QuantileDMatrix(X_sp_tr, max_bin=63)
    ddev = xgb.DMatrix(X_sp_dev)

    def xgb_fit(labels, depth, extra=None):
        params = dict(base, max_depth=depth)
        if extra:
            params.update(extra)
        dfull.set_label(labels)
        return xgb.train(params, dfull, num_boost_round=400)

    multi_extra = {
        "multi_strategy": "multi_output_tree",
        "grow_policy": "lossguide",
        "max_leaves": 2**5,
        "max_cached_hist_node": 8,
    }
    bst_score = xgb_fit(scores_tr, 5, multi_extra)
    bst_score.save_model(str(OUT / "xgb" / "score-multi.json"))
    xgb_score_dev = bst_score.predict(ddev)
    xgb_lin_dev = np.empty((X_sp_dev.shape[0], 3))
    xgb_lout_dev = np.empty((X_sp_dev.shape[0], 3))
    for j in range(3):
        bst = xgb_fit(np.log(in_tr[:, j]), 7)
        bst.save_model(str(OUT / "xgb" / f"lin-{j}.json"))
        xgb_lin_dev[:, j] = bst.predict(ddev)
        bst = xgb_fit(np.log(out_tr[:, j]), 7)
        bst.save_model(str(OUT / "xgb" / f"lout-{j}.json"))
        xgb_lout_dev[:, j] = bst.predict(ddev)
    xgb_cost_dev = monotone(
        np.exp(xgb_lin_dev) * RATES[:, 0] / 1e6 + np.exp(xgb_lout_dev) * RATES[:, 1] / 1e6
    )
    xgb_score_dev = np.clip(xgb_score_dev, 0, 1)
    check("xgb-mono score", xgb_score_dev, "xgb-mono-dev.npz", "score", 5e-5)
    check("xgb-mono cost", xgb_cost_dev, "xgb-mono-dev.npz", "cost", 5e-5)

    # ---------------- IRT member ----------------
    log("[4/6] IRT1d")
    sys.path.insert(0, str(ROOT / "exp" / "models"))
    import irt as irt_mod

    scaler = StandardScaler().fit(dense_tr)
    X_irt_tr = np.hstack(
        [scaler.transform(dense_tr), svd_w.transform(word_tr), svd_c.transform(char_tr)]
    )
    X_irt_dev = np.hstack(
        [scaler.transform(dense_dev), svd_w.transform(word_dev), svd_c.transform(char_dev)]
    )
    W, a, b, loss, iters = irt_mod.fit_irt_linear(X_irt_tr, scores_tr, 1, 0.01)
    np.savez(
        OUT / "irt.npz",
        W=W, a=a, b=b,
        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
    )
    irt_score_dev = irt_mod.predict_irt(X_irt_dev, W, a, b)
    check("irt1d score", irt_score_dev, "irt1d-dev.npz", "score", 5e-6)

    # ---------------- kNN member ----------------
    log("[5/6] kNN index")
    x_tr_n = normalize(sp.hstack([word_tr, char_tr], format="csr"), norm="l2", axis=1, copy=False)
    x_dev_n = normalize(sp.hstack([word_dev, char_dev], format="csr"), norm="l2", axis=1, copy=False)
    logcost_tr = np.log(costs_tr)
    sp.save_npz(OUT / "knn" / "index.npz", x_tr_n)
    np.savez(
        OUT / "knn" / "outcomes.npz",
        scores=scores_tr, logcost=logcost_tr,
        fb_score=scores_tr.mean(axis=0), fb_logcost=logcost_tr.mean(axis=0),
    )
    sim = np.asarray((x_dev_n @ x_tr_n.T).todense(), dtype=np.float64)
    part = np.argpartition(sim, -40, axis=1)[:, -40:]
    rows = np.arange(sim.shape[0])[:, None]
    vals = sim[rows, part]
    order = np.argsort(-vals, axis=1, kind="stable")
    idx = part[rows, order]
    sims = sim[rows, idx]
    w = np.clip(sims, 0.0, None) ** 3
    wsum = w.sum(axis=1, keepdims=True)
    ok = wsum[:, 0] > 0
    knn_score_dev = np.where(
        ok[:, None],
        (w[:, :, None] * scores_tr[idx]).sum(axis=1) / np.maximum(wsum, 1e-300),
        scores_tr.mean(axis=0)[None, :],
    )
    knn_logc_dev = np.where(
        ok[:, None],
        (w[:, :, None] * logcost_tr[idx]).sum(axis=1) / np.maximum(wsum, 1e-300),
        logcost_tr.mean(axis=0)[None, :],
    )
    check("knn-k40 score", knn_score_dev, "knn-k40-dev.npz", "score", 1e-9)
    check("knn-k40 cost", monotone(np.exp(knn_logc_dev)), "knn-k40-dev.npz", "cost", 1e-9)

    # ---------------- blend + policy verification ----------------
    log("[6/6] blend/policy verification vs E049")
    blend_score = (xgb_score_dev + irt_score_dev + knn_score_dev + lgbm_score_dev) / 4.0
    cost_mean = monotone(np.exp((np.log(lgbm_cost_dev) + np.log(xgb_cost_dev)) / 2.0))
    check("blend4 score", blend_score, "blend4-cost2b-dev.npz", "score", 5e-5)
    check("blend4 cost", cost_mean, "blend4-cost2b-dev.npz", "cost", 5e-5)
    q90_compose = cost_mean.copy()
    q90_compose[:, 2] = lgbm_q90_dev[:, 2]
    q90_compose = monotone(q90_compose)
    check("blend4 q90k1 cost", q90_compose, "blend4-q90k1-dev.npz", "cost", 5e-5)

    import alloc_lib
    from freeze_final import final_assignment

    ref = final_assignment("dev")
    got = {
        "fast": alloc_lib.greedy_allocate(blend_score, cost_mean, multiplier=1.25, utilization=0.93, allow_k1=False),
        "balanced": alloc_lib.greedy_allocate(blend_score, cost_mean, multiplier=2.0, utilization=0.88, allow_k1=False),
        "premium": alloc_lib.two_stage_premium(blend_score, q90_compose, multiplier=4.0, k1_utilization=0.65, fill_utilization=0.70, k1_cost_cap=0.1),
    }
    for tier in ("fast", "balanced", "premium"):
        diff = int((got[tier] != ref[tier]).sum())
        log(f"  policy {tier}: {diff} decision diffs vs E049")
        if diff:
            raise AssertionError(f"{tier} 배정 불일치 {diff}건")

    # ---------------- lookup table ----------------
    log("lookup table (train+dev)")
    from harness import load_split

    hashes = []
    look_scores = []
    look_costs = []
    for split in ("train", "dev"):
        table = load_split(split)
        for text, s_row, c_row in zip(table["texts"], table["scores"], table["costs"]):
            hashes.append(hashlib.sha256(text.encode("utf-8")).digest())
            look_scores.append(s_row)
            look_costs.append(c_row)
    key = np.frombuffer(b"".join(hashes), dtype=np.uint8).reshape(-1, 32)
    # order by lexicographic hash for binary search
    order = np.lexsort(key.T[::-1])
    np.savez_compressed(
        OUT / "lookup.npz",
        key=key[order],
        scores=np.asarray(look_scores)[order],
        costs=np.asarray(look_costs)[order],
    )

    policy = {
        "blend_members": ["xgb-mono", "irt1d", "knn-k40", "lgbm"],
        "cost_mean_members": ["lgbm", "xgb-mono"],
        "rates": RATES.tolist(),
        "fast": {"decision": "greedy", "allow_k1": False, "utilization": 0.93},
        "balanced": {"decision": "greedy", "allow_k1": False, "utilization": 0.88},
        "premium": {
            "decision": "two-stage",
            "k1_utilization": 0.65,
            "fill_utilization": 0.70,
            "k1_cost_cap": 0.1,
            "k1_cost_source": "lgbm-q90",
        },
        "feature_version": 2,
        "registry_ref": "E049",
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=1), encoding="utf-8")

    manifest = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            manifest[str(p.relative_to(OUT)).replace("\\", "/")] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    log(f"bundle complete: {len(manifest)} files, {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
