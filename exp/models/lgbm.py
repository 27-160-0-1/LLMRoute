# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""M2: LightGBM variants (lgbm, lgbm-q75, lgbm-q90).

Feature modes (both evaluated, best OOF score MSE adopted; both recorded):
  (a) "sparse": dense(81) + word(65536) + char(65536) hstacked CSR
  (b) "svd":    dense(81) + SVD(word,128) + SVD(char,128)  (SVD fit on train only)

Heads (all LightGBM, lr=0.05, fixed tree count, no early stopping):
  - score:   3 regression heads on score in [0,1]
  - in_tok:  3 regression heads on log(in_tok)
  - out_tok: 3 regression heads on log(out_tok); quantile objective for -q75/-q90

cost = exp(log_in_pred)*rin/1e6 + exp(log_out_pred)*rout/1e6, then monotone
correction cost[:,1] >= cost[:,0], cost[:,2] >= cost[:,1].

OOF protocol: 5 folds, fold_id = np.arange(N) % 5, validation fold never seen
in training. Dev preds come from models fit on the full train split.

Hyperparameter sweep (score heads, mean OOF score MSE over 3 models):
num_leaves {31,63} x n_estimators {300,600} x min_child_samples {10,20},
lr 0.05. 600-round models are trained once per (mode, leaves, mcs) and also
evaluated at the 300-tree prefix (boosting trees are identical prefixes).

Reproducibility: seed 42, deterministic=True, force_col_wise=True,
num_threads fixed (default 4). `python exp/models/lgbm.py all` reproduces the
entire pipeline sequentially; the individual subcommands (prep / sweep /
select / tokens / finalize) produce byte-identical results when run as
parallel processes because every boost is independently seeded.

Efficiency notes: feature_pre_filter=True (default) with min_data_in_leaf set
at Dataset construction losslessly drops hashed columns that can never be
split (document frequency below min_data_in_leaf), which is what makes the
raw-sparse mode tractable; Datasets are built once per (mode, mcs) and reused
across heads via set_label.
"""

from __future__ import annotations

import argparse
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
WORK_DEFAULT = ROOT / "build" / "lgbm-work"

SEED = 42
N_FOLDS = 5
LR = 0.05
MODELS = ["ax31-light", "ax31", "axk1-think"]
RATES = np.array([[1.0, 4.0], [2.127, 8.509], [6.565, 26.260]])  # (rin, rout)

SWEEP_LEAVES = [31, 63]
SWEEP_TREES = [300, 600]
SWEEP_MCS = [10, 20]
TOKEN_HEADS = ("in", "out", "q75", "q90")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_params(num_leaves: int, mcs: int, threads: int,
                 objective: str = "regression", alpha: float | None = None) -> dict:
    p = dict(
        objective=objective,
        learning_rate=LR,
        num_leaves=num_leaves,
        min_data_in_leaf=mcs,
        seed=SEED,
        verbosity=-1,
        force_col_wise=True,
        deterministic=True,
        num_threads=threads,
    )
    if alpha is not None:
        p["alpha"] = alpha
    return p


# ---------------------------------------------------------------- features


def load_targets(split: str) -> dict:
    return dict(np.load(FEATS / f"targets-{split}.npz"))


def build_sparse(split: str):
    dense = np.load(FEATS / f"dense-{split}.npy")
    word = sp.load_npz(FEATS / f"word-{split}.npz").tocsr()
    char = sp.load_npz(FEATS / f"char-{split}.npz").tocsr()
    return sp.hstack([sp.csr_matrix(dense), word, char], format="csr")


def cmd_prep(work: Path) -> None:
    """Fit TruncatedSVD(128) on train word/char and cache svd-mode features."""
    work.mkdir(parents=True, exist_ok=True)
    if (work / "svd-train.npy").exists() and (work / "svd-dev.npy").exists():
        log("prep: svd features already cached")
        return
    dense = {s: np.load(FEATS / f"dense-{s}.npy") for s in ("train", "dev")}
    word = {s: sp.load_npz(FEATS / f"word-{s}.npz").tocsr() for s in ("train", "dev")}
    char = {s: sp.load_npz(FEATS / f"char-{s}.npz").tocsr() for s in ("train", "dev")}
    log("prep: fitting TruncatedSVD(128) on word/char (train only)...")
    svd_w = TruncatedSVD(n_components=128, random_state=SEED).fit(word["train"])
    svd_c = TruncatedSVD(n_components=128, random_state=SEED).fit(char["train"])
    for s in ("train", "dev"):
        X = np.hstack([dense[s], svd_w.transform(word[s]), svd_c.transform(char[s])])
        np.save(work / f"svd-{s}.npy", X)
    log("prep: cached svd features")


def get_X(mode: str, work: Path):
    if mode == "svd":
        if not (work / "svd-train.npy").exists():
            cmd_prep(work)
        return np.load(work / "svd-train.npy"), np.load(work / "svd-dev.npy")
    return build_sparse("train"), build_sparse("dev")


# ---------------------------------------------------------------- training


class FoldData:
    """5 fold-train Datasets + 1 full-train Dataset, built once per (X, mcs)."""

    def __init__(self, X_train, X_dev, mcs: int):
        self.X = X_train
        self.X_dev = X_dev
        self.fids = np.arange(X_train.shape[0]) % N_FOLDS
        ds_params = dict(min_data_in_leaf=mcs, verbosity=-1)
        self.folds = []
        for f in range(N_FOLDS):
            tr = self.fids != f
            ds = lgb.Dataset(X_train[tr], params=dict(ds_params), free_raw_data=False)
            self.folds.append((tr, ds))
        self.full = lgb.Dataset(X_train, params=dict(ds_params), free_raw_data=False)

    def run_head(self, y: np.ndarray, params: dict, iters_list):
        """OOF + dev preds for one target vector at each iteration prefix."""
        n = self.X.shape[0]
        nround = max(iters_list)
        oof = {it: np.zeros(n) for it in iters_list}
        for tr, ds in self.folds:
            ds.set_label(y[tr])
            bst = lgb.train(params, ds, num_boost_round=nround)
            X_va = self.X[~tr]
            for it in iters_list:
                oof[it][~tr] = bst.predict(X_va, num_iteration=it)
        self.full.set_label(y)
        bst = lgb.train(params, self.full, num_boost_round=nround)
        dev = {it: bst.predict(self.X_dev, num_iteration=it) for it in iters_list}
        return {it: (oof[it], dev[it]) for it in iters_list}


def sweep_file(work: Path, mode: str, leaves: int, mcs: int) -> Path:
    return work / f"sweep-{mode}-l{leaves}-m{mcs}.npz"


def cmd_sweep(mode: str, leaves: int, mcs: int, threads: int, work: Path) -> None:
    """Score heads for one (mode, leaves, mcs); OOF/dev saved at 300/600 trees."""
    out = sweep_file(work, mode, leaves, mcs)
    if out.exists():
        log(f"sweep {out.name}: already done")
        return
    scores_true = load_targets("train")["scores"]
    X_train, X_dev = get_X(mode, work)
    log(f"sweep mode={mode} leaves={leaves} mcs={mcs}: building datasets...")
    fd = FoldData(X_train, X_dev, mcs)
    n, m = X_train.shape[0], X_dev.shape[0]
    oof3 = {it: np.zeros((n, 3)) for it in SWEEP_TREES}
    dev3 = {it: np.zeros((m, 3)) for it in SWEEP_TREES}
    params = train_params(leaves, mcs, threads)
    for j in range(3):
        t0 = time.time()
        res = fd.run_head(scores_true[:, j], params, SWEEP_TREES)
        for it in SWEEP_TREES:
            oof3[it][:, j], dev3[it][:, j] = res[it]
        log(f"sweep mode={mode} leaves={leaves} mcs={mcs} head={MODELS[j]} "
            f"({time.time() - t0:.0f}s)")
    np.savez(out, oof300=oof3[300], oof600=oof3[600],
             dev300=dev3[300], dev600=dev3[600])
    for it in SWEEP_TREES:
        mse = float(np.mean((np.clip(oof3[it], 0, 1) - scores_true) ** 2))
        log(f"sweep mode={mode} leaves={leaves} mcs={mcs} trees={it} "
            f"oof_score_mse={mse:.5f}")


def cmd_select(work: Path) -> dict:
    """Aggregate sweep files, pick best config by OOF score MSE."""
    scores_true = load_targets("train")["scores"]
    rows = []
    for mode in ("svd", "sparse"):
        for leaves in SWEEP_LEAVES:
            for mcs in SWEEP_MCS:
                f = sweep_file(work, mode, leaves, mcs)
                if not f.exists():
                    raise SystemExit(f"missing sweep file {f}")
                data = np.load(f)
                for it in SWEEP_TREES:
                    oof = np.clip(data[f"oof{it}"], 0.0, 1.0)
                    mse = float(np.mean((oof - scores_true) ** 2))
                    rows.append(dict(mode=mode, num_leaves=leaves,
                                     min_child_samples=mcs, n_estimators=it,
                                     oof_score_mse=round(mse, 6)))
    best = min(rows, key=lambda r: r["oof_score_mse"])
    chosen = dict(best)
    with open(work / "sweep.json", "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, chosen=chosen), f, indent=2)
    log(f"select: chosen {chosen}")
    return chosen


def cmd_tokens(head: str, threads: int, work: Path) -> None:
    """Token heads (3 models) at the chosen config. head in {in,out,q75,q90}."""
    out_file = work / f"tokens-{head}.npz"
    if out_file.exists():
        log(f"tokens {head}: already done")
        return
    chosen = json.load(open(work / "sweep.json", encoding="utf-8"))["chosen"]
    mode, leaves = chosen["mode"], chosen["num_leaves"]
    mcs, trees = chosen["min_child_samples"], chosen["n_estimators"]
    targ = load_targets("train")
    if head == "in":
        Y = np.log(np.maximum(targ["in_tokens"].astype(float), 1.0))
        params = train_params(leaves, mcs, threads)
    else:
        Y = np.log(np.maximum(targ["out_tokens"].astype(float), 1.0))
        if head == "out":
            params = train_params(leaves, mcs, threads)
        else:
            alpha = {"q75": 0.75, "q90": 0.90}[head]
            params = train_params(leaves, mcs, threads,
                                  objective="quantile", alpha=alpha)
    X_train, X_dev = get_X(mode, work)
    log(f"tokens {head}: building datasets (mode={mode}, mcs={mcs})...")
    fd = FoldData(X_train, X_dev, mcs)
    oof = np.zeros((X_train.shape[0], 3))
    dev = np.zeros((X_dev.shape[0], 3))
    for j in range(3):
        t0 = time.time()
        res = fd.run_head(Y[:, j], params, [trees])
        oof[:, j], dev[:, j] = res[trees]
        log(f"tokens {head} head={MODELS[j]} ({time.time() - t0:.0f}s)")
    np.savez(out_file, oof=oof, dev=dev)


# ---------------------------------------------------------------- finalize


def monotone_cost(cost: np.ndarray) -> np.ndarray:
    cost = cost.copy()
    cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1 + 1e-12))
    cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1 + 1e-12))
    return cost


def cost_from_tokens(log_in_pred: np.ndarray, log_out_pred: np.ndarray) -> np.ndarray:
    cost = (np.exp(log_in_pred) * RATES[None, :, 0] / 1e6
            + np.exp(log_out_pred) * RATES[None, :, 1] / 1e6)
    return np.maximum(cost, 1e-9)


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


def cmd_finalize(work: Path) -> None:
    PREDS.mkdir(parents=True, exist_ok=True)
    sweep = json.load(open(work / "sweep.json", encoding="utf-8"))
    chosen = sweep["chosen"]
    mode, leaves = chosen["mode"], chosen["num_leaves"]
    mcs, trees = chosen["min_child_samples"], chosen["n_estimators"]
    targ = load_targets("train")
    scores_true = targ["scores"]

    sdata = np.load(sweep_file(work, mode, leaves, mcs))
    score_oof = np.clip(sdata[f"oof{trees}"], 0.0, 1.0)
    score_dev = np.clip(sdata[f"dev{trees}"], 0.0, 1.0)

    tok = {h: np.load(work / f"tokens-{h}.npz") for h in TOKEN_HEADS}
    in_oof, in_dev = tok["in"]["oof"], tok["in"]["dev"]
    variants = {
        "lgbm": ("regression", tok["out"]),
        "lgbm-q75": ("quantile-0.75", tok["q75"]),
        "lgbm-q90": ("quantile-0.90", tok["q90"]),
    }
    hp = dict(feature_mode=mode, num_leaves=leaves, n_estimators=trees,
              min_child_samples=mcs, learning_rate=LR, seed=SEED,
              deterministic=True, force_col_wise=True,
              svd_components=128 if mode == "svd" else None)
    for tag, (obj, t) in variants.items():
        cost_oof = monotone_cost(cost_from_tokens(in_oof, t["oof"]))
        cost_dev = monotone_cost(cost_from_tokens(in_dev, t["dev"]))
        np.savez(PREDS / f"{tag}-train.npz", score=score_oof, cost=cost_oof)
        np.savez(PREDS / f"{tag}-dev.npz", score=score_dev, cost=cost_dev)
        oof_metrics = metrics(score_oof, scores_true, cost_oof, targ["costs"])
        summary = dict(
            variant=tag,
            hyperparams=hp,
            out_tok_objective=obj,
            oof=oof_metrics,
            sweep=sweep["rows"],
            notes=("score/in_tok heads shared across lgbm/-q75/-q90; only the "
                   "out_tok objective differs. cost from predicted log tokens."),
        )
        with open(PREDS / f"{tag}-summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log(f"{tag}: oof score_mse={oof_metrics['score_mse_mean']:.5f} "
            f"logcost_mse={oof_metrics['logcost_mse_mean']:.5f}")


def cmd_all(threads: int, work: Path) -> None:
    cmd_prep(work)
    for mode in ("svd", "sparse"):
        for leaves in SWEEP_LEAVES:
            for mcs in SWEEP_MCS:
                cmd_sweep(mode, leaves, mcs, threads, work)
    cmd_select(work)
    for head in TOKEN_HEADS:
        cmd_tokens(head, threads, work)
    cmd_finalize(work)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["prep", "sweep", "select", "tokens",
                                    "finalize", "all"])
    ap.add_argument("--mode", choices=["svd", "sparse"])
    ap.add_argument("--leaves", type=int)
    ap.add_argument("--mcs", type=int)
    ap.add_argument("--head", choices=list(TOKEN_HEADS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--work", type=Path, default=WORK_DEFAULT)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    if args.cmd == "prep":
        cmd_prep(args.work)
    elif args.cmd == "sweep":
        cmd_sweep(args.mode, args.leaves, args.mcs, args.threads, args.work)
    elif args.cmd == "select":
        cmd_select(args.work)
    elif args.cmd == "tokens":
        cmd_tokens(args.head, args.threads, args.work)
    elif args.cmd == "finalize":
        cmd_finalize(args.work)
    elif args.cmd == "all":
        cmd_all(args.threads, args.work)


if __name__ == "__main__":
    main()
