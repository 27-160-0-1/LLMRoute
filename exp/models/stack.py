# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""M6 stacking: meta-model over base prediction sets (OOF, leak-free).

Usage: python exp/models/stack.py base1 base2 ... [--variant name]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from harness import ROOT

PREDS = ROOT / "build" / "preds"
FEATS = ROOT / "build" / "feats"
RATES = np.array([[1.0, 4.0], [2.127, 8.509], [6.565, 26.260]]) / 1e6


def load_base(names, split):
    blocks = []
    for name in names:
        data = np.load(PREDS / f"{name}-{split}.npz")
        blocks.append(np.clip(data["score"], 0, 1))
        blocks.append(np.log(np.maximum(data["cost"], 1e-9)))
    return np.hstack(blocks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bases", nargs="+")
    parser.add_argument("--variant", default="stack")
    parser.add_argument("--meta", default="lgbm", choices=["lgbm", "ridge"])
    parser.add_argument("--with-dense", action="store_true")
    args = parser.parse_args()

    X_train = load_base(args.bases, "train")
    X_dev = load_base(args.bases, "dev")
    if args.with_dense:
        X_train = np.hstack([X_train, np.load(FEATS / "dense-train.npy")])
        X_dev = np.hstack([X_dev, np.load(FEATS / "dense-dev.npy")])
    tgt_train = np.load(FEATS / "targets-train.npz")
    y_score = tgt_train["scores"]
    y_logcost = np.log(tgt_train["costs"])

    n = X_train.shape[0]
    folds = np.arange(n) % 5

    def fit_predict(X_fit, Y_fit, X_apply):
        preds = np.empty((X_apply.shape[0], Y_fit.shape[1]))
        if args.meta == "ridge":
            from sklearn.linear_model import Ridge

            model = Ridge(alpha=1.0)
            model.fit(X_fit, Y_fit)
            return model.predict(X_apply)
        import lightgbm as lgb

        for j in range(Y_fit.shape[1]):
            model = lgb.LGBMRegressor(
                n_estimators=250,
                num_leaves=15,
                learning_rate=0.05,
                min_child_samples=20,
                random_state=42,
                verbose=-1,
            )
            model.fit(X_fit, Y_fit[:, j])
            preds[:, j] = model.predict(X_apply)
        return preds

    Y = np.hstack([y_score, y_logcost])
    oof = np.empty_like(Y)
    for f in range(5):
        mask = folds == f
        oof[mask] = fit_predict(X_train[~mask], Y[~mask], X_train[mask])
    dev_pred = fit_predict(X_train, Y, X_dev)

    def to_contract(raw):
        score = np.clip(raw[:, :3], 0, 1)
        cost = np.exp(np.clip(raw[:, 3:], -20, 5))
        cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1 + 1e-12))
        cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1 + 1e-12))
        return score, cost

    for split, raw in (("train", oof), ("dev", dev_pred)):
        score, cost = to_contract(raw)
        np.savez(PREDS / f"{args.variant}-{split}.npz", score=score, cost=cost)

    oof_score, _ = to_contract(oof)
    mse = ((oof_score - y_score) ** 2).mean(axis=0)
    summary = {
        "variant": args.variant,
        "bases": args.bases,
        "meta": args.meta,
        "with_dense": args.with_dense,
        "oof_score_mse": mse.tolist(),
        "oof_score_mse_mean": float(mse.mean()),
    }
    (PREDS / f"{args.variant}-summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
