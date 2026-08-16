# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""A3/A7 ablation: surface-noise augmentation + down-weighting on LGBM backbone.

Augmented copies inherit the source row's fold (no leakage) and labels
(label-preservation assumption). Usage:
  python exp/augment/train_aug.py --variants 2 --weight 0.5 --name lgbm-aug2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import sparse

import feat_lib
from harness import ROOT, load_split
from augment.noise_lib import perturb_batch

PREDS = ROOT / "build" / "preds"
FEATS = ROOT / "build" / "feats"
RATES = np.array([[1.0, 4.0], [2.127, 8.509], [6.565, 26.260]]) / 1e6


def featurize(texts):
    dense, word, char = feat_lib.featurize_batch(texts)
    return dense, word, char


def build_matrix(dense, word, char, svd_word, svd_char):
    return np.hstack([dense, svd_word.transform(word), svd_char.transform(char)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    name = args.name or f"lgbm-aug{args.variants}w{args.weight}"

    import lightgbm as lgb
    from sklearn.decomposition import TruncatedSVD

    table = load_split("train")
    texts = table["texts"]
    n = len(texts)
    folds = np.arange(n) % 5
    tgt = np.load(FEATS / "targets-train.npz")
    y_score = tgt["scores"]
    y_log_out = np.log1p(tgt["out_tokens"])
    y_log_in = np.log(np.maximum(tgt["in_tokens"], 1))

    dense0 = np.load(FEATS / "dense-train.npy")
    word0 = sparse.load_npz(FEATS / "word-train.npz")
    char0 = sparse.load_npz(FEATS / "char-train.npz")

    print("featurizing augmented variants...")
    start = time.time()
    aug_sets = []
    for v, batch in enumerate(perturb_batch(texts, args.variants, strength=args.strength)):
        aug_sets.append(featurize(batch))
        print(f"  variant {v}: {time.time() - start:.0f}s")

    svd_word = TruncatedSVD(n_components=128, random_state=42).fit(word0)
    svd_char = TruncatedSVD(n_components=128, random_state=42).fit(char0)
    X0 = build_matrix(dense0, word0, char0, svd_word, svd_char)
    X_aug = [build_matrix(d, w, c, svd_word, svd_char) for d, w, c in aug_sets]

    dense_dev = np.load(FEATS / "dense-dev.npy")
    word_dev = sparse.load_npz(FEATS / "word-dev.npz")
    char_dev = sparse.load_npz(FEATS / "char-dev.npz")
    X_dev = build_matrix(dense_dev, word_dev, char_dev, svd_word, svd_char)

    def make_model():
        return lgb.LGBMRegressor(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            min_child_samples=15, random_state=42, verbose=-1,
        )

    heads = [("score", y_score), ("log_out", y_log_out), ("log_in", y_log_in)]
    oof = {h: np.zeros_like(y) for h, y in heads}
    dev = {h: np.zeros((X_dev.shape[0], 3)) for h, _ in heads}

    def stacked(mask):
        Xs = [X0[mask]] + [Xa[mask] for Xa in X_aug]
        X = np.vstack(Xs)
        w = np.concatenate([np.ones(mask.sum())] + [np.full(mask.sum(), args.weight)] * args.variants)
        return X, w

    for head, Y in heads:
        for j in range(3):
            for f in range(5):
                tr = folds != f
                X_tr, w_tr = stacked(tr)
                y_tr = np.concatenate([Y[tr, j]] * (1 + args.variants))
                model = make_model()
                model.fit(X_tr, y_tr, sample_weight=w_tr)
                oof[head][folds == f, j] = model.predict(X0[folds == f])
            X_tr, w_tr = stacked(np.ones(n, dtype=bool))
            y_tr = np.concatenate([Y[:, j]] * (1 + args.variants))
            model = make_model()
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            dev[head][:, j] = model.predict(X_dev)
        print(f"head {head} done")

    def to_contract(scores_raw, log_out, log_in):
        score = np.clip(scores_raw, 0, 1)
        out_tok = np.maximum(np.expm1(log_out), 0)
        in_tok = np.maximum(np.exp(log_in), 1)
        cost = in_tok * RATES[:, 0][None, :] + out_tok * RATES[:, 1][None, :]
        cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1 + 1e-12))
        cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1 + 1e-12))
        return score, cost

    s_tr, c_tr = to_contract(oof["score"], oof["log_out"], oof["log_in"])
    s_de, c_de = to_contract(dev["score"], dev["log_out"], dev["log_in"])
    np.savez(PREDS / f"{name}-train.npz", score=s_tr, cost=c_tr)
    np.savez(PREDS / f"{name}-dev.npz", score=s_de, cost=c_de)
    mse = ((s_tr - y_score) ** 2).mean(axis=0)
    summary = {
        "variant": name,
        "augment": {"kind": "A3-surface-noise", "variants": args.variants, "weight": args.weight, "strength": args.strength},
        "oof_score_mse": mse.tolist(),
        "oof_score_mse_mean": float(mse.mean()),
    }
    (PREDS / f"{name}-summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
