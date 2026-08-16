# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Audit #2: regression vs ordinal classification vs binary classification.

score in {0, 0.25, 0.5, 0.75, 1} is ordinal.  Three prediction variants:

  ordinal     per model, 4 cumulative binary LGBMClassifiers
              P(s>=0.25), P(s>=0.5), P(s>=0.75), P(s>=1.0)
              (num_leaves 31, n_estimators 300, lr 0.05, seed 42).
              Cumulative probabilities are made monotone non-increasing via a
              running minimum (later thresholds can never exceed earlier ones),
              then expected score = 0.25 * sum_k P(s>=t_k).
  binary      per model, the single P(s>=0.5) classifier's raw probability is
              used directly as the score prediction.
  ordinal-iso the ordinal expected score passed through a per-model isotonic
              regression fit on train OOF predictions.  Train OOF stays honest:
              for fold f the isotonic map is fit on the OOF pairs of the other
              4 folds only; dev uses an isotonic map fit on the full train OOF.

Cost head: none of these variants model cost; the cost arrays are copied
verbatim from build/preds/mlp-ens-{split}.npz (noted in each summary) and
re-passed through the monotone correction for safety.

OOF protocol: 5 folds, fold_id = np.arange(N) % 5, validation fold never seen
during training.  Dev predictions come from classifiers fit on the full train
split.  Features: dense81 + TruncatedSVD(word,128) + TruncatedSVD(char,128),
SVD fit on train only (reuses the cache in build/lgbm-work when present).

Reproducibility: seed 42, deterministic=True, force_col_wise=True, n_jobs=4.
Run: .venv\\Scripts\\python.exe exp\\models\\ordinal.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from lightgbm import LGBMClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
FEATS = ROOT / "build" / "feats"
PREDS = ROOT / "build" / "preds"
LGBM_WORK = ROOT / "build" / "lgbm-work"
OWN_WORK = ROOT / "build" / "ordinal-work"

SEED = 42
N_FOLDS = 5
MODELS = ["ax31-light", "ax31", "axk1-think"]
THRESHOLDS = [0.25, 0.5, 0.75, 1.0]

HP = dict(num_leaves=31, n_estimators=300, learning_rate=0.05,
          random_state=SEED, deterministic=True, force_col_wise=True,
          verbosity=-1, n_jobs=4)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_features():
    """dense81 + SVD128(word) + SVD128(char); reuse lgbm-work cache if present."""
    for work in (LGBM_WORK, OWN_WORK):
        if (work / "svd-train.npy").exists() and (work / "svd-dev.npy").exists():
            log(f"features: using cache {work}")
            return np.load(work / "svd-train.npy"), np.load(work / "svd-dev.npy")
    log("features: fitting TruncatedSVD(128) on train word/char...")
    dense = {s: np.load(FEATS / f"dense-{s}.npy") for s in ("train", "dev")}
    word = {s: sp.load_npz(FEATS / f"word-{s}.npz").tocsr() for s in ("train", "dev")}
    char = {s: sp.load_npz(FEATS / f"char-{s}.npz").tocsr() for s in ("train", "dev")}
    svd_w = TruncatedSVD(n_components=128, random_state=SEED).fit(word["train"])
    svd_c = TruncatedSVD(n_components=128, random_state=SEED).fit(char["train"])
    OWN_WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in ("train", "dev"):
        out[s] = np.hstack([dense[s], svd_w.transform(word[s]),
                            svd_c.transform(char[s])])
        np.save(OWN_WORK / f"svd-{s}.npy", out[s])
    return out["train"], out["dev"]


def fit_binary(X_tr, y_bin, X_va_list):
    """Fit one binary classifier; return P(class=1) for each matrix in X_va_list."""
    if y_bin.min() == y_bin.max():  # degenerate: single class in training fold
        const = float(y_bin[0])
        return [np.full(X.shape[0], const) for X in X_va_list]
    clf = LGBMClassifier(**HP)
    clf.fit(X_tr, y_bin)
    return [clf.predict_proba(X)[:, 1] for X in X_va_list]


def monotone_cost(cost: np.ndarray) -> np.ndarray:
    cost = cost.copy()
    cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1 + 1e-12))
    cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1 + 1e-12))
    return cost


def score_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    out = {"score_mse": {}, "score_mae": {}}
    for j, m in enumerate(MODELS):
        out["score_mse"][m] = float(np.mean((pred[:, j] - true[:, j]) ** 2))
        out["score_mae"][m] = float(np.mean(np.abs(pred[:, j] - true[:, j])))
    out["score_mse_mean"] = float(np.mean(list(out["score_mse"].values())))
    out["score_mae_mean"] = float(np.mean(list(out["score_mae"].values())))
    return out


def main() -> None:
    PREDS.mkdir(parents=True, exist_ok=True)
    X_train, X_dev = load_features()
    scores_true = np.load(FEATS / "targets-train.npz")["scores"]
    n, m = X_train.shape[0], X_dev.shape[0]
    fids = np.arange(n) % N_FOLDS

    # ---- cumulative binary classifiers: OOF + dev ----
    cum_oof = np.zeros((n, 3, 4))
    cum_dev = np.zeros((m, 3, 4))
    for j in range(3):
        for k, t in enumerate(THRESHOLDS):
            t0 = time.time()
            y_bin = (scores_true[:, j] >= t - 1e-9).astype(int)
            for f in range(N_FOLDS):
                tr = fids != f
                (p_va,) = fit_binary(X_train[tr], y_bin[tr], [X_train[~tr]])
                cum_oof[~tr, j, k] = p_va
            (p_dev,) = fit_binary(X_train, y_bin, [X_dev])
            cum_dev[:, j, k] = p_dev
            log(f"model={MODELS[j]} thr>={t}: pos_rate={y_bin.mean():.3f} "
                f"({time.time() - t0:.0f}s)")

    # raw P(>=0.5) for the binary variant (before monotone correction)
    binary_oof = cum_oof[:, :, 1].copy()
    binary_dev = cum_dev[:, :, 1].copy()

    # monotone non-increasing cumulative probabilities (running minimum)
    cum_oof = np.minimum.accumulate(cum_oof, axis=2)
    cum_dev = np.minimum.accumulate(cum_dev, axis=2)
    ordinal_oof = np.clip(0.25 * cum_oof.sum(axis=2), 0.0, 1.0)
    ordinal_dev = np.clip(0.25 * cum_dev.sum(axis=2), 0.0, 1.0)

    # ---- ordinal-iso: per-model isotonic calibration on train OOF ----
    iso_oof = np.zeros_like(ordinal_oof)
    iso_dev = np.zeros_like(ordinal_dev)
    for j in range(3):
        for f in range(N_FOLDS):  # honest OOF: fit on the other folds' OOF pairs
            tr = fids != f
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(ordinal_oof[tr, j], scores_true[tr, j])
            iso_oof[~tr, j] = iso.predict(ordinal_oof[~tr, j])
        iso_full = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso_full.fit(ordinal_oof[:, j], scores_true[:, j])
        iso_dev[:, j] = iso_full.predict(ordinal_dev[:, j])
    iso_oof = np.clip(iso_oof, 0.0, 1.0)
    iso_dev = np.clip(iso_dev, 0.0, 1.0)

    # ---- cost head: copied from mlp-ens ----
    cost_train = monotone_cost(np.load(PREDS / "mlp-ens-train.npz")["cost"])
    cost_dev = monotone_cost(np.load(PREDS / "mlp-ens-dev.npz")["cost"])

    variants = {
        "ordinal": (ordinal_oof, ordinal_dev,
                    "expected score from 4 monotone cumulative P(s>=t)"),
        "binary": (np.clip(binary_oof, 0, 1), np.clip(binary_dev, 0, 1),
                   "score = raw P(s>=0.5) from a single binary classifier"),
        "ordinal-iso": (iso_oof, iso_dev,
                        "ordinal expected score + per-model isotonic calibration "
                        "(train OOF fit; fold-honest for the OOF file)"),
    }
    for tag, (s_oof, s_dev, desc) in variants.items():
        np.savez(PREDS / f"{tag}-train.npz", score=s_oof, cost=cost_train)
        np.savez(PREDS / f"{tag}-dev.npz", score=s_dev, cost=cost_dev)
        met = score_metrics(s_oof, scores_true)
        summary = dict(
            variant=tag,
            description=desc,
            hyperparams=dict(
                classifier="LGBMClassifier", **{k: v for k, v in HP.items()},
                thresholds=THRESHOLDS, folds=N_FOLDS,
                features="dense81+SVD128(word)+SVD128(char), train-fit svd",
            ),
            oof=met,
            cost_head="copied from mlp-ens predictions (no own cost model)",
        )
        with open(PREDS / f"{tag}-summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        log(f"{tag}: oof_score_mse={met['score_mse_mean']:.5f} "
            f"per-model={[round(met['score_mse'][mm], 5) for mm in MODELS]}")


if __name__ == "__main__":
    main()
