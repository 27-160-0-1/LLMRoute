# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Blend-weight search scored on the train-internal holdout (dev untouched)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import alloc_lib
from eval_preds import UTIL_GRID
from harness import TIER_MULTIPLIERS, TIERS, load_split

PREDS = Path(__file__).resolve().parents[1] / "build" / "preds"
WEIGHTS_TIER = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}


def holdout_weighted(score_pred, cost_pred, real_s, real_c, folds):
    a_mask = folds != 0
    b_mask = folds == 0
    total = 0.0
    for tier in TIERS:
        mult = TIER_MULTIPLIERS[tier]
        best = None
        for u in UTIL_GRID[::2]:
            pick = alloc_lib.greedy_allocate(
                score_pred[a_mask], cost_pred[a_mask], multiplier=mult, utilization=u
            )
            na = a_mask.sum()
            r = real_c[a_mask][np.arange(na), pick].sum() / real_c[a_mask][:, 0].sum()
            s = real_s[a_mask][np.arange(na), pick].mean()
            if r <= mult and (best is None or s > best[0]):
                best = (s, u)
        u_star = best[1] if best else 1.0 / mult
        pick = alloc_lib.greedy_allocate(
            score_pred[b_mask], cost_pred[b_mask], multiplier=mult, utilization=u_star
        )
        nb = b_mask.sum()
        r = real_c[b_mask][np.arange(nb), pick].sum() / real_c[b_mask][:, 0].sum()
        s = real_s[b_mask][np.arange(nb), pick].mean()
        total += WEIGHTS_TIER[tier] * (s if r <= mult else 0.0)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bases", nargs="+")
    parser.add_argument("--cost-from", required=True)
    parser.add_argument("--trials", type=int, default=40)
    args = parser.parse_args()

    table = load_split("train")
    real_s = np.asarray(table["scores"])
    real_c = np.asarray(table["costs"])
    n = len(real_s)
    folds = np.arange(n) % 5

    base_scores = [
        np.clip(np.load(PREDS / f"{b}-train.npz")["score"], 0, 1) for b in args.bases
    ]
    cost_pred = np.load(PREDS / f"{args.cost_from}-train.npz")["cost"]

    rng = np.random.default_rng(20260816)
    results = []
    # include equal weights + one-hot corners
    candidates = [np.ones(len(args.bases))]
    for i in range(len(args.bases)):
        w = np.ones(len(args.bases)) * 0.25
        w[i] = 1.0
        candidates.append(w)
    while len(candidates) < args.trials:
        candidates.append(rng.dirichlet(np.ones(len(args.bases)) * 1.5))
    for w in candidates:
        w = np.asarray(w, dtype=float)
        w = w / w.sum()
        score_pred = sum(wi * s for wi, s in zip(w, base_scores))
        val = holdout_weighted(score_pred, cost_pred, real_s, real_c, folds)
        results.append((val, w.tolist()))
        print(f"{val:.4f}  {np.round(w, 3).tolist()}")
    results.sort(reverse=True)
    print("\nTOP5:")
    for val, w in results[:5]:
        print(f"  {val:.4f}  {np.round(np.asarray(w), 3).tolist()}")
    out = {"bases": args.bases, "cost_from": args.cost_from, "top": results[:5]}
    (PREDS / "blend-weight-search.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
