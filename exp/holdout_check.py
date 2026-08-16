# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Rank-stability gate: Train-internal holdout, dev untouched.

Fold 0 (20% of train) = pseudo-dev B; folds 1-4 = A.
OOF predictions on B come from models fit on A only, so using B here is
leak-free. Utilization is calibrated on A(OOF) and applied to B.
Numbers are float (internal ranking only, NOT official-report metrics).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import alloc_lib
from eval_preds import UTIL_GRID, load_preds
from harness import TIER_MULTIPLIERS, TIERS, load_split


def subset_eval(pick, real_s, real_c, mult):
    n = len(pick)
    ratio = real_c[np.arange(n), pick].sum() / real_c[:, 0].sum()
    return real_s[np.arange(n), pick].mean(), ratio, ratio <= mult + 1e-12


def run(name: str) -> dict:
    score_pred, cost_pred = load_preds(name, "train")
    table = load_split("train")
    real_s = np.asarray(table["scores"])
    real_c = np.asarray(table["costs"])
    n = len(real_s)
    folds = np.arange(n) % 5
    a_mask = folds != 0
    b_mask = folds == 0

    out = {}
    weighted = 0.0
    weights = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
    for tier in TIERS:
        mult = TIER_MULTIPLIERS[tier]
        best = None
        for u in UTIL_GRID:
            pick_a = alloc_lib.greedy_allocate(
                score_pred[a_mask], cost_pred[a_mask], multiplier=mult, utilization=u
            )
            s, r, passed = subset_eval(pick_a, real_s[a_mask], real_c[a_mask], mult)
            if passed and (best is None or (s, -r) > (best[0], -best[1])):
                best = (s, r, u)
        u_star = best[2] if best else 1.0 / mult
        pick_b = alloc_lib.greedy_allocate(
            score_pred[b_mask], cost_pred[b_mask], multiplier=mult, utilization=u_star
        )
        s, r, passed = subset_eval(pick_b, real_s[b_mask], real_c[b_mask], mult)
        out[tier] = {"u_star": u_star, "B_score": round(s, 4), "B_ratio": round(r, 4), "B_pass": bool(passed)}
        weighted += weights[tier] * (s if passed else 0.0)
    out["B_weighted"] = round(weighted, 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    for name in args.names:
        print(name, run(name))


if __name__ == "__main__":
    main()
