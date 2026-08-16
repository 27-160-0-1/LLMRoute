# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Budget-constrained oracle (perfect knowledge) per tier + reference rows."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from harness import (
    MODEL_IDS,
    TIER_MULTIPLIERS,
    TIERS,
    append_result,
    evaluate,
    load_split,
)


def lagrangian_select(scores, costs, cap):
    """Multi-choice knapsack via lambda binary search + greedy fill."""

    n = scores.shape[0]

    def choose(lam):
        util = scores - lam * costs
        # tie-break: prefer cheaper model on equal utility
        pick = np.zeros(n, dtype=int)
        for j in (1, 2):
            better = util[:, j] > util[np.arange(n), pick] + 1e-15
            cheaper_equal = (
                np.isclose(util[:, j], util[np.arange(n), pick])
                & (costs[:, j] < costs[np.arange(n), pick])
            )
            pick = np.where(better | cheaper_equal, j, pick)
        return pick

    pick = choose(0.0)
    if costs[np.arange(n), pick].sum() <= cap:
        best = pick
    else:
        low, high = 0.0, 1.0
        while costs[np.arange(n), choose(high)].sum() > cap and high < 2**60:
            low, high = high, high * 2
        for _ in range(200):
            mid = (low + high) / 2
            if costs[np.arange(n), choose(mid)].sum() <= cap:
                high = mid
            else:
                low = mid
        best = choose(high)
    # greedy fill of leftover budget with best score-per-cost upgrades
    total = costs[np.arange(n), best].sum()
    upgrades = []
    for i in range(n):
        for j in range(3):
            ds = scores[i, j] - scores[i, best[i]]
            dc = costs[i, j] - costs[i, best[i]]
            if ds > 0:
                upgrades.append((ds / max(dc, 1e-12), ds, dc, i, j))
    upgrades.sort(reverse=True)
    for _, ds, dc, i, j in upgrades:
        cur = costs[i, best[i]]
        nds = scores[i, j] - scores[i, best[i]]
        ndc = costs[i, j] - cur
        if nds <= 0:
            continue
        if total + ndc <= cap:
            total += ndc
            best[i] = j
    return best


def main() -> None:
    for split in ("train", "dev"):
        table = load_split(split)
        scores = np.asarray(table["scores"])
        costs = np.asarray(table["costs"])
        light_total = costs[:, 0].sum()
        assignment_by_tier = {}
        for tier in TIERS:
            cap = light_total * TIER_MULTIPLIERS[tier]
            start = time.time()
            pick = lagrangian_select(scores.copy(), costs, cap)
            elapsed = time.time() - start
            assignment_by_tier[tier] = [MODEL_IDS[j] for j in pick]
            print(f"{split} {tier}: solved in {elapsed:.1f}s")
        report = evaluate(split, assignment_by_tier)
        row = append_result(
            f"[기준] budget-oracle ({split})",
            report,
            notes="실측 outcome 완전정보 하 등급별 예산제약 상한",
        )
        print(row)


if __name__ == "__main__":
    main()
