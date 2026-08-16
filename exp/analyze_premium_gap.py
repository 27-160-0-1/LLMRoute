# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Where does premium lose vs the budget-oracle? Category-level diagnosis."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import alloc_lib
from eda import categorize
from eval_preds import load_preds
from harness import MODEL_IDS, TIER_MULTIPLIERS, load_split
from oracle_budget import lagrangian_select


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "knn-k40"
    table = load_split("dev")
    scores = np.asarray(table["scores"])
    costs = np.asarray(table["costs"])
    texts = table["texts"]
    cats = np.asarray([categorize(t) for t in texts])
    n = len(texts)
    mult = TIER_MULTIPLIERS["premium"]
    light_total = costs[:, 0].sum()

    oracle = lagrangian_select(scores.copy(), costs, light_total * mult)

    dev_score, dev_cost = load_preds(name, "dev")
    best = None
    grid = [round(0.50 + 0.05 * i, 2) for i in range(13)]
    for k1_u in grid:
        for fill_u in grid:
            pick = alloc_lib.two_stage_premium(
                dev_score, dev_cost, multiplier=mult, k1_utilization=k1_u, fill_utilization=fill_u
            )
            s = scores[np.arange(n), pick].mean()
            r = costs[np.arange(n), pick].sum() / light_total
            if r <= mult and (best is None or s > best[0]):
                best = (s, r, pick, (k1_u, fill_u))
    ours = best[2]
    print(f"{name} premium D4 best: score {best[0]:.4f} ratio {best[1]:.4f} params {best[3]}")

    for label, pick in (("oracle", oracle), ("ours", ours)):
        counts = {m: int((pick == j).sum()) for j, m in enumerate(MODEL_IDS)}
        spend = {m: float(costs[pick == j, j].sum()) for j, m in enumerate(MODEL_IDS)}
        print(f"{label}: counts {counts} spend {spend}")

    print("\n카테고리별 [oracle k1 수 / ours k1 수 / oracle-ours 점수차합]:")
    for cat in sorted(set(cats.tolist())):
        mask = cats == cat
        o_k1 = int((oracle[mask] == 2).sum())
        u_k1 = int((ours[mask] == 2).sum())
        gap = float(
            scores[mask, :][np.arange(mask.sum()), oracle[mask]].sum()
            - scores[mask, :][np.arange(mask.sum()), ours[mask]].sum()
        )
        print(f"  {cat:16s}: {o_k1:4d} / {u_k1:4d} / {gap:+.1f}")

    # where we pick k1 wrongly (k1 chosen, score gain <= 0)
    wasted = ((ours == 2) & (scores[:, 2] <= scores[:, 0])).sum()
    missed = ((oracle == 2) & (ours != 2) & (scores[:, 2] > scores[:, 0])).sum()
    print(f"\nours k1 낭비(이득<=0): {wasted}, oracle k1 누락: {missed}")


if __name__ == "__main__":
    main()
