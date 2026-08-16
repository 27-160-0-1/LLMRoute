# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Fast-tier gap diagnosis vs budget-oracle."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import alloc_lib
from eda import categorize
from eval_preds import UTIL_GRID, load_preds
from harness import MODEL_IDS, TIER_MULTIPLIERS, load_split
from oracle_budget import lagrangian_select


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "knn-k40"
    table = load_split("dev")
    scores = np.asarray(table["scores"])
    costs = np.asarray(table["costs"])
    cats = np.asarray([categorize(t) for t in table["texts"]])
    n = len(scores)
    mult = TIER_MULTIPLIERS["fast"]
    light_total = costs[:, 0].sum()

    oracle = lagrangian_select(scores.copy(), costs, light_total * mult)

    dev_score, dev_cost = load_preds(name, "dev")
    best = None
    for u in UTIL_GRID:
        for fn in (alloc_lib.greedy_allocate, alloc_lib.lagrangian_allocate):
            pick = fn(dev_score, dev_cost, multiplier=mult, utilization=u)
            s = scores[np.arange(n), pick].mean()
            r = costs[np.arange(n), pick].sum() / light_total
            if r <= mult and (best is None or s > best[0]):
                best = (s, r, pick)
    ours = best[2]
    print(f"{name} fast best: score {best[0]:.4f} ratio {best[1]:.4f}")

    for label, pick in (("oracle", oracle), ("ours", ours)):
        counts = {m: int((pick == j).sum()) for j, m in enumerate(MODEL_IDS)}
        spend_extra = float(
            (costs[np.arange(n), pick] - costs[:, 0]).sum()
        )
        print(f"{label}: counts {counts} 추가지출 {spend_extra:.3f} (한도 {0.25 * light_total:.3f})")

    print("\n카테고리별 [oracle 승격수(ax31/k1) / ours 승격수 / 점수차합]:")
    for cat in sorted(set(cats.tolist())):
        mask = cats == cat
        o_up = (int((oracle[mask] == 1).sum()), int((oracle[mask] == 2).sum()))
        u_up = (int((ours[mask] == 1).sum()), int((ours[mask] == 2).sum()))
        gap = float(
            scores[mask, :][np.arange(mask.sum()), oracle[mask]].sum()
            - scores[mask, :][np.arange(mask.sum()), ours[mask]].sum()
        )
        print(f"  {cat:16s}: {o_up} / {u_up} / {gap:+.1f}")

    # oracle's k1 picks in fast: how cheap are they?
    k1_rows = oracle == 2
    if k1_rows.any():
        rel = costs[k1_rows, 2] / costs[k1_rows, 0]
        print(
            f"\noracle fast의 k1 {k1_rows.sum()}건: k1비용/light비용 중앙값 {np.median(rel):.1f}, "
            f"평균 {rel.mean():.1f} — 절대 k1 비용 중앙값 {np.median(costs[k1_rows,2]):.5f}"
        )


if __name__ == "__main__":
    main()
