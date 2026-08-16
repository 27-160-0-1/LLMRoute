# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Decision-axis comparison on one prediction set.

Premium: D4 two-stage grid (k1_util x fill_util) vs D2/D3.
Fast/Balanced: D2 vs D3 fine grid.
Registers the best combined policy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import alloc_lib
import registry_lib
from eval_preds import UTIL_GRID, fast_eval, load_preds, realized
from harness import MODEL_IDS, TIER_MULTIPLIERS, TIERS, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    dev_score, dev_cost = load_preds(args.name, "dev")
    real_s, real_c = realized("dev")

    assignment = {}
    chosen = {}
    over_probs = {}
    for tier in ("fast", "balanced"):
        mult = TIER_MULTIPLIERS[tier]
        best = None
        for allocator in ("lagrangian", "greedy"):
            for u in UTIL_GRID:
                fn = (
                    alloc_lib.greedy_allocate
                    if allocator == "greedy"
                    else alloc_lib.lagrangian_allocate
                )
                pick = fn(dev_score, dev_cost, multiplier=mult, utilization=u)
                score, ratio, passed = fast_eval(pick, real_s, real_c, mult)
                if passed and (best is None or (score, -ratio) > (best[0], -best[1])):
                    best = (score, ratio, pick, f"{allocator}@{u}")
        assignment[tier] = [MODEL_IDS[j] for j in best[2]]
        chosen[tier] = best[3]

    mult = TIER_MULTIPLIERS["premium"]
    best = None
    grid = [round(0.50 + 0.05 * i, 2) for i in range(13)]  # 0.50..1.10
    for k1_u in grid:
        for fill_u in grid:
            pick = alloc_lib.two_stage_premium(
                dev_score, dev_cost, multiplier=mult, k1_utilization=k1_u, fill_utilization=fill_u
            )
            score, ratio, passed = fast_eval(pick, real_s, real_c, mult)
            if passed and (best is None or (score, -ratio) > (best[0], -best[1])):
                best = (score, ratio, pick, f"D4@k1:{k1_u},fill:{fill_u}")
    # also plain D2/D3 for comparison
    for allocator in ("lagrangian", "greedy"):
        for u in UTIL_GRID:
            fn = (
                alloc_lib.greedy_allocate
                if allocator == "greedy"
                else alloc_lib.lagrangian_allocate
            )
            pick = fn(dev_score, dev_cost, multiplier=mult, utilization=u)
            score, ratio, passed = fast_eval(pick, real_s, real_c, mult)
            if passed and (best is None or (score, -ratio) > (best[0], -best[1])):
                best = (score, ratio, pick, f"{allocator}@{u}")
    assignment["premium"] = [MODEL_IDS[j] for j in best[2]]
    chosen["premium"] = best[3]

    for tier in TIERS:
        pick = np.asarray([MODEL_IDS.index(m) for m in assignment[tier]])
        over_probs[tier] = alloc_lib.bootstrap_over_probability(
            real_c[np.arange(len(pick)), pick], real_c[:, 0], TIER_MULTIPLIERS[tier]
        )

    report = evaluate("dev", assignment)
    metrics = registry_lib.metrics_from_report(report, over_probs)
    exp_id = registry_lib.register(
        family="decision",
        config={
            "name": f"{args.name}+best-decision",
            "pred_set": args.name,
            "decision": chosen,
            "calibration": "dev",
        },
        metrics=metrics,
        notes=f"등급별 최적 배분 {chosen}. {args.note}",
    )
    print(f"registered {exp_id}")
    tiers = report["tiers"]
    print({t: (tiers[t]["quality_score"], tiers[t]["budget_ratio"]) for t in TIERS})
    print("final:", report["final_score"])
    import subprocess

    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "make_report.py")], check=False)


if __name__ == "__main__":
    main()
