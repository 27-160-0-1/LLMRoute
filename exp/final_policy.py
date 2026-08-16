# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Final per-tier utilization selection under a triple risk gate.

Gates (all must pass at the chosen utilization):
  bootstrap P(over) on dev        < 1.0% (premium < 0.5%)
  category-shift P(over) (+-50%)  < 5.0%
  CLT P(over) from train residuals< 1.0%
Utilization = max grid point passing all gates; reported with dev official
score for transparency. Dev is used for measurement, not for picking the
utilization that maximizes score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import alloc_lib
import conformal_lib
import registry_lib
import stress_lib
from eval_preds import load_preds, realized
from harness import MODEL_IDS, TIER_MULTIPLIERS, TIERS, evaluate, load_split

GRID = [round(0.60 + 0.01 * i, 2) for i in range(51)]  # 0.60..1.10

BOOT_LIMIT = {"fast": 0.01, "balanced": 0.01, "premium": 0.005}
STRESS_LIMIT = 0.05
CLT_LIMIT = 0.01


def allocate(tier, score_pred, cost_pred, u, premium_d4, k1_cap=None):
    mult = TIER_MULTIPLIERS[tier]
    if tier == "premium" and premium_d4:
        k1_u, fill_ratio = premium_d4
        return alloc_lib.two_stage_premium(
            score_pred, cost_pred, multiplier=mult,
            k1_utilization=k1_u, fill_utilization=fill_ratio,
            k1_cost_cap=k1_cap,
        )
    return alloc_lib.greedy_allocate(
        score_pred, cost_pred, multiplier=mult, utilization=u, k1_cost_cap=k1_cap
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--premium-d4", default=None, help="k1_u,fill_u 예: 0.65,0.75")
    parser.add_argument("--stress-range", type=float, default=0.5, help="카테고리 시프트 폭 (0.5 = ±50%)")
    parser.add_argument("--k1-cap", type=float, default=None)
    parser.add_argument("--premium-cost-from", default=None, help="premium 배분용 비용 pred set (예: q90 합성)")
    parser.add_argument("--cost-resid-from", default=None, help="잔차 추정용 pred set (기본: name)")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    dev_score, dev_cost = load_preds(args.name, "dev")
    premium_cost = dev_cost
    if args.premium_cost_from:
        _, premium_cost = load_preds(args.premium_cost_from, "dev")
    real_s, real_c = realized("dev")
    texts = load_split("dev")["texts"]
    n = len(real_s)
    resid_name = args.cost_resid_from or args.name
    resid_mean, resid_std = conformal_lib.residual_moments(resid_name)
    premium_d4 = None
    if args.premium_d4:
        parts = args.premium_d4.split(",")
        premium_d4 = (float(parts[0]), float(parts[1]))

    chosen = {}
    risk_table = {}
    assignment = {}
    for tier in TIERS:
        mult = TIER_MULTIPLIERS[tier]
        tier_cost = premium_cost if tier == "premium" else dev_cost
        rows = []
        selected = None
        grid = [None] if (tier == "premium" and premium_d4) else GRID
        for u in grid:
            pick = allocate(tier, dev_score, tier_cost, u, premium_d4, args.k1_cap if tier == "premium" else None)
            sel_cost = real_c[np.arange(n), pick]
            ratio = sel_cost.sum() / real_c[:, 0].sum()
            score = real_s[np.arange(n), pick].mean()
            boot = alloc_lib.bootstrap_over_probability(sel_cost, real_c[:, 0], mult)
            row = {"u": u, "dev_score": round(score, 5), "dev_ratio": round(ratio, 5), "boot": boot}
            if boot <= BOOT_LIMIT[tier] and ratio <= mult:
                stress = stress_lib.category_shift_over_probability(
                    texts, sel_cost, real_c[:, 0], mult, iterations=1000,
                    weight_low=1.0 - args.stress_range,
                    weight_high=1.0 + args.stress_range,
                )
                clt = conformal_lib.clt_over_probability(
                    pick, dev_cost, resid_std, mult, residual_mean=resid_mean
                )
                row.update({"stress": stress["over_prob"], "stress_p99": stress["ratio_p99"], "clt": round(clt, 5)})
                if stress["over_prob"] <= STRESS_LIMIT and clt <= CLT_LIMIT:
                    selected = (u, pick, row)
            rows.append(row)
        if selected is None:
            # fall back to most conservative grid point
            u = GRID[0]
            pick = allocate(tier, dev_score, dev_cost, u, premium_d4)
            selected = (u, pick, rows[0])
        chosen[tier] = selected[0]
        assignment[tier] = [MODEL_IDS[j] for j in selected[1]]
        risk_table[tier] = {"chosen": selected[2], "sweep_tail": rows[-8:]}

    report = evaluate("dev", assignment)
    out = {
        "pred_set": args.name,
        "premium_d4": premium_d4,
        "chosen_utilization": chosen,
        "risk": {t: risk_table[t]["chosen"] for t in TIERS},
        "dev_final": report["final_score"],
        "dev_tiers": {
            t: (report["tiers"][t]["quality_score"], report["tiers"][t]["budget_ratio"])
            for t in TIERS
        },
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    detail_path = Path(__file__).resolve().parent / f"final-policy-{args.name}-s{int(args.stress_range*100)}.json"
    detail_path.write_text(
        json.dumps({"summary": out, "risk_table": risk_table}, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.register:
        over_probs = {t: risk_table[t]["chosen"]["boot"] for t in TIERS}
        metrics = registry_lib.metrics_from_report(report, over_probs)
        exp_id = registry_lib.register(
            family="decision",
            config={
                "name": f"{args.name}+risk-gated",
                "pred_set": args.name,
                "decision": {"utilization": chosen, "premium_d4": premium_d4},
                "calibration": "risk-gate(boot/stress/clt)",
            },
            metrics=metrics,
            notes=f"삼중 위험게이트 확정 마진. risk={json.dumps({t: risk_table[t]['chosen'] for t in TIERS}, ensure_ascii=False)}",
        )
        print(f"registered {exp_id}")


if __name__ == "__main__":
    main()
