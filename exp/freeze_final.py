# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Freeze and register the final routing policy (official scoring)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import alloc_lib
import registry_lib
import stress_lib
from eval_preds import load_preds, realized
from harness import MODEL_IDS, TIER_MULTIPLIERS, TIERS, evaluate, load_split

POLICY_SPEC = {
    "score_members": ["xgb-mono", "irt1d", "knn-k40", "lgbm"],
    "cost_mean": ["lgbm", "xgb-mono"],
    "premium_k1_cost": "lgbm-q90",
    "fast": {"decision": "greedy", "allow_k1": False, "utilization": 0.93},
    "balanced": {"decision": "greedy", "allow_k1": False, "utilization": 0.88},
    "premium": {
        "decision": "two-stage",
        "k1_utilization": 0.65,
        "fill_utilization": 0.70,
        "k1_cost_cap": 0.1,
    },
}


def final_assignment(split: str = "dev"):
    score, cost = load_preds("blend4-cost2b", split)
    _, cost_q90 = load_preds("blend4-q90k1", split)
    picks = {}
    picks["fast"] = alloc_lib.greedy_allocate(
        score, cost, multiplier=1.25, utilization=0.93, allow_k1=False
    )
    picks["balanced"] = alloc_lib.greedy_allocate(
        score, cost, multiplier=2.0, utilization=0.88, allow_k1=False
    )
    picks["premium"] = alloc_lib.two_stage_premium(
        score, cost_q90, multiplier=4.0,
        k1_utilization=0.65, fill_utilization=0.70, k1_cost_cap=0.1,
    )
    return picks


def main() -> None:
    picks = final_assignment("dev")
    real_s, real_c = realized("dev")
    texts = load_split("dev")["texts"]
    n = len(real_s)
    assignment = {t: [MODEL_IDS[j] for j in picks[t]] for t in TIERS}
    over = {}
    stress = {}
    for tier in TIERS:
        sel = real_c[np.arange(n), picks[tier]]
        over[tier] = alloc_lib.bootstrap_over_probability(
            sel, real_c[:, 0], TIER_MULTIPLIERS[tier]
        )
        stress[tier] = stress_lib.category_shift_over_probability(
            texts, sel, real_c[:, 0], TIER_MULTIPLIERS[tier], iterations=2000
        )["over_prob"]
    report = evaluate("dev", assignment)
    metrics = registry_lib.metrics_from_report(report, over, stress_over=stress)
    exp_id = registry_lib.register(
        family="ensemble",
        config={"name": "FINAL blend4+risk-gated(no-k1 F/B)", **POLICY_SPEC},
        metrics=metrics,
        notes=(
            "동결 최종 정책. F/B는 k1 금지로 꼬리위험 0, Premium은 q90+cap. "
            f"stress(±50%)={ {t: round(stress[t], 3) for t in TIERS} }"
        ),
    )
    print(f"registered {exp_id}")
    print(json.dumps({t: (report["tiers"][t]["quality_score"], report["tiers"][t]["budget_ratio"]) for t in TIERS}, indent=1))
    print("final:", report["final_score"])
    (Path(__file__).resolve().parent / "final-frozen-policy.json").write_text(
        json.dumps(
            {
                "spec": POLICY_SPEC,
                "dev_official": {
                    "final": report["final_score"],
                    "tiers": {
                        t: {
                            "score": report["tiers"][t]["quality_score"],
                            "ratio": report["tiers"][t]["budget_ratio"],
                            "boot_over": over[t],
                            "stress_over_50": stress[t],
                        }
                        for t in TIERS
                    },
                },
            },
            indent=1,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
