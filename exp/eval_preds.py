# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a prediction set: margin sweep -> official scoring -> results.md.

Prediction set contract (produced by model experiments):
  build/preds/{name}-train.npz  score (N,3) OOF, cost (N,3) OOF absolute credits
  build/preds/{name}-dev.npz    score (N,3),     cost (N,3)   fit on full train
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import alloc_lib
from harness import (
    MODEL_IDS,
    TIER_MULTIPLIERS,
    TIERS,
    append_result,
    evaluate,
    load_split,
)

PREDS = Path(__file__).resolve().parents[1] / "build" / "preds"

UTIL_GRID = [round(0.60 + 0.01 * i, 2) for i in range(56)]  # 0.60..1.15


def load_preds(name: str, split: str):
    data = np.load(PREDS / f"{name}-{split}.npz")
    return data["score"], data["cost"]


def realized(split: str):
    table = load_split(split)
    return np.asarray(table["scores"]), np.asarray(table["costs"])


def fast_eval(pick: np.ndarray, real_scores, real_costs, multiplier: float):
    n = len(pick)
    sel_cost = real_costs[np.arange(n), pick]
    ratio = sel_cost.sum() / real_costs[:, 0].sum()
    score = real_scores[np.arange(n), pick].mean()
    return score, ratio, ratio <= multiplier + 1e-12


def sweep_tier(
    score_pred,
    cost_pred,
    real_scores,
    real_costs,
    tier: str,
    allocator: str = "lagrangian",
):
    """Return list of (utilization, pick, score, ratio, passed)."""

    multiplier = TIER_MULTIPLIERS[tier]
    rows = []
    for u in UTIL_GRID:
        if allocator == "greedy":
            pick = alloc_lib.greedy_allocate(
                score_pred, cost_pred, multiplier=multiplier, utilization=u
            )
        else:
            pick = alloc_lib.lagrangian_allocate(
                score_pred, cost_pred, multiplier=multiplier, utilization=u
            )
        score, ratio, passed = fast_eval(pick, real_scores, real_costs, multiplier)
        rows.append((u, pick, score, ratio, passed))
    return rows


def best_row(rows):
    passing = [r for r in rows if r[4]]
    if not passing:
        return None
    return max(passing, key=lambda r: (r[2], -r[3]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--allocator", default="lagrangian", choices=["lagrangian", "greedy"])
    parser.add_argument("--note", default="")
    parser.add_argument("--no-md", action="store_true")
    args = parser.parse_args()

    train_score_pred, train_cost_pred = load_preds(args.name, "train")
    dev_score_pred, dev_cost_pred = load_preds(args.name, "dev")
    train_real_s, train_real_c = realized("train")
    dev_real_s, dev_real_c = realized("dev")

    assignment_dev_cal = {}
    assignment_train_cal = {}
    over_probs = {}
    picked_util = {}
    for tier in TIERS:
        rows_dev = sweep_tier(
            dev_score_pred, dev_cost_pred, dev_real_s, dev_real_c, tier, args.allocator
        )
        rows_train = sweep_tier(
            train_score_pred, train_cost_pred, train_real_s, train_real_c, tier, args.allocator
        )
        best_dev = best_row(rows_dev)
        best_train = best_row(rows_train)
        # train-calibrated utilization applied to dev predictions
        u_train = best_train[0] if best_train else 1.0 / TIER_MULTIPLIERS[tier]
        pick_transfer = None
        for u, pick, *_ in rows_dev:
            if u == u_train:
                pick_transfer = pick
                break
        if pick_transfer is None:
            pick_transfer = rows_dev[0][1]
        assignment_dev_cal[tier] = [MODEL_IDS[j] for j in best_dev[1]] if best_dev else [MODEL_IDS[0]] * len(dev_real_s)
        assignment_train_cal[tier] = [MODEL_IDS[j] for j in pick_transfer]
        picked_util[tier] = (best_dev[0] if best_dev else None, u_train)
        if best_dev:
            n = len(best_dev[1])
            over_probs[tier] = alloc_lib.bootstrap_over_probability(
                dev_real_c[np.arange(n), best_dev[1]],
                dev_real_c[:, 0],
                TIER_MULTIPLIERS[tier],
            )

    start = time.time()
    report_dev_cal = evaluate("dev", assignment_dev_cal)
    report_train_cal = evaluate("dev", assignment_train_cal)
    _ = time.time() - start

    util_note = " ".join(
        f"{tier[0]}:{picked_util[tier][0]}/{picked_util[tier][1]}" for tier in TIERS
    )
    if not args.no_md:
        import registry_lib
        import subprocess

        metrics = registry_lib.metrics_from_report(
            report_dev_cal,
            over_probs,
            train_cal_weighted=report_train_cal["final_score"],
        )
        # per-experiment detail for dashboard: picks + sweep curves
        detail_dir = Path(__file__).resolve().parents[1] / "exp"
        sweep_payload = {}
        for tier in TIERS:
            rows_dev = sweep_tier(
                dev_score_pred, dev_cost_pred, dev_real_s, dev_real_c, tier, args.allocator
            )
            sweep_payload[tier] = {
                "util": [r[0] for r in rows_dev],
                "score": [r[2] for r in rows_dev],
                "ratio": [r[3] for r in rows_dev],
            }
        exp_id = registry_lib.register(
            family="model",
            config={
                "name": f"{args.name}+{args.allocator}",
                "pred_set": args.name,
                "decision": args.allocator,
                "calibration": "dev",
                "utilization": {t: picked_util[t][0] for t in TIERS},
                "train_cal_utilization": {t: picked_util[t][1] for t in TIERS},
            },
            metrics=metrics,
            artifacts=[f"build/preds/{args.name}-dev.npz", f"build/preds/{args.name}-summary.json"],
            notes=f"util(dev/train-cal) {util_note}. {args.note}",
        )
        exp_dir = detail_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            exp_dir / "detail.npz",
            picks_dev=np.asarray(
                [[MODEL_IDS.index(m) for m in assignment_dev_cal[t]] for t in TIERS],
                dtype=np.int8,
            ),
            **{
                f"sweep_{t}_{k}": np.asarray(sweep_payload[t][k])
                for t in TIERS
                for k in ("util", "score", "ratio")
            },
        )
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "make_report.py")],
            check=False,
        )
        print(f"registered {exp_id}")
    for label, report in (("dev-cal", report_dev_cal), ("train-cal", report_train_cal)):
        tiers = report["tiers"]
        print(
            label,
            {t: (tiers[t]["quality_score"], tiers[t]["budget_ratio"]) for t in TIERS},
            report["final_score"],
        )


if __name__ == "__main__":
    main()
