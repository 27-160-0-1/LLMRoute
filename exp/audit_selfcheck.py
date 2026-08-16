# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Audit 3: harness scores must match `ossp_router.cli self-check` exactly.

Generates three experiment submissions (hash-regex, budget-oracle, naive-D3),
runs the official CLI self-check, and diffs every tier score/ratio digit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import alloc_lib
from harness import (
    MODEL_IDS,
    POLICY,
    ROOT,
    TIER_MULTIPLIERS,
    TIERS,
    evaluate,
    load_split,
    make_submission,
)
from ossp_router.heuristic import write_submission_atomic
from oracle_budget import lagrangian_select


def submissions_for(name: str, split: str = "dev"):
    table = load_split(split)
    n = len(table["texts"])
    scores = np.asarray(table["scores"])
    costs = np.asarray(table["costs"])
    if name == "hash-regex":
        import hash_regex

        artifact = hash_regex.load_artifact(ROOT / "baselines" / "hash-regex-public.v1.json")
        return {
            tier: [
                d.model_id
                for d in hash_regex.make_hash_regex_submission(
                    table["inputs"], POLICY, artifact, tier
                ).submission.decisions
            ]
            for tier in TIERS
        }
    if name == "budget-oracle":
        light_total = costs[:, 0].sum()
        return {
            tier: [
                MODEL_IDS[j]
                for j in lagrangian_select(
                    scores.copy(), costs, light_total * TIER_MULTIPLIERS[tier]
                )
            ]
            for tier in TIERS
        }
    if name == "naive-d3":
        data = np.load(ROOT / "build" / "preds" / f"naive-{split}.npz")
        return {
            tier: [
                MODEL_IDS[j]
                for j in alloc_lib.lagrangian_allocate(
                    data["score"], data["cost"], multiplier=TIER_MULTIPLIERS[tier], utilization=0.9
                )
            ]
            for tier in TIERS
        }
    raise ValueError(name)


def main() -> None:
    split = "dev"
    ok_all = True
    for name in ("hash-regex", "budget-oracle", "naive-d3"):
        assignment = submissions_for(name, split)
        harness_report = evaluate(split, assignment)
        outdir = ROOT / "build" / "audit" / name
        outdir.mkdir(parents=True, exist_ok=True)
        for tier in TIERS:
            write_submission_atomic(
                outdir / f"{tier}.json", make_submission(split, tier, assignment[tier])
            )
        report_path = outdir / "self-check-report.json"
        cmd = [
            sys.executable,
            "-m",
            "ossp_router.cli",
            "self-check",
            "--input",
            str(ROOT / "data" / "materialized" / split / "inputs.json"),
            "--outcomes",
            str(ROOT / "data" / split / "outcomes.json"),
            "--submissions",
            str(outdir),
            "--report",
            str(report_path),
        ]
        env = {"PYTHONPATH": str(ROOT / "src")}
        import os

        run_env = dict(os.environ)
        run_env.update(env)
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=run_env)
        if proc.returncode != 0:
            print(f"{name}: self-check 실행 실패\n{proc.stderr[-800:]}")
            ok_all = False
            continue
        official = json.loads(report_path.read_text(encoding="utf-8"))
        for tier in TIERS:
            h = harness_report["tiers"][tier]
            o = official["tiers"][tier]
            fields = ("quality_score", "budget_ratio", "tier_score", "total_cost", "budget_passed")
            mismatch = [f for f in fields if h[f] != o[f]]
            status = "일치" if not mismatch else f"불일치 {mismatch}"
            if mismatch:
                ok_all = False
            print(
                f"{name} {tier}: harness={h['quality_score']}/{h['budget_ratio']} "
                f"cli={o['quality_score']}/{o['budget_ratio']} -> {status}"
            )
        if harness_report["final_score"] != official["final_score"]:
            ok_all = False
            print(f"{name}: final_score 불일치!")
        else:
            print(f"{name}: final_score {official['final_score']} 일치")
    print("\nAUDIT3:", "PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
