# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Retro-register reference rows (baselines/oracles) into registry.jsonl."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import registry_lib
from harness import MODEL_IDS, POLICY, TIER_MULTIPLIERS, TIERS, evaluate, load_split
from oracle_budget import lagrangian_select
from alloc_lib import bootstrap_over_probability


def over_probs_for(split, assignment):
    table = load_split(split)
    costs = np.asarray(table["costs"])
    out = {}
    for tier in TIERS:
        pick = np.asarray([MODEL_IDS.index(m) for m in assignment[tier]])
        out[tier] = bootstrap_over_probability(
            costs[np.arange(len(pick)), pick], costs[:, 0], TIER_MULTIPLIERS[tier]
        )
    return out


def main() -> None:
    existing = {row["config"].get("name") for row in registry_lib.load_all()}

    def add(name, family, split, assignment, notes, runtime=None):
        if name in existing:
            print(f"skip {name} (등록됨)")
            return
        report = evaluate(split, assignment)
        probs = over_probs_for(split, assignment)
        metrics = registry_lib.metrics_from_report(report, probs)
        exp_id = registry_lib.register(
            family=family,
            config={"name": name, "split": split},
            metrics=metrics,
            runtime=runtime,
            notes=notes,
        )
        print(f"registered {exp_id}: {name}")

    for split in ("dev",):
        table = load_split(split)
        inputs = table["inputs"]
        n = len(inputs.episodes)
        scores = np.asarray(table["scores"])
        costs = np.asarray(table["costs"])

        add(
            f"all-light ({split})",
            "reference",
            split,
            {t: [MODEL_IDS[0]] * n for t in TIERS},
            "전 문항 light",
        )

        from ossp_router.heuristic import extract_features, select_model

        add(
            f"prompt-heuristic ({split})",
            "reference",
            split,
            {
                t: [select_model(extract_features(e), t) for e in inputs.episodes]
                for t in TIERS
            },
            "공식 약한 baseline",
        )

        import hash_regex

        artifact = hash_regex.load_artifact(
            registry_lib.ROOT / "baselines" / "hash-regex-public.v1.json"
        )
        start = time.time()
        hr_assign = {
            t: [
                d.model_id
                for d in hash_regex.make_hash_regex_submission(
                    inputs, POLICY, artifact, t
                ).submission.decisions
            ]
            for t in TIERS
        }
        per_tier = (time.time() - start) / 3
        add(
            f"hash-regex ({split})",
            "reference",
            split,
            hr_assign,
            "공식 최강 baseline (제공 artifact)",
            runtime={"per_tier_sec": round(per_tier, 2), "measured_env": "host"},
        )

        light_total = costs[:, 0].sum()
        add(
            f"budget-oracle ({split})",
            "reference",
            split,
            {
                t: [
                    MODEL_IDS[j]
                    for j in lagrangian_select(
                        scores.copy(), costs, light_total * TIER_MULTIPLIERS[t]
                    )
                ]
                for t in TIERS
            },
            "실측 outcome 완전정보 상한 (예산제약)",
        )


if __name__ == "__main__":
    main()
