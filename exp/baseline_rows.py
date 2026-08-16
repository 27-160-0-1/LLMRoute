# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Reference rows: all-light, prompt-heuristic, hash-regex (public artifact)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import MODEL_IDS, ROOT, TIERS, append_result, evaluate, load_split


def main() -> None:
    from ossp_router.heuristic import extract_features, select_model

    import hash_regex

    for split in ("dev",):
        table = load_split(split)
        inputs = table["inputs"]
        n = len(inputs.episodes)

        report = evaluate(split, {})
        print(append_result(f"[기준] all-light ({split})", report, notes="전 문항 light"))

        assignment = {
            tier: [
                select_model(extract_features(episode), tier)
                for episode in inputs.episodes
            ]
            for tier in TIERS
        }
        report = evaluate(split, assignment)
        print(
            append_result(
                f"[기준] prompt-heuristic ({split})", report, notes="공식 약한 baseline"
            )
        )

        artifact = hash_regex.load_artifact(
            ROOT / "baselines" / "hash-regex-public.v1.json"
        )
        from harness import POLICY

        assignment = {}
        start = time.time()
        for tier in TIERS:
            plan = hash_regex.make_hash_regex_submission(inputs, POLICY, artifact, tier)
            assignment[tier] = [d.model_id for d in plan.submission.decisions]
        elapsed = (time.time() - start) / len(TIERS)
        report = evaluate(split, assignment)
        print(
            append_result(
                f"[기준] hash-regex ({split})",
                report,
                runtime_seconds=f"{elapsed:.1f}s(로컬)",
                notes="공식 최강 baseline, 제공 artifact",
            )
        )


if __name__ == "__main__":
    main()
