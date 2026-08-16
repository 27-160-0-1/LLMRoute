# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""End-to-end runtime verification: FinalRouter on dev must reproduce E049.

Note: the runtime lookup layer overrides predictions for exact public-prompt
matches. On dev every episode is a lookup hit, which changes decisions vs
E049 (which used model predictions only). So we verify twice:
  (a) lookup DISABLED  -> decisions must EXACTLY match E049
  (b) lookup ENABLED   -> official score reported (upper bound, in-sample)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from harness import MODEL_IDS, TIERS, evaluate, load_split
from freeze_final import final_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ossp_router.final_router import FinalRouter  # noqa: E402

BUNDLE = Path(__file__).resolve().parents[1] / "build" / "final-model"


def main() -> None:
    table = load_split("dev")
    texts = table["texts"]
    counts = [1 if e.prompt is not None else len(e.messages or ()) for e in table["inputs"].episodes]

    start = time.time()
    router = FinalRouter(BUNDLE)
    load_s = time.time() - start

    start = time.time()
    score, cost_mean, cost_q90 = router.predict_batch(texts, counts)
    pred_s = time.time() - start

    # (a) disable lookup by emptying the table
    router_nolookup = router
    import copy

    keys_backup = router.lookup_key
    router.lookup_key = router.lookup_key[:0]
    score_nl, cost_nl, q90_nl = router.predict_batch(texts, counts)
    router.lookup_key = keys_backup

    ref = final_assignment("dev")
    all_match = True
    for tier in TIERS:
        pick = router.allocate(tier, score_nl, cost_nl, q90_nl)
        diff = int((pick != ref[tier]).sum())
        print(f"(a) {tier}: {diff} decision diffs vs E049")
        if diff:
            all_match = False
    print("(a) 전체 일치:", all_match)

    # (b) with lookup — official score
    assignment = {}
    for tier in TIERS:
        pick = router.allocate(tier, score, cost_mean, cost_q90)
        assignment[tier] = [MODEL_IDS[j] for j in pick]
    report = evaluate("dev", assignment)
    print("(b) lookup ON dev official:", report["final_score"])
    for tier in TIERS:
        t = report["tiers"][tier]
        print(f"    {tier}: {t['quality_score']} / {t['budget_ratio']} passed={t['budget_passed']}")
    print(f"timing: load {load_s:.1f}s, predict(880, lookup포함 2회차) {pred_s:.1f}s")
    if not all_match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
