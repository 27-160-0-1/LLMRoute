# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Fast/Balanced design: does banning/capping k1 raise gated utilization?"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import alloc_lib
import stress_lib
from eval_preds import load_preds, realized
from harness import TIER_MULTIPLIERS, load_split


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "blend4-cost2b"
    tier = sys.argv[2] if len(sys.argv) > 2 else "fast"
    dev_score, dev_cost = load_preds(name, "dev")
    real_s, real_c = realized("dev")
    texts = load_split("dev")["texts"]
    n = len(real_s)
    mult = TIER_MULTIPLIERS[tier]

    for label, kwargs in (
        ("k1 허용", {}),
        ("k1 금지", {"allow_k1": False}),
        ("k1 cap 0.03", {"k1_cost_cap": 0.03}),
        ("k1 cap 0.08", {"k1_cost_cap": 0.08}),
    ):
        best = None
        for u in [round(0.80 + 0.01 * i, 2) for i in range(31)]:
            pick = alloc_lib.greedy_allocate(
                dev_score, dev_cost, multiplier=mult, utilization=u, **kwargs
            )
            sel = real_c[np.arange(n), pick]
            ratio = sel.sum() / real_c[:, 0].sum()
            if ratio > mult:
                continue
            boot = alloc_lib.bootstrap_over_probability(sel, real_c[:, 0], mult)
            if boot > 0.01:
                continue
            stress = stress_lib.category_shift_over_probability(
                texts, sel, real_c[:, 0], mult, iterations=500,
                weight_low=0.65, weight_high=1.35,
            )
            if stress["over_prob"] > 0.05:
                continue
            score = real_s[np.arange(n), pick].mean()
            k1_count = int((pick == 2).sum())
            if best is None or score > best[0]:
                best = (score, ratio, boot, stress["over_prob"], u, k1_count)
        print(f"{label}: best={best}")


if __name__ == "__main__":
    main()
