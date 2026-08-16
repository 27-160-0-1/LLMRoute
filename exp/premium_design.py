# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Premium tier design: quantile k1 cost + per-episode k1 cap vs tail risk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import alloc_lib
from eval_preds import load_preds, realized
from harness import TIER_MULTIPLIERS, load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--boot-limit", type=float, default=0.005)
    args = parser.parse_args()

    dev_score, dev_cost = load_preds(args.name, "dev")
    real_s, real_c = realized("dev")
    n = len(real_s)
    mult = TIER_MULTIPLIERS["premium"]

    print(f"{'cap':>6} {'u':>5} {'fill':>5} | {'score':>7} {'ratio':>6} {'boot%':>6} | pass")
    best = None
    for cap in (0.1, 0.15, 0.2, 0.3, 0.5, 1.0, None):
        for u in [round(0.55 + 0.05 * i, 2) for i in range(10)]:
            for fill in (0.6, 0.65, 0.7, 0.75, 0.8, 0.9):
                pick = alloc_lib.two_stage_premium(
                    dev_score, dev_cost, multiplier=mult,
                    k1_utilization=u, fill_utilization=fill,
                    k1_cost_cap=cap,
                )
                sel = real_c[np.arange(n), pick]
                ratio = sel.sum() / real_c[:, 0].sum()
                if ratio > mult:
                    continue
                score = real_s[np.arange(n), pick].mean()
                boot = alloc_lib.bootstrap_over_probability(sel, real_c[:, 0], mult)
                ok = boot <= args.boot_limit
                if ok and (best is None or score > best[0]):
                    best = (score, ratio, boot, cap, u, fill)
                    print(
                        f"{str(cap):>6} {u:>5} {fill:>5} | {score:.5f} {ratio:.3f} "
                        f"{100*boot:>5.1f} | {'OK' if ok else '--'}"
                    )
    print("\nBEST (boot 게이트 통과):", best)


if __name__ == "__main__":
    main()
