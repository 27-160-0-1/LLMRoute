# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Distribution-shift stress tests for budget risk beyond plain bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from eda import categorize


def category_shift_over_probability(
    texts: Sequence[str],
    realized_selected: np.ndarray,
    realized_light: np.ndarray,
    multiplier: float,
    *,
    iterations: int = 2000,
    weight_low: float = 0.5,
    weight_high: float = 1.5,
    seed: int = 20260816,
) -> Dict[str, float]:
    """P(over budget) when category mix shifts by random +-50% reweighting.

    Each iteration draws one weight per category in [low, high], resamples
    N episodes with replacement using those weights, and checks the realized
    budget ratio. Harsher and more realistic than plain bootstrap.
    """

    rng = np.random.default_rng(seed)
    cats = np.asarray([categorize(t) for t in texts])
    unique = sorted(set(cats.tolist()))
    cat_index = {c: np.where(cats == c)[0] for c in unique}
    n = len(texts)
    over = 0
    ratios = []
    for _ in range(iterations):
        weights = {c: rng.uniform(weight_low, weight_high) for c in unique}
        p = np.empty(n)
        for c in unique:
            p[cat_index[c]] = weights[c]
        p /= p.sum()
        idx = rng.choice(n, size=n, replace=True, p=p)
        ratio = realized_selected[idx].sum() / realized_light[idx].sum()
        ratios.append(ratio)
        if ratio > multiplier:
            over += 1
    ratios = np.asarray(ratios)
    return {
        "over_prob": float(over / iterations),
        "ratio_p50": float(np.percentile(ratios, 50)),
        "ratio_p95": float(np.percentile(ratios, 95)),
        "ratio_p99": float(np.percentile(ratios, 99)),
        "ratio_max": float(ratios.max()),
    }
