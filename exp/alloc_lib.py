# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Decision layer: budget allocation from predicted scores/costs.

All allocators are order/ID-invariant: identical (score, cost) prediction
rows always receive identical decisions (group promotion).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

MODEL_IDS = ("ax31-light", "ax31", "axk1-think")


def lagrangian_allocate(
    score_pred: np.ndarray,
    cost_pred: np.ndarray,
    *,
    multiplier: float,
    utilization: float,
    allow_k1: bool = True,
) -> np.ndarray:
    """D3: batch Lagrangian with binary search on the cost penalty.

    Order-invariant: decisions depend only on per-row predictions and the
    global lambda. Returns int array of model indices.
    """

    n = score_pred.shape[0]
    models = (0, 1, 2) if allow_k1 else (0, 1)
    light_total = cost_pred[:, 0].sum()
    cap = light_total * max(1.0, multiplier * utilization)

    def choose(lam: float) -> np.ndarray:
        best = np.zeros(n, dtype=int)
        best_util = score_pred[:, 0] - lam * cost_pred[:, 0] / light_total * n
        for j in models[1:]:
            util = score_pred[:, j] - lam * cost_pred[:, j] / light_total * n
            better = util > best_util + 1e-15
            best = np.where(better, j, best)
            best_util = np.where(better, util, best_util)
        return best

    pick = choose(0.0)
    if cost_pred[np.arange(n), pick].sum() <= cap:
        return pick
    low, high = 0.0, 1.0
    while cost_pred[np.arange(n), choose(high)].sum() > cap and high < 2**60:
        low, high = high, high * 2.0
    pick = choose(high)
    for _ in range(100):
        mid = (low + high) / 2.0
        cand = choose(mid)
        if cost_pred[np.arange(n), cand].sum() <= cap:
            high, pick = mid, cand
        else:
            low = mid
    if cost_pred[np.arange(n), pick].sum() > cap:
        return np.zeros(n, dtype=int)
    return pick


def greedy_allocate(
    score_pred: np.ndarray,
    cost_pred: np.ndarray,
    *,
    multiplier: float,
    utilization: float,
    allow_k1: bool = True,
    from_pick: np.ndarray | None = None,
    k1_cost_cap: float | None = None,
) -> np.ndarray:
    """D2: group-wise greedy promotion by marginal score/cost ratio.

    Rows with identical (rounded) prediction signatures are promoted as one
    group, so ties cannot depend on input order. ``k1_cost_cap`` excludes
    k1 for rows whose predicted k1 cost exceeds the cap (tail-risk control).
    """

    n = score_pred.shape[0]
    pick = np.zeros(n, dtype=int) if from_pick is None else from_pick.copy()
    light_total = cost_pred[:, 0].sum()
    cap = light_total * max(1.0, multiplier * utilization)
    total = cost_pred[np.arange(n), pick].sum()
    models = (0, 1, 2) if allow_k1 else (0, 1)

    # candidate upgrades: (row, target) with positive score gain
    groups: Dict[Tuple, List[Tuple[int, int]]] = {}
    for i in range(n):
        for j in models:
            if j <= pick[i]:
                continue
            if j == 2 and k1_cost_cap is not None and cost_pred[i, 2] > k1_cost_cap:
                continue
            ds = score_pred[i, j] - score_pred[i, pick[i]]
            dc = cost_pred[i, j] - cost_pred[i, pick[i]]
            if ds <= 1e-12:
                continue
            ratio = ds / max(dc, 1e-12)
            key = (round(ratio, 12), round(ds, 12), round(dc, 12), j)
            groups.setdefault(key, []).append((i, j))
    # order: high ratio first, then small dc, small ds
    ordered = sorted(groups.items(), key=lambda kv: (-kv[0][0], kv[0][2], kv[0][1], kv[0][3]))
    for key, members in ordered:
        rows = [i for i, _ in members if pick[i] < members[0][1]]
        if not rows:
            continue
        j = members[0][1]
        dc_total = sum(cost_pred[i, j] - cost_pred[i, pick[i]] for i, _ in members)
        if total + dc_total <= cap:
            for i, _ in members:
                pick[i] = j
            total += dc_total
    return pick


def two_stage_premium(
    score_pred: np.ndarray,
    cost_pred: np.ndarray,
    *,
    multiplier: float,
    k1_utilization: float,
    fill_utilization: float,
    k1_cost_cap: float | None = None,
) -> np.ndarray:
    """D4: commit K1 picks under a conservative cap, then fill with ax31."""

    if k1_cost_cap is not None:
        masked = score_pred.copy()
        masked[cost_pred[:, 2] > k1_cost_cap, 2] = 0.0
        score_pred = masked
    pick = lagrangian_allocate(
        score_pred,
        cost_pred,
        multiplier=multiplier,
        utilization=k1_utilization,
        allow_k1=True,
    )
    n = score_pred.shape[0]
    light_total = cost_pred[:, 0].sum()
    cap = light_total * max(1.0, multiplier * fill_utilization)
    total = cost_pred[np.arange(n), pick].sum()
    groups: Dict[Tuple, List[int]] = {}
    for i in range(n):
        if pick[i] != 0:
            continue
        ds = score_pred[i, 1] - score_pred[i, 0]
        dc = cost_pred[i, 1] - cost_pred[i, 0]
        if ds <= 1e-12:
            continue
        key = (round(ds / max(dc, 1e-12), 12), round(dc, 12), round(ds, 12))
        groups.setdefault(key, []).append(i)
    for key, rows in sorted(groups.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        dc_total = sum(cost_pred[i, 1] - cost_pred[i, 0] for i in rows)
        if total + dc_total <= cap:
            for i in rows:
                pick[i] = 1
            total += dc_total
    return pick


def bootstrap_over_probability(
    realized_selected: np.ndarray,
    realized_light: np.ndarray,
    multiplier: float,
    *,
    iterations: int = 1000,
    seed: int = 20260816,
) -> float:
    rng = np.random.default_rng(seed)
    n = len(realized_selected)
    idx = rng.integers(0, n, size=(iterations, n))
    ratios = realized_selected[idx].sum(axis=1) / realized_light[idx].sum(axis=1)
    return float((ratios > multiplier).mean())
