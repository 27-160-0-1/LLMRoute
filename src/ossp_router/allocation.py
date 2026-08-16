# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Order-invariant budget allocation for the final router.

Ports of the experiment-verified allocators (exp/alloc_lib.py). Rows with
identical prediction signatures are always promoted together, so decisions
cannot depend on episode order or IDs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def lagrangian_allocate(
    score_pred: np.ndarray,
    cost_pred: np.ndarray,
    *,
    multiplier: float,
    utilization: float,
    allow_k1: bool = True,
) -> np.ndarray:
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
    k1_cost_cap: float | None = None,
) -> np.ndarray:
    n = score_pred.shape[0]
    pick = np.zeros(n, dtype=int)
    light_total = cost_pred[:, 0].sum()
    cap = light_total * max(1.0, multiplier * utilization)
    total = cost_pred[np.arange(n), pick].sum()
    models = (0, 1, 2) if allow_k1 else (0, 1)

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
