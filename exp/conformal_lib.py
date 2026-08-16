# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""P6: conformal / CLT-based budget guarantees from train-OOF cost residuals.

Two mechanisms, both calibrated WITHOUT touching dev:
- per-episode split-conformal multiplicative cost inflation (conservative)
- sum-level Gaussian (CLT) utilization solving P(over) <= alpha (tighter)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

PREDS = Path(__file__).resolve().parents[1] / "build" / "preds"
FEATS = Path(__file__).resolve().parents[1] / "build" / "feats"


def oof_log_residuals(name: str) -> np.ndarray:
    """(N,3) log(real cost) - log(pred cost) on train OOF."""

    pred = np.load(PREDS / f"{name}-train.npz")["cost"]
    real = np.load(FEATS / "targets-train.npz")["costs"]
    return np.log(real) - np.log(np.maximum(pred, 1e-12))


def conformal_factors(name: str, alpha: float = 0.9) -> np.ndarray:
    """Per-model multiplicative inflation exp(quantile_alpha(residual))."""

    residuals = oof_log_residuals(name)
    return np.exp(np.quantile(residuals, alpha, axis=0))


def inflate_costs(cost_pred: np.ndarray, factors: np.ndarray) -> np.ndarray:
    out = cost_pred * factors[None, :]
    out[:, 1] = np.maximum(out[:, 1], out[:, 0] * (1 + 1e-12))
    out[:, 2] = np.maximum(out[:, 2], out[:, 1] * (1 + 1e-12))
    return out


def clt_over_probability(
    pick: np.ndarray,
    cost_pred: np.ndarray,
    residual_std: np.ndarray,
    multiplier: float,
    *,
    residual_mean: np.ndarray | None = None,
) -> float:
    """P(sum realized_sel > multiplier * sum realized_light) under lognormal errors.

    Per-episode realized cost ~ pred * exp(eps), eps ~ N(mu_j, sd_j^2) i.i.d.
    g_i = c_sel_i - m * c_light_i ; approximate Sum(g) as Gaussian.
    Light-column correlation between numerator and denominator for rows picked
    light is handled exactly (their g_i = (1-m) * c_light_i * exp(eps) with
    same eps).
    """

    from math import erf, sqrt

    n = len(pick)
    mu = residual_mean if residual_mean is not None else np.zeros(3)
    sd = residual_std
    mean_sum = 0.0
    var_sum = 0.0
    for i in range(n):
        j = pick[i]
        if j == 0:
            # g = (1-m) * c_light * exp(eps0): exact same shock in both terms
            base = (1.0 - multiplier) * cost_pred[i, 0]
            m1 = np.exp(mu[0] + 0.5 * sd[0] ** 2)
            m2 = np.exp(2 * mu[0] + 2 * sd[0] ** 2)
            mean_sum += base * m1
            var_sum += (base**2) * (m2 - m1**2)
        else:
            for coef, jj in ((1.0, j), (-multiplier, 0)):
                base = coef * cost_pred[i, jj]
                m1 = np.exp(mu[jj] + 0.5 * sd[jj] ** 2)
                m2 = np.exp(2 * mu[jj] + 2 * sd[jj] ** 2)
                mean_sum += base * m1
                var_sum += (base**2) * (m2 - m1**2)
    if var_sum <= 0:
        return 0.0 if mean_sum <= 0 else 1.0
    z = mean_sum / sqrt(var_sum)
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def residual_moments(name: str) -> tuple[np.ndarray, np.ndarray]:
    residuals = oof_log_residuals(name)
    return residuals.mean(axis=0), residuals.std(axis=0)
