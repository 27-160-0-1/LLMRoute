# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""A7 supplement: does the shipped score model actually rank the routing
decision?  Uses the saved shipped lookup-OFF dev predictions.

Output: build/a7/lift.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "exp" / "audit"))

import alloc_lib  # noqa: E402
from a7_proxy import OUT, load_raw, monotone  # noqa: E402
from harness import load_split  # noqa: E402


def main() -> None:
    d = np.load(OUT / "pred-shipped-nolookup-dev.npz")
    score, cost, costq = d["score"], d["cost"], d["costq"]
    t = load_split("dev")
    real_s = np.asarray(t["scores"])
    n = real_s.shape[0]

    gain_true = real_s[:, 1] - real_s[:, 0]
    gain_pred = score[:, 1] - score[:, 0]
    k1_true = real_s[:, 2] - real_s[:, 1]
    k1_pred = score[:, 2] - score[:, 1]

    from scipy.stats import spearmanr

    out = {"n_dev": n,
           "true_score_values_light": sorted(set(real_s[:, 0].tolist()))[:6],
           "gain_ax31_vs_light": {
               "true_dist": {str(v): int((gain_true == v).sum())
                             for v in sorted(set(gain_true.tolist()))},
               "base_rate_positive": float((gain_true > 0).mean()),
               "pearson_r": float(np.corrcoef(gain_pred, gain_true)[0, 1]),
               "spearman_r": float(spearmanr(gain_pred, gain_true).statistic),
           },
           "gain_k1_vs_ax31": {
               "base_rate_positive": float((k1_true > 0).mean()),
               "pearson_r": float(np.corrcoef(k1_pred, k1_true)[0, 1]),
               "spearman_r": float(spearmanr(k1_pred, k1_true).statistic),
           }}

    # fast tier: which episodes did the shipped policy upgrade to ax31?
    pick = alloc_lib.greedy_allocate(score, cost, multiplier=1.25,
                                     utilization=0.93, allow_k1=False)
    sel = pick == 1
    k = int(sel.sum())
    # counterfactual: same k upgrades chosen by TRUE gain/cost ratio (oracle)
    ratio_true = gain_true / np.maximum(cost[:, 1] - cost[:, 0], 1e-12)
    oracle_sel = np.zeros(n, dtype=bool)
    oracle_sel[np.argsort(-ratio_true)[:k]] = True
    # counterfactual: cheapest-first (no score information at all)
    cheap_sel = np.zeros(n, dtype=bool)
    cheap_sel[np.argsort(cost[:, 1] - cost[:, 0])[:k]] = True
    out["fast_tier_pick_quality"] = {
        "n_upgraded": k,
        "base_rate_positive_gain": float((gain_true > 0).mean()),
        "router_picks_positive_gain": float((gain_true[sel] > 0).mean()),
        "router_picks_negative_gain": float((gain_true[sel] < 0).mean()),
        "cheapfirst_picks_positive_gain": float((gain_true[cheap_sel] > 0).mean()),
        "oracle_picks_positive_gain": float((gain_true[oracle_sel] > 0).mean()),
        "mean_true_gain_router": float(gain_true[sel].mean()),
        "mean_true_gain_cheapfirst": float(gain_true[cheap_sel].mean()),
        "mean_true_gain_oracle": float(gain_true[oracle_sel].mean()),
        "mean_true_gain_all": float(gain_true.mean()),
    }
    (OUT / "lift.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
