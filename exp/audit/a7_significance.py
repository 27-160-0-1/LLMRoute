# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""A7: is the shipped score model's contribution distinguishable from zero?

Compares the SHIPPED lookup-OFF allocation (A) against the same pipeline with
the 4-member score blend replaced by a train-marginal constant (B).
Allocations are computed once on the full dev split (as in a real run) and
then held fixed; the paired bootstrap resamples the 880 dev episodes to put a
CI on the realized weighted-score difference.

Output: build/a7/significance.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "exp" / "audit"))

from a7_proxy import OUT, allocate, load_raw, monotone  # noqa: E402
from harness import TIERS, load_split  # noqa: E402

W = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
MULT = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}


def main() -> None:
    d = np.load(OUT / "pred-shipped-nolookup-dev.npz")
    s_real, c_real, q_real = d["score"], d["cost"], d["costq"]
    _, _, _, t_tr = load_raw("train")
    n = s_real.shape[0]
    s_const = np.repeat(t_tr["scores"].mean(axis=0)[None, :], n, axis=0)
    c_const = monotone(np.repeat(
        np.exp(np.log(t_tr["costs"]).mean(axis=0))[None, :], n, axis=0))

    pick_A = allocate(s_real, c_real, q_real)      # shipped, honest
    pick_B = allocate(s_const, c_real, q_real)     # score head killed
    pick_C = allocate(s_real, c_const, c_const)    # cost head killed

    table = load_split("dev")
    real_s = np.asarray(table["scores"])
    real_c = np.asarray(table["costs"])
    rows = np.arange(n)

    out = {"n_dev": n, "decision_diffs": {}, "per_tier": {}, "bootstrap": {}}
    for t in TIERS:
        out["decision_diffs"][t] = {
            "A_vs_B_changed_decisions": int((pick_A[t] != pick_B[t]).sum()),
            "A_vs_C_changed_decisions": int((pick_A[t] != pick_C[t]).sum()),
            "A_counts": np.bincount(pick_A[t], minlength=3).tolist(),
            "B_counts": np.bincount(pick_B[t], minlength=3).tolist(),
        }
        out["per_tier"][t] = {
            "A_score": float(real_s[rows, pick_A[t]].mean()),
            "B_score": float(real_s[rows, pick_B[t]].mean()),
            "A_ratio": float(real_c[rows, pick_A[t]].sum() / real_c[:, 0].sum()),
            "B_ratio": float(real_c[rows, pick_B[t]].sum() / real_c[:, 0].sum()),
        }

    # per-episode weighted realized score for each policy
    def weighted(pick):
        return sum(W[t] * real_s[rows, pick[t]] for t in TIERS)

    wA, wB, wC = weighted(pick_A), weighted(pick_B), weighted(pick_C)
    out["full_dev_weighted"] = {"A": float(wA.mean()), "B": float(wB.mean()),
                                "C": float(wC.mean()),
                                "A_minus_B": float(wA.mean() - wB.mean()),
                                "A_minus_C": float(wA.mean() - wC.mean())}

    rng = np.random.default_rng(20260817)
    B = 10000
    idx = rng.integers(0, n, size=(B, n))
    dAB = (wA - wB)[idx].mean(axis=1)
    dAC = (wA - wC)[idx].mean(axis=1)
    for name, arr in (("A_minus_B_score_head", dAB), ("A_minus_C_cost_head", dAC)):
        out["bootstrap"][name] = {
            "mean": float(arr.mean()),
            "ci2.5": float(np.percentile(arr, 2.5)),
            "ci97.5": float(np.percentile(arr, 97.5)),
            "p_le_zero": float((arr <= 0).mean()),
        }
    (OUT / "significance.json").write_text(json.dumps(out, indent=1),
                                           encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
