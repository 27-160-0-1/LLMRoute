# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""A7: were the frozen utilization constants selected ON the dev split?

Sweeps the frozen policy's utilization knobs using the SHIPPED lookup-OFF dev
predictions and locates where the frozen values (fast 0.93, balanced 0.88,
premium k1 0.65 / fill 0.70) sit relative to the dev-optimal value.
If frozen == dev argmax, the reported dev score is a selection-biased estimate.

Output: build/a7/util_sweep.json
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
from a7_proxy import OUT, monotone  # noqa: E402
from harness import load_split  # noqa: E402

GRID = [round(0.60 + 0.01 * i, 2) for i in range(56)]  # exp/eval_preds.UTIL_GRID


def evaluate_pick(pick, real_s, real_c, mult):
    n = len(pick)
    rows = np.arange(n)
    ratio = real_c[rows, pick].sum() / real_c[:, 0].sum()
    return (float(real_s[rows, pick].mean()), float(ratio),
            bool(ratio <= mult + 1e-12))


def main() -> None:
    d = np.load(OUT / "pred-shipped-nolookup-dev.npz")
    score, cost, costq = d["score"], d["cost"], d["costq"]
    q = cost.copy()
    q[:, 2] = costq[:, 2]
    q = monotone(q)
    t = load_split("dev")
    real_s = np.asarray(t["scores"])
    real_c = np.asarray(t["costs"])

    out = {}
    for tier, mult, frozen in (("fast", 1.25, 0.93), ("balanced", 2.0, 0.88)):
        rows = []
        for u in GRID:
            pick = alloc_lib.greedy_allocate(score, cost, multiplier=mult,
                                             utilization=u, allow_k1=False)
            s, r, ok = evaluate_pick(pick, real_s, real_c, mult)
            rows.append({"u": u, "score": s, "ratio": r, "passed": ok})
        passing = [r for r in rows if r["passed"]]
        best = max(passing, key=lambda r: (r["score"], -r["ratio"]))
        froz = next(r for r in rows if abs(r["u"] - frozen) < 1e-9)
        out[tier] = {
            "frozen_u": frozen, "frozen": froz,
            "dev_argmax_u": best["u"], "dev_argmax": best,
            "frozen_is_dev_argmax": abs(best["u"] - frozen) < 1e-9,
            "gap_to_dev_argmax": round(best["score"] - froz["score"], 12),
            "n_grid": len(GRID), "n_passing": len(passing),
            "curve": rows,
        }

    # premium: sweep k1_utilization (fill fixed 0.70) and fill (k1 fixed 0.65)
    prem = {}
    for name, fixed in (("k1_utilization", "fill"), ("fill_utilization", "k1")):
        rows = []
        for u in GRID:
            if name == "k1_utilization":
                pick = alloc_lib.two_stage_premium(
                    score, q, multiplier=4.0, k1_utilization=u,
                    fill_utilization=0.70, k1_cost_cap=0.1)
            else:
                pick = alloc_lib.two_stage_premium(
                    score, q, multiplier=4.0, k1_utilization=0.65,
                    fill_utilization=u, k1_cost_cap=0.1)
            s, r, ok = evaluate_pick(pick, real_s, real_c, 4.0)
            rows.append({"u": u, "score": s, "ratio": r, "passed": ok})
        passing = [r for r in rows if r["passed"]]
        best = max(passing, key=lambda r: (r["score"], -r["ratio"]))
        frozen = 0.65 if name == "k1_utilization" else 0.70
        froz = next(r for r in rows if abs(r["u"] - frozen) < 1e-9)
        prem[name] = {"frozen_u": frozen, "frozen": froz,
                      "dev_argmax_u": best["u"], "dev_argmax": best,
                      "frozen_is_dev_argmax": abs(best["u"] - frozen) < 1e-9,
                      "gap_to_dev_argmax": round(best["score"] - froz["score"], 12),
                      "n_passing": len(passing), "curve": rows}
    out["premium"] = prem

    (OUT / "util_sweep.json").write_text(json.dumps(out, indent=1),
                                         encoding="utf-8")
    for k in ("fast", "balanced"):
        v = out[k]
        print(f"{k}: frozen u={v['frozen_u']} score={v['frozen']['score']:.6f} "
              f"ratio={v['frozen']['ratio']:.4f} | dev argmax u={v['dev_argmax_u']} "
              f"score={v['dev_argmax']['score']:.6f} | frozen==argmax "
              f"{v['frozen_is_dev_argmax']} gap={v['gap_to_dev_argmax']:.6f}")
    for k, v in prem.items():
        print(f"premium.{k}: frozen u={v['frozen_u']} score={v['frozen']['score']:.6f} "
              f"| dev argmax u={v['dev_argmax_u']} score={v['dev_argmax']['score']:.6f} "
              f"| frozen==argmax {v['frozen_is_dev_argmax']} gap={v['gap_to_dev_argmax']:.6f}")


if __name__ == "__main__":
    main()
