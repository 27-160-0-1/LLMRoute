# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""A7 channel decomposition: which of {score head, cost head} actually earns
the dev score?  Uses the already-trained baseline proxy predictions
(build/a7/pred-baseline-dev.npz) and swaps one channel for a constant
(marginal-only) predictor fitted on train labels.

No retraining.  Writes build/a7/ablate.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "exp" / "audit"))

from a7_proxy import OUT, load_raw, monotone, run_and_score  # noqa: E402


def main() -> None:
    d = np.load(OUT / "pred-baseline-dev.npz")
    s_real, c_real, q_real = d["score"], d["cost"], d["costq"]
    _, _, _, t_tr = load_raw("train")
    n = s_real.shape[0]
    s_const = np.repeat(t_tr["scores"].mean(axis=0)[None, :], n, axis=0)
    c_const = monotone(np.repeat(
        np.exp(np.log(t_tr["costs"]).mean(axis=0))[None, :], n, axis=0))

    res = []
    res.append(run_and_score(s_real, c_real, q_real, "dev",
                             "A  score=real  cost=real   (proxy baseline)"))
    res.append(run_and_score(s_const, c_real, q_real, "dev",
                             "B  score=CONST cost=real   (cost channel only)"))
    res.append(run_and_score(s_real, c_const, c_const, "dev",
                             "C  score=real  cost=CONST  (score channel only)"))
    res.append(run_and_score(s_const, c_const, c_const, "dev",
                             "D  score=CONST cost=CONST  (no information)"))
    (OUT / "ablate.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote build/a7/ablate.json")


if __name__ == "__main__":
    main()
