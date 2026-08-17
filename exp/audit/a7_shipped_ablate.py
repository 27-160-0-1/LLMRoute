# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""A7: channel ablation on the ACTUAL SHIPPED 4-member blend (not the proxy).

Loads the shipped router from build/bundle-nolookup (lookup.npz emptied, so
every number here is lookup-OFF) via the shipped FinalRouter class, takes its
real dev predictions, then re-scores after replacing ONE channel at a time
with a constant (train-marginal) predictor.

READ-ONLY: nothing under models/final-v1, src/ or build/bundle-nolookup is
written.  Output: build/a7/shipped_ablate.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "exp" / "audit"))

from ossp_router.final_router import FinalRouter  # noqa: E402

from a7_proxy import OUT, load_raw, monotone, run_and_score  # noqa: E402
from harness import load_split  # noqa: E402

BUNDLE = ROOT / "build" / "bundle-nolookup"


def main() -> None:
    table = load_split("dev")
    texts = table["texts"]
    counts = [1 if e.prompt is not None else len(e.messages or ())
              for e in table["inputs"].episodes]
    router = FinalRouter(BUNDLE)
    print(f"lookup rows in bundle: {len(router.lookup_key)}", flush=True)
    score, cost, costq = router.predict_batch(texts, counts)
    np.savez(OUT / "pred-shipped-nolookup-dev.npz",
             score=score, cost=cost, costq=costq)

    _, _, _, t_tr = load_raw("train")
    n = score.shape[0]
    s_const = np.repeat(t_tr["scores"].mean(axis=0)[None, :], n, axis=0)
    c_const = monotone(np.repeat(
        np.exp(np.log(t_tr["costs"]).mean(axis=0))[None, :], n, axis=0))

    res = []
    res.append(run_and_score(score, cost, costq, "dev",
                             "SHIPPED-A score=blend4 cost=real  (= honest run)"))
    res.append(run_and_score(s_const, cost, costq, "dev",
                             "SHIPPED-B score=CONST  cost=real"))
    res.append(run_and_score(score, c_const, c_const, "dev",
                             "SHIPPED-C score=blend4 cost=CONST"))
    res.append(run_and_score(s_const, c_const, c_const, "dev",
                             "SHIPPED-D score=CONST  cost=CONST"))
    # score-model discrimination diagnostics
    real = np.asarray(table["scores"])
    diag = {}
    for j, m in enumerate(("ax31-light", "ax31", "axk1-think")):
        diag[m] = {
            "pearson_r": float(np.corrcoef(score[:, j], real[:, j])[0, 1]),
            "pred_mean": float(score[:, j].mean()),
            "true_mean": float(real[:, j].mean()),
            "pred_std": float(score[:, j].std()),
            "true_std": float(real[:, j].std()),
        }
    gain_true = real[:, 1] - real[:, 0]
    gain_pred = score[:, 1] - score[:, 0]
    diag["gain_ax31_minus_light"] = {
        "pearson_r": float(np.corrcoef(gain_pred, gain_true)[0, 1]),
        "pred_mean": float(gain_pred.mean()),
        "true_mean": float(gain_true.mean()),
    }
    payload = {"runs": res, "score_model_diagnostics": diag}
    (OUT / "shipped_ablate.json").write_text(json.dumps(payload, indent=1),
                                             encoding="utf-8")
    print(json.dumps(diag, indent=1))
    print("wrote build/a7/shipped_ablate.json")


if __name__ == "__main__":
    main()
