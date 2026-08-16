# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Blending: weighted average of base prediction sets (robust vs stacking).

Usage: python exp/models/blend.py a b c --weights 1,1,1 --cost-from mlp-ens --name blend-x
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from harness import ROOT

PREDS = ROOT / "build" / "preds"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bases", nargs="+")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--cost-from", default=None, help="단일 셋 비용 사용; 기본은 log-space 가중평균")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    weights = (
        np.asarray([float(w) for w in args.weights.split(",")])
        if args.weights
        else np.ones(len(args.bases))
    )
    weights = weights / weights.sum()

    summary = {"variant": args.name, "bases": args.bases, "weights": weights.tolist(), "cost_from": args.cost_from}
    for split in ("train", "dev"):
        scores = []
        log_costs = []
        for base in args.bases:
            data = np.load(PREDS / f"{base}-{split}.npz")
            scores.append(np.clip(data["score"], 0, 1))
            log_costs.append(np.log(np.maximum(data["cost"], 1e-12)))
        score = sum(w * s for w, s in zip(weights, scores))
        if args.cost_from:
            cost = np.load(PREDS / f"{args.cost_from}-{split}.npz")["cost"]
        else:
            cost = np.exp(sum(w * c for w, c in zip(weights, log_costs)))
        cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1 + 1e-12))
        cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1 + 1e-12))
        np.savez(PREDS / f"{args.name}-{split}.npz", score=score, cost=cost)
        if split == "train":
            tgt = np.load(ROOT / "build" / "feats" / "targets-train.npz")["scores"]
            mse = ((score - tgt) ** 2).mean(axis=0)
            summary["oof_score_mse"] = mse.tolist()
            summary["oof_score_mse_mean"] = float(mse.mean())
    (PREDS / f"{args.name}-summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
