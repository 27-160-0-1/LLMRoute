# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Compose a cost prediction set column-wise from different sources.

Usage: python exp/models/compose_cost.py --light cost2 --ax31 cost2 --k1 lgbm-q90 \
       --score-from blend6-cost2 --name blend6-q90k1
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
    parser.add_argument("--light", required=True)
    parser.add_argument("--ax31", required=True)
    parser.add_argument("--k1", required=True)
    parser.add_argument("--score-from", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    for split in ("train", "dev"):
        score = np.load(PREDS / f"{args.score_from}-{split}.npz")["score"]
        cost = np.empty_like(score)
        for j, src in enumerate((args.light, args.ax31, args.k1)):
            cost[:, j] = np.load(PREDS / f"{src}-{split}.npz")["cost"][:, j]
        cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1 + 1e-12))
        cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1 + 1e-12))
        np.savez(PREDS / f"{args.name}-{split}.npz", score=score, cost=cost)
    (PREDS / f"{args.name}-summary.json").write_text(
        json.dumps(
            {
                "variant": args.name,
                "score_from": args.score_from,
                "cost_cols": {"light": args.light, "ax31": args.ax31, "k1": args.k1},
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"{args.name} written")


if __name__ == "__main__":
    main()
