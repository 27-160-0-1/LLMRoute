# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Naive constant prediction set — smoke test for the eval pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from harness import ROOT, load_split


def main() -> None:
    out = ROOT / "build" / "preds"
    out.mkdir(parents=True, exist_ok=True)
    train = load_split("train")
    mean_scores = np.asarray(train["scores"]).mean(axis=0)
    mean_costs = np.asarray(train["costs"]).mean(axis=0)
    mean_chars = np.mean([len(x) for x in train["texts"]])
    for split in ("train", "dev"):
        t = load_split(split)
        n = len(t["texts"])
        chars = np.asarray([len(x) for x in t["texts"]], dtype=float)
        score_pred = np.tile(mean_scores, (n, 1))
        cost_pred = np.outer(chars / mean_chars, mean_costs)
        np.savez(out / f"naive-{split}.npz", score=score_pred, cost=cost_pred)
    print("naive preds written")


if __name__ == "__main__":
    main()
