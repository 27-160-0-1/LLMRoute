# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Precompute shared features/targets for all experiments → build/feats/."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import sparse

import feat_lib
from harness import ROOT, load_split

OUT = ROOT / "build" / "feats"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        table = load_split(split)
        texts = table["texts"]
        message_counts = [
            1 if e.prompt is not None else len(e.messages)
            for e in table["inputs"].episodes
        ]
        start = time.time()
        dense, word, char = feat_lib.featurize_batch(texts, message_counts)
        elapsed = time.time() - start
        np.save(OUT / f"dense-{split}.npy", dense)
        sparse.save_npz(OUT / f"word-{split}.npz", word)
        sparse.save_npz(OUT / f"char-{split}.npz", char)
        np.savez(
            OUT / f"targets-{split}.npz",
            scores=np.asarray(table["scores"]),
            costs=np.asarray(table["costs"]),
            in_tokens=np.asarray(table["in_tokens"]),
            out_tokens=np.asarray(table["out_tokens"]),
            gens=np.asarray(table["gens"]),
        )
        print(
            f"{split}: dense {dense.shape}, word nnz {word.nnz}, char nnz {char.nnz}, "
            f"featurize {elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
