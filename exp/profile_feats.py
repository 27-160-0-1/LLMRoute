# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Profile feature extraction blocks over the combined 2,640-episode workload."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import feat_lib
from harness import load_split


def main() -> None:
    texts = load_split("train")["texts"] + load_split("dev")["texts"]
    print(f"episodes: {len(texts)}, total chars: {sum(len(t) for t in texts):,}")

    start = time.time()
    for t in texts:
        feat_lib.dense_features(t)
    print(f"dense: {time.time() - start:.2f}s")

    start = time.time()
    for t in texts:
        feat_lib.word_gram_hash(t)
    print(f"word-gram: {time.time() - start:.2f}s")

    start = time.time()
    for t in texts:
        feat_lib.char_gram_hash(t)
    print(f"char-gram: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
