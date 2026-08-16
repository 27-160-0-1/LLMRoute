# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""A3: deterministic surface-noise augmentation (label-preserving assumption).

Perturbations mimic private-set surface variation: whitespace, punctuation,
casing, typos, markdown tweaks, word drops. Never redistributed — used
in-memory for training only.
"""

from __future__ import annotations

import random
import re
from typing import List

_WS = re.compile(r" {2,}")


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4 or not word.isalpha():
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.random()
    if op < 0.4:  # swap
        return word[: i - 1] + word[i] + word[i - 1] + word[i + 1 :]
    if op < 0.7:  # drop
        return word[:i] + word[i + 1 :]
    return word[:i] + word[i] + word[i:]  # double


def perturb(text: str, seed: int, *, strength: float = 1.0) -> str:
    rng = random.Random(seed)
    out = text

    # markdown/whitespace tweaks
    if rng.random() < 0.5 * strength:
        out = out.replace("\r\n", "\n")
        out = _WS.sub(" ", out)
    if rng.random() < 0.3 * strength:
        out = re.sub(r"\n{3,}", "\n\n", out)
    if rng.random() < 0.2 * strength:
        out = out.replace("**", "")
    # punctuation jitter
    if rng.random() < 0.3 * strength:
        out = out.replace(" ,", ",").replace(" .", ".")
    if rng.random() < 0.2 * strength:
        out = re.sub(r"\.\s", lambda m: ". " if rng.random() < 0.9 else " ", out)
    # casing
    if rng.random() < 0.15 * strength and out and out[0].isalpha():
        out = (out[0].lower() if out[0].isupper() else out[0].upper()) + out[1:]

    # word-level: typos + drops on a small fraction (avoid numbers/code)
    words = out.split(" ")
    if len(words) > 8:
        n_edit = max(1, int(0.02 * strength * len(words)))
        for _ in range(n_edit):
            i = rng.randrange(len(words))
            w = words[i]
            if any(c.isdigit() for c in w) or "`" in w or "$" in w:
                continue
            r = rng.random()
            if r < 0.6:
                words[i] = _typo(w, rng)
            elif r < 0.8 and len(w) < 12 and w.lower() not in ("not", "no", "without", "must", "only"):
                words[i] = ""  # drop (guard negation/constraint words)
        out = " ".join(w for w in words if w != "")
    return out if out.strip() else text


def perturb_batch(texts: List[str], n_variants: int = 1, base_seed: int = 42, strength: float = 1.0) -> List[List[str]]:
    """Return [variant][row] perturbed texts, deterministic per (row, variant)."""

    return [
        [perturb(t, base_seed + 1000 * v + i, strength=strength) for i, t in enumerate(texts)]
        for v in range(n_variants)
    ]
