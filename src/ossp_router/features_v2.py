# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Deterministic content-only feature extraction for the final router.

Byte-for-byte port of the training-time extractor (exp/feat_lib.py, v2).
Word n-grams are hashed with zlib.crc32; char n-grams with a vectorized
numpy polynomial rolling hash. No randomness, no ID/order dependence.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Dict, List, Sequence, Tuple

import numpy as np

FEATURE_VERSION = 2

WORD_BINS = 1 << 16
CHAR_BINS = 1 << 16
CHAR_NS = (3, 4)
CHAR_CAP = 4096

_TOKEN = re.compile(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]")

_RE = {
    "code_fence": re.compile(r"```"),
    "code_kw": re.compile(
        r"(?:^|\s)(?:def |class |function |import |#include|SELECT |INSERT |public static|=>|\breturn\b)",
        re.M,
    ),
    "latex": re.compile(r"\\(?:frac|sum|int|sqrt|cdot|times|left|right|begin|end|mathbb|angle|triangle|pi\b)"),
    "math_sym": re.compile(r"[=+\-*/^∑∫√≤≥≠±×÷]"),
    "dollar_math": re.compile(r"\$[^$\n]{1,200}\$"),
    "mcq_opt": re.compile(r"(?:^|\n)\s*(?:\(?[A-J]\)|[A-J]\.)\s+\S"),
    "which_following": re.compile(r"which of the following|다음 중", re.I),
    "translate": re.compile(r"\btranslat|번역", re.I),
    "summarize": re.compile(r"\bsummari[sz]|요약", re.I),
    "rewrite": re.compile(r"\brewrite|바꿔|고쳐|수정", re.I),
    "extract_list": re.compile(r"\bextract|\blist\b|나열|추출", re.I),
    "prove": re.compile(r"\bprove\b|\bproof\b|증명", re.I),
    "step": re.compile(r"step[- ]by[- ]step|단계별", re.I),
    "explain": re.compile(r"\bexplain|\bwhy\b|설명|이유", re.I),
    "reasoning": re.compile(
        r"\b(?:derive|reason|analyze|algorithm|complexity|theorem|lemma|induction|counterexample)\b|유도|추론|분석|알고리즘|복잡도|정리|귀납|반례",
        re.I,
    ),
    "constraint": re.compile(
        r"\b(?:exactly|at least|at most|must|only|without|no more than)\b|정확히|이상|이하|반드시|오직|제외",
        re.I,
    ),
    "find_number": re.compile(r"\bfind the\b|\bhow many\b|\bcompute\b|\bcalculate\b|구하시오|구하라|계산", re.I),
    "remainder_mod": re.compile(r"\bremainder\b|\bmodulo\b|\bdivisible\b|나머지|나누어", re.I),
    "json_marker": re.compile(r"[{\[]\s*\"|json", re.I),
    "table_marker": re.compile(r"\|.+\|.+\|"),
    "html_marker": re.compile(r"</?[a-z]+[^>]*>"),
    "url": re.compile(r"https?://"),
    "question": re.compile(r"[?？]"),
    "roman_choice": re.compile(r"\b(?:I{1,3}|IV|V)\.\s"),
    "aime_style": re.compile(r"\b(?:AIME|integer answer|answer is an integer)\b", re.I),
    "geometry": re.compile(r"\btriangle|circle|polygon|angle|radius|perimeter|vertex|삼각형|원의|각도", re.I),
    "probability": re.compile(r"\bprobability|expected value|확률|기댓값", re.I),
    "sequence": re.compile(r"\bsequence|series|recurrence|수열|점화식", re.I),
    "wordproblem": re.compile(r"\bhow much|\bhow many|얼마나|몇\s", re.I),
    "python_ref": re.compile(r"\bpython|파이썬", re.I),
    "bugfix": re.compile(r"\bbug|\bfix|\berror|\btraceback|\bexception|오류|버그|예외", re.I),
    "implement": re.compile(r"\bimplement|\bwrite a (?:function|program|class)|작성하시오|구현", re.I),
    "multi_part": re.compile(r"\([a-e]\)\s|\b(?:part|question)\s+\d|문항|\(1\)|\(2\)"),
    "roleplay": re.compile(r"\byou are\b|\bact as\b|역할", re.I),
    "format_req": re.compile(r"\bformat\b|\bjson\b|\bmarkdown\b|\btable\b|형식|표로", re.I),
}

DENSE_DIM = 46 + len(_RE)


def dense_features(text: str, message_count: int = 1) -> List[float]:
    n = len(text)
    nonspace = sum(not c.isspace() for c in text[:20000]) + max(0, n - 20000) // 2
    lower = text[:20000]
    hangul = sum("\uac00" <= c <= "\ud7a3" for c in lower)
    latin = sum(("a" <= c <= "z") or ("A" <= c <= "Z") for c in lower)
    digits = sum(c.isdigit() for c in lower)
    upper = sum("A" <= c <= "Z" for c in lower)
    lines = text.split("\n")
    n_lines = len(lines)
    indent_lines = sum(1 for ln in lines[:2000] if ln.startswith(("    ", "\t")))
    words = _TOKEN.findall(text[:40000])
    n_words = len(words)
    n_numbers = sum(1 for w in words if w.isdigit())
    avg_word = (sum(len(w) for w in words) / n_words) if n_words else 0.0
    sentences = max(1, len(re.findall(r"[.!?。！？]", text[:40000])))
    denom = max(1, min(n, 20000))

    feats = [
        math.log1p(n),
        math.log1p(n_words),
        math.log1p(sentences),
        math.log1p(message_count),
        math.log1p(n_lines),
        math.log1p(indent_lines),
        hangul / denom,
        latin / denom,
        digits / denom,
        upper / denom,
        n_numbers / max(1, n_words),
        avg_word,
        float(n >= 2000),
        float(n >= 8000),
        float(n >= 32000),
        n / 70000.0,
        math.log1p(n) ** 2 / 100.0,
        float(message_count >= 2),
        float(message_count >= 3),
        (n_words / sentences) if sentences else 0.0,
        float(text[:1].isdigit()),
        float(bool(re.match(r"\s*(?:Round|Solve|Simplify|Calculate|What|Let|Suppose|Evaluate|Convert|Work out|Find)", text[:64], re.I))),
    ]
    feats.extend([0.0] * (46 - len(feats)))
    for key in sorted(_RE):
        feats.append(math.log1p(len(_RE[key].findall(text[:40000]))))
    return feats


def word_gram_hash(text: str, bins: int = WORD_BINS) -> Dict[int, float]:
    tokens = []
    for token in _TOKEN.findall(text[:40000]):
        token = token.casefold()
        if token.isdecimal():
            token = "<num>"
        tokens.append(token)
    acc: Dict[int, float] = {}
    for prefix, seq in (
        (b"w1:", tokens),
        (b"w2:", [f"{a}\x1f{b}" for a, b in zip(tokens, tokens[1:])]),
    ):
        for token in seq:
            h = zlib.crc32(prefix + token.encode("utf-8"))
            index = h & (bins - 1)
            sign = 1.0 if h & 0x80000000 else -1.0
            acc[index] = acc.get(index, 0.0) + sign
    return acc


_P = np.uint64(1099511628211)


def char_gram_hash(text: str, bins: int = CHAR_BINS, ns: Sequence[int] = CHAR_NS, cap: int = CHAR_CAP) -> Dict[int, float]:
    snippet = re.sub(r"\s+", " ", text[:cap].casefold())
    raw = snippet.encode("utf-8", "ignore")[: cap + 512]
    if len(raw) < min(ns):
        return {}
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.uint64)
    acc: Dict[int, float] = {}
    for order, gram in enumerate(ns):
        if len(arr) < gram:
            continue
        h = np.zeros(len(arr) - gram + 1, dtype=np.uint64)
        for offset in range(gram):
            h = h * _P + arr[offset : len(arr) - gram + 1 + offset]
        h = h ^ (h >> np.uint64(29))
        h = h * _P
        h = (h ^ (h >> np.uint64(32))) + np.uint64(order * 0x9E3779B9)
        index = (h & np.uint64(bins - 1)).astype(np.int64)
        sign = np.where((h >> np.uint64(63)) & np.uint64(1), 1.0, -1.0)
        for i, s in zip(index.tolist(), sign.tolist()):
            acc[i] = acc.get(i, 0.0) + s
    return acc


def _l2_dict(d: Dict[int, float]) -> Dict[int, float]:
    norm = math.sqrt(sum(v * v for v in d.values()))
    if norm:
        return {k: v / norm for k, v in d.items()}
    return d


def featurize_one(text: str, message_count: int) -> Tuple[List[float], Dict[int, float], Dict[int, float]]:
    return (
        dense_features(text, message_count),
        _l2_dict(word_gram_hash(text)),
        _l2_dict(char_gram_hash(text)),
    )


def featurize_batch(texts: Sequence[str], message_counts: Sequence[int]):
    """Return dense (N,D) + word/char scipy CSR (row-L2, same as training)."""

    from scipy import sparse

    results = [featurize_one(t, m) for t, m in zip(texts, message_counts)]
    dense = np.asarray([r[0] for r in results], dtype=np.float64)

    def to_csr(dicts, bins):
        indptr = [0]
        indices: List[int] = []
        data: List[float] = []
        for d in dicts:
            keys = sorted(d)
            indices.extend(keys)
            data.extend(d[k] for k in keys)
            indptr.append(len(indices))
        return sparse.csr_matrix(
            (np.asarray(data), np.asarray(indices, dtype=np.int64), np.asarray(indptr, dtype=np.int64)),
            shape=(len(dicts), bins),
        )

    word = to_csr([r[1] for r in results], WORD_BINS)
    char = to_csr([r[2] for r in results], CHAR_BINS)
    return dense, word, char
