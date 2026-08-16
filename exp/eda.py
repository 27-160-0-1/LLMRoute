# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""EDA + baseline-number verification. Writes analysis/eda.md."""

from __future__ import annotations

import collections
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from harness import MODEL_IDS, ROOT, TIERS, evaluate, load_split

LIGHT, AX31, K1 = 0, 1, 2

_CODE = re.compile(r"```|(?:^|\s)(?:def |class |function |SELECT |import |#include)", re.M)
_MATH = re.compile(r"[=+\-*/^∑∫√≤≥]|\\(?:frac|sum|int|sqrt)|\$[^$]+\$")
_KOREAN = re.compile(r"[가-힣]")
_TRANSLATE = re.compile(r"\b(?:translate|번역)\b", re.I)
_SUMMARIZE = re.compile(r"\b(?:summari[sz]e|요약)\b", re.I)
_MCQ = re.compile(r"(?:^|\n)\s*(?:\(?[A-E]\)|[A-E]\.)\s+\S", re.M)
_AIME = re.compile(r"\b(?:AIME|aime)\b")
_PROOF = re.compile(r"\b(?:prove|proof|증명)\b", re.I)


def categorize(text: str) -> str:
    korean = len(_KOREAN.findall(text)) / max(1, len(text))
    if _CODE.search(text):
        return "code"
    if _TRANSLATE.search(text):
        return "translate"
    if _SUMMARIZE.search(text):
        return "summarize"
    if len(text) >= 8000:
        return "long-context"
    math_hits = len(_MATH.findall(text))
    digits = sum(ch.isdigit() for ch in text)
    if math_hits >= 3 or digits / max(1, len(text)) > 0.06:
        return "math"
    if _MCQ.search(text):
        return "mcq"
    if korean > 0.2:
        return "korean-general"
    return "english-general"


def pct(values, qs=(50, 75, 90, 95, 99, 100)):
    arr = np.asarray(values, dtype=float)
    return {q: float(np.percentile(arr, q)) for q in qs}


def split_stats(split: str, lines: list) -> None:
    table = load_split(split)
    scores = np.asarray(table["scores"])  # N x 3
    costs = np.asarray(table["costs"])
    out_tok = np.asarray(table["out_tokens"])
    in_tok = np.asarray(table["in_tokens"])
    gens = np.asarray(table["gens"])
    texts = table["texts"]
    n = len(texts)
    light_total = costs[:, LIGHT].sum()

    lines.append(f"\n## Split: {split} ({n} episodes)\n")

    # --- official baseline verification ---
    lines.append("### 기준 수치 (공식 채점기 재계산)\n")
    lines.append("| 배정 | 점수 | 비용비 |")
    lines.append("| --- | ---: | ---: |")
    for label, index in (("all-light", LIGHT), ("all-ax31", AX31), ("all-k1", K1)):
        assignment = [MODEL_IDS[index]] * n
        report = evaluate(split, {"premium": assignment})
        tier_report = report["tiers"]["premium"]
        lines.append(
            f"| {label} | {float(tier_report['quality_score']):.4f} "
            f"| {float(tier_report['budget_ratio']):.4f} |"
        )
    # oracle: budget-free upper bound with cheapest max-score model
    best = scores.max(axis=1)
    oracle_sel = []
    for i in range(n):
        candidates = [j for j in range(3) if scores[i, j] == best[i]]
        cheapest = min(candidates, key=lambda j: costs[i, j])
        oracle_sel.append(MODEL_IDS[cheapest])
    report = evaluate(split, {"premium": oracle_sel})
    tier_report = report["tiers"]["premium"]
    lines.append(
        f"| oracle(min-cost) | {float(tier_report['quality_score']):.4f} "
        f"| {float(tier_report['budget_ratio']):.4f} |"
    )

    # --- score value structure ---
    lines.append("\n### score 값 구조\n")
    uniq = collections.Counter(scores.flatten().tolist())
    lines.append(
        "unique scores: "
        + ", ".join(f"{k}:{v}" for k, v in sorted(uniq.items()))
    )
    lines.append(
        "\nnum_generations unique: "
        + str(sorted(set(gens.flatten().tolist())))
    )

    # --- per-model marginals ---
    lines.append("\n### 모델별 score/토큰 요약\n")
    lines.append("| 모델 | 평균 score | 평균 in_tok | 평균 out_tok | out_tok p50/p90/p99/max | 평균 cost |")
    lines.append("| --- | ---: | ---: | ---: | --- | ---: |")
    for j, model in enumerate(MODEL_IDS):
        p = pct(out_tok[:, j])
        lines.append(
            f"| {model} | {scores[:, j].mean():.4f} | {in_tok[:, j].mean():.0f} "
            f"| {out_tok[:, j].mean():.0f} | {p[50]:.0f}/{p[90]:.0f}/{p[99]:.0f}/{p[100]:.0f} "
            f"| {costs[:, j].mean():.6f} |"
        )

    # --- upgrade structure ---
    light_max = scores[:, LIGHT] == best
    ax31_max = scores[:, AX31] == best
    k1_only = (scores[:, K1] > scores[:, LIGHT]) & (scores[:, K1] > scores[:, AX31])
    ax31_beats_light = scores[:, AX31] > scores[:, LIGHT]
    light_beats_k1 = scores[:, LIGHT] > scores[:, K1]
    lines.append("\n### 승격 구조\n")
    lines.append(f"- light가 최고점 동률 포함: {light_max.mean():.3f}")
    lines.append(f"- ax31 > light: {ax31_beats_light.mean():.3f}")
    lines.append(f"- k1이 유일 최고: {k1_only.mean():.3f}")
    lines.append(f"- light > k1 (역전): {light_beats_k1.mean():.3f}")
    lines.append(
        f"- k1 비용/light 비용 비율 평균: {(costs[:, K1] / costs[:, LIGHT]).mean():.1f}"
    )

    # --- category clusters ---
    lines.append("\n### 카테고리별 분석\n")
    lines.append(
        "| 카테고리 | n | light | ax31 | k1 | light充分% | k1유일% | k1 cost 평균 | light cost 평균 |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    cats = [categorize(t) for t in texts]
    for cat in sorted(set(cats)):
        mask = np.asarray([c == cat for c in cats])
        m = mask.sum()
        lines.append(
            f"| {cat} | {m} | {scores[mask, LIGHT].mean():.3f} "
            f"| {scores[mask, AX31].mean():.3f} | {scores[mask, K1].mean():.3f} "
            f"| {100 * light_max[mask].mean():.0f} | {100 * k1_only[mask].mean():.0f} "
            f"| {costs[mask, K1].mean():.5f} | {costs[mask, LIGHT].mean():.5f} |"
        )

    # --- length vs tokens ---
    chars = np.asarray([len(t) for t in texts])
    lines.append("\n### 길이/토큰 관계\n")
    lines.append(f"- 프롬프트 길이 p50/p90/p99/max: {pct(chars)[50]:.0f}/{pct(chars)[90]:.0f}/{pct(chars)[99]:.0f}/{chars.max()}")
    ratio = in_tok[:, LIGHT] / np.maximum(1, chars)
    lines.append(f"- light in_tok/char 평균 {ratio.mean():.3f} (p10 {np.percentile(ratio,10):.3f} / p90 {np.percentile(ratio,90):.3f})")
    corr = np.corrcoef(chars, in_tok[:, LIGHT])[0, 1]
    lines.append(f"- char vs in_tok 상관 {corr:.4f}")
    # k1 output tokens vs category/scores
    lines.append(
        f"- k1 out_tok 총합의 상위 10% 문항 비중: "
        f"{np.sort(out_tok[:, K1])[-max(1, n // 10):].sum() / out_tok[:, K1].sum():.3f}"
    )
    return None


def cross_split_notes(lines: list) -> None:
    train = load_split("train")
    dev = load_split("dev")
    lines.append("\n## Train ↔ Dev 분포 비교\n")
    for split, table in (("train", train), ("dev", dev)):
        cats = collections.Counter(categorize(t) for t in table["texts"])
        total = sum(cats.values())
        dist = ", ".join(
            f"{k}:{100 * v / total:.1f}%" for k, v in sorted(cats.items(), key=lambda kv: -kv[1])
        )
        lines.append(f"- {split}: {dist}")
    # exact prompt overlap
    train_set = set(train["texts"])
    dev_set = set(dev["texts"])
    overlap = train_set & dev_set
    lines.append(f"- Train/Dev 프롬프트 정확 중복: {len(overlap)}")
    dup_train = len(train["texts"]) - len(train_set)
    dup_dev = len(dev["texts"]) - len(dev_set)
    lines.append(f"- Train 내부 중복: {dup_train}, Dev 내부 중복: {dup_dev}")


def main() -> None:
    lines = ["# EDA — SKT OSSP 2026 LLM Router Challenge\n"]
    for split in ("train", "dev"):
        split_stats(split, lines)
    cross_split_notes(lines)
    out = ROOT / "analysis" / "eda.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {out}")
    print("\n".join(lines[:80]))


if __name__ == "__main__":
    main()
