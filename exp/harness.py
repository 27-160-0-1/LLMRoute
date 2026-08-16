# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Experiment harness wrapping the official Decimal scorer.

Every experiment must be evaluated through ``evaluate`` (official scorer).
Float arrays from ``episode_table`` are for optimization loops only; the
numbers recorded in exp/results.md always come from the official scorer.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT / "src"), str(ROOT / "baselines")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    Decision,
    Submission,
    load_bundled_policy,
    load_input,
    load_outcomes,
    parse_submission,
    submission_to_dict,
)
from ossp_router.scoring import score_submissions  # noqa: E402

POLICY = load_bundled_policy()
RESULTS_MD = ROOT / "exp" / "results.md"
TIER_MULTIPLIERS = {tier: float(POLICY.tiers[tier].budget_multiplier) for tier in TIERS}
TIER_WEIGHTS = {tier: float(POLICY.tiers[tier].weight) for tier in TIERS}

_CACHE: Dict[str, dict] = {}


def _episode_text(episode) -> str:
    if episode.prompt is not None:
        return episode.prompt
    return "\n".join(message.content for message in episode.messages)


def load_split(split: str) -> dict:
    """Load inputs + outcomes for a split with aligned per-episode arrays."""

    if split in _CACHE:
        return _CACHE[split]
    inputs = load_input(ROOT / "data" / "materialized" / split / "inputs.json")
    outcomes = load_outcomes(ROOT / "data" / split / "outcomes.json")
    outcome_index = {
        (item.episode_id, item.model_id): item for item in outcomes.outcomes
    }
    unit = Decimal(POLICY.token_unit)
    texts: List[str] = []
    scores: List[List[float]] = []
    costs: List[List[float]] = []
    in_tokens: List[List[int]] = []
    out_tokens: List[List[int]] = []
    gens: List[List[int]] = []
    for episode in inputs.episodes:
        texts.append(_episode_text(episode))
        row_scores, row_costs, row_in, row_out, row_gen = [], [], [], [], []
        for model_id in MODEL_IDS:
            outcome = outcome_index[(episode.episode_id, model_id)]
            rates = POLICY.models[model_id]
            cost = (
                rates.fixed_cost
                + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
                + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
            )
            row_scores.append(float(outcome.score))
            row_costs.append(float(cost))
            row_in.append(outcome.input_tokens)
            row_out.append(outcome.output_tokens)
            row_gen.append(outcome.num_generations)
        scores.append(row_scores)
        costs.append(row_costs)
        in_tokens.append(row_in)
        out_tokens.append(row_out)
        gens.append(row_gen)
    table = {
        "inputs": inputs,
        "outcomes": outcomes,
        "episode_ids": [episode.episode_id for episode in inputs.episodes],
        "texts": texts,
        "scores": scores,
        "costs": costs,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "gens": gens,
    }
    _CACHE[split] = table
    return table


def make_submission(split: str, tier: str, selected: Sequence[str]) -> Submission:
    table = load_split(split)
    inputs = table["inputs"]
    if len(selected) != len(inputs.episodes):
        raise ValueError("selection length mismatch")
    submission = Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=POLICY.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(
            Decision(episode.episode_id, model_id)
            for episode, model_id in zip(inputs.episodes, selected)
        ),
    )
    return parse_submission(submission_to_dict(submission))


def evaluate(split: str, assignment_by_tier: Mapping[str, Sequence[str]]) -> dict:
    """Officially score one assignment per tier; missing tiers use all-light."""

    table = load_split(split)
    inputs = table["inputs"]
    outcomes = table["outcomes"]
    all_light = [MODEL_IDS[0]] * len(inputs.episodes)
    submissions = [
        make_submission(split, tier, assignment_by_tier.get(tier, all_light))
        for tier in TIERS
    ]
    return score_submissions(inputs, outcomes, submissions, POLICY)


def summarize(report: dict) -> dict:
    """Compact per-tier summary of an official report."""

    out = {}
    for tier in TIERS:
        tier_report = report["tiers"][tier]
        out[tier] = {
            "score": tier_report["quality_score"],
            "ratio": tier_report["budget_ratio"],
            "passed": tier_report["budget_passed"],
            "counts": tier_report["model_counts"],
        }
    out["final"] = report["final_score"]
    return out


def bootstrap_over_probability(
    selected_costs: Sequence[float],
    light_costs: Sequence[float],
    multiplier: float,
    *,
    iterations: int = 1000,
    seed: int = 20260816,
) -> float:
    """P(budget ratio > multiplier) under episode resampling with replacement."""

    import random

    rng = random.Random(seed)
    n = len(selected_costs)
    over = 0
    for _ in range(iterations):
        total_sel = 0.0
        total_light = 0.0
        for _ in range(n):
            index = rng.randrange(n)
            total_sel += selected_costs[index]
            total_light += light_costs[index]
        if total_sel > total_light * multiplier:
            over += 1
    return over / iterations


def append_result(
    name: str,
    report: dict,
    *,
    over_probs: Optional[Mapping[str, float]] = None,
    runtime_seconds: Optional[str] = None,
    notes: str = "",
) -> str:
    """Append one experiment row to exp/results.md and return the row."""

    cells = []
    for tier in TIERS:
        tier_report = report["tiers"][tier]
        score = tier_report["quality_score"]
        ratio = tier_report["budget_ratio"]
        flag = "" if tier_report["budget_passed"] else " **[초과→0]**"
        cells.append(f"{float(score):.4f} / {float(ratio):.4f}{flag}")
    final = float(report["final_score"])
    if over_probs:
        risk = "/".join(
            f"{100.0 * float(over_probs.get(tier, float('nan'))):.1f}%" for tier in TIERS
        )
    else:
        risk = "-"
    runtime = runtime_seconds if runtime_seconds is not None else "-"
    stamp = _dt.datetime.now().strftime("%m-%d %H:%M")
    row = (
        f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | "
        f"**{final:.4f}** | {risk} | {runtime} | {notes} ({stamp}) |"
    )
    header = (
        "| 접근법 | Fast 점수/비용비 | Balanced 점수/비용비 | Premium 점수/비용비 "
        "| 가중 최종점수 | 예산초과확률(F/B/P) | 등급당 추론시간 | 비고 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    if not RESULTS_MD.exists():
        RESULTS_MD.write_text(
            "# 실험 결과 (공식 Decimal 채점기 기준)\n\n" + header,
            encoding="utf-8",
        )
    with RESULTS_MD.open("a", encoding="utf-8") as handle:
        handle.write(row + "\n")
    return row


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
