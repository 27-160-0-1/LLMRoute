# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Experiment registry: one JSONL line per experiment in exp/registry.jsonl."""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "exp" / "registry.jsonl"
TIERS = ("fast", "balanced", "premium")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        ).stdout.strip()
    except OSError:
        return "unknown"


def load_all() -> list:
    if not REGISTRY.exists():
        return []
    rows = []
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def next_exp_id() -> str:
    rows = load_all()
    highest = 0
    for row in rows:
        eid = row.get("exp_id", "E000")
        try:
            highest = max(highest, int(eid.lstrip("E")))
        except ValueError:
            continue
    return f"E{highest + 1:03d}"


def metrics_from_report(
    report: Mapping[str, Any],
    over_probs: Optional[Mapping[str, float]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for tier in TIERS:
        t = report["tiers"][tier]
        metrics[tier] = {
            "score": t["quality_score"],
            "cost_ratio": t["budget_ratio"],
            "budget_pass": t["budget_passed"],
            "overrun_prob_bootstrap": (
                float(over_probs[tier]) if over_probs and tier in over_probs else None
            ),
        }
    metrics["weighted_final"] = report["final_score"]
    metrics.update(extra)
    return metrics


def register(
    *,
    family: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    runtime: Optional[Mapping[str, Any]] = None,
    artifacts: Sequence[str] = (),
    parent_exp: Optional[str] = None,
    notes: str = "",
    exp_id: Optional[str] = None,
) -> str:
    if exp_id is None:
        exp_id = next_exp_id()
    row = {
        "exp_id": exp_id,
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "family": family,
        "config": dict(config),
        "metrics": dict(metrics),
        "runtime": dict(runtime) if runtime else {"measured_env": "host", "per_tier_sec": None},
        "artifacts": list(artifacts),
        "parent_exp": parent_exp,
        "notes": notes,
    }
    with REGISTRY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return exp_id
