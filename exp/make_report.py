# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Regenerate exp/results.md from exp/registry.jsonl (no manual edits)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import registry_lib
from registry_lib import ROOT, TIERS


def fmt_tier(row, tier):
    m = row["metrics"].get(tier)
    if not m:
        return "-"
    flag = "" if m.get("budget_pass") else " **[초과→0]**"
    cell = f"{float(m['score']):.4f} / {float(m['cost_ratio']):.4f}{flag}"
    return cell


def fmt_risk(row):
    parts = []
    for tier in TIERS:
        m = row["metrics"].get(tier, {})
        p = m.get("overrun_prob_bootstrap")
        parts.append("-" if p is None else f"{100 * p:.1f}%")
    return "/".join(parts)


def fmt_runtime(row):
    rt = row.get("runtime") or {}
    per = rt.get("per_tier_sec")
    env = rt.get("measured_env", "host")
    if not per:
        return "미실측"
    if isinstance(per, dict):
        avg = sum(float(v) for v in per.values()) / len(per)
    else:
        avg = float(per)
    return f"{avg:.1f}s ({env})"


def main() -> None:
    rows = registry_lib.load_all()
    rows.sort(key=lambda r: r["exp_id"])
    lines = [
        "# 실험 결과 (공식 Decimal 채점기; exp/make_report.py가 registry.jsonl에서 자동 생성)",
        "",
        "| exp_id | family | 접근법 | Fast 점수/비용비 | Balanced 점수/비용비 | Premium 점수/비용비 "
        "| 가중 최종점수 | 예산초과확률(F/B/P) | 등급당 추론시간 | 비고 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    best = None
    for row in rows:
        name = row["config"].get("name", row["exp_id"])
        final = float(row["metrics"]["weighted_final"])
        is_ref = row["family"] == "reference"
        marker = ""
        if not is_ref:
            if best is None or final > best:
                best = final
                marker = " 🔺"
        lines.append(
            f"| {row['exp_id']} | {row['family']} | {name} | "
            f"{fmt_tier(row, 'fast')} | {fmt_tier(row, 'balanced')} | {fmt_tier(row, 'premium')} | "
            f"**{final:.4f}**{marker} | {fmt_risk(row)} | {fmt_runtime(row)} | {row.get('notes', '')} |"
        )
    (ROOT / "exp" / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"results.md regenerated with {len(rows)} rows")


if __name__ == "__main__":
    main()
