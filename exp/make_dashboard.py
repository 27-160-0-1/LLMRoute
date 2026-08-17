# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Dashboard: registry.jsonl -> exp/figs/*.png + self-contained exp/dashboard.html."""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import registry_lib
from registry_lib import ROOT, TIERS

FIGS = ROOT / "exp" / "figs"
MULTIPLIERS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
ORACLE = {"fast": 0.7594, "balanced": 0.8074, "premium": 0.8591, "final": 0.8037}

plt.rcParams.update(
    {
        "font.family": ["Malgun Gothic", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "#f8f8f8",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 9,
    }
)

CHARTS = []


def save(fig, name, title):
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    CHARTS.append((title, base64.b64encode(buf.getvalue()).decode()))


# 2026-08-17 감사(exp/audit/, exp/ceiling-check.md)로 확정한 실측값.
# MEASURED: 동결 정책을 공식 채점기로 직접 재측정한 값.
#   조회표 OFF = 일반화 추정치, 조회표 ON = Dev 전 문항 적중(암기)이므로 성능 지표가 아님.
# NULL_RANDOM_FEATURES: 특징을 난수로 대체해도 나오는 점수. 진짜 무정보 기준선은
#   all-light 0.6193이 아니라 이 값이며, 이득 대부분이 비용모델+배분기에서 나옴을 뜻한다.
MEASURED = [
    ("검증 실측 — 조회표 OFF (정직한 일반화)", 0.684318, "#1d9e75"),
    ("검증 실측 — 조회표 ON (암기, 성능 아님)", 0.760284, "#d03b3b"),
]
NULL_RANDOM_FEATURES = 0.656297


def chart_leaderboard(rows):
    entries = [
        (
            f"{r['exp_id']} {r['config'].get('name', '')[:44]}",
            float(r["metrics"]["weighted_final"]),
            "#888888" if r["family"] == "reference" else "#3b7dd8",
        )
        for r in rows
    ]
    entries.extend((n, v, c) for n, v, c in MEASURED)
    entries.sort(key=lambda e: e[1])
    names = [e[0] for e in entries]
    finals = [e[1] for e in entries]
    colors = [e[2] for e in entries]
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.32 * len(entries))))
    ax.barh(names, finals, color=colors)
    ax.axvline(ORACLE["final"], color="red", ls="--", lw=1, label=f"budget-oracle {ORACLE['final']}")
    ax.axvline(0.6954, color="orange", ls=":", lw=1, label="hash-regex 0.6954")
    ax.axvline(
        NULL_RANDOM_FEATURES,
        color="#7a4fbf",
        ls="-.",
        lw=1.2,
        label=f"무정보 null (랜덤 특징) {NULL_RANDOM_FEATURES:.4f}",
    )
    ax.axvspan(0.55, NULL_RANDOM_FEATURES, color="#7a4fbf", alpha=0.06)
    for i, v in enumerate(finals):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=7)
    for i, n in enumerate(names):
        if n.startswith("검증 실측"):
            ax.get_yticklabels()[i].set_fontweight("bold")
    ax.set_xlabel("가중 최종점수 (dev) — 음영 구간은 무정보 null 이하")
    ax.set_xlim(0.55, 0.85)
    ax.legend(loc="lower right", fontsize=7)
    save(fig, "leaderboard", "1. 리더보드 — 전 실험 가중 최종점수 (2026-08-17 감사 실측 반영)")


def chart_pareto(rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    families = sorted({r["family"] for r in rows})
    cmap = {f: c for f, c in zip(families, plt.cm.tab10.colors)}
    for ax, tier in zip(axes, TIERS):
        for r in rows:
            m = r["metrics"].get(tier)
            if not m:
                continue
            ax.scatter(
                float(m["cost_ratio"]),
                float(m["score"]),
                color=cmap[r["family"]],
                s=22,
                alpha=0.8,
            )
            ax.annotate(r["exp_id"], (float(m["cost_ratio"]), float(m["score"])), fontsize=5, alpha=0.6)
        ax.axvline(MULTIPLIERS[tier], color="red", ls="--", lw=1)
        ax.axhline(ORACLE[tier], color="red", ls=":", lw=0.8)
        ax.set_title(f"{tier} (한도 {MULTIPLIERS[tier]})")
        ax.set_xlabel("비용비")
        ax.set_ylabel("점수")
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap[f], label=f) for f in families]
    axes[2].legend(handles=handles, fontsize=6, loc="lower right")
    save(fig, "pareto", "2. 등급별 Pareto frontier (x=비용비, y=점수)")


def chart_risk(rows):
    cand = [r for r in rows if any(
        (r["metrics"].get(t) or {}).get("overrun_prob_bootstrap") is not None for t in TIERS
    )]
    cand = sorted(cand, key=lambda r: -float(r["metrics"]["weighted_final"]))[:12]
    if not cand:
        return
    fig, ax = plt.subplots(figsize=(9, 3.2))
    width = 0.25
    x = np.arange(len(cand))
    for k, tier in enumerate(TIERS):
        probs = [
            100 * ((r["metrics"].get(tier) or {}).get("overrun_prob_bootstrap") or 0.0)
            for r in cand
        ]
        ax.bar(x + (k - 1) * width, probs, width, label=tier)
    ax.axhline(1.0, color="red", ls="--", lw=1, label="한도 1%")
    ax.set_xticks(x, [r["exp_id"] for r in cand], fontsize=7)
    ax.set_ylabel("부트스트랩 예산초과확률 (%)")
    ax.legend(fontsize=7)
    save(fig, "risk", "3. 예산 위험도 — 부트스트랩 초과확률")


def chart_sweep(rows):
    detailed = [
        (r, ROOT / "exp" / r["exp_id"] / "detail.npz")
        for r in rows
        if (ROOT / "exp" / r["exp_id"] / "detail.npz").exists()
    ]
    if not detailed:
        return
    r, path = max(detailed, key=lambda t: float(t[0]["metrics"]["weighted_final"]))
    data = np.load(path)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, tier in zip(axes, TIERS):
        util = data[f"sweep_{tier}_util"]
        score = data[f"sweep_{tier}_score"]
        ratio = data[f"sweep_{tier}_ratio"]
        ax.plot(util, score, "o-", ms=3, label="dev 점수")
        ax2 = ax.twinx()
        ax2.plot(util, ratio, "s--", ms=3, color="tomato", label="실현 비용비")
        ax2.axhline(MULTIPLIERS[tier], color="red", ls=":", lw=1)
        ax.set_title(f"{tier} — {r['exp_id']}")
        ax.set_xlabel("목표 사용률")
    save(fig, "sweep", f"4. 마진 sweep — 최고 실험 {r['exp_id']} ({r['config'].get('name')})")


def _best_pred_row(rows):
    cands = [r for r in rows if r["family"] == "model" and r["config"].get("pred_set")]
    if not cands:
        return None
    return max(cands, key=lambda r: float(r["metrics"]["weighted_final"]))


def chart_pred_quality(rows):
    r = _best_pred_row(rows)
    if r is None:
        return
    name = r["config"]["pred_set"]
    pred_path = ROOT / "build" / "preds" / f"{name}-dev.npz"
    tgt_path = ROOT / "build" / "feats" / "targets-dev.npz"
    if not pred_path.exists() or not tgt_path.exists():
        return
    pred = np.load(pred_path)
    tgt = np.load(tgt_path)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    labels = ["light", "ax31", "k1"]
    # calibration
    ax = axes[0]
    for j, lab in enumerate(labels):
        p = pred["score"][:, j]
        y = tgt["scores"][:, j]
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(p, bins) - 1
        xs, ys = [], []
        for b in range(10):
            mask = idx == b
            if mask.sum() >= 10:
                xs.append(p[mask].mean())
                ys.append(y[mask].mean())
        ax.plot(xs, ys, "o-", label=lab)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_title(f"calibration ({r['exp_id']} {name})")
    ax.set_xlabel("예측 score")
    ax.set_ylabel("실제 score")
    ax.legend(fontsize=7)
    # confusion of best-model choice
    ax = axes[1]
    true_best = tgt["scores"].argmax(axis=1)
    pred_best = pred["score"].argmax(axis=1)
    mat = np.zeros((3, 3))
    for a, b in zip(true_best, pred_best):
        mat[a, b] += 1
    im = ax.imshow(mat, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("argmax 예측")
    ax.set_ylabel("argmax 실제")
    ax.set_title("best-model confusion (dev)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    # k1 cost scatter
    ax = axes[2]
    rate_out = 26.260 / 1e6
    real_cost = tgt["costs"][:, 2]
    pred_cost = pred["cost"][:, 2]
    ax.loglog(real_cost, pred_cost, ".", ms=3, alpha=0.5)
    lims = [min(real_cost.min(), pred_cost.min()), max(real_cost.max(), pred_cost.max())]
    ax.plot(lims, lims, "k:", lw=1)
    ax.set_xlabel("실제 k1 cost")
    ax.set_ylabel("예측 k1 cost")
    ax.set_title("k1 비용 예측 (log-log)")
    save(fig, "pred_quality", f"5. 예측 품질 — {r['exp_id']} {name}")


def chart_domain(rows):
    r = _best_pred_row(rows)
    if r is None:
        return
    detail_path = ROOT / "exp" / r["exp_id"] / "detail.npz"
    if not detail_path.exists():
        return
    from eda import categorize
    from harness import load_split

    table = load_split("dev")
    cats = np.asarray([categorize(t) for t in table["texts"]])
    scores = np.asarray(table["scores"])
    picks = np.load(detail_path)["picks_dev"]
    unique = sorted(set(cats.tolist()))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, tier, pick in zip(axes, TIERS, picks):
        n = len(pick)
        gain = scores[np.arange(n), pick] - scores[:, 0]
        vals = [gain[cats == c].sum() for c in unique]
        base = [scores[cats == c, 0].sum() for c in unique]
        ax.bar(range(len(unique)), base, color="#cccccc", label="light 기본점수")
        ax.bar(range(len(unique)), vals, bottom=base, color="#3b7dd8", label="라우팅 이득")
        ax.set_xticks(range(len(unique)), unique, rotation=45, ha="right", fontsize=6)
        ax.set_title(tier)
    axes[0].legend(fontsize=7)
    save(fig, "domain", f"6. 도메인 분해 — {r['exp_id']} 점수 이득 기여")


def chart_timeline(rows):
    prog = [r for r in rows if r["family"] != "reference"]
    if not prog:
        return
    finals = [float(r["metrics"]["weighted_final"]) for r in prog]
    best = np.maximum.accumulate(finals)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(range(len(prog)), finals, "o", ms=4, alpha=0.5, label="실험")
    ax.plot(range(len(prog)), best, "-", color="green", label="최고 갱신")
    ax.axhline(0.6954, color="orange", ls=":", label="hash-regex")
    ax.axhline(ORACLE["final"], color="red", ls="--", label="budget-oracle")
    ax.set_xticks(range(len(prog)), [r["exp_id"] for r in prog], rotation=90, fontsize=6)
    ax.set_ylabel("가중 최종점수")
    ax.legend(fontsize=7)
    save(fig, "timeline", "7. 진행 타임라인")


def chart_runtime(rows):
    pts = []
    for r in rows:
        rt = r.get("runtime") or {}
        per = rt.get("per_tier_sec")
        if per is None:
            continue
        if isinstance(per, dict):
            per = sum(float(v) for v in per.values()) / len(per)
        pts.append((float(per), float(r["metrics"]["weighted_final"]), r["exp_id"], rt.get("measured_env", "host")))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for per, final, eid, env in pts:
        color = "#3b7dd8" if env.startswith("docker") else "#999999"
        ax.scatter(per, final, color=color, s=25)
        ax.annotate(f"{eid}", (per, final), fontsize=6)
    ax.axvline(90, color="red", ls="--", label="90초 한도")
    ax.set_xlabel("등급당 추론시간 (s) — 회색=host, 파랑=docker-arm64")
    ax.set_ylabel("가중 최종점수")
    ax.legend(fontsize=7)
    save(fig, "runtime", "8. 런타임 vs 점수")


def main() -> None:
    rows = registry_lib.load_all()
    if not rows:
        print("registry 비어 있음")
        return
    chart_leaderboard(rows)
    chart_pareto(rows)
    chart_risk(rows)
    chart_sweep(rows)
    chart_pred_quality(rows)
    chart_domain(rows)
    chart_timeline(rows)
    chart_runtime(rows)
    best = max(
        (r for r in rows if r["family"] != "reference"),
        key=lambda r: float(r["metrics"]["weighted_final"]),
        default=None,
    )
    head = (
        f"<p>실험 수: {len(rows)} · 현재 1위: "
        f"{best['exp_id'] if best else '-'} "
        f"{best['config'].get('name') if best else ''} "
        f"<b>{float(best['metrics']['weighted_final']):.4f}</b>"
        f" (baseline 0.6954 / oracle {ORACLE['final']})</p>"
        if best
        else ""
    )
    html = [
        "<!doctype html><meta charset='utf-8'><title>OSSP Router 실험 대시보드</title>",
        "<style>body{font-family:sans-serif;max-width:1200px;margin:20px auto;padding:0 16px;"
        "background:#fafafa;color:#222}@media(prefers-color-scheme:dark){body{background:#1e1e1e;color:#ddd}}"
        "img{max-width:100%;background:white;border:1px solid #ccc;border-radius:6px;padding:4px}"
        "h2{border-bottom:1px solid #ccc;padding-bottom:4px}</style>",
        "<h1>OSSP 2026 LLM Router — 실험 대시보드</h1>",
        head,
    ]
    for title, b64 in CHARTS:
        html.append(f"<h2>{title}</h2><img src='data:image/png;base64,{b64}'>")
    (ROOT / "exp" / "dashboard.html").write_text("\n".join(html), encoding="utf-8")
    print(f"dashboard.html written with {len(CHARTS)} charts; PNGs in exp/figs/")


if __name__ == "__main__":
    main()
