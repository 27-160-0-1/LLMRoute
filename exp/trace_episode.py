# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""단일 episode에 대한 라우터 결정 추적기 (white-box tracer).

한 episode_id를 받아 다음을 순서대로 출력한다.

  [0] 실행 컨텍스트 (번들 / 조회표 항목 수 / 정책 상수)
  [1] 프롬프트 요약
  [2] 주요 특징값 상위 10개 — 분포 대비 |z| 순 + 모델 gain 순
  [3] xgb-mono 점수 3개 · 토큰 헤드 · 비용 예측 (mean / q90)
  [4] 등급별 결정과 그 이유 (이득/비용 비율, 라그랑주 승수)
  [5] 실제 outcome (모델별 실제 score와 실제 비용)
  [6] 판정 — 옳은 선택이었나 / 손실은 얼마인가

점수 예측은 **xgb-mono 단독**이다 (models/final-v1/policy.json의
``blend_members = ["xgb-mono"]``). 이전 구성이던 4멤버 블렌드
(xgb-mono + irt1d + knn-k40 + lgbm)는 공개 Dev에서 기여가
+0.00159 (95% CI [-0.00483, +0.00830], p(<=0)=0.31)로 0과 구별되지
않아 폐기했다. 근거는 exp/audit/A7-overfit-falsification.md.
비용 예측만 2헤드(lgbm + xgb) 로그공간 평균을 유지하고, premium 등급의
k1 상한에는 lgbm q90 분위수 헤드를 쓴다.

기본 번들은 ``build/bundle-nolookup``이다. 정확 매칭 조회표가 켜져 있으면
공개 Train/Dev 문항의 예측이 실현값으로 덮어써져 "예측 능력"이 아니라
"암기"를 보게 되기 때문이다. 조회표 ON 동작을 보려면
``--bundle models/final-v1``을 명시한다.

산술 경로는 ``src/ossp_router/final_router.py``의
``FinalRouter.predict_batch``(:98-157)를 그대로 옮긴 것이며,
``--verify``(기본 ON)로 원본 함수와 비트 단위 일치를 매 실행마다 확인한다.

사용 예::

    $env:PYTHONPATH='src'
    .\\.venv\\Scripts\\python.exe exp\\trace_episode.py --episode dev-0042
    .\\.venv\\Scripts\\python.exe exp\\trace_episode.py --rank regret-fast --top 15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ossp_router import allocation, features_v2
from ossp_router.final_router import MODEL_IDS, FinalRouter
from ossp_router.heuristic import episode_text
from ossp_router.protocol import load_input

REPO = Path(__file__).resolve().parents[1]

# 등급별 예산 배수 (src/ossp_router/resources/routing-policy.v1.json:31-42)
TIER_MULTIPLIER = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
TIER_WEIGHT = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}

# dense_features()가 채우는 46칸 중 앞 22칸의 이름
# (src/ossp_router/features_v2.py:97-121, 나머지 24칸은 항상 0.0인 예약 슬롯)
DENSE_NAMES: List[str] = [
    "log1p(chars)",
    "log1p(words)",
    "log1p(sentences)",
    "log1p(message_count)",
    "log1p(lines)",
    "log1p(indent_lines)",
    "hangul_ratio",
    "latin_ratio",
    "digit_ratio",
    "upper_ratio",
    "numeric_word_ratio",
    "avg_word_len",
    "chars>=2000",
    "chars>=8000",
    "chars>=32000",
    "chars/70000",
    "log1p(chars)^2/100",
    "messages>=2",
    "messages>=3",
    "words_per_sentence",
    "starts_with_digit",
    "starts_with_task_verb",
]
DENSE_NAMES += [f"reserved-{i}" for i in range(len(DENSE_NAMES), 46)]
DENSE_NAMES += [f"re:{key}" for key in sorted(features_v2._RE)]
assert len(DENSE_NAMES) == features_v2.DENSE_DIM


# --------------------------------------------------------------------------
# 특징 -> 모델 gain 매핑
# --------------------------------------------------------------------------


def gain_table(router: FinalRouter) -> Dict:
    """xgb-mono score 헤드의 total_gain을 원본 특징 공간으로 되돌린다.

    학습 시 열 순서는 [dense 81 | word-hash 65536 | char-hash 65536]이고
    (final_router.py:115), ``keep-cols.npy``가 그중 살아남은 열의 원본
    인덱스를 담는다 (final_router.py:116). get_score()의 'fN'은 keep 이후의
    위치이므로 keep[N]으로 되돌려야 원본 특징을 가리킨다.
    """

    raw = router.xgb_score.get_score(importance_type="total_gain")
    keep = router.xgb_keep
    d = features_v2.DENSE_DIM
    dense_gain = np.zeros(d, dtype=np.float64)
    word_gain = 0.0
    char_gain = 0.0
    for key, val in raw.items():
        orig = int(keep[int(key[1:])])
        if orig < d:
            dense_gain[orig] += val
        elif orig < d + features_v2.WORD_BINS:
            word_gain += val
        else:
            char_gain += val
    total = float(dense_gain.sum() + word_gain + char_gain)
    return {
        "dense_gain": dense_gain,
        "word_gain": float(word_gain),
        "char_gain": float(char_gain),
        "total_gain": total,
        "n_split_feats": len(raw),
        "n_kept_cols": int(keep.size),
    }


# --------------------------------------------------------------------------
# 예측 — final_router.FinalRouter.predict_batch(:98-157)의 중간값 노출판
# --------------------------------------------------------------------------


def predict_verbose(router: FinalRouter, texts: Sequence[str], counts: Sequence[int]) -> Dict:
    """predict_batch와 동일한 연산을 수행하되 모든 중간값을 반환한다.

    점수는 xgb-mono 단독이므로 blend_score == xgb_score이다. 다만
    final_router.py:133의 ``blend_score = xgb_score``는 복사가 아니라
    별칭이고 조회표 적중 시 제자리 수정되므로, 여기서는 조회표 적용 전
    값을 반드시 따로 복사해 둔다 (pre_lookup_score).
    """

    import xgboost as xgb
    from scipy import sparse

    dense, word, char = features_v2.featurize_batch(texts, counts)
    x_svd = np.hstack([dense, word @ router.svd_word.T, char @ router.svd_char.T])

    # LightGBM은 비용 전용이다. score-* 부스터는 번들에서 제거됐다.
    lgbm_lin = np.column_stack([router.lgbm[f"lin-{j}"].predict(x_svd) for j in range(3)])
    lgbm_lout = np.column_stack([router.lgbm[f"lout-{j}"].predict(x_svd) for j in range(3)])
    lgbm_q90 = np.column_stack([router.lgbm[f"q90-{j}"].predict(x_svd) for j in range(3)])
    lgbm_cost = router._monotone(
        np.exp(lgbm_lin) * router.rates[:, 0] / 1e6 + np.exp(lgbm_lout) * router.rates[:, 1] / 1e6
    )
    lgbm_q90_cost = router._monotone(
        np.exp(lgbm_lin) * router.rates[:, 0] / 1e6 + np.exp(lgbm_q90) * router.rates[:, 1] / 1e6
    )

    x_sp = sparse.hstack([sparse.csr_matrix(dense), word, char], format="csr").astype(np.float32)
    x_sp = x_sp[:, router.xgb_keep].tocsr()
    dmat = xgb.DMatrix(x_sp)
    xgb_score = np.clip(router.xgb_score.predict(dmat), 0.0, 1.0)
    xgb_lin = np.column_stack([router.xgb_tokens[f"lin-{j}"].predict(dmat) for j in range(3)])
    xgb_lout = np.column_stack([router.xgb_tokens[f"lout-{j}"].predict(dmat) for j in range(3)])
    xgb_cost = router._monotone(
        np.exp(xgb_lin) * router.rates[:, 0] / 1e6 + np.exp(xgb_lout) * router.rates[:, 1] / 1e6
    )

    # 점수 헤드는 xgb-mono 하나뿐이다 (final_router.py:129-133).
    blend_score = xgb_score.copy()
    cost_mean = router._monotone(np.exp((np.log(lgbm_cost) + np.log(xgb_cost)) / 2.0))
    cost_q90 = cost_mean.copy()
    cost_q90[:, 2] = lgbm_q90_cost[:, 2]
    cost_q90 = router._monotone(cost_q90)

    if len(router.lookup_key):
        keys = np.frombuffer(
            b"".join(hashlib.sha256(t.encode("utf-8")).digest() for t in texts), dtype=np.uint8
        ).reshape(-1, 32)
        keys_v = np.ascontiguousarray(keys).view("V32").ravel()
        pos = np.searchsorted(router.lookup_key, keys_v)
        pos = np.minimum(pos, len(router.lookup_key) - 1)
        hit = router.lookup_key[pos] == keys_v
    else:
        pos = np.zeros(len(texts), dtype=np.int64)
        hit = np.zeros(len(texts), dtype=bool)
    pre_lookup_score = blend_score.copy()
    if hit.any():
        blend_score[hit] = router.lookup_scores[pos[hit]]
        cost_mean[hit] = router.lookup_costs[pos[hit]]
        cost_q90[hit] = router.lookup_costs[pos[hit]]
        cost_mean = router._monotone(cost_mean)
        cost_q90 = router._monotone(cost_q90)

    return {
        "dense": dense,
        "xgb_score": xgb_score,
        "lgbm_in": np.exp(lgbm_lin),
        "lgbm_out": np.exp(lgbm_lout),
        "lgbm_out_q90": np.exp(lgbm_q90),
        "xgb_in": np.exp(xgb_lin),
        "xgb_out": np.exp(xgb_lout),
        "lgbm_cost": lgbm_cost,
        "xgb_cost": xgb_cost,
        "lgbm_q90_cost": lgbm_q90_cost,
        "pre_lookup_score": pre_lookup_score,
        "lookup_hit": hit,
        "blend_score": blend_score,
        "cost_mean": cost_mean,
        "cost_q90": cost_q90,
    }


# --------------------------------------------------------------------------
# 배분기 추적 — allocation.py의 계기판 부착판
# --------------------------------------------------------------------------


def trace_greedy(
    score_pred: np.ndarray,
    cost_pred: np.ndarray,
    *,
    multiplier: float,
    utilization: float,
    allow_k1: bool,
    target: int,
) -> Dict:
    """allocation.greedy_allocate(:60-101)를 그대로 재현하며 경계를 기록한다."""

    n = score_pred.shape[0]
    pick = np.zeros(n, dtype=int)
    light_total = cost_pred[:, 0].sum()
    cap = light_total * max(1.0, multiplier * utilization)
    total = cost_pred[np.arange(n), pick].sum()
    models = (0, 1, 2) if allow_k1 else (0, 1)

    groups: Dict[Tuple, List[Tuple[int, int]]] = {}
    for i in range(n):
        for j in models:
            if j <= pick[i]:
                continue
            ds = score_pred[i, j] - score_pred[i, pick[i]]
            dc = cost_pred[i, j] - cost_pred[i, pick[i]]
            if ds <= 1e-12:
                continue
            ratio = ds / max(dc, 1e-12)
            key = (round(ratio, 12), round(ds, 12), round(dc, 12), j)
            groups.setdefault(key, []).append((i, j))
    ordered = sorted(groups.items(), key=lambda kv: (-kv[0][0], kv[0][2], kv[0][1], kv[0][3]))

    target_events: List[Dict] = []
    last_accept: Optional[Dict] = None
    first_reject: Optional[Dict] = None
    n_accept = 0
    for rank, (key, members) in enumerate(ordered):
        rows = [i for i, _ in members if pick[i] < members[0][1]]
        if not rows:
            continue
        j = members[0][1]
        dc_total = sum(cost_pred[i, j] - cost_pred[i, pick[i]] for i, _ in members)
        accepted = total + dc_total <= cap
        event = {
            "rank": rank,
            "ratio": key[0],
            "ds": key[1],
            "dc": key[2],
            "to_model": j,
            "group_size": len(members),
            "dc_total": float(dc_total),
            "total_before": float(total),
            "cap": float(cap),
            "accepted": bool(accepted),
        }
        if accepted:
            for i, _ in members:
                pick[i] = j
            total += dc_total
            n_accept += 1
            last_accept = event
        elif first_reject is None:
            first_reject = event
        if any(i == target for i, _ in members):
            target_events.append(event)

    return {
        "pick": pick,
        "light_total": float(light_total),
        "cap": float(cap),
        "cap_formula": f"light_total x max(1, {multiplier} x {utilization})",
        "total_spent": float(total),
        "n_groups": len(ordered),
        "n_accepted_groups": n_accept,
        "target_events": target_events,
        "last_accept": last_accept,
        "first_reject": first_reject,
    }


def trace_lagrangian(
    score_pred: np.ndarray,
    cost_pred: np.ndarray,
    *,
    multiplier: float,
    utilization: float,
) -> Dict:
    """allocation.lagrangian_allocate(:18-57)를 재현하고 최종 lambda*를 기록한다."""

    n = score_pred.shape[0]
    models = (0, 1, 2)
    light_total = cost_pred[:, 0].sum()
    cap = light_total * max(1.0, multiplier * utilization)

    def choose(lam: float) -> np.ndarray:
        best = np.zeros(n, dtype=int)
        best_util = score_pred[:, 0] - lam * cost_pred[:, 0] / light_total * n
        for j in models[1:]:
            util = score_pred[:, j] - lam * cost_pred[:, j] / light_total * n
            better = util > best_util + 1e-15
            best = np.where(better, j, best)
            best_util = np.where(better, util, best_util)
        return best

    pick = choose(0.0)
    if cost_pred[np.arange(n), pick].sum() <= cap:
        return {
            "pick": pick,
            "lam": 0.0,
            "cap": float(cap),
            "unconstrained": True,
            "total": float(cost_pred[np.arange(n), pick].sum()),
            "choose": choose,
            "n": n,
            "light_total": float(light_total),
        }
    low, high = 0.0, 1.0
    while cost_pred[np.arange(n), choose(high)].sum() > cap and high < 2**60:
        low, high = high, high * 2.0
    pick = choose(high)
    lam = high
    for _ in range(100):
        mid = (low + high) / 2.0
        cand = choose(mid)
        if cost_pred[np.arange(n), cand].sum() <= cap:
            high, pick, lam = mid, cand, mid
        else:
            low = mid
    if cost_pred[np.arange(n), pick].sum() > cap:
        pick = np.zeros(n, dtype=int)
    return {
        "pick": pick,
        "lam": float(lam),
        "cap": float(cap),
        "unconstrained": False,
        "total": float(cost_pred[np.arange(n), pick].sum()),
        "choose": choose,
        "n": n,
        "light_total": float(light_total),
    }


def trace_two_stage(
    score_pred: np.ndarray,
    cost_pred: np.ndarray,
    *,
    multiplier: float,
    k1_utilization: float,
    fill_utilization: float,
    k1_cost_cap: float,
    target: int,
) -> Dict:
    """allocation.two_stage_premium(:104-144) 재현 + 2단계 근거 기록."""

    masked_rows = cost_pred[:, 2] > k1_cost_cap
    masked = score_pred.copy()
    masked[masked_rows, 2] = 0.0
    stage1 = trace_lagrangian(masked, cost_pred, multiplier=multiplier, utilization=k1_utilization)
    pick = stage1["pick"].copy()

    n = masked.shape[0]
    light_total = cost_pred[:, 0].sum()
    cap = light_total * max(1.0, multiplier * fill_utilization)
    total = cost_pred[np.arange(n), pick].sum()
    groups: Dict[Tuple, List[int]] = {}
    for i in range(n):
        if pick[i] != 0:
            continue
        ds = masked[i, 1] - masked[i, 0]
        dc = cost_pred[i, 1] - cost_pred[i, 0]
        if ds <= 1e-12:
            continue
        key = (round(ds / max(dc, 1e-12), 12), round(dc, 12), round(ds, 12))
        groups.setdefault(key, []).append(i)

    target_event = None
    last_accept = None
    first_reject = None
    for key, rows in sorted(groups.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        dc_total = sum(cost_pred[i, 1] - cost_pred[i, 0] for i in rows)
        accepted = total + dc_total <= cap
        event = {
            "ratio": key[0],
            "ds": key[2],
            "dc": key[1],
            "group_size": len(rows),
            "dc_total": float(dc_total),
            "total_before": float(total),
            "cap": float(cap),
            "accepted": bool(accepted),
        }
        if accepted:
            for i in rows:
                pick[i] = 1
            total += dc_total
            last_accept = event
        elif first_reject is None:
            first_reject = event
        if target in rows:
            target_event = event

    return {
        "pick": pick,
        "stage1": stage1,
        "stage1_pick_target": int(stage1["pick"][target]),
        "k1_masked": bool(masked_rows[target]),
        "n_k1_masked": int(masked_rows.sum()),
        "fill_cap": float(cap),
        "fill_total": float(total),
        "fill_target_event": target_event,
        "fill_last_accept": last_accept,
        "fill_first_reject": first_reject,
        "masked_scores": masked,
    }


# --------------------------------------------------------------------------
# 데이터 로딩
# --------------------------------------------------------------------------


def load_outcomes_map(path: Path) -> Dict[str, Dict[str, Dict]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {e["episode_id"]: e["models"] for e in raw["episodes"]}


def actual_cost(rec: Dict, rates: np.ndarray, j: int) -> float:
    return rec["input_tokens"] * rates[j, 0] / 1e6 + rec["output_tokens"] * rates[j, 1] / 1e6


# --------------------------------------------------------------------------
# 출력 헬퍼
# --------------------------------------------------------------------------


def rule(title: str = "", char: str = "=") -> str:
    if not title:
        return char * 78
    return f"{char * 3} {title} " + char * max(3, 74 - len(title))


def summarize_prompt(text: str, limit: int, mask: bool) -> List[str]:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    lines = [
        f"길이               : {len(text):,} chars, {len(text.splitlines())} lines",
        f"sha256             : {digest}",
    ]
    if mask:
        head = text[:120].replace("\n", "\\n")
        lines.append(f"본문(마스킹)       : {head!r} ... [+{len(text) - 120} chars 생략]")
    else:
        body = text[:limit]
        lines.append("본문:")
        for ln in body.splitlines()[:20]:
            lines.append(f"  | {ln}")
        if len(text) > limit:
            lines.append(f"  | ... [+{len(text) - limit} chars 생략]")
    return lines


# --------------------------------------------------------------------------
# 메인 추적
# --------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    inputs = load_input(args.input)
    episodes = list(inputs.episodes)
    ids = [e.episode_id for e in episodes]
    texts = [episode_text(e) for e in episodes]
    counts = [1 if e.prompt is not None else len(e.messages or ()) for e in episodes]

    router = FinalRouter(args.bundle)
    rates = router.rates
    P = predict_verbose(router, texts, counts)
    t_pred = time.time() - t0

    if args.verify:
        ref_s, ref_cm, ref_cq = router.predict_batch(texts, counts)
        same = (
            np.array_equal(ref_s, P["blend_score"])
            and np.array_equal(ref_cm, P["cost_mean"])
            and np.array_equal(ref_cq, P["cost_q90"])
        )
        if not same:
            print("치명적: 추적기 산술이 FinalRouter.predict_batch와 다릅니다.", file=sys.stderr)
            return 3
    verify_note = "OK (bit-exact)" if args.verify else "생략됨 (--no-verify)"

    outcomes = load_outcomes_map(args.outcomes) if args.outcomes else {}

    score = P["blend_score"]
    cost_mean = P["cost_mean"]
    cost_q90 = P["cost_q90"]

    # 세 등급 배분을 한 번씩 실행 (배분은 배치 전역 연산이다)
    picks: Dict[str, np.ndarray] = {}
    for tier in ("fast", "balanced", "premium"):
        picks[tier] = router.allocate(tier, score, cost_mean, cost_q90)

    if args.rank:
        return do_rank(args, ids, texts, P, picks, outcomes, rates)

    if args.episode not in ids:
        print(f"오류: episode_id '{args.episode}'를 {args.input}에서 찾을 수 없습니다.", file=sys.stderr)
        return 2
    i = ids.index(args.episode)

    out = sys.stdout
    w = out.write
    w(rule() + "\n")
    w(f" OSSP-2026 라우터 결정 추적 — episode {args.episode}\n")
    w(rule() + "\n\n")

    # ---- [0] 실행 컨텍스트 -------------------------------------------------
    w(rule("[0] 실행 컨텍스트") + "\n")
    w(f"입력               : {args.input} ({len(ids)} episodes, split={inputs.split})\n")
    w(f"모델 번들          : {router.bundle}\n")
    w(f"조회표 항목 수     : {len(router.lookup_key)} "
      f"({'ON — 예측이 실현값으로 덮어써짐' if len(router.lookup_key) else 'OFF — 순수 예측'})\n")
    w(f"이 문항 조회표 적중: {bool(P['lookup_hit'][i])}\n")
    w(f"predict_batch 대조 : {verify_note}\n")
    w("단가표 rates       : "
      + " ".join(
          f"{mid}=({rates[j, 0]:g},{rates[j, 1]:g})" for j, mid in enumerate(MODEL_IDS)
      )
      + "  (input,output credits per 1e6 tokens)\n")
    w(f"특징 버전          : v{router.policy_spec['feature_version']}, "
      f"dense {features_v2.DENSE_DIM}차원 + word-SVD 128 + char-SVD 128 = "
      f"{features_v2.DENSE_DIM + 256}차원\n")
    w(f"예측 소요          : {t_pred:.1f}s (배치 {len(ids)}건 일괄)\n\n")

    # ---- [1] 프롬프트 요약 -------------------------------------------------
    w(rule("[1] 프롬프트 요약") + "\n")
    ep = episodes[i]
    w(f"episode_id         : {ep.episode_id}\n")
    w(f"메시지 수          : {counts[i]} ({'prompt 단일' if ep.prompt is not None else 'messages'})\n")
    for ln in summarize_prompt(texts[i], args.prompt_chars, args.mask):
        w(ln + "\n")
    w("\n")

    # ---- [2] 특징값 상위 10 ------------------------------------------------
    dense = P["dense"]
    mu = dense.mean(axis=0)
    sd = dense.std(axis=0)
    sd_safe = np.where(sd > 1e-12, sd, np.nan)
    z = (dense[i] - mu) / sd_safe
    z = np.nan_to_num(z, nan=0.0)
    order = np.argsort(-np.abs(z))
    w(rule("[2] 추출된 주요 특징값 상위 10개") + "\n")
    w(f"기준: split 전체 {len(ids)}건 분포 대비 |z| 내림차순 "
      "(dense 81차원 중, 값이 아니라 '얼마나 이례적인가'로 정렬)\n")
    w(f"{'#':>2}  {'feature':<28} {'value':>12} {'split mean':>12} {'z':>8}\n")
    shown = 0
    for k in order:
        if shown >= 10:
            break
        if sd[k] <= 1e-12:
            continue
        w(f"{shown + 1:>2}  {DENSE_NAMES[k]:<28} {dense[i, k]:>12.4f} {mu[k]:>12.4f} {z[k]:>+8.2f}\n")
        shown += 1
    w("\n주: re:* 는 정규식 적중 횟수의 log1p 값이다 "
      "(src/ossp_router/features_v2.py:122-123). 예: 1.0986 = log1p(2) = 2회 적중.\n\n")

    # ---- [3] 예측 ----------------------------------------------------------
    w(rule("[3] 예측") + "\n")
    w("멤버별 정답 확률 예측 (4개 평균이 blend, final_router.py:170-172)\n")
    w(f"{'model':<12} {'xgb-mono':>9} {'irt1d':>9} {'knn-k40':>9} {'lgbm':>9} {'blend':>9}\n")
    for j, mid in enumerate(MODEL_IDS):
        w(f"{mid:<12} {P['xgb_score'][i, j]:>9.4f} {P['irt_score'][i, j]:>9.4f} "
          f"{P['knn_score'][i, j]:>9.4f} {P['lgbm_score'][i, j]:>9.4f} "
          f"{P['pre_lookup_score'][i, j]:>9.4f}\n")
    w(f"\nirt1d 잠재변수 theta = {P['irt_theta'][i]:+.4f} "
      "(1차원 난이도 축; 높을수록 세 모델 모두 정답 확률 상승)\n")
    w(f"kNN 최근접 이웃 유사도 top5 = "
      + ", ".join(f"{s:.3f}" for s in P["knn_sim"][i, :5])
      + f"  (k=40, 가중치 = sim^3)\n")
    if len(router.lookup_key) and P["lookup_hit"][i]:
        w("\n조회표 적중 -> 위 blend 값이 아래 실현값으로 대체됨:\n")
        w("  " + "  ".join(f"{mid}={score[i, j]:.4f}" for j, mid in enumerate(MODEL_IDS)) + "\n")
    w("\n토큰 헤드 예측 (생성 전체 합계 기준)\n")
    w(f"{'model':<12} {'in(lgbm)':>10} {'in(xgb)':>10} {'out(lgbm)':>10} {'out(xgb)':>10} {'out q90':>10}\n")
    for j, mid in enumerate(MODEL_IDS):
        w(f"{mid:<12} {P['lgbm_in'][i, j]:>10.0f} {P['xgb_in'][i, j]:>10.0f} "
          f"{P['lgbm_out'][i, j]:>10.0f} {P['xgb_out'][i, j]:>10.0f} {P['lgbm_out_q90'][i, j]:>10.0f}\n")
    w("\n비용 예측 (credits)\n")
    w(f"{'model':<12} {'lgbm':>12} {'xgb':>12} {'cost_mean':>12} {'cost_q90':>12}\n")
    for j, mid in enumerate(MODEL_IDS):
        w(f"{mid:<12} {P['lgbm_cost'][i, j]:>12.6f} {P['xgb_cost'][i, j]:>12.6f} "
          f"{cost_mean[i, j]:>12.6f} {cost_q90[i, j]:>12.6f}\n")
    w("cost_mean = exp(mean(log lgbm_cost, log xgb_cost)) 후 단조 보정 "
      "(final_router.py:173-176). cost_q90의 k1 열만 lgbm q90 출력토큰으로 대체.\n\n")

    # ---- [4] 등급별 결정 ---------------------------------------------------
    w(rule("[4] 등급별 결정과 그 이유") + "\n")
    for tier in ("fast", "balanced", "premium"):
        spec = router.policy_spec[tier]
        mult = TIER_MULTIPLIER[tier]
        w(f"\n--- {tier.upper()} (budget_multiplier={mult}, weight={TIER_WEIGHT[tier]}) ---\n")
        if tier in ("fast", "balanced"):
            tr = trace_greedy(
                score[:, :], cost_mean, multiplier=mult,
                utilization=spec["utilization"], allow_k1=spec["allow_k1"], target=i,
            )
            assert np.array_equal(tr["pick"], picks[tier]), "greedy 추적이 본 배분기와 불일치"
            w(f"배분기             : greedy, allow_k1={spec['allow_k1']}, "
              f"utilization={spec['utilization']}\n")
            w(f"all-light 기준비용 : {tr['light_total']:.6f} credits\n")
            w(f"내부 지출 상한 cap : {tr['cap']:.6f} = {tr['cap_formula']}\n")
            w(f"실제 지출          : {tr['total_spent']:.6f} "
              f"({tr['total_spent'] / tr['cap'] * 100:.1f}% of cap, "
              f"{tr['total_spent'] / tr['light_total']:.4f}x all-light)\n")
            w(f"승급 그룹          : {tr['n_groups']}개 중 {tr['n_accepted_groups']}개 채택\n")
            ds = score[i, 1] - score[i, 0]
            dc = cost_mean[i, 1] - cost_mean[i, 0]
            w(f"이 문항의 후보     : light -> ax31, dScore={ds:+.4f}, dCost={dc:+.6f}, "
              f"ratio={ds / max(dc, 1e-12):.4f} (점수/credit)\n")
            if tr["last_accept"]:
                w(f"채택 경계 ratio    : 마지막 채택 {tr['last_accept']['ratio']:.4f}")
                if tr["first_reject"]:
                    w(f" / 첫 탈락 {tr['first_reject']['ratio']:.4f}")
                w("\n")
            if tr["target_events"]:
                ev = tr["target_events"][0]
                w(f"이 문항의 판정     : ratio 순위 {ev['rank'] + 1}/{tr['n_groups']}, "
                  f"동률 그룹 크기 {ev['group_size']}, "
                  f"{'채택' if ev['accepted'] else '탈락'}\n")
                if not ev["accepted"]:
                    w(f"  탈락 사유        : 잔여 예산 {ev['cap'] - ev['total_before']:.6f} < "
                      f"그룹 소요 {ev['dc_total']:.6f}\n")
            else:
                w("이 문항의 판정     : 승급 후보 없음 "
                  "(dScore <= 0 이므로 그룹 생성 자체가 안 됨) -> light 유지\n")
            w(f"=> 선택: {MODEL_IDS[picks[tier][i]]}\n")
        else:
            tr = trace_two_stage(
                score, cost_q90, multiplier=mult,
                k1_utilization=spec["k1_utilization"],
                fill_utilization=spec["fill_utilization"],
                k1_cost_cap=spec["k1_cost_cap"], target=i,
            )
            assert np.array_equal(tr["pick"], picks[tier]), "two-stage 추적이 본 배분기와 불일치"
            s1 = tr["stage1"]
            w(f"배분기             : two-stage (라그랑주 k1 -> greedy fill)\n")
            w(f"k1_cost_cap        : {spec['k1_cost_cap']} credits (cost_q90 기준), "
              f"배치에서 {tr['n_k1_masked']}건 차단\n")
            w(f"  이 문항 cost_q90[k1] = {cost_q90[i, 2]:.6f} -> "
              f"{'차단됨 (k1 점수를 0으로 마스킹)' if tr['k1_masked'] else '통과'}\n")
            w(f"1단계 cap          : {s1['cap']:.6f} "
              f"= light_total x max(1, {mult} x {spec['k1_utilization']})\n")
            w(f"1단계 lambda*      : {s1['lam']:.6g}"
              f"{' (예산 여유, 무제약 argmax)' if s1['unconstrained'] else ' (이분탐색 100회)'}\n")
            util = score[:, :] if not tr["k1_masked"] else tr["masked_scores"]
            lam = s1["lam"]
            n = len(ids)
            w("  이 문항의 라그랑주 효용 = score - lambda* x cost / light_total x N:\n")
            best_u = None
            for j, mid in enumerate(MODEL_IDS):
                u = tr["masked_scores"][i, j] - lam * cost_q90[i, j] / s1["light_total"] * n
                if j != 2 or not tr["k1_masked"]:
                    best_u = u if best_u is None else max(best_u, u)
                w(f"    {mid:<12} score={tr['masked_scores'][i, j]:.4f} "
                  f"cost={cost_q90[i, j]:.6f} -> utility={u:+.4f}"
                  f"{'   <== 최대' if j == tr['stage1_pick_target'] else ''}\n")
            if tr["k1_masked"]:
                # 반사실: k1_cost_cap이 없었다면 1단계에서 k1이 이겼을까?
                u_k1 = score[i, 2] - lam * cost_q90[i, 2] / s1["light_total"] * n
                w(f"  반사실 (cap 제거 시): axk1-think score={score[i, 2]:.4f} -> "
                  f"utility={u_k1:+.4f} vs 차선 {best_u:+.4f} -> "
                  f"{'그래도 탈락 (cap은 결정적이지 않았음)' if u_k1 <= best_u else 'k1 승리 (cap이 결정적이었음)'}\n")
            w(f"1단계 결과         : {MODEL_IDS[tr['stage1_pick_target']]} "
              f"(배치 지출 {s1['total']:.6f} / cap {s1['cap']:.6f})\n")
            w(f"2단계 fill cap     : {tr['fill_cap']:.6f} "
              f"= light_total x max(1, {mult} x {spec['fill_utilization']})\n")
            w(f"2단계 지출         : {tr['fill_total']:.6f} "
              f"({tr['fill_total'] / tr['fill_cap'] * 100:.1f}% of cap)\n")
            if tr["fill_target_event"]:
                ev = tr["fill_target_event"]
                w(f"2단계 이 문항      : light->ax31 ratio={ev['ratio']:.4f}, "
                  f"동률 그룹 {ev['group_size']}건, {'채택' if ev['accepted'] else '탈락'}\n")
            elif tr["stage1_pick_target"] == 0:
                w("2단계 이 문항      : 승급 후보 없음 (dScore <= 0) -> light 유지\n")
            else:
                w("2단계 이 문항      : 1단계에서 이미 승급됨 -> fill 대상 아님\n")
            if tr["fill_last_accept"]:
                w(f"2단계 경계 ratio   : 마지막 채택 {tr['fill_last_accept']['ratio']:.4f}")
                if tr["fill_first_reject"]:
                    w(f" / 첫 탈락 {tr['fill_first_reject']['ratio']:.4f}")
                w("\n")
            w(f"=> 선택: {MODEL_IDS[picks[tier][i]]}\n")
    w("\n")

    # ---- [5] 실제 outcome --------------------------------------------------
    if args.episode not in outcomes:
        w(rule("[5] 실제 outcome") + "\n(outcomes 미제공 — --outcomes 로 지정)\n\n")
        return 0
    rec = outcomes[args.episode]
    w(rule("[5] 실제 outcome (data/*/outcomes.json)") + "\n")
    w(f"{'model':<12} {'actual':>8} {'pred':>8} {'err':>8} {'in_tok':>8} {'out_tok':>8} "
      f"{'gen':>4} {'actual cost':>12} {'pred cost':>10} {'cost x':>7}\n")
    a_score = np.zeros(3)
    a_cost = np.zeros(3)
    for j, mid in enumerate(MODEL_IDS):
        r = rec[mid]
        a_score[j] = float(r["score"])
        a_cost[j] = actual_cost(r, rates, j)
        err = P["pre_lookup_score"][i, j] - a_score[j]
        w(f"{mid:<12} {a_score[j]:>8.2f} {P['pre_lookup_score'][i, j]:>8.4f} {err:>+8.4f} "
          f"{r['input_tokens']:>8} {r['output_tokens']:>8} {r['num_generations']:>4} "
          f"{a_cost[j]:>12.6f} {cost_mean[i, j]:>10.6f} "
          f"{cost_mean[i, j] / max(a_cost[j], 1e-12):>7.2f}\n")
    w("\npred 열은 조회표 적용 전의 순수 예측값이다 "
      "(조회표 ON 번들에서도 예측력을 볼 수 있도록 분리).\n")
    w(f"실제 출력토큰 비율 : k1/light = {rec[MODEL_IDS[2]]['output_tokens'] / max(1, rec[MODEL_IDS[0]]['output_tokens']):.2f}x, "
      f"ax31/light = {rec[MODEL_IDS[1]]['output_tokens'] / max(1, rec[MODEL_IDS[0]]['output_tokens']):.2f}x\n\n")

    # ---- [6] 판정 ----------------------------------------------------------
    w(rule("[6] 판정 — 옳은 선택이었나 / 손실은 얼마인가") + "\n")
    best = float(a_score.max())
    # 최고점 동률이면 가장 싼 모델이 오라클
    cands = [j for j in range(3) if a_score[j] >= best - 1e-12]
    oracle = min(cands, key=lambda j: a_cost[j])
    w(f"문항 오라클        : {MODEL_IDS[oracle]} (score={best:.2f}, cost={a_cost[oracle]:.6f})\n")
    if len(cands) > 1:
        w(f"  동점 모델        : {', '.join(MODEL_IDS[j] for j in cands)} "
          "-> 최저비용 모델을 오라클로 본다\n")
    w("\n")
    w(f"{'tier':<10} {'선택':<12} {'실제 score':>10} {'실제 cost':>11} {'regret':>8} "
      f"{'초과지출':>10} {'판정':<10}\n")
    verdicts = {}
    for tier in ("fast", "balanced", "premium"):
        j = int(picks[tier][i])
        regret = best - a_score[j]
        # 같은 점수를 낼 수 있는 가장 싼 모델 대비 초과 지출
        same = [k for k in range(3) if a_score[k] >= a_score[j] - 1e-12]
        cheapest_same = min(same, key=lambda k: a_cost[k])
        overspend = a_cost[j] - a_cost[cheapest_same]
        if regret <= 1e-12 and overspend <= 1e-12:
            v = "최적"
        elif regret <= 1e-12:
            v = "과지출"
        elif a_score[j] <= 0 and best > 0:
            v = "실패"
        else:
            v = "과소투자"
        verdicts[tier] = v
        w(f"{tier:<10} {MODEL_IDS[j]:<12} {a_score[j]:>10.2f} {a_cost[j]:>11.6f} "
          f"{regret:>8.2f} {overspend:>10.6f} {v:<10}\n")
    w("\n판정 정의: 최적 = regret 0 이고 초과지출 0 / 과지출 = 더 싼 모델로 같은 점수 가능 /\n"
      "           과소투자 = 더 비싼 모델이 더 높은 점수 / 실패 = 선택 모델이 0점인데 정답 가능\n")
    w(f"\n가중 손실(이 문항 1건) = "
      + " + ".join(
          f"{TIER_WEIGHT[t]}x{best - a_score[int(picks[t][i])]:.2f}" for t in ("fast", "balanced", "premium")
      )
      + f" = {sum(TIER_WEIGHT[t] * (best - a_score[int(picks[t][i])]) for t in ('fast', 'balanced', 'premium')):.4f}"
      f" (최종점수 기여 손실 = 이 값 / {len(ids)})\n")
    w(rule() + "\n")

    if args.json:
        payload = {
            "episode_id": args.episode,
            "bundle": str(router.bundle),
            "lookup_entries": int(len(router.lookup_key)),
            "pred_score_prelookup": P["pre_lookup_score"][i].tolist(),
            "pred_cost_mean": cost_mean[i].tolist(),
            "pred_cost_q90": cost_q90[i].tolist(),
            "pred_out_tokens_lgbm": P["lgbm_out"][i].tolist(),
            "actual_score": a_score.tolist(),
            "actual_cost": a_cost.tolist(),
            "decisions": {t: MODEL_IDS[int(picks[t][i])] for t in ("fast", "balanced", "premium")},
            "verdicts": verdicts,
        }
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[json] {args.json}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------
# 실패 사례 탐색기
# --------------------------------------------------------------------------


def do_rank(args, ids, texts, P, picks, outcomes, rates) -> int:
    if not outcomes:
        print("오류: --rank 에는 --outcomes 가 필요합니다.", file=sys.stderr)
        return 2
    n = len(ids)
    a_score = np.zeros((n, 3))
    a_cost = np.zeros((n, 3))
    for r, eid in enumerate(ids):
        rec = outcomes[eid]
        for j, mid in enumerate(MODEL_IDS):
            a_score[r, j] = float(rec[mid]["score"])
            a_cost[r, j] = actual_cost(rec[mid], rates, j)
    pred = P["pre_lookup_score"]
    best = a_score.max(axis=1)

    mode = args.rank
    if mode == "score-error":
        metric = np.abs(pred - a_score).mean(axis=1)
        label = "mean |pred - actual| (3모델 평균)"
    elif mode == "cost-error":
        metric = np.abs(np.log(P["cost_mean"] / np.maximum(a_cost, 1e-12))).mean(axis=1)
        label = "mean |log(pred_cost / actual_cost)|"
    elif mode.startswith("regret-"):
        tier = mode.split("-", 1)[1]
        sel = picks[tier]
        metric = best - a_score[np.arange(n), sel]
        label = f"regret = best_actual - actual[{tier} 선택]"
    else:
        print(f"오류: 알 수 없는 --rank {mode}", file=sys.stderr)
        return 2

    order = np.argsort(-metric)[: args.top]
    print(f"# rank={mode}  기준: {label}  (상위 {args.top}건 / {n}건)")
    print(f"{'episode':<12} {'metric':>8} {'fast':<12} {'bal':<12} {'prem':<12} "
          f"{'actual(l/a/k)':<18} {'pred(l/a/k)':<20} chars")
    for r in order:
        if metric[r] <= 0:
            continue
        act = "/".join(f"{a_score[r, j]:.2f}" for j in range(3))
        pr = "/".join(f"{pred[r, j]:.2f}" for j in range(3))
        print(f"{ids[r]:<12} {metric[r]:>8.4f} "
              f"{MODEL_IDS[int(picks['fast'][r])]:<12} "
              f"{MODEL_IDS[int(picks['balanced'][r])]:<12} "
              f"{MODEL_IDS[int(picks['premium'][r])]:<12} "
              f"{act:<18} {pr:<20} {len(texts[r])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trace_episode",
        description="단일 episode에 대한 라우터 결정 추적기",
    )
    p.add_argument("--episode", help="추적할 episode_id (예: dev-0042)")
    p.add_argument("--input", type=Path, default=REPO / "data/materialized/dev/inputs.json")
    p.add_argument("--outcomes", type=Path, default=REPO / "data/dev/outcomes.json")
    p.add_argument(
        "--bundle", type=Path, default=REPO / "build/bundle-nolookup",
        help="모델 번들 경로. 기본은 조회표 OFF 번들 (예측 능력을 보기 위함)",
    )
    p.add_argument("--prompt-chars", type=int, default=700, help="본문 표시 길이")
    p.add_argument("--mask", action="store_true", help="재배포 불가 원문 마스킹 (해시+앞 120자만)")
    p.add_argument("--json", type=Path, help="구조화 결과를 JSON으로도 저장")
    p.add_argument("--no-verify", dest="verify", action="store_false",
                   help="FinalRouter.predict_batch 대조 생략 (2배 빠름)")
    p.add_argument("--rank", help="실패 사례 탐색: score-error | cost-error | regret-fast|balanced|premium")
    p.add_argument("--top", type=int, default=20)
    p.set_defaults(verify=True)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    if not args.episode and not args.rank:
        print("오류: --episode 또는 --rank 중 하나가 필요합니다.", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
