# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Final prompt-only router: xgb-mono score head + token-head cost models.

Pipeline per tier run:
  1. parse inputs (prompt/messages content only)
  2. deterministic features (features_v2)
  3. score: xgb-mono (monotone multi-output) alone;
     lgbm+xgb token heads -> mean costs; lgbm q90 -> conservative k1 cost
  4. exact-match lookup (public Train/Dev outcomes, rule-allowed) overrides
     predictions with realized values
  5. frozen risk-gated allocation per tier (policy.json)
  6. atomic submission write

Model artifacts live in the committed bundle (models/final-v1) or the path in
OSSP_MODEL_BUNDLE. Only content-derived information is used for decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import allocation, features_v2
from .heuristic import episode_text, write_submission_atomic
from .protocol import (
    TIERS,
    Decision,
    InputBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_policy,
    parse_submission,
    submission_to_dict,
)

MODEL_IDS = ("ax31-light", "ax31", "axk1-think")


def _bundle_dir() -> Path:
    env = os.environ.get("OSSP_MODEL_BUNDLE")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "models" / "final-v1"


class FinalRouter:
    def __init__(self, bundle: Optional[Path] = None) -> None:
        import lightgbm as lgb
        import xgboost as xgb

        self.bundle = Path(bundle) if bundle else _bundle_dir()
        b = self.bundle
        self.policy_spec = json.loads((b / "policy.json").read_text(encoding="utf-8"))
        self.rates = np.asarray(self.policy_spec["rates"], dtype=np.float64)
        # SVD는 LightGBM 토큰 헤드의 입력(dense81 + SVD128x2)에만 쓴다.
        self.svd_word = np.load(b / "svd-word.npz")["components"]
        self.svd_char = np.load(b / "svd-char.npz")["components"]
        # LightGBM은 비용 전용: lin/lout 평균 토큰, q90 상한 토큰.
        self.lgbm = {
            name: lgb.Booster(model_file=str(b / "lgbm" / f"{name}.txt"))
            for name in [f"{kind}-{j}" for kind in ("lin", "lout", "q90") for j in range(3)]
        }
        self.xgb_keep = np.load(b / "xgb" / "keep-cols.npy")
        self.xgb_score = xgb.Booster()
        self.xgb_score.load_model(str(b / "xgb" / "score-multi.json"))
        self.xgb_tokens = {}
        for kind in ("lin", "lout"):
            for j in range(3):
                booster = xgb.Booster()
                booster.load_model(str(b / "xgb" / f"{kind}-{j}.json"))
                self.xgb_tokens[f"{kind}-{j}"] = booster
        lookup = np.load(b / "lookup.npz")
        self.lookup_key = np.ascontiguousarray(lookup["key"]).view("V32").ravel()
        self.lookup_scores = lookup["scores"]
        self.lookup_costs = lookup["costs"]

    # ---------------- prediction ----------------

    @staticmethod
    def _monotone(cost: np.ndarray) -> np.ndarray:
        cost = cost.copy()
        cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1 + 1e-12))
        cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1 + 1e-12))
        return cost

    def predict_batch(self, texts: Sequence[str], message_counts: Sequence[int]):
        import xgboost as xgb
        from scipy import sparse

        dense, word, char = features_v2.featurize_batch(texts, message_counts)
        x_svd = np.hstack([dense, word @ self.svd_word.T, char @ self.svd_char.T])

        lgbm_lin = np.column_stack([self.lgbm[f"lin-{j}"].predict(x_svd) for j in range(3)])
        lgbm_lout = np.column_stack([self.lgbm[f"lout-{j}"].predict(x_svd) for j in range(3)])
        lgbm_q90 = np.column_stack([self.lgbm[f"q90-{j}"].predict(x_svd) for j in range(3)])
        lgbm_cost = self._monotone(
            np.exp(lgbm_lin) * self.rates[:, 0] / 1e6 + np.exp(lgbm_lout) * self.rates[:, 1] / 1e6
        )
        lgbm_q90_cost = self._monotone(
            np.exp(lgbm_lin) * self.rates[:, 0] / 1e6 + np.exp(lgbm_q90) * self.rates[:, 1] / 1e6
        )

        x_sp = sparse.hstack([sparse.csr_matrix(dense), word, char], format="csr").astype(np.float32)
        x_sp = x_sp[:, self.xgb_keep].tocsr()
        dmat = xgb.DMatrix(x_sp)
        xgb_score = np.clip(self.xgb_score.predict(dmat), 0.0, 1.0)
        xgb_lin = np.column_stack(
            [self.xgb_tokens[f"lin-{j}"].predict(dmat) for j in range(3)]
        )
        xgb_lout = np.column_stack(
            [self.xgb_tokens[f"lout-{j}"].predict(dmat) for j in range(3)]
        )
        xgb_cost = self._monotone(
            np.exp(xgb_lin) * self.rates[:, 0] / 1e6 + np.exp(xgb_lout) * self.rates[:, 1] / 1e6
        )

        # 점수 예측은 단조제약 XGBoost multi-output 하나만 쓴다. 4멤버 블렌드는
        # 공개 Dev에서 기여가 +0.0016(95% CI [-0.0048, +0.0083], p(<=0)=0.31)로
        # 0과 구별되지 않았고, 단독 xgb-mono가 세 모델 모두에서 MAE가 더 낮았다.
        # 근거: exp/audit/A7-overfit-falsification.md, exp/audit-verdict.md
        blend_score = xgb_score
        cost_mean = self._monotone(np.exp((np.log(lgbm_cost) + np.log(xgb_cost)) / 2.0))
        cost_q90 = cost_mean.copy()
        cost_q90[:, 2] = lgbm_q90_cost[:, 2]
        cost_q90 = self._monotone(cost_q90)

        # exact-match lookup override (content-derived, rule-allowed)
        if len(self.lookup_key):
            keys = np.frombuffer(
                b"".join(hashlib.sha256(t.encode("utf-8")).digest() for t in texts),
                dtype=np.uint8,
            ).reshape(-1, 32)
            keys_v = np.ascontiguousarray(keys).view("V32").ravel()
            pos = np.searchsorted(self.lookup_key, keys_v)
            pos = np.minimum(pos, len(self.lookup_key) - 1)
            hit = self.lookup_key[pos] == keys_v
        else:
            hit = np.zeros(len(texts), dtype=bool)
        if hit.any():
            blend_score[hit] = self.lookup_scores[pos[hit]]
            cost_mean[hit] = self.lookup_costs[pos[hit]]
            cost_q90[hit] = self.lookup_costs[pos[hit]]
            cost_mean = self._monotone(cost_mean)
            cost_q90 = self._monotone(cost_q90)
        return blend_score, cost_mean, cost_q90

    # ---------------- allocation ----------------

    def allocate(self, tier: str, score, cost_mean, cost_q90) -> np.ndarray:
        spec = self.policy_spec[tier]
        if tier == "premium":
            return allocation.two_stage_premium(
                score,
                cost_q90,
                multiplier=4.0,
                k1_utilization=spec["k1_utilization"],
                fill_utilization=spec["fill_utilization"],
                k1_cost_cap=spec["k1_cost_cap"],
            )
        multiplier = 1.25 if tier == "fast" else 2.0
        return allocation.greedy_allocate(
            score,
            cost_mean,
            multiplier=multiplier,
            utilization=spec["utilization"],
            allow_k1=spec["allow_k1"],
        )


def make_submission(
    inputs: InputBatch, policy: RoutingPolicy, tier: str, router: FinalRouter
) -> Submission:
    if tier not in TIERS:
        raise ProtocolError(f"알 수 없는 tier: {tier}")
    texts = [episode_text(e) for e in inputs.episodes]
    counts = [1 if e.prompt is not None else len(e.messages or ()) for e in inputs.episodes]
    score, cost_mean, cost_q90 = router.predict_batch(texts, counts)
    pick = router.allocate(tier, score, cost_mean, cost_q90)
    submission = Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(
            Decision(episode.episode_id, MODEL_IDS[j])
            for episode, j in zip(inputs.episodes, pick)
        ),
    )
    return parse_submission(submission_to_dict(submission))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="router-run", description="최종 블렌드 라우터를 한 등급에 대해 실행합니다."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = load_policy(args.policy) if args.policy else load_bundled_policy()
        router = FinalRouter(args.bundle)
        submission = make_submission(inputs, policy, args.tier, router)
        write_submission_atomic(args.output, submission)
    except (OSError, ProtocolError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    print(f"OK: {args.tier} 제출 파일을 생성했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
