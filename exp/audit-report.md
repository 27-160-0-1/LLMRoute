<!--
SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
SPDX-License-Identifier: Apache-2.0
-->

# 감사 보고서 (v2-1) — 2026-08-16

모든 항목은 실제 실행으로 검증. 재현 명령: `.venv\Scripts\python.exe exp\selftest.py`,
`.venv\Scripts\python.exe exp\audit_selfcheck.py`.

## 1. 데이터 정합성 — PASS
- materialize 후 Train **1,760** / Dev **880** 확인 (base 파일 1,736/868 → AIME 결합 완료).
- SHA-256 대조: train `029a0fb1…` / dev `5920f9ea…` — `data/public-data.v1.json`과 **완전 일치**.
- 모든 학습·평가는 `data/materialized/{split}/inputs.json`만 사용 (exp/harness.py 경로 고정).
  base 파일로 학습한 실험 **없음**.

## 2. score 라벨 구조 — PASS (후속 실험 등록)
- unique scores: {0: 1357, 0.25: 20, 0.5: 459, 0.75: 18, 1.0: 3426} (train, 3모델 합산).
- num_generations ∈ {2, 4} — score는 생성 시도별 정답률.
- 현 예측기는 연속 회귀. **회귀 vs 서수분류 vs 이진분류 비교는 실험 매트릭스에 등록**
  (logit-cls variant가 1차 비교; 서수 분류는 P2/IRT 계열과 함께 진행).

## 3. 채점 일치성 — PASS
`exp/audit_selfcheck.py`: 하네스 점수와 `ossp_router.cli self-check` 리포트를
3개 실험 × 3등급에서 대조 — quality_score, budget_ratio, tier_score, total_cost,
budget_passed, final_score **전부 12자리 문자열 완전 일치**:
- hash-regex: final 0.695369318182 일치
- budget-oracle: final 0.803693181818 일치
- naive-d3(의도적 예산초과 케이스): 예산초과 → final 0 처리도 일치

## 4. 결정성·규칙 준수 — 부분 PASS (최종 라우터에서 재감사 예정)
- (a,b,c) 순서/ID/split 불변성: 배분 레이어(`alloc_lib`)의 **순서 셔플 property test PASS**
  (동일 예측행 → 동일 결정, 그룹 승격으로 동률 처리). 최종 라우터 통합 후
  전체 파이프라인 셔플 감사를 tests/로 추가 예정 (Phase 4 게이트).
- (d) 학습 산출물 점검: `build/preds/*.npz`는 순수 예측 배열(score/cost)만 포함.
  summary.json은 하이퍼파라미터·지표만. **프롬프트 원문·episode_id 미포함** 확인.
- (e) AIME 원문 비커밋: `/data/materialized/`, `build/`, `.venv/` 모두 `.gitignore` 등재
  (`git check-ignore -v` 확인). untracked는 자체 코드 `analysis/`, `exp/`뿐이며
  프롬프트 원문 미포함.

## 5. 예산 계산 검증 — PASS
- Dev light_baseline_cost를 정책 계수(Decimal)로 독립 재계산 = `4.381428` —
  공식 리포트 값과 **정확 일치** (selftest `_budget`).
- 공식 채점기는 비용을 반올림 전 Decimal(160자리 문맥)로 비교함을 소스에서 확인
  (`scoring.py: budget_passed = total_cost <= budget_limit`, quantize는 표시용).

## 6. 런타임 실측 — 미실측 (전 행)
- 현재 results.md의 모든 행은 **호스트(Windows x64) 측정** 또는 미측정.
  arm64/2CPU/2GiB/90초 실측은 Phase 4에서 `check_runtime.py`로 수행 예정.
- 주의: check_runtime은 Train+Dev **결합 2,640문항**을 등급마다 한 번에 처리 —
  특징 추출 예산 설계 기준.

## 버전 확인 (selftest 13/13 PASS)
| 구성요소 | 버전 | 검증 내용 |
| --- | --- | --- |
| Python | 3.11.9 (AMD64 호스트) | ≥3.10, 컨테이너는 3.11.15-alpine 예정 |
| numpy | 2.0.2 | uint64 연산, 2.x 제거 API 미사용 |
| scipy | 1.17.1 | CSR save/load roundtrip |
| scikit-learn | 1.9.0 | Ridge/Logistic/Isotonic/SVD/MLP fit |
| lightgbm | 4.7.0 | dense+sparse fit, quantile objective |
| xgboost | 3.2.0 | hist + sparse |
| feat_lib | v2 | 결정성, crc32 플랫폼 안정성, numpy 롤링해시 = 순수 파이썬 참조 구현 일치 |

## 미해결 리스크 (추적)
1. 최종 라우터의 컨테이너 결정성 감사 (Phase 4 게이트)
2. arm64 90초 실측 (Phase 4 게이트)
3. 서수분류 vs 회귀 비교 실험 (실험 매트릭스 등록됨)
