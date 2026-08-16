# 최종 보고서 — SKT OSSP 2026 Efficient LLM Routing Challenge

작성: 2026-08-16. 모든 수치는 공식 Decimal 채점기(`ossp_router.scoring` /
`ossp_router.cli self-check`) 실행 결과이며, `exp/registry.jsonl`(49개 실험)에서
재현할 수 있다.

## 1. 최종 제출 아키텍처 (registry E049, 런타임 검증 완료)

**Score 예측** — 4멤버 균등 블렌드 (전부 Train 1,760만으로 학습):
| 멤버 | 방식 | OOF score MSE |
| --- | --- | ---: |
| xgb-mono | XGBoost multi-output tree (3 score 공동), hist, raw sparse 117,575열 | 0.1342 |
| irt1d | IRT: score≈σ(a_j·θ+b_j), θ=W·x 선형, L-BFGS | 0.1390 |
| knn-k40 | 코사인 kNN(k=40, w=sim³) over word+char 해시 TF | 0.1501 |
| lgbm | LightGBM 3-head 회귀, dense81+SVD128×2 | 0.1387 |

**Cost 예측** — 토큰 헤드 분리: log(in_tok)·log(out_tok) 회귀 → 요율 결합,
lgbm·xgb 두 계열의 log-공간 평균. Premium의 k1 배분 비용만 **LightGBM
quantile(α=0.90)** 상한 사용.

**결정 레이어** (등급별 동결, `models/final-v1/policy.json`):
| 등급 | 배분 | 파라미터 | dev 실측 |
| --- | --- | --- | --- |
| Fast | greedy 그룹승격, **k1 금지** | 목표사용률 0.93 | 0.6591 / 1.1804 |
| Balanced | greedy 그룹승격, **k1 금지** | 목표사용률 0.88 | 0.6918 / 1.6774 |
| Premium | 2단계: k1 확정(q90 비용, 개별캡 0.1cr) → ax31 fill | k1_u 0.65 / fill 0.70 | 0.7105 / 2.8239 |

**정확 매칭 조회표(D5)** — 공개 Train+Dev 2,640문항의 SHA-256(프롬프트 원문) →
실측 score/cost. 적중 시 예측을 실측값으로 대체(규칙 명시 허용). Dev에서 조회표
활성 시 공식 self-check **0.7603** (전 문항 적중의 상한; 비공개셋에서는 공개
문항이 섞인 만큼만 이득).

**런타임**: python:3.11-slim-bookworm(digest 고정) + numpy/scipy/lightgbm/xgboost.
호스트 실측 로드 6.0s + 880문항 예측 8.3s. 번들 136MB (개별 파일 <100MB).

## 2. 성능 요약 (Dev 880, 공식 채점기)

| 항목 | Fast | Balanced | Premium | 가중 최종 |
| --- | ---: | ---: | ---: | ---: |
| all-light | 0.6193/1.000 | 동일 | 동일 | 0.6193 |
| 공식 최강 baseline hash-regex | 0.6631/1.236 | 0.6938/1.962 | 0.7401/3.985 | 0.6954 |
| dev-cal 탐색 최고 (E041, 참고) | 0.6744/1.237 | 0.6946/1.992 | 0.7460/3.935 | 0.7020 |
| **최종 E049 (위험게이트)** | 0.6591/1.180 | 0.6918/1.677 | 0.7105/2.824 | **0.6843** |
| 최종 + 조회표 (dev 상한) | 0.7386/1.157 | 0.7420/1.205 | 0.8074/2.087 | 0.7603 |
| budget-oracle 상한 | 0.7594/1.249 | 0.8074/1.989 | 0.8591/3.995 | 0.8037 |

**E049가 dev-cal 최고(0.7020)가 아니라 위험게이트(0.6843)를 최종으로 삼은 이유**:
- 공식 문서 전례: baseline이 dev 3.985 → 비공개셋 4.2(+5.4%)로 Premium 0점.
- Premium은 k1 출력토큰 롱테일(1문항 최대 130k tok = light 총비용의 78%) 때문에
  낮은 사용률에서도 부트스트랩 초과확률이 큼. 승자의 저주(배분기가 비용
  과소예측 문항을 선택; 선택집합 실현비용 +32% 편향)도 실측 확인.
- E049의 각 등급은 삼중 게이트 통과: **부트스트랩 초과확률** F 0.1%/B 0%/P 0.5%,
  **±50% 카테고리 시프트** 전부 ≤2.9%, **CLT(train 잔차)** ≈0%. Fast/Balanced는
  k1 완전 배제로 꼬리위험 자체를 제거.
- baseline 방식(dev 아슬아슬 보정)이 비공개셋에서 한 등급이라도 0점이 되면
  가중 0.3~0.4를 통째로 잃음 — 기대값 계산상 게이트형이 우월.

Train 내부 홀드아웃(Dev 미접촉) 순위 검증: blend4 계열이 전 후보 중 최상위
(0.6625~0.6629) — dev 선택 과적합 아님 확인.

## 3. 금지 전략 준수 체크리스트

| 금지 항목 | 준수 근거 |
| --- | --- |
| 모델 순차 호출/답변 비교 | 라우터는 사전계산 artifact만 사용, 모델 호출 코드 없음 |
| 선택 후 재시도/변경 | 단일 패스 배분, 재시도 로직 없음 |
| challenge_id/split/episode_id/순서 의존 | `tests/test_final_router.py` 5개 감사 테스트 통과 (순서 셔플·ID 개명·split 변경 → 프롬프트별 선택 동일). 배분은 그룹 승격으로 동률도 내용 기반 처리 |
| 비공개 자료 사용 | 학습 입력은 공개 Train/Dev + 공개 비용 정책만. `data/materialized` SHA-256이 공개 registry와 일치 |
| 실행 중 네트워크 | 이미지에 pip 제거, 실행 시 다운로드 없음. 조회표·모델 전부 이미지 내장 |
| 소스≠이미지 | Dockerfile이 committed src/ + models/final-v1만 COPY. `build_final.py`가 재학습→검증(스냅샷 오차 0) 재현 경로 제공 |

프롬프트 원문/episode_id는 어떤 학습 산출물에도 저장하지 않음(조회표는 SHA-256
해시만 — 규칙이 명시 허용하는 "프롬프트 해시" 조회). AIME 원문은 git 비추적
경로(`/data/materialized/`)에만 존재, 이미지에도 미포함.

## 4. 라이선스 표

| 구성요소 | 버전 | 라이선스 | 용도 |
| --- | --- | --- | --- |
| numpy | 2.0.2 | BSD-3-Clause | 런타임/학습 |
| scipy | 1.17.1 | BSD-3-Clause | 런타임/학습 |
| lightgbm | 4.7.0 | MIT | 런타임/학습 |
| xgboost | 3.2.0 | Apache-2.0 | 런타임/학습 |
| scikit-learn | 1.9.0 | BSD-3-Clause | 학습 전용 (SVD/scaler 계수만 산출물로 내보냄, 이미지 미포함) |
| matplotlib | (호스트) | PSF-기반 (matplotlib license) | 대시보드 전용, 이미지·제출물 미포함 |
| python:3.11-slim-bookworm | digest 고정 | PSF + Debian (표준 구성요소) | 기반 이미지 |
| 학습 산출물 (models/final-v1) | v1 | Apache-2.0 (본 저장소) | 공개 Train/Dev에서만 유도 |

외부 벤치마크 데이터 추가 사용 없음(P7 미채택 — 아래 §6). 사전학습 언어모델
가중치 사용 없음 → 가중치 공개 요건은 models/final-v1 커밋으로 충족.

## 5. 위험·마진 설계 요약

- 부트스트랩(880 복원추출 1,000회) + 카테고리 ±50% 리샘플 2,000회 + CLT 잔차
  모델 삼중 게이트. 수치는 `exp/final-frozen-policy.json`, 스윕 전체는
  `exp/final-policy-*.json`.
- Premium 전용 보호: q90 quantile 비용으로 k1 선정, 문항당 예측 k1 비용 0.1
  credit 초과 시 k1 배제(꼬리 컷), ax31 fill은 예산 70%까지만.
- 예산 여유: Fast 5.6% / Balanced 16.1% / Premium 29.4% — baseline 전례의
  +5.4% 드리프트가 세 등급 동시에 와도 전 등급 통과.

## 6. 탐색 요약 (registry 49실험, exp/results.md·dashboard.html 자동 생성)

- 특징: dense81 + word 1-2gram/char 3-4gram signed 해시(crc32/롤링해시, 결정적).
- 모델 축: ridge/logistic < kNN < MLP < LightGBM ≈ IRT < XGBoost-mono.
  서수분류(0.6887)>이진(0.6761)이나 회귀 대비 이득 없음(감사#2 결론).
- 논문 기법: P1 Δ헤드(점수 이득 없음, 비용 토큰헤드는 채택) / P2 IRT(irt1d 채택,
  2d는 MSE만 개선) / P3·P4(과제 구조상 P1 재파라미터화 — delta-costaware +0.0003,
  미채택) / P5(=D3 라그랑주, D2 greedy가 우세) / **P6 conformal·CLT(마진 설계에
  채택)** / P7 외부 데이터 사전학습(기대이득 대비 라이선스·재현 리스크로 미채택,
  kNN 유사도 p50=0.61로 공개 데이터 커버리지 이미 높음).
- 증강: A3 표면노이즈 +0.004(잡음 수준, 미채택), A1·A2(패러프레이즈/역번역)는
  라벨 보존 가정 위험 대비 이득 근거 부족으로 미채택. A6 불확실성은 q90
  quantile로 대체 구현.
- 스태킹(meta-LGBM)은 OOF/full-fit 분포차로 역효과(0.6876) — 균등 블렌딩 채택.

## 7. 남은 제출 절차 (참가자 수행 필요)

1. 이 저장소를 팀 GitHub로 fork 후 본 작업 커밋 push (models/final-v1 포함).
2. `docker build --pull --platform linux/arm64 -f container/Dockerfile -t <registry>/<repo>:v1 .`
   → 공개 레지스트리 push → **이미지 다이제스트 기록**.
3. `submission-ossp-skt.json`에 커밋 SHA + 다이제스트 기재, 별도 커밋
   (`tools/validate_technical_submission.py`로 검증).
4. osscontest.kr에 결과보고서 업로드 (마감 2026-08-27 18:00 KST).

## 8. 검증 상태

- [x] 데이터 SHA-256 일치, 참가자 관련 단위테스트 전부 통과
- [x] 하네스↔공식 self-check 12자리 일치 (3실험×3등급)
- [x] 재학습 재현: 번들 재빌드 시 스냅샷 예측과 오차 0
- [x] 런타임 CLI가 E049 배정 완전 재현 (조회표 OFF)
- [x] 결정성 감사 5테스트 통과 (순서/ID/split/반복/단일문항)
- [x] 공식 self-check (조회표 ON): 0.7603, 전 등급 통과, near_budget 없음
- [x] linux/arm64 이미지 실행 검증 (§9)

## 9. 런타임 실측 (linux/arm64 이미지, 커밋 3e95eea)

측정 환경: Windows x64 호스트에서 QEMU 에뮬레이션(공식 환경은 네이티브 Apple
Silicon — 로컬 측정은 규정상 참고값). 공식 자원한도 동일 적용:
`--cpus 2 --memory 2g --memory-swap 2g --pids-limit 32 --network none
--read-only --ipc none --tmpfs /tmp:size=256m`, 입력은 공개 Train+Dev 결합
2,640문항 (check_runtime.py와 동일 워크로드; check_runtime.py 자체는 POSIX 전용
모듈(fcntl) 때문에 Windows 호스트에서 실행 불가 → 동일 조건 수동 측정).

| 등급 | QEMU 벽시계 | 종료코드 | 출력 검증 |
| --- | ---: | --- | --- |
| fast | 562.6s | 0 | 2,640 결정 전부 존재, v1 형식 유효 |
| balanced | 463.2s | 0 | 동일 |
| premium | 313.7s | 0 | 동일 |

- 네이티브 x64 호스트(비컨테이너) 실측: 로드 6.0s + 2,640문항 예측 ~25s ≈
  **등급당 ~33s** — QEMU 감속계수 ~10-17×. 공식 Apple Silicon 네이티브에서는
  90초 한도 대비 충분한 여유로 판단. 최종 확인은 공식 환경 실행을 따름.
- **크로스 플랫폼 결정성**: arm64 컨테이너 출력과 x64 호스트 출력의
  결정 7,920개(3등급×2,640) **완전 일치** — 부동소수점/플랫폼 차이로 인한
  선택 변동 없음.
- 메모리: 컨테이너 2GiB 한도 내 정상 종료(초과 시 OOM kill로 실패했을 것).
- 이미지: 로컬 매니페스트 리스트 `sha256:5da033b8…` (레지스트리 push 후의
  다이제스트를 submission-ossp-skt.json에 기재할 것).
