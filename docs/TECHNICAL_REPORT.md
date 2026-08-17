<!--
SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
SPDX-License-Identifier: Apache-2.0
-->

# 예산 제약 프롬프트 라우팅: 사용자 제어 품질–비용 트레이드오프의 쌍대(dual) 구성

**SKT OSSP 2026 Efficient LLM Routing Challenge — 기술 보고서**

이 문서는 본 저장소의 라우터를 Feng et al. (2025)의 *IPR: Intelligent Prompt
Routing with User-Controlled Quality-Cost Trade-offs* 논문 구조에 맞추어
기술한다. 절 번호와 표제는 원 논문의 구성을 따르며, 각 절 첫머리에 **[IPR
§x]** 로 대응 관계를 명시한다.

> **인용 논문.** Aosong Feng, Balasubramaniam Srinivasan, Yun Zhou, Zhichao Xu,
> Kang Zhou, Sheng Guan, Yueyan Chen, Xian Wu, Ninad Kulkarni, Yi Zhang,
> Zhengyuan Shen, Dmitriy Bespalov, Soumya Smruti Mishra, Yifei Teng,
> Darren Yow-Bang Wang, Haibo Ding, Lin Lee Cheong. 2025.
> **IPR: Intelligent Prompt Routing with User-Controlled Quality-Cost
> Trade-offs.** In *Proceedings of the 2025 Conference on Empirical Methods in
> Natural Language Processing: Industry Track*, pages 2484–2498, Suzhou, China.
> Association for Computational Linguistics.
> DOI [10.18653/v1/2025.emnlp-industry.170](https://doi.org/10.18653/v1/2025.emnlp-industry.170) ·
> ACL Anthology `2025.emnlp-industry.170` · preprint
> [arXiv:2509.06274](https://arxiv.org/abs/2509.06274).
>
> 이 저장소는 위 논문의 코드·데이터·모델 가중치를 일절 포함하지 않는다.
> 인용은 **문제 정식화와 보고 구조**에 한정하며, 구현은 전부 독립적이다.

---

## 초록

프롬프트만 보고 세 개의 평가용 모델 프로필(`ax31-light`, `ax31`,
`axk1-think`) 중 하나를 고르는 라우터를 제시한다. IPR이 *예측 품질 하한을
제약으로 두고 비용을 최소화*하는 반면, 본 과제는 *비용 상한을 하드 제약으로
두고 품질을 최대화*한다. 두 문제는 서로 쌍대이며, 이 차이가 설계 전반을
바꾼다. 하드 예산은 (i) 품질뿐 아니라 **비용까지 예측**해야 하고,
(ii) 문항별 독립 판정이 아니라 **배치 전역 배분** 문제가 되며,
(iii) 예산 초과 시 해당 등급이 0점이므로 **분포 이동에 대한 마진 설계**가
점수 최적화보다 우선한다는 것을 뜻한다.

구현은 4-멤버 품질 블렌드(XGBoost multi-output, 1-D IRT, 코사인 kNN,
LightGBM)와 분리된 토큰-헤드 비용 모델, 그리고 등급별로 동결한 순서 불변
배분기로 구성된다. 공개 Dev 880문항 공식 채점기 기준 가중 최종 점수
**0.684318**(95 % CI [0.656175, 0.710795], 전 등급 예산 통과)을 재현했다.
공개 Train+Dev를 담은 정확 매칭 조회표를 켜면 0.760284가 나오지만 Dev는 그
조회표에 100 % 포함되므로 이는 암기이며 일반화 지표가 아니다(§4.3.2).

---

## 1. 서론 — 네 가지 운영 제약

**[IPR §1]** IPR은 상용 서빙에서의 네 가지 도전 과제를 든다. 본 과제에도
그대로 적용되지만, 세 번째 항목의 성격이 다르다.

| # | IPR이 제기한 과제 | 본 과제에서의 형태 |
| --- | --- | --- |
| i | **생성 없는 품질 예측** — 후보를 호출하지 않고 프롬프트만으로 품질 추정 | 동일. 규칙이 모델 호출·답변 비교·순차 승격을 명시적으로 금지 ([CHALLENGE_RULES.md](CHALLENGE_RULES.md) §금지하는 전략) |
| ii | **지연 예산** — 라우팅 결정당 200 ms 미만 | 등급당 90초 안에 전체 배치 처리, CPU 2코어·메모리 2 GiB·프로세스 32개 ([RUNTIME.md](RUNTIME.md)) |
| iii | **비용 절감** — 품질 저하를 허용 범위로 묶고 비용을 낮춤 (연속적 트레이드오프) | **하드 예산.** 등급별 비용 비율 상한 1.25 / 2.0 / 4.0을 조금이라도 넘으면 해당 등급 **0점** ([SCORING.md](SCORING.md)) |
| iv | **확장성** — 새 모델 온보딩 비용 | 모델 집합이 3개로 고정. 대신 **비공개 평가셋으로의 일반화**가 그 자리를 차지 |

(iii)의 차이가 결정적이다. IPR에서 품질 저하 허용량을 조금 잘못 잡으면
점수가 매끄럽게 나빠지지만, 본 과제에서 예산을 1 credit이라도 넘기면 그
등급의 가중치(0.4 / 0.3 / 0.3)를 통째로 잃는다. 손실 함수가 **불연속**이다.

### 1.1 기여

1. IPR의 tolerance 게이팅을 하드 예산 체제로 옮긴 **쌍대 정식화**(§2).
2. IPR에 없는 **비용 예측기** — 모델별 입·출력 토큰 수를 로그 공간에서
   회귀하고 공개 요율과 결합. 상위 등급에는 q90 분위 회귀로 꼬리를 상향
   추정(§3.3).
3. 동점 문항을 그룹 단위로 승격시켜 **문항 ID·입력 순서에 대한 불변성**을
   구조적으로 보장하는 배분기(§3.4).
4. 부트스트랩·카테고리 시프트·CLT **삼중 위험 게이트**로 마진을 정한 정책
   동결 절차(§3.5). dev 최적점(0.7020) 대신 게이트 통과 구성(0.6843)을
   선택한 근거를 정량화한다.

### 1.2 문서 규모에 대한 참고

대회 제출물 규모 제한을 고려하여 이 보고서는 **순수 마크다운**이며 새로운
그림·바이너리 자산을 추가하지 않는다. 실행 이미지에는 사전학습 언어모델
가중치가 들어가지 않는다(§6.2 참조).

---

## 2. 문제 정식화와 라우팅 프레임워크

### 2.1 라우팅 정식화

**[IPR §2.1]** IPR의 표기를 따른다. 프롬프트 $x_i$, 후보 모델 집합
$\mathcal{C}$, 후보 $c$의 응답 $y_{i,c}$에 대해

- 실제 품질 $r_{i,c} = R(x_i, y_{i,c}) \in [0,1]$
- 호출 비용 $v_{i,c}$
- 예측 품질 $\hat{r}_{i,c} = R_\theta(x_i, c)$ — 프롬프트와 후보 식별자만 사용

IPR의 목적함수는 **품질 제약하 비용 최소화**다.

$$c^{*}_i = \arg\min_{c \in \mathcal{C}_\tau} v_{i,c} \qquad \text{(IPR Eq. 1)}$$

본 과제의 목적함수는 그 **쌍대**, 즉 **비용 제약하 품질 최대화**다. 등급
$t \in \{\text{fast}, \text{balanced}, \text{premium}\}$, 배치
$\mathcal{I}$, 배수 $M_t \in \{1.25, 2.0, 4.0\}$에 대해

$$\max_{\{c_i\}_{i \in \mathcal{I}}} \sum_{i \in \mathcal{I}} r_{i,c_i}
\quad \text{s.t.} \quad \sum_{i \in \mathcal{I}} v_{i,c_i} \;\le\; M_t \sum_{i \in \mathcal{I}} v_{i,\text{light}}$$

두 가지가 구조적으로 달라진다.

**(a) 비용도 예측 대상이다.** IPR은 $v_{i,c}$를 모델 등록소의 가격표에서
읽는다. 프롬프트 길이가 주어지면 비용이 결정되는 구조이기 때문이다. 본
과제에서 $v_{i,c}$는 **출력 토큰 수에 좌우**되고, 출력 토큰 수는 모델이
실제로 생성해 봐야 안다. 공식 평가에서는 문항별 실제 비용이 제공되지
않는다([CHALLENGE_RULES.md](CHALLENGE_RULES.md) §사용할 수 있는 정보). 따라서
$\hat{v}_{i,c}$ 역시 학습해야 한다.

**(b) 결정이 분리되지 않는다.** IPR의 제약은 문항별(per-prompt)이므로
$c^*_i$를 문항마다 독립적으로 풀 수 있다. 본 과제의 제약은 배치 전역
합계에 걸리므로, 한 문항을 승격시키면 다른 문항의 여지가 줄어든다. 이는
다중선택 배낭문제(multiple-choice knapsack)이며 §3.4에서 다룬다.

### 2.2 라우팅 전략 — 노브(knob)의 대응

**[IPR §2.2]** IPR은 사용자 노브로 tolerance $\tau \in [0,1]$을 노출하고,
문항별 동적 임계값을 쓴다.

$$G(\hat{r}_{i,c}, \tau) = \hat{r}_{i,c} - r_{i,\text{th}} \ge 0, \qquad
r_{i,\text{th}} = \hat{r}_{i,\max} - \tau\,(\hat{r}_{i,\max} - \hat{r}_{i,\min})
\qquad \text{(IPR Eqs. 3–4)}$$

배포판(‘Dynamic Max’)에서는 $\hat{r}_{i,\min} = 0$이므로
$r_{i,\text{th}} = (1-\tau)\hat{r}_{i,\max}$로 축약되고, 서빙 시에는 안전
마진 $\delta \ge 0$을 빼서 $r_{\text{th}} \leftarrow (1-\tau)\hat{r}_{\max} - \delta$
로 쓴다.

본 과제의 사용자 노브는 **등급 $t$** 이며, 대응 관계는 다음과 같다.

| IPR | 본 과제 | 성격 |
| --- | --- | --- |
| tolerance $\tau \in [0,1]$ (연속) | 등급 $t$ (이산 3단계) | 사용자 노출 노브 |
| $\tau = 0$: 예측 최상 모델 강제 | Premium ($M = 4.0$) | 품질 우선단 |
| $\tau = 1$: 항상 최저가 | Fast ($M = 1.25$) | 비용 우선단 |
| 문항별 임계값 $r_{i,\text{th}}$ | 전역 라그랑주 승수 $\lambda_t$ / 비율 컷오프 | 실효 결정 경계 |
| 안전 마진 $\delta$ | 목표 사용률 $u_t$ + k1 개별 비용 캡 | 추정 오차 흡수 |

핵심 차이는 **임계값이 어디서 오는가**다. IPR은 문항 $i$ 안에서 예측
점수의 스프레드로 임계값을 만든다 — 문항끼리 상호작용이 없다. 본 과제의
실효 임계값은 배치 전체 예산을 만족시키는 지점에서 결정되므로, 같은
프롬프트라도 함께 들어온 배치의 구성에 따라 달라질 수 있다.

> **규칙 준수 관점.** 이 배치 의존성은 규칙 위반이 아니다. 규칙이 금지하는
> 것은 `challenge_id`·`split`·`episode_id`·**입력 순서**에 대한 의존이며
> ([CHALLENGE_RULES.md](CHALLENGE_RULES.md) §라우터 실행 입력), 예산은 본래
> 배치 전체에 대해 정의된다. 순서 불변성은 §3.4에서 구조적으로 보장한다.

### 2.3 평가 지표

**[IPR §2.3]** IPR은 품질 예측 정확도(Top-K Accuracy / Top-K F1)와
종단 라우팅 성능(Bounded-ARQGC, Cost Save Ratio)을 나눈다.

Bounded-ARQGC는 비용 예산 $\alpha$를 0에서 1까지 훑으며 품질 곡선의 정규화
면적을 적분한다.

$$\text{Bounded-ARQGC} = \int_0^1 \frac{Q(\alpha) - Q_{\min}}{Q_{\max} - Q_{\min}}\, d\alpha$$

본 과제의 공식 지표는 **곡선이 아니라 세 개의 지정 지점**이다
([SCORING.md](SCORING.md)).

$$\text{final\_score} = \frac{0.4 \cdot P_{\text{fast}} + 0.3 \cdot P_{\text{bal}} + 0.3 \cdot P_{\text{prem}}}{|\mathcal{I}|},
\qquad
P_t = \begin{cases} \sum_i r_{i,c_i} & \text{예산 통과} \\ 0 & \text{예산 초과}\end{cases}$$

즉 Bounded-ARQGC를 $\alpha \in \{1.25, 2.0, 4.0\}/M_{\max}$ 세 점에서만
샘플링하되, **각 점에 절벽(cliff)을 붙인 형태**다. 이 절벽 때문에 곡선
면적 최적화와 실제 최적 정책이 갈라진다(§4.4).

내부 실험에서는 IPR의 오라클/랜덤 기준선 관행을 따라 두 상한을 함께
기록했다: 전 문항 light(0.6193, 곡선 하단)과 실측 outcome을 아는
budget-oracle(0.8037, 곡선 상단).

---

## 3. 시스템 — 예산 제약 라우터

### 3.1 시스템 개요

**[IPR §3.1]** IPR은 Quality Estimator(QE) · Decision Optimization(DO) ·
Model Registry의 3-컴포넌트다. 본 시스템도 같은 3분할을 쓰되, QE가
**품질과 비용 두 갈래**로 나뉜다.

```
container/entrypoint.py
  └─ ossp_router.final_router.main(--input, --tier, --output)
       ├─ protocol.load_input()                     # 스키마 검증, prompt/messages만 추출
       ├─ protocol.load_bundled_policy()            # 동결 v1 정책 (Model Registry 대응)
       └─ FinalRouter
            ├─ features_v2.featurize_batch()        # 결정적 내용 기반 특징
            ├─ predict_batch()                      # QE: 품질 + 비용
            │    ├─ 품질  : xgb-mono ⊕ irt1d ⊕ knn-k40 ⊕ lgbm  (균등 블렌드)
            │    ├─ 비용  : lgbm/xgb 토큰 헤드 → 요율 결합 → 로그공간 평균
            │    ├─ 꼬리  : lgbm q90 분위 → premium k1 상향 비용
            │    └─ 조회  : SHA-256(프롬프트) 정확 매칭 시 실측값으로 대체
            ├─ allocate(tier, ...)                  # DO: 등급별 동결 배분기
            └─ heuristic.write_submission_atomic()  # 임시파일 → os.replace, 0644
```

`Model Registry`에 해당하는 것은 `models/final-v1/policy.json`이며, 모델별
요율 행렬과 등급별 배분 파라미터를 담는다.

| 모델 | 입력 토큰 요율 | 출력 토큰 요율 |
| --- | ---: | ---: |
| `ax31-light` | 1.0 | 4.0 |
| `ax31` | 2.127 | 8.509 |
| `axk1-think` | 6.565 | 26.260 |

### 3.2 품질 추정기 구조

**[IPR §3.2]** IPR의 QE는 Prompt Encoder($d = 768$) + LLM Identity
Encoder($d' = 128$) + 2층 FFN이다. 본 시스템은 **신경망을 쓰지 않는다.**
2 GiB 메모리·2 코어·네트워크 차단·`linux/arm64` 조건과, 사전학습 가중치를
이미지에 넣을 경우 발생하는 라이선스·재배포 의무를 피하기 위해서다.

#### 3.2.1 특징 추출 (Prompt Encoder 대응)

`src/ossp_router/features_v2.py`. 학습 시 추출기(`exp/feat_lib.py` v2)의
바이트 단위 포팅이며, 난수·ID·순서 의존이 전혀 없다.

- **Dense 81차원** — 길이·단어수·문장수·메시지수·줄수·들여쓰기 로그 스케일,
  한글/라틴/숫자/대문자 비율, 길이 구간 지시자, 그리고 35개 정규식 패턴
  적중 수의 $\log(1+n)$. 패턴은 코드 펜스, LaTeX 매크로, 객관식 보기,
  번역/요약/재작성 지시, 증명·단계별 추론 요구, 기하·확률·수열 어휘,
  AIME 문체 등을 포괄한다.
- **Word n-gram 해시** — 1-2gram을 `zlib.crc32`로 $2^{16}$ 빈에 부호 해싱.
  숫자 토큰은 `<num>`으로 정규화.
- **Char n-gram 해시** — 3-4gram을 numpy 벡터화 다항 롤링 해시로 $2^{16}$
  빈에 부호 해싱(앞 4096자).

세 블록 모두 행 L2 정규화 후, word/char는 SVD로 각 128차원에 사영하여
트리 모델 입력(dense 81 + 128 + 128 = 337차원)을 만든다.

#### 3.2.2 블렌드 멤버 (Quality Predictor 대응)

전부 공개 Train 1,760문항만으로 학습했다. OOF score MSE는
`exp/registry.jsonl`(49개 실험 기록)에서 재현된다.

| 멤버 | 방식 | OOF score MSE |
| --- | --- | ---: |
| `xgb-mono` | XGBoost multi-output tree, hist, raw sparse 117,575열 | 0.1342 |
| `irt1d` | IRT: $\hat{r}_{i,j} = \sigma(a_j \theta_i + b_j)$, $\theta_i = W x_i$, L-BFGS | 0.1390 |
| `lgbm` | LightGBM 3-head 회귀, dense81 + SVD128×2 | 0.1387 |
| `knn-k40` | 코사인 kNN($k=40$, 가중치 $\text{sim}^3$) over word+char 해시 TF | 0.1501 |

최종 품질 예측은 네 멤버의 **균등 평균**이다. 스태킹(meta-LGBM)은 OOF와
full-fit의 분포 차이 때문에 오히려 역효과였다(0.6876, §4.4).

> **IRT ↔ LLM Identity Encoder.** `irt1d`는 IPR의 LLM Identity Encoder와
> 같은 역할을 훨씬 작은 파라미터로 수행한다. 문항의 잠재 난이도
> $\theta_i$는 프롬프트에서 선형으로 뽑고, 모델 $j$의 정체성은 판별도
> $a_j$와 절편 $b_j$ 단 두 개의 스칼라로 표현된다. IPR의 $e_c \in
> \mathbb{R}^{128}$에 대응하지만, 후보가 3개뿐이고 그 서열이 안정적인
> 본 과제에서는 2-파라미터로 충분하다. `irt.npz` 전체가 8 KB다.

### 3.3 비용 추정기 — IPR에 없는 구성요소

IPR은 비용을 가격표에서 읽는다. 본 과제는 출력 토큰 수가 사전에 알려지지
않으므로 비용을 예측해야 한다. **토큰 헤드를 분리**하는 방식을 쓴다.

모델 $j$, 문항 $i$에 대해 로그 공간에서 입력·출력 토큰을 각각 회귀한
$\widehat{\ell^{\text{in}}_{ij}}, \widehat{\ell^{\text{out}}_{ij}}$를 얻고,
공개 요율 $(\rho^{\text{in}}_j, \rho^{\text{out}}_j)$과 토큰 단위
$T = 10^6$로 결합한다.

$$\hat{v}_{ij} = \frac{\exp(\widehat{\ell^{\text{in}}_{ij}})\,\rho^{\text{in}}_j
+ \exp(\widehat{\ell^{\text{out}}_{ij}})\,\rho^{\text{out}}_j}{T}$$

LightGBM 계열과 XGBoost 계열로 각각 추정한 뒤 **로그 공간 평균**을 취하고,
$\hat{v}_{i,0} \le \hat{v}_{i,1} \le \hat{v}_{i,2}$ 단조성을 사후 강제한다
(`FinalRouter._monotone`). 비용을 직접 회귀하지 않고 토큰을 회귀하는 이유는,
요율이 정책으로 고정되어 있어 토큰 수가 유일한 미지수이고, 로그 공간에서
분산이 안정되기 때문이다.

**꼬리 위험 처리.** Premium 등급의 `axk1-think` 배분에는 평균 대신
**LightGBM 분위 회귀($\alpha = 0.90$)** 출력 토큰 추정치를 쓴다. 한 문항의
`axk1-think` 출력이 130k 토큰(= light 총비용의 78%)에 달하는 롱테일이
실측되었기 때문이다. 이는 IPR의 안전 마진 $\delta$를 **분포의 꼬리에
비대칭적으로** 적용한 형태로 볼 수 있다.

**정확 매칭 조회표.** 공개 Train+Dev 2,640문항의 SHA-256(프롬프트 원문) →
실측 score/cost 표를 이미지에 포함한다. 적중 시 예측을 실측값으로 대체한다.
규칙이 "정확한 프롬프트나 프롬프트 해시를 사용하는 공개 자료 조회"를 명시적으로
허용한다([CHALLENGE_RULES.md](CHALLENGE_RULES.md) §사용할 수 있는 정보).
프롬프트 원문은 저장하지 않고 해시만 보관한다(`lookup.npz`, 137 KB).
비공개 평가셋에서는 공개 문항이 섞인 비율만큼만 이득이 발생한다.

### 3.4 결정 최적화 — 순서 불변 배분

**[IPR §3.1 Algorithm 1]** IPR의 결정은 문항별 게이팅이므로 알고리즘이
한 줄이다. 본 과제는 다중선택 배낭문제를 풀어야 한다.

`src/ossp_router/allocation.py`에 세 배분기가 있다.

**(a) `lagrangian_allocate`** — 승수 $\lambda$에 대해 문항별로
$\hat{r}_{ij} - \lambda \hat{v}_{ij} \cdot n / V_{\text{light}}$를 최대화하는
$j$를 고르고, 예산을 만족하는 최소 $\lambda$를 이분 탐색(100회)한다.
연속 완화의 최적해이며 정수 갭만큼 손해를 본다.

**(b) `greedy_allocate`** — 승격 후보 $(i, j)$마다 비용 대비 품질 이득
$\Delta r / \Delta v$를 계산하고, 비율 내림차순으로 예산이 허용하는 한
승격한다. Fast·Balanced에 채택.

**(c) `two_stage_premium`** — 1단계에서 q90 비용 기준으로 `axk1-think`를
라그랑주 배분(사용률 0.65, 문항당 예측 k1 비용 0.1 credit 초과 시 제외),
2단계에서 남은 문항을 `ax31`로 greedy fill(사용률 0.70). Premium에 채택.

> **순서 불변성의 구조적 보장.** greedy 계열은 정렬을 쓰므로 순진하게
> 구현하면 동점 문항의 처리 순서가 입력 순서에 의존하게 되고, 이는 규칙이
> 금지하는 **입력 순서 의존**에 해당한다. 이를 막기 위해 배분기는 예측
> 서명 $(\text{round}(\Delta r/\Delta v, 12), \text{round}(\Delta r, 12),
> \text{round}(\Delta v, 12), j)$가 동일한 문항들을 **하나의 그룹으로 묶어
> 전부 승격시키거나 전부 보류**한다(`allocation.py:88-101`). 그룹 단위
> 예산 검사이므로 그룹 내부 순서가 결과에 영향을 주지 못한다. 정렬 키도
> 값 자체로만 구성되어 인덱스를 포함하지 않는다.

`tests/test_final_router.py`가 이 성질을 감사한다: 입력 역순, ID 개명,
`split`·`challenge_id` 변경, 반복 실행, 단일 문항 배치 — 다섯 경우 모두
프롬프트별 선택이 동일해야 통과한다.

### 3.5 위험 보정 — 안전 마진의 설계

**[IPR §3.1의 $\delta$]** IPR의 $\delta$는 스칼라 하나다. 하드 예산
체제에서는 마진을 **통계적으로 정당화**해야 한다. 정책 동결 전에 세
독립 게이트를 통과시켰다.

1. **부트스트랩** — Dev 880에서 복원추출 1,000회, 예산 초과 확률 측정.
2. **카테고리 시프트 스트레스** — 도메인 구성을 ±50% 리샘플 2,000회.
3. **CLT 잔차 모델** — Train 잔차 분포 기반 해석적 초과 확률.

동결된 최종 정책(`registry E049`)의 게이트 통과 수치:

| 등급 | 배분 | 파라미터 | 부트스트랩 초과확률 | 스트레스 | 여유(중앙값) | 여유(p99) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Fast | greedy, **k1 금지** | $u = 0.93$ | 0.221 % | 2.1 % | 5.6 % | 1.0 % |
| Balanced | greedy, **k1 금지** | $u = 0.88$ | 0.000 % | 0.0 % | 16.1 % | 7.0 % |
| Premium | 2단계 (q90 + 캡 0.1) | $u_{k1} = 0.65$, $u_{\text{fill}} = 0.70$ | 0.631 % | 1.6 % | 29.4 % | **2.6 %** |

> 초과확률은 2026-08-17 감사(`exp/audit/A10-significance-stability.md`)의
> 재측정값이다. 저장소가 이전에 기록한 0.1 % / 0.0 % / 0.5 %는 과소평가였다.
> **중앙값 기준 "여유"는 꼬리위험을 과소표현한다** — Premium의 p99 비용 비율은
> 3.89로 한도 4.00 대비 여유가 2.6 %에 불과하다.

Fast·Balanced에서 `axk1-think`를 **완전히 배제**한 것은 점수 손실을
감수하고 꼬리 위험을 0으로 만드는 선택이다. 실제로 §4.3의 Dev 실행에서
두 등급의 `axk1-think` 선택 수는 0이다.

**승자의 저주.** 배분기는 정의상 "비용 대비 이득이 커 보이는" 문항을
고르므로, 비용을 과소 예측한 문항이 선택될 확률이 높다. 선택 집합의 실현
비용이 예측 대비 **+32 %** 편향되는 것을 실측했다. 사용률 $u_t$는 이
편향을 흡수하는 마진이다.

---

## 4. 실험

### 4.1 데이터

**[IPR §4.1]** IPR은 11개 후보 모델에 대해 약 150만 프롬프트를 보상
모델(Skywork-Reward-Gemma-2-27B)로 주석한 IPRBench를 구축했다. 본 과제는
운영자가 제공한 고정 데이터를 쓴다.

| 구분 | 문항 수 | 내용 |
| --- | ---: | --- |
| 공개 Train | 1,760 | 프롬프트 + 모델별 `score`, `input_tokens`, `output_tokens`, `num_generations` |
| 공개 Dev | 880 | 동일 |
| 비공개 평가 | 미공개 | 구성·분할 기준 비공개 |

재배포 불가한 AIME 원문은 SHA-256으로 고정한 공개 출처에서 내려받아
`data/materialized/`(git 비추적)에 결합한다. 이 경로는 이미지에 포함되지
않는다.

### 4.2 실험 설정

**[IPR §4.2]** IPR의 기준선은 static/random/Budget-Aware Random/
RouteLLM-style/Oracle이다. 본 과제의 대응 기준선은 운영자가 제공한다.

| 기준선 | Fast | Balanced | Premium | 가중 최종 |
| --- | --- | --- | --- | ---: |
| all-light (하단) | 0.6193 / 1.000 | 동일 | 동일 | 0.6193 |
| prompt-heuristic | 0.6259 / 1.072 | 0.6582 / 1.368 | 0.6918 / 2.102 | 0.6553 |
| hash-regex (공식 최강) | 0.6631 / 1.236 | 0.6937 / 1.962 | 0.7401 / 3.985 | 0.6954 |
| budget-oracle (상한) | 0.7594 / 1.249 | 0.8074 / 1.989 | 0.8591 / 3.995 | 0.8037 |

총 49개 실험을 `exp/registry.jsonl`에 기록했고, 모든 수치는 공식 Decimal
채점기(`ossp_router.scoring`)로 산출했다.

### 4.3 결과

#### 4.3.1 동결 정책 (조회표 OFF)

| 항목 | Fast | Balanced | Premium | 가중 최종 |
| --- | --- | --- | --- | ---: |
| E041 (dev 탐색 최고, 참고) | 0.6744 / 1.237 | 0.6946 / 1.992 | 0.7460 / 3.935 | 0.7020 |
| **E049 (동결, 위험게이트)** | 0.6591 / 1.180 | 0.6918 / 1.677 | 0.7105 / 2.824 | **0.6843** |

#### 4.3.2 재현 실행 — 정직한 일반화 점수 (2026-08-17 독립 검증)

`.venv`(numpy 2.0.2 / scipy 1.17.1 / lightgbm 4.7.0 / xgboost 3.2.0)에서
materialize한 Dev 880을 세 등급에 돌리고 공식 self-check로 채점했다.
**조회표를 끈 번들**(`build/bundle-nolookup`)이 유일한 일반화 지표다.

| 등급 | tier_score | 비용 비율 | 한도 | 오라클 천장 | 달성률 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fast | 0.659091 | 1.180437 | 1.25 | 0.759469 | 86.8 % |
| Balanced | 0.691761 | 1.677356 | 2.0 | 0.807915 | 85.6 % |
| Premium | 0.710511 | 2.823888 | 4.0 | 0.859256 | 82.7 % |
| **최종** | **0.684318** | | | 0.803939 | **85.1 %** |

95 % 부트스트랩 신뢰구간은 **[0.656175, 0.710795]**이며, §4.3.1의 동결 정책
E049(0.6843)를 소수 4자리까지 재현한다. 즉 registry 기록은 원래부터 조회표를
끈 값이었고 과장되지 않았다.

> **조회표를 켜면 0.760284가 나오지만 이는 성능이 아니다.**
> `lookup.npz`는 공개 Train+Dev 2,640문항의 SHA-256 → 실측 outcome을 담고,
> 실측 적중률이 **train 1760/1760, dev 880/880 = 모두 100 %**다. 즉 Dev에서는
> 예측이 아니라 실측값을 되읽는다. 규정이 명시 허용하는 구성이지만
> ([`CHALLENGE_RULES.md`](CHALLENGE_RULES.md) §사용할 수 있는 정보), 공개 Dev
> 점수로는 일반화를 측정할 수 없다. 조회표는 비공개 평가셋에 공개 문항이
> 섞였을 때를 위한 보험으로만 의미가 있으며, 기대 점수는 적중률 0 %를 가정해
> 계산해야 한다. 상세는 [`../exp/audit-verdict.md`](../exp/audit-verdict.md).

**학습에 Dev를 쓰지 않았다.** `exp/build_final.py:80`은 `targets-train.npz`
하나만 로드하고, SVD(`:88-89`)·StandardScaler(`:197`)·LightGBM(`:108-109`)·
XGBoost(`:165`)가 전부 Train 특징으로만 적합된다. kNN 인덱스도 (1760, 131072)로
Train 전용이다. Dev가 들어가는 곳은 조회표(`:273-294`) 하나뿐이며 이는 파라미터
적합이 아니라 해시 캐시다.

#### 4.3.2.1 진짜 무정보 기준선

감사(`exp/audit/A7-overfit-falsification.md`)가 밝힌 사실이다.

| 조건 | Dev 가중 최종 |
| --- | ---: |
| all-light | 0.619318 |
| **랜덤 특징** + 비용모델 + 배분기 | **0.656297** ← 무정보 null |
| 라벨 셔플 + 비용모델 + 배분기 | 0.666515 |
| 출하 번들 (조회표 OFF) | 0.684318 |

라벨을 섞어도 특징을 난수로 바꿔도 점수가 붕괴하지 않는다. 이득의 대부분은
점수 예측이 아니라 **비용모델 + 예산 배분기**에서 나오며, 점수 모델의 실제
기여는 **+0.028**이다. 4멤버 블렌드만 떼면 +0.00159
(95 % CI [−0.00483, +0.00830], p(≤0)=0.31)로 0과 구별되지 않는다 — §3.2의
블렌드는 이 대회 지표 위에서 통계적으로 정당화되지 않는다.

#### 4.3.3 지연

**[IPR §4.3]** IPR은 라우팅 결정당 P90 85 ms / P99 108 ms를 보고한다.
본 시스템은 배치 단위로 측정한다.

| 환경 | 워크로드 | 등급당 벽시계 | 문항당 |
| --- | --- | ---: | ---: |
| 호스트 x64 네이티브 | Dev 880 | 7.3 – 11.1 s | 8 – 13 ms |
| `linux/arm64` 컨테이너 (QEMU 에뮬레이션) | Dev 880 | 59.9 – 67 s | 68 – 76 ms |

컨테이너 측정은 공식 자원 프로필(`--cpus 2 --memory 2g --memory-swap 2g
--pids-limit 32 --network none --read-only --ipc none --cgroupns private
--tmpfs /tmp:size=256m`)을 그대로 적용했다. 공식 환경은 네이티브 Apple
Silicon이므로 QEMU 수치는 상한으로 보아야 하며, 그 상한조차 90초 한도
안이다. 다만 비공개 평가셋의 문항 수가 Dev보다 크면 여유가 줄어든다 —
§6.1 참조.

### 4.4 절제 실험

**[IPR §H]** IPR은 손실 함수, 인코더 공유 방식, 라우팅 전략을 절제한다.
본 실험의 대응 결과는 다음과 같다.

**모델 축.** ridge/logistic < kNN < MLP < LightGBM ≈ IRT < XGBoost-mono.
서수 분류(0.6887)가 이진 분류(0.6761)보다 낫지만 회귀 대비 이득은 없었다.

**논문 기법 이식.** Δ-헤드(점수 이득 없음, 다만 **비용 토큰 헤드는 채택**),
IRT(1-D 채택, 2-D는 MSE만 개선), 라그랑주 배분(greedy가 우세),
conformal/CLT(**마진 설계에 채택**). 외부 데이터 사전학습은 kNN 유사도
중앙값이 이미 0.61로 공개 데이터 커버리지가 높아 기대 이득 대비
라이선스·재현 리스크가 커서 미채택.

**증강.** 표면 노이즈 +0.004(잡음 수준), 패러프레이즈·역번역은 라벨 보존
가정의 위험 대비 근거 부족으로 미채택.

**결합 방식.** 균등 블렌드 > 스태킹(0.6876). OOF와 full-fit의 분포 차이가
메타 학습기를 오도했다.

**가장 중요한 절제 — 곡선 최적점 대신 게이트 통과 정책.** dev 탐색 최고
구성(E041, 0.7020)은 세 등급 모두 예산 한도의 98–99 %를 쓴다. 공식
문서에 기록된 선례에서 hash-regex baseline이 dev 3.985 → 비공개셋
4.2(+5.4 %)로 드리프트하여 Premium 0점을 받았다. E041에 같은 드리프트가
오면 세 등급이 동시에 0점이 된다. 기대값 계산상 **0.6843을 확실히 얻는
쪽이 0.7020을 확률적으로 얻는 쪽보다 우월**하다. Train 내부 홀드아웃(Dev
미접촉)에서도 blend4 계열이 전 후보 중 최상위(0.6625–0.6629)여서, dev
선택 과적합이 아님을 확인했다.

---

## 5. 관련 연구와 위치

**[IPR §5]** IPR은 HybridLLM, RouteLLM, Zooter, GraphRouter, OmniRouter,
PickLLM 등 30여 접근을 정리하고, 산업 규모 감독(150만 보상 주석)과
프리즈-인코더 + 어댑터 기반 모듈식 확장성으로 차별화한다.

본 시스템의 위치는 다음과 같다.

- **IPR과 공유하는 것** — 생성 없는 프롬프트-only 품질 예측, 사용자 노출
  품질–비용 노브, 모델 정체성의 저차원 임베딩(IRT), 추정 오차를 흡수하는
  안전 마진.
- **IPR과 다른 것** — (i) 제약 방향이 반대(하드 예산 vs 품질 하한),
  (ii) 비용 자체를 예측(IPR은 가격표), (iii) 문항별 게이팅이 아니라 배치
  전역 배낭 배분, (iv) 신경 인코더 대신 해시 n-gram + 그래디언트 부스팅
  — 2 GiB·CPU-only·네트워크 차단 제약과 가중치 재배포 의무 회피 때문.
- **RouteLLM 계열과 다른 것** — 강/약 2모델 선호 데이터가 아니라 3모델
  절대 품질을 각각 회귀. IPR의 $\hat{r}_{i,c} = R_\theta(x_i, c)$ 형태를
  따른다.

---

## 6. 결론과 한계

프롬프트-only 라우팅을 하드 예산 체제로 옮기면, 문제는 "품질을 얼마나 잘
예측하는가"에서 "**비용 예측의 꼬리를 얼마나 잘 통제하는가**"로 이동한다.
품질 예측 성능이 가장 좋았던 구성이 최종 선택이 아니었다는 점(§4.4)이
이를 단적으로 보여준다.

### 6.1 한계

1. **조회표 이득의 불확실성.** Dev에서 관측된 0.7603은 조회표가 전 문항을
   적중시킨 상한이다. 비공개셋에서의 실제 이득은 공개 문항 혼입 비율에
   달려 있고, 그 비율은 알 수 없다. 조회표를 끈 0.6843을 하한으로 본다.
2. **평가셋 크기에 대한 시간 여유.** §4.3.3의 컨테이너 측정은 Dev 880
   기준이다. 로딩 고정비(호스트 6.0 s)가 크므로 문항 수가 늘어도 선형으로
   늘지는 않지만, 공식 환경에서의 최종 확인은 운영자 실행을 따른다.
3. **다중 턴 문맥 미반영.** IPR도 같은 한계를 든다. `messages` 입력은
   전체 텍스트를 이어붙여 처리하며, 턴 구조 자체는 메시지 수 특징 외에는
   쓰지 않는다.
4. **모델 집합 고정.** IPR의 모듈식 온보딩(어댑터 + 일관성 손실)에 대응하는
   기능이 없다. 후보가 3개로 고정된 과제 조건에서는 불필요하다.
5. **보상 모델 품질.** `score`는 운영자가 사전 계산해 제공한 값이며,
   그 산출 방식은 공개되지 않는다.

### 6.2 AI 모델 사용 고지

실행 이미지에는 **사전학습 언어모델 가중치를 포함하지 않는다.** 포함된
학습 산출물(`models/final-v1`, 142.5 MB)은 전부 공개 Train/Dev에서 유도한
그래디언트 부스팅 트리·선형 계수·SVD 성분·해시 색인이며, 본 저장소의
Apache-2.0으로 배포한다. 따라서 결과보고서의 AI 모델 항목에는
`해당 없음 — 실행 이미지에 AI 모델을 탑재하지 않음`이 적용된다.

### 6.3 라이선스

| 구성요소 | 버전 | 라이선스 | 용도 |
| --- | --- | --- | --- |
| numpy | 2.0.2 | BSD-3-Clause | 런타임 · 학습 |
| scipy | 1.17.1 | BSD-3-Clause | 런타임 · 학습 |
| lightgbm | 4.7.0 | MIT | 런타임 · 학습 |
| xgboost | 3.2.0 | Apache-2.0 | 런타임 · 학습 |
| scikit-learn | 1.9.0 | BSD-3-Clause | 학습 전용 (계수만 산출, 이미지 미포함) |
| `python:3.11-slim-bookworm` | digest 고정 | PSF + Debian 표준 구성요소 | 기반 이미지 |
| `models/final-v1` | v1 | Apache-2.0 (본 저장소) | 공개 Train/Dev에서만 유도 |

전부 [CHALLENGE_RULES.md](CHALLENGE_RULES.md) §최종 평가와 제출 저장소의
허용 목록(Apache-2.0 / MIT / BSD-2 / BSD-3 / ISC / 0BSD / BSL-1.0 / Zlib)
안에 있다.

---

## 부록 A. 재현 절차

[REPRODUCE.md](REPRODUCE.md)에 단계별 명령을 정리했다.

## 부록 B. IPR 절 대응표

| IPR 절 | 본 문서 | 대응 관계 |
| --- | --- | --- |
| §1 Introduction | §1 | 네 운영 제약 중 (iii)이 연속→불연속으로 바뀜 |
| §2.1 LLM Routing Formulation | §2.1 | Eq. 1의 쌍대 |
| §2.2 Routing Strategy | §2.2 | $\tau$ ↔ 등급, $\delta$ ↔ 사용률 |
| §2.3 Evaluation | §2.3 | Bounded-ARQGC의 3점 샘플 + 절벽 |
| §3.1 System Overview | §3.1 | QE / DO / Registry 3분할 유지 |
| §3.2 Quality Estimator | §3.2 | 신경 인코더 → 해시 n-gram + GBDT, LIE → IRT $(a_j, b_j)$ |
| — (해당 없음) | §3.3 | **비용 추정기 — 본 과제 고유** |
| §3.1 Algorithm 1 | §3.4 | 문항별 게이팅 → 배치 배낭 |
| §3.1의 $\delta$ | §3.5 | 스칼라 마진 → 삼중 위험 게이트 |
| §4.1 Dataset | §4.1 | IPRBench 150만 → 운영자 제공 2,640 |
| §4.2 Setup | §4.2 | 기준선 대응 |
| §4.3 Results | §4.3 | Bounded-ARQGC → 등급별 점수/비율 |
| §H Ablations | §4.4 | 손실·결합·전략 절제 |
| §5 Related Works | §5 | 위치 정리 |
| §6 Conclusions | §6 | 결론 |
| Limitations | §6.1 | 한계 |
| §F Cost Formula | §3.1 표 | 요율표 |
