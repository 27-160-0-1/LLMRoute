<!--
SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
SPDX-License-Identifier: Apache-2.0
-->

# A6 — Independent cost / budget re-implementation

**Date** 2026-08-17 · **Verdict: PASS — 0 disagreements in 161 compared fields**
(70 frozen-policy dev + 41 train + 50 registry-replay), plus 2 constructed
floating-point boundary cases and a 3-point `near_budget` threshold probe, all
in agreement.

A clean-room cost/budget/score calculator was written from `docs/SCORING.md`
alone and placed at `build/independent_scorer.py`. It reproduces every number
the official scorer publishes, to the full 12 displayed decimals and to the
exact unrounded `Decimal` where the official report exposes one.

`src/ossp_router/scoring.py` was **never opened**. It was *imported and called*
(§4, §6) purely as a black box to answer "what does the official scorer
actually do". Everything else — the formula, the rates, the multipliers, the
weights, the rounding rule, the `near_budget` threshold — came from the written
spec.

## Artifacts produced (all under `build/`, nothing outside it was touched)

| File | Purpose |
| --- | --- |
| `build/independent_scorer.py` | the clean-room calculator (Decimal, prec 160) |
| `build/a6_compare.py` | frozen policy ×2 + top-5 replayable registry experiments vs official |
| `build/a6_train_check.py` | train-split cross-check incl. an over-budget tier |
| `build/a6_boundary.py` | float-vs-Decimal boundary constructions |
| `build/a6_near_budget.py` | 95 % `near_budget` threshold probe |
| `build/a6_consistency.py` | all-49-row registry aggregation + `num_generations` counterfactual |
| `build/a6_double_rounding.py` | resolves the 7 apparent registry deltas |

---

## 1. What the spec pins down, and the two things it does not

Transcribed verbatim into `build/independent_scorer.py`:

```text
episode_cost = fixed_cost
             + input_tokens  * input_token_rate  / token_unit
             + output_tokens * output_token_rate / token_unit
```

| model | fixed | in-rate | out-rate |
| --- | ---: | ---: | ---: |
| `ax31-light` | 0 | 1 | 4 |
| `ax31` | 0 | 2.127 | 8.509 |
| `axk1-think` | 0 | 6.565 | 26.260 |

`token_unit = 1 000 000`; `budget_limit = light_baseline_cost * multiplier`;
Fast 1.25 / w 0.4, Balanced 2.0 / w 0.3, Premium 4.0 / w 0.3;
`tier_points_total = quality_points_total if budget_passed else 0`;
`final_score = (0.4·F + 0.3·B + 0.3·P) / n`; 160-digit `Decimal`; display
quantized at 12 dp `ROUND_HALF_EVEN` then trailing zeros stripped.

Two things the spec leaves ambiguous, and how they resolved:

1. **`num_generations` is in `outcomes.json` but not in the cost formula.**
   The formula literally has no such term, so the clean-room implementation
   ignores it. Confirmed empirically in §5 — including it inflates dev cost by
   ~2.3× and disagrees with every official `total_cost`.
2. **`near_budget` at exactly 95 %.** Spec says `한도의 95% 이상` ("95 % or
   more"), so `>=`. Probed directly in §6c — official agrees.

---

## 2. `light_baseline_cost`, computed independently

```
$ $env:PYTHONPATH='src'; .venv/Scripts/python.exe build/independent_scorer.py baseline dev
{
  "split": "dev",
  "num_episodes": 880,
  "light_baseline_cost_exact": "4.381428",
  "budget_limit_fast": "5.47678500",
  "budget_limit_balanced": "8.7628560",
  "budget_limit_premium": "17.5257120"
}

$ ... build/independent_scorer.py baseline train
{
  "split": "train",
  "num_episodes": 1760,
  "light_baseline_cost_exact": "8.603786",
  "budget_limit_fast": "10.75473250",
  "budget_limit_balanced": "17.2075720",
  "budget_limit_premium": "34.4151440"
}
```

| split | n | independent `light_baseline_cost` | official | source of official |
| --- | ---: | --- | --- | --- |
| dev | 880 | **4.381428** | 4.381428 | `build/dev-final-report.json` |
| train | 1760 | **8.603786** | 8.603786 | `score_submissions` on `build/a6-train/*` (§3c) |

Both baselines are exact to the last digit; no train report existed before this
audit, so the train figure was cross-checked by driving the official scorer on
a freshly built train submission.

---

## 3. Side-by-side against the official scorer

### 3a. Frozen policy — lookup ON (`build/dev-final/*.json`)

`.venv/Scripts/python.exe build/a6_compare.py`

| tier | field | INDEPENDENT | OFFICIAL (`dev-final-report.json`) | match |
| --- | --- | --- | --- | --- |
| fast | light_baseline_cost | 4.381428 | 4.381428 | OK |
| fast | budget_limit | 5.476785 | 5.476785 | OK |
| fast | total_cost | 5.069558494 | 5.069558494 | OK |
| fast | budget_ratio | 1.157056214093 | 1.157056214093 | OK |
| fast | budget_passed | True | True | OK |
| fast | near_budget | False | False | OK |
| fast | quality_points_total | 650 | 650 | OK |
| fast | quality_score | 0.738636363636 | 0.738636363636 | OK |
| fast | tier_points_total | 650 | 650 | OK |
| fast | tier_score | 0.738636363636 | 0.738636363636 | OK |
| fast | model_counts | 152 / 728 / 0 | 152 / 728 / 0 | OK |
| balanced | budget_limit | 8.762856 | 8.762856 | OK |
| balanced | total_cost | 5.279213067 | 5.279213067 | OK |
| balanced | budget_ratio | 1.204906954308 | 1.204906954308 | OK |
| balanced | quality_points_total | 653 | 653 | OK |
| balanced | tier_score | 0.742045454545 | 0.742045454545 | OK |
| balanced | model_counts | 158 / 722 / 0 | 158 / 722 / 0 | OK |
| premium | budget_limit | 17.525712 | 17.525712 | OK |
| premium | total_cost | 9.142871128 | 9.142871128 | OK |
| premium | budget_ratio | 2.08673316736 | 2.08673316736 | OK |
| premium | quality_points_total | 710.5 | 710.5 | OK |
| premium | tier_score | 0.807386363636 | 0.807386363636 | OK |
| premium | model_counts | 140 / 668 / 72 | 140 / 668 / 72 | OK |
| FINAL | final_weighted_points_total | 669.05 | 669.05 | OK |
| FINAL | **final_score** | **0.760284090909** | **0.760284090909** | **OK** |

Full unrounded value from the independent scorer, for the record:

```
final_score_exact = 0.7602840909090909090909090909090909090909090909090909…
budget_ratio_exact(balanced) = 1.20490695430804751327649341721466152131222971140915701456237555427…
```

### 3b. Frozen policy — lookup OFF (`build/dev-nolookup/*.json`)

| tier | field | INDEPENDENT | OFFICIAL (`dev-nolookup-report.json`) | match |
| --- | --- | --- | --- | --- |
| fast | total_cost | 5.172000263 | 5.172000263 | OK |
| fast | budget_ratio | 1.180437123011 | 1.180437123011 | OK |
| fast | tier_points_total | 580 | 580 | OK |
| fast | tier_score | 0.659090909091 | 0.659090909091 | OK |
| balanced | total_cost | 7.349213591 | 7.349213591 | OK |
| balanced | budget_ratio | 1.677355782407 | 1.677355782407 | OK |
| balanced | tier_points_total | 608.75 | 608.75 | OK |
| balanced | tier_score | 0.691761363636 | 0.691761363636 | OK |
| premium | total_cost | 12.372663763 | 12.372663763 | OK |
| premium | budget_ratio | 2.823888413321 | 2.823888413321 | OK |
| premium | tier_points_total | 625.25 | 625.25 | OK |
| premium | tier_score | 0.710511363636 | 0.710511363636 | OK |
| FINAL | final_weighted_points_total | 602.2 | 602.2 | OK |
| FINAL | **final_score** | **0.684318181818** | **0.684318181818** | **OK** |

11 fields × 3 tiers × 2 runs + 2 final fields × 2 runs = **70 fields, 0 mismatches.**

### 3c. Train split, including a genuinely over-budget tier

`build/a6_train_check.py` builds a content-free deterministic routing
(`i%5`, `i%3`, `i%7`/`i%2`) over all 1760 train episodes and scores it with
both calculators. Premium deliberately blows the 4× limit.

| tier | field | INDEPENDENT | OFFICIAL | match |
| --- | --- | --- | --- | --- |
| fast | light_baseline_cost | 8.603786 | 8.603786 | OK |
| fast | budget_limit | 10.7547325 | 10.7547325 | OK |
| fast | total_cost | 10.582553523 | 10.582553523 | OK |
| fast | budget_ratio | 1.229987998655 | 1.229987998655 | OK |
| fast | near_budget | True | True | OK |
| fast | tier_score | 0.616193181818 | 0.616193181818 | OK |
| balanced | total_cost | 11.839654527 | 11.839654527 | OK |
| balanced | budget_ratio | 1.376098211532 | 1.376098211532 | OK |
| balanced | tier_score | 0.622727272727 | 0.622727272727 | OK |
| **premium** | total_cost | 36.780910108 | 36.780910108 | OK |
| **premium** | budget_ratio | 4.274968032445 | 4.274968032445 | OK |
| **premium** | budget_passed | **False** | **False** | OK |
| **premium** | over_budget_zero_applied | **True** | **True** | OK |
| **premium** | quality_points_total | **1133** (preserved) | **1133** | OK |
| **premium** | quality_score | **0.64375** (preserved) | **0.64375** | OK |
| **premium** | tier_points_total | **0** (zeroed) | **0** | OK |
| **premium** | tier_score | **0** | **0** | OK |
| FINAL | final_weighted_points_total | 762.6 | 762.6 | OK |
| FINAL | final_score | 0.433295454545 | 0.433295454545 | OK |

13 fields × 3 tiers + 2 final = **41 fields, 0 mismatches**. This confirms the spec sentence
`예산을 초과하면 품질 점수 합계는 감사용으로 보존하되, 최종 점수에 반영하는
등급 점수 합계는 0입니다` — `quality_*` survives, `tier_*` is zeroed, and the
final score still counts the two passing tiers.

### 3d. Top-5 registry experiments

Ranking `exp/registry.jsonl` by `weighted_final` gives
E004 (0.803693) · E041 (0.701960) · E039 (0.701506) · E044 (0.701364) ·
E040 (0.700994). **Only E039 of those five has its per-episode decisions
still on disk.** E004/E041/E044/E040/E046 were registered with
`"artifacts": []`, and the `build/preds/*.npz` files that the decision-family
runs consumed no longer exist, so their decision vectors are not recoverable.

Two complementary checks were therefore run.

**(i) Full per-episode replay** for the top 5 experiments that *do* keep
decisions (`exp/E*/detail.npz['picks_dev']`, an int8 (3, 880) array indexed
into `('ax31-light','ax31','axk1-think')`, rows in fast/balanced/premium
order). These are E039, E014, E017, E011, E010. Registry `score` is the
official `quality_score` and `cost_ratio` is the official `budget_ratio`
(`exp/registry_lib.py:metrics_from_report`), both produced by
`ossp_router.scoring` via `exp/harness.py:evaluate`.

| exp | name | tier | IND quality_score | REG score | m | IND budget_ratio | REG cost_ratio | m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E039 | xgb-mono+lagrangian | fast | 0.673295454545 | 0.673295454545 | OK | 1.246978817408 | 1.246978817408 | OK |
| E039 | | balanced | 0.697727272727 | 0.697727272727 | OK | 1.789707813297 | 1.789707813297 | OK |
| E039 | | premium | 0.742897727273 | 0.742897727273 | OK | 3.990814093944 | 3.990814093944 | OK |
| E039 | | **weighted** | **0.701505681818** | **0.701505681818** | **OK** | | | |
| E014 | irt1d+lagrangian | fast | 0.665340909091 | 0.665340909091 | OK | 1.236250206554 | 1.236250206554 | OK |
| E014 | | balanced | 0.697159090909 | 0.697159090909 | OK | 1.963204578964 | 1.963204578964 | OK |
| E014 | | premium | 0.731818181818 | 0.731818181818 | OK | 3.983409010715 | 3.983409010715 | OK |
| E014 | | **weighted** | **0.694829545455** | **0.694829545455** | **OK** | | | |
| E017 | irt2d+lagrangian | fast | 0.667329545455 | 0.667329545455 | OK | 1.246419567319 | 1.246419567319 | OK |
| E017 | | balanced | 0.691193181818 | 0.691193181818 | OK | 1.952033875942 | 1.952033875942 | OK |
| E017 | | premium | 0.731534090909 | 0.731534090909 | OK | 3.915564586249 | 3.915564586249 | OK |
| E017 | | **weighted** | **0.69375** | **0.69375** | **OK** | | | |
| E011 | knn-k20+greedy | fast | 0.664772727273 | 0.664772727273 | OK | 1.24687695313 | 1.24687695313 | OK |
| E011 | | balanced | 0.694034090909 | 0.694034090909 | OK | 1.926081884034 | 1.926081884034 | OK |
| E011 | | premium | 0.725852272727 | 0.725852272727 | OK | 3.937880570444 | 3.937880570444 | OK |
| E011 | | **weighted** | **0.691875** | **0.691875** | **OK** | | | |
| E010 | knn-k20+lagrangian | fast | 0.664772727273 | 0.664772727273 | OK | 1.234790856314 | 1.234790856314 | OK |
| E010 | | balanced | 0.691761363636 | 0.691761363636 | OK | 1.96945732647 | 1.96945732647 | OK |
| E010 | | premium | 0.723579545455 | 0.723579545455 | OK | 3.951374387300 | 3.9513743873 | OK |
| E010 | | **weighted** | **0.690511363636** | **0.690511363636** | **OK** | | | |

`budget_pass` also matched on all 15 tier rows. 3 fields × 3 tiers × 5 experiments
+ 5 weighted finals = **50 fields, 0 mismatches.**

**(ii) Aggregation check on all 49 rows** including the unreplayable top-5
(`build/a6_consistency.py`). Independent 0.4/0.3/0.3 weighting reproduces
`weighted_final` on 42/49 rows exactly, including E004 0.803693181818,
E041 0.701960227273, E044 0.701363636364, E046 0.700909090909.
The 7 that differ by exactly 1 in the 12th decimal (E007, E014, E016, E019,
E024, E033, E040) are a **double-rounding artefact of the check, not of the
scorer** — see §7. Note E014 is in that list yet its *full replay* above
matched exactly, which is itself the proof.

---

## 4. The budget comparison is on the UNROUNDED `Decimal`

> `docs/SCORING.md` — "모든 비용·품질 연산은 160자리 `Decimal` 문맥에서
> 수행합니다. **비용 한도는 반올림하기 전 값으로 비교합니다.**"
> and "비용이 한도와 정확히 같으면 통과합니다. 한도를 조금이라도 초과하면
> 해당 등급의 점수는 `0`입니다."

`build/a6_boundary.py` builds two synthetic 2-episode splits and hands them to
the **official** `score_submissions`. Every cost is an exact integer number of
nano-credits (rates carry 3 decimals, `token_unit` is 1e6), and
`gcd(2127, 8509) = 1`, so any target cost can be hit exactly.

### Case A — exact tie: Decimal PASS, float FAIL

ep1 light `in=2`, ep1 ax31 `in=1000`, ep2 light `in=2123`; balanced tier picks
ax31 on ep1.

```
EXACT total_cost   = 0.00425 credits   (4 250 000 nano)
EXACT budget_limit = 0.00425 credits   (4 250 000 nano)
EXACT total-limit  = 0                 (0 nano)

naive-float total       = 0.00425
naive-float limit       = 0.0042499999999999994
naive-float total-limit = 8.673617379884035e-19      -> float says OVER

float (naive IEEE-754) verdict : FAIL
Decimal (spec) verdict         : PASS
    official budget_limit             0.00425
    official total_cost               0.00425
    official budget_ratio             2
    official budget_passed            True
    official near_budget              True
    official tier_points_total        2
    official tier_score               1
  OFFICIAL scorer verdict        : PASS
  >>> OFFICIAL AGREES WITH       : Decimal
```

A float implementation would have zeroed a tier that the spec says passes.

### Case B — 1 nano-credit over: Decimal FAIL, float PASS

To make float *miss* a real overrun the magnitudes must be large enough that
one double ulp exceeds 1e-9 credits (~2.5e7 credits). ep1 light `in=1e13`,
ep1 ax31 `in=7052186176931 out=196`, ep2 light `in=1e13`; fast tier (×1.25).

```
EXACT total_cost   = 25000000.000000001 credits   (25 000 000 000 000 001 nano)
EXACT budget_limit = 25000000           credits   (25 000 000 000 000 000 nano)
EXACT total-limit  = 1E-9 credits                 (1 nano)

naive-float total       = 25000000.0
naive-float limit       = 25000000.0
naive-float total-limit = 0.0                        -> float says WITHIN

float (naive IEEE-754) verdict : PASS
Decimal (spec) verdict         : FAIL
    official light_baseline_cost      20000000
    official budget_limit             25000000
    official total_cost               25000000.000000001
    official budget_ratio             1.25
    official budget_passed            False
    official near_budget              True
    official over_budget_zero_applied True
    official quality_points_total     2
    official tier_points_total        0
    official tier_score               0
  OFFICIAL scorer verdict        : FAIL
  >>> OFFICIAL AGREES WITH       : Decimal
  official final_score           : 0.6
```

**Both directions constructed; the official scorer follows `Decimal` in both.**

> **Trap worth flagging.** In Case B the *displayed* `budget_ratio` is
> `"1.25"` — exactly the multiplier — because the 12-dp quantization of
> 1.25000000000000005 rounds to 1.25, while `budget_passed` is `False`.
> A reviewer eyeballing the published ratio against the limit would conclude
> the tier passed. Only `budget_passed` / `over_budget_zero_applied` are
> authoritative. This is precisely why the spec says the comparison uses the
> value *before* rounding.

Independent scorer on the same two splits (`build/a6_consistency.py` part c):

```
caseA-exact-tie     tier=balanced total=0.004250            passed=True   tier_score=1
    exact total-limit = 0.0000000    (float would compare 0.00425 <= 0.00425 -> True)
caseB-nano-overrun  tier=fast     total=25000000.000000001  passed=False  tier_score=0
    exact total-limit = 0.000000001  (float would compare 25000000.0 <= 25000000.0 -> True)
```

---

## 5. `num_generations` is not a cost multiplier

`outcomes.json` carries `num_generations` (dev totals 2006 selections over 880
episodes) but the spec formula has no such factor. Counterfactual:

| run | tier | spec formula | × num_generations | official `total_cost` | spec matches |
| --- | --- | ---: | ---: | ---: | --- |
| lookup ON | fast | 5.069558494 | 12.715149082 | 5.069558494 | True |
| lookup ON | balanced | 5.279213067 | 13.134458228 | 5.279213067 | True |
| lookup ON | premium | 9.142871128 | 20.861774350 | 9.142871128 | True |
| lookup OFF | fast | 5.172000263 | 12.749934272 | 5.172000263 | True |
| lookup OFF | balanced | 7.349213591 | 18.150348930 | 7.349213591 | True |
| lookup OFF | premium | 12.372663763 | 28.162028840 | 12.372663763 | True |

`input_tokens` / `output_tokens` in `outcomes.json` are already the per-episode
totals. `num_generations` is reporting metadata only.

---

## 6. `near_budget` threshold probe (95 %)

Spec: "self-check는 등급 한도의 **95% 이상**을 사용하면 `near_budget`을
표시합니다." — inclusive. `build/a6_near_budget.py` constructs exact ratios:

```
exactly-95pct      total/limit = 0.0070357   / 0.007406 = 0.95
    official budget_passed=True  near_budget=True
just-under-95pct   total/limit = 0.028499997 / 0.03     = 0.9499999
    official budget_passed=True  near_budget=False
just-over-95pct    total/limit = 0.028500003 / 0.03     = 0.9500001
    official budget_passed=True  near_budget=True
```

Inclusive `>=`, as the spec wording implies and as the clean-room
implementation guessed. **0 disagreements.**

---

## 7. The 7 apparent registry deltas — resolved, no scorer bug

Re-aggregating the *published 12-dp tier scores* with 0.4/0.3/0.3 gives a
value 1 unit off in the 12th decimal for E007, E014, E016, E019, E024, E033,
E040. That is double rounding introduced by the check, not by the scorer: the
spec's formula divides the **exact** points sum, rounding once at the end.

Per-episode outcome scores are drawn from `{0, 0.25, 0.5, 0.75, 1}` (verified
over train+dev: 2002 / 26 / 671 / 32 / 5189 occurrences), so
`quality_points_total` is always a multiple of 0.25 and is recoverable from a
published 12-dp score. Recomputing from those exact points
(`build/a6_double_rounding.py`):

| exp | recovered exact points f/b/p | agg-of-ROUNDED | agg-of-EXACT | registry | |
| --- | --- | --- | --- | --- | --- |
| E007 | 582 / 598.5 / 631.75 | 0.683948863637 | 0.683948863636 | 0.683948863636 | OK |
| E014 | 585.5 / 613.5 / 644 | 0.694829545454 | 0.694829545455 | 0.694829545455 | OK |
| E016 | 583.75 / 603.5 / 635.25 | 0.687642045454 | 0.687642045455 | 0.687642045455 | OK |
| E019 | 583.5 / 614 / 645.25 | 0.694517045454 | 0.694517045455 | 0.694517045455 | OK |
| E024 | 581 / 596.75 / 633.75 | 0.683579545454 | 0.683579545455 | 0.683579545455 | OK |
| E033 | 584.75 / 594.25 / 632.25 | 0.683920454546 | 0.683920454545 | 0.683920454545 | OK |
| E040 | 589.5 / 614.5 / 655.75 | 0.700994318181 | 0.700994318182 | 0.700994318182 | OK |

**7 / 7 explained.** The official scorer is right; the naive re-aggregation is
wrong. Practical consequence: **never reconstruct a final score by weighting
published tier scores** — use `tier_points_total`.

---

## 8. Nothing disagreed, so the "arbitrate by quoting the spec" step is moot

No arbitration was needed: the independent implementation and the official
scorer agree on every field of every comparison, and on both sides of both
constructed floating-point boundary cases. The two spec ambiguities (§1) were
resolved by direct empirical probe and both landed on the reading the spec's
wording implies.

## 9. Scope limits (not verified)

- **E004 / E041 / E044 / E040 / E046 were not replayed per-episode.** Their
  decision vectors are gone (`artifacts: []`, `build/preds/` absent). They were
  only checked for weighting consistency (§3d-ii). If the lead needs a true
  per-episode replay of the top-5 *overall*, those runs must be regenerated
  first.
- This track verified the **scorer**, not the router. It says nothing about
  whether 0.760284 is generalization (it is not — see the lookup analysis) or
  whether 0.684318 will hold on the private set.
- Only the three shipped model ids and the three shipped tiers were exercised;
  a policy with a non-zero `fixed_cost` was never scored, because no such
  policy exists in the repo.
- **Out-of-scope observation for the lead, not caused by A6:**
  `git status` shows `models/final-v1/manifest.json` (mtime 13:36:56) and
  `setup.cfg` (mtime 13:44:03) as modified. Both are on the declared read-only
  list. A6 wrote only to `build/*` (earliest 14:05:03) and
  `exp/audit/A6-cost-reimplementation.md`, so these predate this track — but
  the session-start snapshot reported the tree as clean, so someone should
  reconcile that.
