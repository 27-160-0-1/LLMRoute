<!--
SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
SPDX-License-Identifier: Apache-2.0
-->

# A10 — Statistical significance and distribution stability

Auditor track A10. Every number below was produced by a script in `build/` against the
frozen artifacts; scripts and raw JSON are listed in §7. Nothing under `models/final-v1/`,
`src/`, `container/`, `tests/`, `data/` was modified.

**Verdict: PARTIAL.** The Dev score is reproducible and its sampling CI is tight, but three
claims in the repo do not survive audit:

1. The repo's per-tier bootstrap overrun probabilities (0.1% / 0.0% / 0.5%) are **understated**.
   High-precision re-measurement gives **0.221% / 0.000% / 0.631%**.
2. The **Premium** tier's 99.9th-percentile cost ratio under an i.i.d. bootstrap is **4.354**,
   i.e. it **exceeds the 4.0 cap**; under a ±50% category-composition shift the **99th** percentile
   is **4.276**, also over cap. The Premium tier is not as safe as the frozen-policy notes imply.
3. The registry leaderboard near the top is **noise plus budget-cliff gambling**, not quality.
   E039/E014/E017 (dev 0.7015 / 0.6948 / 0.6938) sit within 0.01 of their caps and are zeroed
   in **47–70%** of bootstrap resamples.

The leakage smell test in §4 came back **negative**: after the lookup table is disabled there is
no localized-memorization signal in any domain.

---

## 0. Replication check (precondition for everything else)

A float re-implementation of the official rule reproduces both shipped reports to 12 decimals,
so bootstrap resampling on the float path is legitimate.

```
$env:PYTHONPATH='src'; .venv/Scripts/python.exe -c "... build/a10_lib.py ..."
n 880 models ['ax31-light', 'ax31', 'axk1-think']
dev-final    final=0.760284090909
    fast score=0.738636363636 ratio=1.157056214093 pass=True
    balanced score=0.742045454545 ratio=1.204906954308 pass=True
    premium score=0.807386363636 ratio=2.086733167360 pass=True
dev-nolookup final=0.684318181818
    fast score=0.659090909091 ratio=1.180437123011 pass=True
    balanced score=0.691761363636 ratio=1.677355782407 pass=True
    premium score=0.710511363636 ratio=2.823888413321 pass=True
```

Matches `build/dev-final-report.json` (0.760284090909) and
`build/dev-nolookup-report.json` (0.684318181818) exactly.

Independently, the frozen knobs were replayed through `FinalRouter.predict_batch` +
`ossp_router.allocation` on the lookup-disabled bundle:

```
"replayed_dev_final": 0.6843181818181818,
"shipped_dev_final":  0.684318181818,
"per_episode_picks_identical_to_shipped": true
```

---

## 1. Bootstrap CI of the weighted final score — lookup-ON vs lookup-OFF

Dev 880 episodes resampled with replacement. B = 200,000, seed 20260817, **paired**
(both policies scored on the same resample). The light baseline cost is recomputed inside
each resample, and the budget-zero rule is applied, exactly as the official scorer does.

| policy | point | boot mean | 2.5% | 97.5% | sd | P(a tier zeroed) |
|---|---:|---:|---:|---:|---:|---:|
| **lookup-ON** (`build/dev-final`) | 0.760284090909 | 0.760174 | **0.734318** | **0.785739** | 0.014597 | 0.0495% |
| **lookup-OFF** (`build/dev-nolookup`) | 0.684318181818 | 0.682439 | **0.653864** | **0.711875** | 0.025648 | 0.822% |

Quality-only (budget-zero rule suppressed — isolates sampling noise in the score itself):

| policy | mean | 2.5% | 97.5% | sd |
|---|---:|---:|---:|---:|
| lookup-ON | 0.760322 | 0.734432 | 0.785739 | 0.013106 |
| lookup-OFF | 0.684358 | 0.656250 | 0.711989 | 0.014205 |

Note the lookup-OFF budgeted sd (0.02565) is **1.8×** its quality-only sd (0.01420). The extra
variance is entirely the Premium budget cliff, not score noise.

Per-tier quality CIs (B = 1000 replicate set, `build/a10_sec123.json`):

| tier | lookup-ON mean [2.5%, 97.5%] | lookup-OFF mean [2.5%, 97.5%] |
|---|---|---|
| fast | 0.738515 [0.711357, 0.765625] | 0.658918 [0.627266, 0.688359] |
| balanced | 0.741957 [0.713920, 0.768473] | 0.691816 [0.665050, 0.719041] |
| premium | 0.807656 [0.782379, 0.830398] | 0.710506 [0.682102, 0.737216] |

### CI of the difference (paired, B = 1000)

```
"boot_diff": {"mean": 0.078565, "sd": 0.027017,
              "lo": 0.065050, "hi": 0.090967,   # 2.5 / 97.5 pct
              "p1": 0.062669, "p99": 0.099978},
"diff_frac_le0": 0.0
```

**ON − OFF = +0.0760 point, 95% CI [+0.0651, +0.0910], P(OFF ≥ ON) = 0/1000.**
The memorization advantage is far outside sampling noise. This is the size of the
lookup table's contribution and it is rule-allowed, but it is **not** generalization —
consistent with the lead's finding of a 100% hash hit rate on all 2640 public prompts.

### Reference: frozen lookup-OFF vs the official strongest baseline

I regenerated the `hash-regex` baseline on Dev to get its per-episode picks
(`baselines/hash_regex.py --artifact baselines/hash-regex-public.v1.json`), reproducing
its published 0.695369318182 exactly, then bootstrapped head-to-head (B = 200,000):

| policy | point | budgeted mean | budgeted [2.5%, 97.5%] | **P(a tier zeroed)** | margins to cap (F/B/P) |
|---|---:|---:|---|---:|---|
| frozen lookup-OFF | 0.684318 | 0.682439 | [0.653864, 0.711875] | **0.82%** | 0.0696 / 0.3226 / 1.1761 |
| hash-regex | 0.695369 | **0.399109** | **[0.000000, 0.714716]** | **66.35%** | 0.0140 / 0.0385 / 0.0148 |
| all-light | 0.619318 | 0.619379 | [0.588920, 0.649716] | 0.00% | 0.2500 / 1.0000 / 3.0000 |

Paired differences:

```
frozen_lookup_off - hash_regex:
  quality-only diff = -0.011073, 95% CI [-0.024290, +0.002102], P(frozen > hash-regex) = 5.00%
  budgeted    diff = +0.283330, 95% CI [-0.019602, +0.700227], P(frozen > hash-regex) = 68.48%
frozen_lookup_off - all_light:
  quality-only diff = +0.064979, 95% CI [+0.045682, +0.084432], P(frozen > all-light) = 100%
```

So: on **pure quality** the frozen router is behind hash-regex by 0.0111 and that deficit is
marginally significant (only 5.0% of resamples favour the router). On the **actual scoring rule**
the router wins two thirds of the time, purely because hash-regex is parked 0.0148 from the
Premium cap and is zeroed in 66% of resamples. Both statements should be reported together;
reporting only one is misleading.

---

## 2. Top registry experiments — the leaderboard near the top is noise

Registry ranking (`exp/registry.jsonl`, 49 entries). Top 10 by `weighted_final`:

| rank | exp | dev final | per-episode detail recoverable? | name |
|---:|---|---:|---|---|
| 1 | E004 | 0.803693 | no | budget-oracle (not a router) |
| 2 | E041 | 0.701960 | no | blend6x-cost3+best-decision |
| 3 | **E039** | 0.701506 | **yes** | xgb-mono+lagrangian |
| 4 | E044 | 0.701364 | no | blend4-cost3+best-decision |
| 5 | E040 | 0.700994 | no | blend7-cost3+best-decision |
| 6 | E046 | 0.700909 | no | blend4-cost2b+best-decision |
| 7 | E036 | 0.700114 | no | blend6-cost2+best-decision |
| 8 | E035 | 0.699943 | no | blend6-costlgbm+best-decision |
| 9 | E028/E034 | 0.699148 | no | blend5/6-costdelta+best-decision |
| 11 | E045 | 0.699063 | no | blend6b-cost3+best-decision |

Only 24 of 49 stored `detail.npz`. The top three **recoverable** experiments are E039 (0.701506),
E014 (0.694830), E017 (0.693750). All three replicate from `picks_dev` to ≤5e-13:

```
E039 registry 0.701505681818 -> replicated 0.7015056818181818 (delta +1.8e-13)
E014 registry 0.694829545455 -> replicated 0.6948295454545456 (delta -4.5e-13)
E017 registry 0.693750000000 -> replicated 0.6937500000000000 (delta  0.0)
```

Bootstrap (B = 200,000):

| exp | point | **budgeted** mean [2.5%, 97.5%] | quality-only mean [2.5%, 97.5%] | **P(any tier zeroed)** |
|---|---:|---|---|---:|
| E039 | 0.701506 | 0.443217 **[0.000000, 0.719972]** | 0.701466 [0.674176, 0.728295] | **67.29%** |
| E014 | 0.694830 | 0.400529 **[0.000000, 0.713523]** | 0.694808 [0.667330, 0.721733] | **68.58%** |
| E017 | 0.693750 | 0.397724 **[0.000000, 0.711903]** | 0.693729 [0.665994, 0.720881] | **69.66%** |
| E049 frozen | 0.684318 | 0.682439 [0.653864, 0.711875] | 0.684358 [0.656250, 0.711989] | **0.82%** |

Why: all three park at the caps.

| exp | fast ratio (cap 1.25) | balanced (cap 2.0) | premium (cap 4.0) | per-tier overrun prob F/B/P |
|---|---:|---:|---:|---|
| E039 | 1.246979 | 1.789708 | 3.990814 | 50.3% / 8.2% / 47.1% |
| E014 | 1.236250 | 1.963205 | 3.983409 | 40.3% / 40.2% / 46.7% |
| E017 | 1.246420 | 1.952034 | 3.915565 | 45.5% / 39.0% / 42.5% |
| frozen | 1.180437 | 1.677356 | 2.823888 | 0.221% / 0.000% / 0.631% |

**Plain statement, as requested:** the CIs of the top experiments overlap essentially completely.
Pairwise, on the quality axis, the 95% CI of every top-3 difference straddles zero:

```
E039 - E014: point +0.02614, 95% CI [-0.41107, +0.43816], P(E039>E014) = 0.662
E039 - E017: point +0.04170, 95% CI [-0.42930, +0.48114], P(E039>E017) = 0.673
E014 - E017: point +0.01556, 95% CI [-0.27728, +0.27477], P(E014>E017) = 0.546
```

CI overlap fraction of the narrower interval = **1.00** for all three pairs. The ordering
E039 > E014 > E017 carries no statistical content; a 0.0016–0.0080 gap sits inside a
±0.014 quality sd and a ±0.25 budget-cliff sd. **Registry ranking near the top is noise, and
worse, the leading entries are entries that bought their dev score by spending the budget
margin — they would score 0 on roughly two of every three alternative draws of the eval set.**

A direct selection experiment confirms the ranking is not informative. Re-selecting the best of
all 25 recoverable candidates independently on each of 1000 bootstrap resamples:

```
"dev_argmax_candidate": "E039", "dev_argmax_score": 0.701506
"frozen_rank_on_dev": 12 (of 25)
"most_selected": E039 281, E049-frozen 275, E023 132, E014 81, E017 51, E011 43
"selection_optimism_mean": -0.000320  (sd 0.026829)
```

The dev "winner" is chosen only 28% of the time; the frozen policy — ranked 12th on dev —
is chosen 27.5% of the time. The candidates are statistically indistinguishable.

---

## 3. Cost-ratio distribution and budget-overrun probability

### 3a. i.i.d. bootstrap, B = 200,000 (frozen lookup-OFF)

| tier | cap | point ratio | mean | sd | 2.5% | 97.5% | **p99** | **p99.9** | max | **overrun prob** | repo claim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fast | 1.25 | 1.180437 | 1.182097 | 0.021493 | 1.142946 | 1.226792 | 1.236277 | 1.257273 | 1.306684 | **0.221% ±0.011%** | 0.1% |
| balanced | 2.00 | 1.677356 | 1.683074 | 0.075302 | 1.536232 | 1.830432 | 1.856398 | 1.905024 | 1.996288 | **0.000%** | 0.0% |
| premium | 4.00 | 2.823888 | 2.840054 | 0.383594 | 2.210853 | 3.700893 | 3.901844 | **4.353972** | 5.360585 | **0.631% ±0.018%** | 0.5% |

**Discrepancy vs the repo's claim.** `exp/registry.jsonl` E049 records
`overrun_prob_bootstrap` 0.001 / 0.0 / 0.005. At B = 200,000 the true values are
**0.00221 / 0.00000 / 0.00631**. Fast is understated by 2.2× and Premium by 1.26×.
The repo values came from a B = 1000 run where the standard error on a 0.005 probability is
±0.0022 — the estimates were simply too noisy to quote at one significant figure. The
direction of the error is unfavourable (real risk is higher than reported), so this should be
corrected rather than left as-is. Absolute magnitudes remain small.

Two additional facts worth recording, neither of which appears in the repo notes:

* **Premium p99.9 = 4.354 > cap 4.0.** The Premium tier is over cap in the top 0.1% of
  i.i.d. draws. Fast p99.9 = 1.257 > cap 1.25 likewise. Balanced has real headroom
  (p99.9 = 1.905, max 1.996 — never crosses 2.0 in 200k draws).
* Distance to cap in sd units: fast **z = 3.14**, balanced **z = 4.10**, premium **z = 3.01**.
  Premium is the thinnest tier in standardized terms despite having the largest absolute margin.

For contrast, the lookup-ON policy has zero measured overruns on every tier
(fast 0.0515%, balanced 0.000%, premium 0.000%; premium p99.9 = 2.632).

### 3b. Category-shift stress: domain composition resampled ±50%

Each iteration draws one weight per domain from U(0.5, 1.5), renormalizes, and resamples 880
episodes with those probabilities (same design as `exp/stress_lib.py`, but with the finer
10-domain partition from §4 rather than the repo's 7-bucket `eda.categorize`). 20,000 iterations.

**Frozen lookup-OFF, 99th-percentile cost ratio vs cap:**

| tier | cap | p50 | p95 | **p99** | p99.9 | max | overrun prob |
|---|---:|---:|---:|---:|---:|---:|---:|
| fast | **1.25** | 1.180758 | 1.253919 | **1.287872** ❌ | 1.326869 | 1.383353 | 5.86% |
| balanced | **2.00** | 1.681382 | 1.847315 | **1.923985** ✅ | 2.001405 | 2.052377 | 0.115% |
| premium | **4.00** | 2.796012 | 3.771866 | **4.276486** ❌ | 4.923501 | 6.198077 | 2.54% |

**Under a ±50% composition shift, Fast and Premium both breach their caps at the 99th
percentile.** Only Balanced holds (p99 = 1.924, 3.8% below cap).

Cross-check with the repo's own coarser 7-bucket `eda.categorize` partition (`math` 190,
`code` 147, `mcq` 230, `english-general` 245, `long-context` 54, `korean-general` 9,
`translate` 5) — same conclusion, milder because the buckets are coarser:

| tier | cap | p99 | overrun prob |
|---|---:|---:|---:|
| fast | 1.25 | **1.261441** ❌ | 2.28% |
| balanced | 2.00 | 1.888718 ✅ | 0.02% |
| premium | 4.00 | **4.082154** ❌ | 1.43% |

E049's recorded `stress_over` = {fast 0.021, balanced 0.0, premium 0.016} matches the repo-partition
run (2.28% / 0.02% / 1.43%) closely, so the repo's stress harness is sound — it is the *partition*
that is optimistic. The finer partition, which better reflects the actual 10 public data sources,
gives 5.86% / 0.115% / 2.54%.

Harsher shifts:

| shift | fast p99 (cap 1.25) | fast over | balanced p99 (cap 2.0) | balanced over | premium p99 (cap 4.0) | premium over |
|---|---:|---:|---:|---:|---:|---:|
| ±50% | 1.2879 | 5.86% | 1.9240 | 0.115% | 4.2765 | 2.54% |
| ±75% | 1.3385 | 12.84% | 1.9978 | 0.945% | 4.6972 | 5.05% |
| ±90% | 1.3832 | 17.47% | 2.0675 | 2.285% | 5.2412 | 7.82% |

Deterministic worst case — double one domain's mass, leave the rest:

| tier | cap | worst domain to double | resulting ratio |
|---|---:|---|---:|
| fast | 1.25 | code-cruxeval | 1.2171 ✅ |
| balanced | 2.00 | logic-ruletaker | 1.7598 ✅ |
| premium | 4.00 | code-cruxeval | 3.4865 ✅ |

No single-domain doubling breaks a cap. The ±50% p99 breaches come from *several* expensive
domains being up-weighted simultaneously.

**Effect on the final score under ±50% shift:**

```
quality-only weighted final : mean 0.684036, [2.5%, 97.5%] = [0.638125, 0.728013]
conditional on all 3 tiers passing (92.92% of draws): mean 0.683870, [0.638068, 0.728011]
unconditional (budget rule applied): mean 0.662855, p50 0.681818, [0.414375, 0.727445]
P(any tier zeroed) = 7.075%
```

The quality itself is stable under composition shift (±0.045). The tail is entirely the budget cliff.

---

## 4. Domain decomposition — leakage smell test → **NEGATIVE**

### 4a. Partition and its validation

Content heuristics only (`build/a10_domains.py`); no outcome, cost or hash information is used.
Labels are named after the likely public source family (`data/sources/source-pins.v1.json` pins
10 sources: aime24, aime25, belebele-korean, cruxeval, gsm8k, hrmcr, ruletaker,
truthfulqa-binary, babilong-4k-16k, deepmind-mathematics).

**Independent validation of the partition.** The 12 AIME episodes are exactly recoverable as
`materialized/dev/inputs.json` (880) minus `data/dev/inputs-base.json` (868), because AIME is
source-fetch-only:

```
extra = ['dev-0008','dev-0031','dev-0059','dev-0140','dev-0149','dev-0240',
         'dev-0245','dev-0394','dev-0420','dev-0421','dev-0422','dev-0423']
heuristic aime precision = 1.000   recall = 0.917 (11/12, missed dev-0140)
```

The purely textual heuristic recovers 11 of 12 with **zero false positives** out of 880. The
ground-truth AIME set is used for the AIME row below. Residual `other` bucket = 8 episodes (0.9%).

### 4b. Frozen lookup-OFF tier scores per domain vs the all-light baseline

`weighted` = 0.4·fast + 0.3·balanced + 0.3·premium within the domain.

| domain | n | all-light | all-ax31 | all-k1 | oracle | **OFF weighted** | OFF fast/bal/prem | **lift vs light** | 95% CI of lift | z | headroom captured |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| mc-truthfulqa | 83 | 0.7048 | 0.8434 | 0.9036 | 0.9277 | 0.8470 | .8434/.8434/.8554 | **+0.1422** | [+0.072, +0.220] | 3.77 | 63.8% |
| math-symbolic-dm | 158 | 0.4193 | 0.5475 | 0.8481 | 0.8766 | 0.5358 | .6177*/.6608*/.6994* | **+0.1165** | [+0.068, +0.168] | 4.56 | 25.5% |
| other | 8 | 0.5000 | 0.5625 | 0.8125 | 0.8750 | 0.5625 | — | +0.0625 | [0.000, +0.188] | 1.09 | 16.7% |
| ko-reason-hrmcr | 20 | 0.0000 | 0.0750 | 0.0750 | 0.1250 | 0.0550 | — | +0.0550 | [0.000, +0.120] | 1.82 | 44.0% |
| code-cruxeval | 119 | 0.4832 | 0.5210 | 0.8529 | 0.9118 | 0.5345 | .5168/.5294/.5630 | +0.0513 | [−0.021, +0.126] | 1.38 | 12.0% |
| ko-mc-belebele | 161 | 0.8230 | 0.8696 | 0.9348 | 0.9472 | 0.8652 | — | +0.0422 | [−0.009, +0.091] | 1.62 | 34.0% |
| logic-ruletaker | 137 | 0.7080 | 0.7701 | 0.7956 | 0.8942 | 0.7493 | .7153/.7737/.7701 | +0.0412 | [+0.006, +0.077] | 2.23 | 22.2% |
| math-word-gsm8k | 102 | 0.8725 | 0.9387 | 0.9559 | 0.9804 | 0.9132 | — | +0.0407 | [+0.014, +0.072] | 2.68 | 37.7% |
| longctx-babilong | 80 | 0.5000 | 0.5312 | 0.5375 | 0.7125 | 0.5319 | .5000/.5500/.5563 | +0.0319 | [0.000, +0.068] | 1.83 | 15.0% |
| math-comp-aime | 12 | 0.0208 | 0.0417 | 0.7292 | 0.7292 | 0.0333 | — | +0.0125 | [0.000, +0.038] | 1.04 | 1.8% |

(*coarse `math` rollup figures; per-domain tier splits are in `build/a10_sec45.json`.)

Coarse rollup:

| coarse | n | all-light | oracle | OFF weighted | OFF lift | ON weighted | ON lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 119 | 0.4832 | 0.9118 | 0.5345 | +0.0513 | 0.6643 | +0.1811 |
| korean | 181 | 0.7320 | 0.8564 | 0.7757 | +0.0436 | 0.8345 | +0.1025 |
| logic | 137 | 0.7080 | 0.8942 | 0.7493 | +0.0412 | 0.8431 | +0.1350 |
| long-context | 80 | 0.5000 | 0.7125 | 0.5319 | +0.0319 | 0.6031 | +0.1031 |
| math | 272 | 0.5717 | 0.9090 | 0.6551 | +0.0835 | 0.7250 | +0.1533 |
| multiple-choice | 83 | 0.7048 | 0.9277 | 0.8470 | +0.1422 | 0.8855 | +0.1807 |
| other | 8 | 0.5000 | 0.8750 | 0.5625 | +0.0625 | 0.5625 | +0.0625 |

### 4c. Investigation of the two anomalously-high domains

`mc-truthfulqa` (+0.1422, z = 3.77) and `math-symbolic-dm` (+0.1165, z = 4.56) sit well above the
+0.03…+0.05 band of the rest. Three tests, all of which come back clean:

**Test 1 — lift over the best *fixed* model inside the domain.** Memorization means per-episode
discrimination beyond what a domain label alone can buy. If the router merely learned "this domain
needs a bigger model", it cannot beat the best single model within the domain. Every domain is
**negative** on this test:

| domain | OFF weighted | best fixed model in-domain | **lift vs best fixed** |
|---|---:|---:|---:|
| longctx-babilong | 0.5319 | 0.5375 (k1) | −0.0056 |
| ko-reason-hrmcr | 0.0550 | 0.0750 | −0.0200 |
| math-word-gsm8k | 0.9132 | 0.9559 (k1) | −0.0426 |
| logic-ruletaker | 0.7493 | 0.7956 (k1) | −0.0464 |
| **mc-truthfulqa** | 0.8470 | 0.9036 (k1) | **−0.0566** |
| ko-mc-belebele | 0.8652 | 0.9348 (k1) | −0.0696 |
| other | 0.5625 | 0.8125 | −0.2500 |
| **math-symbolic-dm** | 0.5358 | 0.8481 (k1) | **−0.3123** |
| code-cruxeval | 0.5345 | 0.8529 (k1) | −0.3185 |
| math-comp-aime | 0.0333 | 0.7292 (k1) | −0.6958 |

There is **no domain in which the lookup-OFF router beats the best fixed in-domain model**. Both
"anomalous" domains are the ones where all-light is unusually weak relative to `axk1-think`
(0.7048 → 0.9036, and 0.4193 → 0.8481). The router is correctly detecting a domain-level signal
and is still leaving 0.06–0.31 on the table inside those domains. That is the signature of
generalization, not of memorization.

**Test 2 — dev→train near-duplicate retrieval.** The lookup-OFF bundle still ships
`knn/index.npz` (1760 train rows, word+char CSR, L2-normalized) and `knn/outcomes.npz`
(train realized per-model scores). `knn-k40` is one of the four blend members, so it is a **soft**
lookup that survives disabling `lookup.npz`. Cosine of each dev prompt to its nearest train row,
computed through the router's own feature path:

```
max_cos: mean 0.6331  p50 0.6119  p90 0.8996  p99 0.9789
frac > 0.90 = 9.89%   > 0.95 = 2.73%   > 0.99 = 0.57%   > 0.999 = 0.00%
```

Splitting dev by near-duplicate status and re-scoring the frozen lookup-OFF policy:

| threshold | near-dup n | near-dup lift vs light | novel n | novel lift vs light |
|---|---:|---:|---:|---:|
| cos > 0.90 | 87 | **+0.06494** | 793 | **+0.06501** |
| cos > 0.95 | 24 | +0.03958 | 856 | +0.06571 |
| cos > 0.99 | 5 | +0.16000 | 875 | +0.06446 |

At the only threshold with enough mass to be meaningful (0.90, n = 87), the lift on
near-duplicates and on novel episodes is **identical to 5 decimal places** (0.06494 vs 0.06501).
The kNN soft-lookup is not carrying the lookup-OFF result.

**Test 3 — near-duplicate density in the two anomalous domains.** If those domains were
memorized, they would be the ones packed with train near-duplicates. The opposite holds:

| domain | nn cos mean | nn cos p90 | frac > 0.95 | frac > 0.99 | OFF lift |
|---|---:|---:|---:|---:|---:|
| **mc-truthfulqa** | **0.403** (lowest of all) | 0.549 | 0.0% | 0.0% | **+0.1422** (highest) |
| math-symbolic-dm | 0.646 | 0.812 | 0.6% | 0.0% | +0.1165 |
| ko-reason-hrmcr | **0.942** (highest) | 0.992 | 50.0% | 20.0% | +0.0550 |
| longctx-babilong | 0.877 | 0.917 | 3.8% | 0.0% | +0.0319 |
| logic-ruletaker | 0.858 | 0.920 | 1.5% | 0.7% | +0.0412 |
| code-cruxeval | 0.644 | 0.859 | 5.9% | 0.0% | +0.0513 |
| ko-mc-belebele | 0.509 | 0.820 | 0.6% | 0.0% | +0.0422 |
| math-word-gsm8k | 0.440 | 0.554 | 0.0% | 0.0% | +0.0407 |
| math-comp-aime | 0.535 | 0.653 | 0.0% | 0.0% | +0.0125 |

The highest-lift domain has the **lowest** train similarity; the highest-similarity domain
(`ko-reason-hrmcr`, 50% of its episodes above cos 0.95) has one of the smaller lifts. The
correlation runs the wrong way for a memorization story.

**Conclusion for §4: no localized-memorization signal survives lookup-OFF.** The
memorization column (ON − OFF weighted per domain: code +0.1298, logic +0.0938, math-symbolic
+0.0870, longctx +0.0712, belebele +0.0637, gsm8k +0.0505, truthfulqa +0.0386, hrmcr +0.0200,
aime +0.0083) is spread across every domain roughly in proportion to remaining headroom, which is
what an exact-hash table over 100% of prompts does. It is confined to the lookup path.

---

## 5. Honest expected score on a private evaluation set, 0% public overlap

At 0% overlap the hash lookup contributes nothing (every probe misses; `hit.any()` is false and the
blend path runs unchanged), so the private score is governed by the **lookup-OFF** behaviour.
The lookup-ON Dev figure of 0.760284 has **no** predictive value for a private set and must not be
quoted as an expectation.

### 5a. Bias budget

**(i) Model-fit optimism — measured, and it is already paid.** `exp/build_final.py` documents
"Members (train-only fits, per frozen spec)". I ran the lookup-disabled bundle over the 1760 train
episodes to get the in-sample number:

```
TRAIN lookup-OFF final=0.729971590909
   fast     score=0.709517045455 ratio=1.099161469265 pass=True
   balanced score=0.723437500000 ratio=1.681512554822 pass=True
   premium  score=0.763778409091 ratio=2.403322836482 pass=True
```

| split | role | final | ceiling | % of ceiling |
|---|---|---:|---:|---:|
| train | in-sample (fit) | 0.729972 | 0.790507 | **92.34%** |
| dev | out-of-sample (fit) | 0.684318 | 0.803939 | **85.12%** |

In-sample minus out-of-sample = **0.045653** absolute, **7.22 pp** of ceiling. Dev has already
paid this; a private set with the same relationship to the train fit pays no *additional*
model-fit penalty. (Dev is the easier split on the light baseline — 0.6193 vs 0.5973 — and has the
higher ceiling, which makes the 7.2 pp drop a conservative reading.)

**(ii) Decision-knob selection optimism — measured, and it is negligible on the quality axis.**
The knobs were tuned on Dev, so a winner's-curse correction is owed. Where the frozen knobs sit on
the Dev tuning surface (predictions frozen, allocation replayed):

| tier | frozen knob | frozen score | dev-optimal knob | dev-peak score | gap |
|---|---|---:|---|---:|---:|
| fast | u = 0.93 | 0.659091 | u = 1.01 | 0.672443 | +0.013352 |
| balanced | u = 0.88 | 0.691761 | **u = 0.88** | 0.691761 | **0.000000** |
| premium | k1 = 0.65, fill = 0.70 | 0.710511 | k1 = 0.95, fill = 0.45 | 0.723864 | +0.013352 |
| **weighted** | | **0.684318** | | **0.693665** | **+0.009347** |

Balanced sits **exactly at the Dev argmax** over a 61-point grid; fast and premium were risk-gated
below their peaks. Optimism bootstrap (knob re-selected on each of 1000 resamples, then scored on
full Dev), quality axis only:

| tier | quality optimism mean | sd | 95% | **P(naively-tuned knob overruns on the real set)** |
|---|---:|---:|---|---:|
| fast | +0.000544 | 0.014842 | [−0.0284, +0.0290] | **25.1%** |
| balanced | +0.001535 | 0.014407 | [−0.0256, +0.0284] | 0.0% |
| premium | +0.000439 | 0.013981 | [−0.0270, +0.0273] | 0.0% |
| **weighted** | **+0.000810** | | | |

Family-selection optimism across all 25 recoverable candidates: **−0.000320** (sd 0.026829).

So the winner's curse on the *score* is ≈ **0.0008**, effectively zero — the candidates and knob
settings are so highly correlated (they share features, models and training data) that maximizing
over them buys almost no in-sample noise. What tuning *does* buy is budget-cliff exposure:
a fast-tier utilization tuned naively on 880 episodes has a **25.1%** chance of exceeding the cap
on the real set. The frozen policy's risk-gating away from u = 1.01 is the single most valuable
decision in the whole pipeline, and it cost only 0.0093 of Dev score.

A cruder half-split retune (tune on 440, score on the other 440, 60 reps) gives optimism
+0.0520 mean [−0.0510, +0.2807] — but that figure is dominated by budget-cliff zeros on the
held-out half, not by score overfit, and it is an upper bound that does not apply to the
risk-gated frozen configuration.

### 5b. Projection

Centre = 0.684318 − 0.000810 (knob optimism) − 0.000000 (family selection, measured ≈ 0)
≈ **0.6835**.

**Case A — private set i.i.d. with the public Dev composition, n ≈ 880, 0% overlap:**

| quantity | value |
|---|---|
| point / centre | **0.6835** |
| sd (sampling, budgeted) | 0.02565 |
| sd (sampling, quality-only) | 0.01420 |
| **95% interval** | **[0.654, 0.712]** |
| 80% interval | [0.651, 0.716] |
| P(any tier zeroed) | 0.82% |
| vs all-light 0.619318 | +0.064 [+0.046, +0.084], P(better) = 100% |
| vs hash-regex 0.695369 (quality axis) | −0.011 [−0.024, +0.002], P(better) = 5.0% |

**Case B — private set with domain composition allowed to shift ±50% (the realistic case; the
data card explicitly warns the public mix does not imply the private mix):**

| quantity | value |
|---|---|
| quality-only final | mean **0.6840**, 95% **[0.638, 0.728]** |
| conditional on all 3 tiers passing (92.9% of draws) | mean 0.6839, 95% [0.638, 0.728] |
| **unconditional (scoring rule applied)** | mean **0.6629**, median 0.6818, 95% **[0.414, 0.727]** |
| **P(any tier zeroed)** | **7.08%** |
| expected loss if Fast zeroes | −0.4 × 0.659 = −0.264 |
| expected loss if Premium zeroes | −0.3 × 0.711 = −0.213 |

**Headline honest answer: expect 0.68 ± 0.03 (95% CI [0.654, 0.712]) on a private set drawn like
the public one, widening to [0.638, 0.728] on the quality axis if the domain mix moves by ±50%,
with a 1–7% probability of a tier being zeroed that turns the score into ~0.42–0.47.**
The single-number expectation, integrating the cliff risk over the ±50% shift, is **0.663**.

Do not quote 0.760284 for a private set. Do not quote 0.684318 without the CI. The router's
edge over all-light is real and significant (+0.065, P = 1.00); its position relative to the
official hash-regex baseline is **behind on quality** (−0.011, P(better) = 5%) and **ahead on
the actual scoring rule** only because hash-regex is parked 1.5% from the Premium cap.

---

## 6. Recommendations

1. Correct `exp/registry.jsonl` E049 `overrun_prob_bootstrap` from 0.001/0.0/0.005 to
   **0.00221/0.00000/0.00631** (B = 200,000), or re-run the risk gate at B ≥ 100,000.
2. Record that Premium's i.i.d. **p99.9 = 4.354 exceeds the 4.0 cap** and that under a ±50%
   composition shift **Fast p99 = 1.288 and Premium p99 = 4.277 both exceed cap**. The current
   notes read as if all three tiers are safe.
3. Switch the stress harness's partition from `eda.categorize` (7 buckets, 245 episodes in a single
   `english-general` bucket) to a source-aligned partition. The coarse partition understates the
   overrun probability by roughly 2×.
4. Never present the registry top-3 ordering as a ranking. Attach `P(any tier zeroed)` to every
   registry row — E039's 0.7015 is a 67%-chance-of-zero result.
5. Report the frozen policy's Dev score as **0.684318 (95% CI [0.654, 0.712], lookup disabled)**
   and state alongside it that the shipped 0.760284 is an exact-hash memorization figure with
   100% hit rate on public prompts and zero expected value on unseen data.

---

## 7. Artifacts and reproduction

Scripts (all new, all under `build/`):

| script | produces |
|---|---|
| `build/a10_lib.py` | shared loader + float replication of the official rule |
| `build/a10_domains.py` | content-heuristic domain labeller |
| `build/a10_boot.py` | `build/a10_sec123.json` — B=1000 paired CIs, registry top-3, pairwise |
| `build/a10_overrun.py` | `build/a10_overrun.json` — B=200,000 cost-ratio + overrun |
| `build/a10_sec45.py` | `build/a10_sec45.json` — domain table, category-shift stress, doubling |
| `build/a10_nnleak.py` | `build/a10_nnleak.json` — dev→train near-duplicate analysis |
| `build/a10_sec5.py` | `build/a10_sec5.json` — knob replay, half-split retune, train-vs-dev |
| `build/a10_surface.py` | `build/a10_surface.json` — dev knob tuning surface |
| `build/a10_optimism2.py` | `build/a10_optimism2.json` — decomposed optimism, family selection |
| `build/a10_vsbaseline.py` | `build/a10_vsbaseline.json` — vs hash-regex / all-light |

Data generated (all under `build/`, nothing shipped touched):

* `build/train-nolookup/{fast,balanced,premium}.json` — lookup-disabled router on train (1760)
* `build/dev-hashregex/{fast,balanced,premium}.json` — official hash-regex baseline on dev
* `build/a10_dev_preds.npz` — frozen blend predictions on dev (score, cost_mean, cost_q90)

Reproduce:

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe build/a10_boot.py
.venv/Scripts/python.exe build/a10_overrun.py
.venv/Scripts/python.exe build/a10_sec45.py
.venv/Scripts/python.exe build/a10_nnleak.py
.venv/Scripts/python.exe build/a10_sec5.py
.venv/Scripts/python.exe build/a10_surface.py
.venv/Scripts/python.exe build/a10_optimism2.py
.venv/Scripts/python.exe build/a10_vsbaseline.py
```

Seeds are fixed (20260817 / 555 / 31337 / 777 / 4242). All bootstraps that compare two policies
share one resample matrix so the comparisons are paired.

### Not verified

* Private evaluation set — not available. §5 is a projection from Dev + Train under stated
  assumptions, not a measurement.
* The 25 registry experiments without `detail.npz` (E041, E044, E040, E046, E036, E035, E028,
  E034, E045, E026, E013, E029, E003, E015, E037, E019, E020, …) could not be bootstrapped; their
  per-episode selections are not recoverable. Their published dev scores are taken at face value
  for the ranking table in §2 only.
* Runtime/latency stability and container-side behaviour — out of scope for A10 (covered by the
  lead's black-box test).
