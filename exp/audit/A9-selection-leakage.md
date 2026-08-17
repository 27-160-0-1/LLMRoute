<!--
SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
SPDX-License-Identifier: Apache-2.0
-->

# TRACK A9 — Hyperparameter / model-selection leakage into Dev

Auditor: A9. Scope: `exp/registry.jsonl` (49 entries), `exp/results.md`,
`exp/eval_preds.py`, `exp/compare_decisions.py`, `exp/final_policy.py`,
`exp/fast_design.py`, `exp/premium_design.py`, `exp/freeze_final.py`,
`exp/holdout_check.py`, `exp/optimize_blend.py`.
All scratch artifacts under `build/a9/`. Nothing outside `exp/audit/` and
`build/` was written; `models/final-v1/**`, `src/**`, `data/**` were read only.

**VERDICT: FAIL on the repo's own claim.** The documented claim
"Train 내부 홀드아웃(Dev 미접촉) … dev 선택 과적합이 아님을 확인" (docs/TECHNICAL_REPORT.md
:457-459, exp/final-report.md:63-64) is **not supported**. Every decision
constant in the frozen policy E049 reproduces exactly as the argmax of a
realised-Dev-score search, and the Train-internal holdout that is cited as
proof (a) ranked a different policy family, (b) is group-leaky, (c) has a
sampling SE ~50x wider than the 4e-4 window it is claimed to resolve, and
(d) is unreproducible from the shipped repo because `build/preds/` is empty.

Two things are NOT in dispute and are confirmed here: the final score
0.684318181818 reproduces exactly, and freezing the conservative policy
instead of the dev-argmax policy was the *right* call — the untouched-holdout
evidence below is much stronger against the dev-argmax configs than the report
itself claims. The defect is in the provenance narrative, not the choice.

---

## 0. Reproduction anchor

The exact shipped blend4 dev predictions were extracted from the lookup-disabled
bundle and the frozen policy re-applied:

```
$ PYTHONPATH=src .venv/Scripts/python.exe build/a9/shipped_dev_preds.py
lookup rows in bundle: 0
score (880, 3) cost (880, 3) q90 (880, 3)
frozen policy on shipped no-lookup dev preds -> final: 0.684318181818
expected (build/dev-nolookup-report.json)      -> 0.684318181818
```

Everything below operates on these verified predictions
(`build/a9/dev-preds.npz`).

---

## 1. How many times was Dev consulted? (multiple-comparisons budget)

### 1.1 Classification of the 49 registry entries

```
$ PYTHONPATH=src .venv/Scripts/python.exe build/a9/count_dev.py
registry entries: 49
official TIER_WEIGHTS: {'fast': 0.4, 'balanced': 0.3, 'premium': 0.3}
families: {'reference': 4, 'model': 24, 'decision': 20, 'ensemble': 1}

--- dev-scored candidate allocations (multiple-comparison budget) ---
reference (no selection)                             n_exp= 4 per_exp=    1 total=     4
model / eval_preds.py argmax-on-dev                  n_exp=24 per_exp=  168 total=  4032
decision / compare_decisions.py argmax-on-dev        n_exp=18 per_exp=  505 total=  9090
decision / final_policy.py dev-risk-gated max-u      n_exp= 2 per_exp=  103 total=   206
ensemble E049 frozen (single dev measurement)        n_exp= 1 per_exp=    1 total=     1
TOTAL dev-scored candidates                                               13333
experiments whose REPORTED config was dev-selected   44/49
```

Per-experiment counts are read off the code, not guessed:

| script | family | what it argmaxes | candidates / experiment |
| --- | --- | --- | --- |
| `exp/eval_preds.py` `best_row()` L81-85 | model (24) | `max` over feasible rows of `sweep_tier` where score/ratio are **realised dev** values | 3 tiers x 56 `UTIL_GRID` = **168** |
| `exp/compare_decisions.py` L39-80 | decision (18) | same `(score,-ratio)` argmax, realised dev | F/B 2x56 each + premium 13x13 D4 + 2x56 = **505** |
| `exp/final_policy.py` L82-116 | decision (2) | last grid point passing dev-measured gates (= max feasible `u`; monotone in score) | 2x51 + 1 = **103** |

`registry.jsonl` records `"calibration": "dev"` on 42 of these entries
explicitly. **44/49 experiments have a reported configuration that was chosen
by looking at a realised Dev score.**

### 1.2 Unregistered dev searches that produced the frozen constants

Three more dev-argmax scripts exist and appear in **no** registry entry:

| script | grid | dev-scored candidates |
| --- | --- | --- |
| `exp/fast_design.py` | {k1 allowed, k1 banned, cap .03, cap .08} x u 0.80..1.10 (31), run for fast and balanced | 124 x 2 = **248** |
| `exp/premium_design.py` | cap(7) x k1_u(10) x fill(6) | **420** |
| `exp/analyze_premium_gap.py` L37-46 | 13x13 D4 grid, `s > best[0]` on realised dev | **169** |
| `exp/analyze_fast_gap.py` L36-42 | 56 x 2 allocators, realised dev | **112** |

**Corrected total dev-scored candidate allocations: 13,333 + 949 = 14,282**,
across at least 47 distinct search episodes. This is the multiple-comparisons
budget against the single 880-episode Dev split.

### 1.3 Measured optimism, not assumed

**(a) Recorded sweeps.** `exp/E0xx/detail.npz` stores each model-family
experiment's full dev sweep. The dev argmax gains, over the *median* feasible
grid point of the same sweep:

```
$ PYTHONPATH=src .venv/Scripts/python.exe build/a9/sweep_curse.py
n=24  mean(argmax - median-feasible) = +0.021238  sd=0.003394  min=+0.009830 max=+0.026875
  fast      score sd across feasible pts: 0.015665   argmax-median: +0.039773   u* range 0.90-1.03
  balanced  score sd across feasible pts: 0.007056   argmax-median: +0.009097   u* range 0.76-1.06
  premium   score sd across feasible pts: 0.005477   argmax-median: +0.008665   u* range 0.70-0.96
```

The selected `u*` wanders over 0.90-1.03 / 0.76-1.06 / 0.70-0.96 across
experiments that differ only in the score head — that instability is the
signature of noise-driven selection.

**(b) The registry already contains the transfer test and it fails.**
`eval_preds.py` also evaluates the *train*-calibrated utilization on dev and
stores it as `train_cal_weighted`. Reported dev-cal vs train-cal for all 24
model-family experiments:

```
  n=24  mean gap=+0.509392  median=+0.442159 sd=0.164025  min=+0.218097  max=+0.685597
  gap>0 in 24/24 experiments
  e.g. E039 xgb-mono  dev-cal=0.701506  train-cal=0.267955  gap=+0.433551
       E030 lgbm      dev-cal=0.682869  train-cal=0.000000  gap=+0.682869
```

Ten of the 24 collapse to **0.000000** — every tier busts the budget when the
utilization is not calibrated on Dev. (This gap mixes selection optimism with
an OOF-vs-full-fit calibration mismatch, so it is an upper bound on selection
optimism alone; the clean measurement is §3.)

**(c) Between-experiment multiplicity.** The 18 `best-decision` experiments
span sd 0.004329; the top 8 span only **0.002812** in total. Expected argmax
inflation for 18 exchangeable candidates is `1.657 x sd = 0.007173` — larger
than the entire observed spread of the top 8. **The ranking among the top blends
carries no information.**

**(d) Paired bootstrap on dev, E046 (dev argmax, 0.700909) vs E049 (frozen,
0.684318):**

```
E049 dev weighted (float) = 0.684318
E046 dev weighted (float) = 0.700909
paired bootstrap E046-E049 diff: mean -0.258207 sd 0.243885  95% [-0.697730, 0.021422]
```

E046's +0.0166 dev "advantage" is a point-estimate artifact: under episode
resampling its expected weighted score is **0.258 lower** than E049's, because
its dev-tuned utilizations sit on the budget boundary and flip to a zeroed tier
under resampling. The 95% interval for the difference is almost entirely
negative.

---

## 2. Was the FROZEN policy E049 itself selected on Dev? — YES

### 2.1 The claim

> `exp/final_policy.py` docstring L11-12: "Dev is used for measurement, not for
> picking the utilization that maximizes score."
> `docs/TECHNICAL_REPORT.md` L457-459 / `exp/final-report.md` L63-64:
> "Train 내부 홀드아웃(Dev 미접촉)에서도 blend4 계열이 전 후보 중 최상위
> (0.6625–0.6629)여서, dev 선택 과적합이 아님을 확인했다."

### 2.2 The frozen constants ARE the dev argmax — reproduced exactly

`exp/freeze_final.py` L26-33 hard-codes fast `greedy, allow_k1=False, u=0.93`;
balanced `greedy, allow_k1=False, u=0.88`; premium `two-stage, k1_u=0.65,
fill=0.70, k1_cost_cap=0.1`. **No registry entry and no script output records
where 0.88, `allow_k1=False`, or `cap=0.1` came from.** Re-running the two
unregistered design searches on the exact shipped dev predictions:

```
$ PYTHONPATH=src .venv/Scripts/python.exe build/a9/provenance.py
=== exp/fast_design.py search re-run (dev argmax under dev risk gates) ===
-- tier fast (multiplier 1.25) --
   k1 allowed   -> best (0.6590909090909091, 1.1804371230110366, 0.001, 0.008, 0.93, 0)
   k1 BANNED    -> best (0.6590909090909091, 1.1804371230110366, 0.001, 0.008, 0.93, 0)
   ARGMAX-ON-DEV = k1 allowed @ u=0.93 (dev score 0.659091)
-- tier balanced (multiplier 2.0) --
   k1 allowed   -> best (0.6911931818181818, 1.6981155260339782, 0.006, 0.014, 0.85, 36)
   k1 BANNED    -> best (0.6917613636363636, 1.6773557824070144, 0.0, 0.0, 0.88, 0)
   k1 cap 0.03  -> best (0.6872159090909091, 1.5289476248382947, 0.0, 0.0, 0.8, 32)
   k1 cap 0.08  -> best (0.6911931818181818, 1.6981155260339782, 0.006, 0.014, 0.85, 36)
   ARGMAX-ON-DEV = k1 BANNED @ u=0.88 (dev score 0.691761)

=== exp/premium_design.py search re-run (dev argmax under boot<=0.005) ===
   candidates enumerated on dev: 420
   ARGMAX-ON-DEV = cap=0.1 k1_u=0.65 fill=0.7 (dev score 0.710511, ratio 2.8239, boot 0.005)
   FROZEN E049 premium spec  = cap=0.1 k1_u=0.65 fill=0.7
   MATCHES dev argmax?        True
```

The recovered dev scores match the registered E049 metrics to 12 decimals:
balanced `0.691761363636` and premium `0.710511363636` are literally the
registry values for E049. Fast `u=0.93` likewise.

### 2.3 The registry timeline confirms dev-in-the-loop iteration at freeze time

```
E046 18:55:28  blend4-cost2b best-decision (dev argmax)          dev 0.700909
E047 18:57:09  risk-gated  F u.93 / B u.85 / P d4(.65,.70)       dev 0.684148
E048 18:58:09  premium d4(.65,.75)                               dev 0.689943  (boot .014 > .005 -> discarded)
E049 19:00:05  FINAL  F u.93 no-k1 / B u.88 no-k1 / P d4+q90+cap dev 0.684318
```

Four dev-scored policy revisions in under five minutes, each one recorded with
its dev number. E049 is E047 with balanced moved 0.85 -> 0.88 and k1 banned —
exactly the change `fast_design.py` selects by dev argmax, worth +0.000568 dev
balanced score.

### 2.4 The Train-internal holdout: four independent defects

**(i) Not reproducible.** `exp/holdout_check.py` reads `build/preds/{name}-train.npz`.
That directory is empty in the shipped repo (`build/` is gitignored):

```
$ PYTHONPATH=src .venv/Scripts/python.exe exp/holdout_check.py blend4-cost2b
FileNotFoundError: [Errno 2] No such file or directory:
  'C:\portable\skt_LLM\LLMRoute\build\preds\blend4-cost2b-train.npz'
```

The script prints and stores nothing — no registry row, no JSON. The
0.6625-0.6629 figures exist only in prose. The candidate list ("전 후보") is
never enumerated anywhere.

**(ii) Group-leaky split.** `folds = np.arange(n) % 5` interleaves; template
families (see §3.1) are split across A and B:

```
$ PYTHONPATH=src .venv/Scripts/python.exe build/a9/holdout_noise.py
arange%5 holdout group leakage: 86/352 B rows (24.4%) share a template family with A
```

84.3% of the B rows that belong to any multi-episode family have a sibling in A.

**(iii) The statistic cannot resolve a 4e-4 window.** B has 352 episodes:

```
B_weighted under a fixed near-oracle allocation: point=0.698684
  bootstrap SE over the 352 B rows            = 0.100158   (budget-pass indicator included)
  claimed ranking window in the report        = 0.000400  (0.6625-0.6629)
SCORE-ONLY B_weighted (no budget zeroing): point=0.734437  bootstrap SE=0.020304
  SE / claimed 4e-4 window = 50.8x
paired-difference SE between two candidate routers that differ on k% of rows:
   differ on  2% of rows: mean diff 0.000858  SE of diff 0.000908
   differ on  5% of rows: mean diff 0.002026  SE of diff 0.001354
   differ on 10% of rows: mean diff 0.004124  SE of diff 0.001958
B_weighted by choice of which arange%5 fold is B: 0.734517 0.745881 0.772869 0.750781 0.784659
  spread across folds = 0.050142 (sd 0.020506)
```

Even in the most favourable paired framing the SE of a candidate *difference*
is 0.0009-0.0020, i.e. 2-5x the claimed 4e-4 discrimination window. Merely
choosing a different `%5` fold as B moves the statistic by 0.050 — 125x the
window. **The holdout ranking is noise.**

**(iv) It ranks the wrong policy family.** `holdout_check.py` L50-59 uses only
`alloc_lib.greedy_allocate(..., allow_k1 defaults True)` with no `k1_cost_cap`
and no `two_stage_premium`. The frozen policy uses `allow_k1=False` for F/B and
two-stage-premium on a q90 cost head. The holdout therefore never evaluated the
policy that was frozen; it compared prediction sets under a decision rule that
was discarded.

**(v) Dev-informed inputs.** The pred sets entering the holdout (blend membership,
lgbm/xgb/knn/irt hyperparameters) were themselves chosen across E005-E046, all
of which report dev-argmax metrics. "Dev untouched" is true only of the final
ranking arithmetic, not of the candidates being ranked.

**Conclusion for §2: the holdout was not truly untouched in any useful sense,
and it is not evidence about the frozen policy.** Only two selection procedures
in the repo never read dev — `exp/optimize_blend.py` (blend weights on train
fold 0) and `exp/holdout_check.py` — and both use the same leaky `arange%5`
split. `optimize_blend.py`'s product, E037 `blendW-cost2`, scored dev 0.694574,
*below* the dev-argmax blends, and was not adopted.

---

## 3. Genuinely untouched inner holdout (group-aware, stratified 80/20)

### 3.1 Grouping

Independent of A4: (1) template signature = lowercased text, digit-runs -> `#`,
non-alphanumerics collapsed, first 300 chars, exact-match buckets;
(2) union-find merge of any pair with cosine >= 0.90 on the existing L2-normalised
word+char hashing features.

```
$ PYTHONPATH=src .venv/Scripts/python.exe build/a9/groups.py
[train] n=1760 sig-only groups=1625 final groups=1451 (thresh=0.9)
[train] group size: max=24 mean=1.21 singletons=1209 groups>=2=242 episodes in multi-groups=551
[dev]   n=880  final groups=752  max=9  singletons=644  episodes in multi-groups=236
```

31.3% of train episodes sit in a multi-episode template family.

### 3.2 Surrogate disclosure (honest limitation)

The frozen blend has four score members. `xgb-mono`'s `multi_output_tree` head
over 117,575 sparse columns was fitted on CUDA in `build_final.py`; this host has
no CUDA and a single CPU fit did not complete in >100 minutes of CPU time
(measured: `build/a9/timing.log` — SVD 28s, lgbm 12 heads ~52s, xgb multi
>20 min), making nested CV with it infeasible. §3-§4 therefore use
**blend3 = mean(lgbm, irt1d, knn-k40)** with the lgbm cost and lgbm-q90 k1-cost
heads. The **decision layer** — the layer whose constants were dev-selected — is
byte-identical to the frozen policy. Absolute levels shift; the *gaps between
configurations on the same predictions*, which is what is being measured, do not
depend on the fourth member. Scoring is float, same convention as
`exp/holdout_check.py`.

### 3.3 Result — group-disjoint 80/20 inner holdout (1401 fit / 359 held out)

```
$ PYTHONPATH=src .venv/Scripts/python.exe build/a9/nested.py     (build/a9/nested.log)
[step3] inner holdout: train 1401 / holdout 359 ; group overlap = 0 (must be 0)
[step3] cf. exp/holdout_check.py arange%5 split leaks 86/352 rows
[step3] selection subsplit: fit 1071 / select 330
[step3] E049-FROZEN      holdout weighted = 0.697841  {'fast': (0.663, 1.147, True),  'balanced': (0.7019, 1.626, True), 'premium': (0.7403, 3.876, True)}
[step3] E046-devargmax   holdout weighted = 0.217270  {'fast': (0.6783, 1.307, False),'balanced': (0.7242, 1.969, True), 'premium': (0.7848, 5.735, False)}
[step3] E047-riskgate    holdout weighted = 0.703691  {'fast': (0.663, 1.147, True),  'balanced': (0.7214, 1.795, True), 'premium': (0.7403, 3.876, True)}
[step3] SEARCH-on-holdout   = 0.720265  {'fast': 'lagrangian@1.02', 'balanced': 'greedy@0.9',  'premium': 'D4@0.55,0.6'}
[step3] SEARCH-on-selset    = 0.498955  {'fast': 'lagrangian@0.99', 'balanced': 'greedy@1.07', 'premium': 'D4@0.55,0.6'}
```

**Dev overfitting amount** (dev score minus untouched-holdout score, same
decision constants):

| configuration | Dev (official) | inner holdout (untouched) | gap = dev - holdout |
| --- | --- | --- | --- |
| E046 blend4-cost2b + best-decision (dev argmax) | 0.700909 | **0.217270** | **+0.483639** |
| E047 risk-gated | 0.684148 | 0.703691 | -0.019543 |
| **E049 FROZEN** | **0.684318** | **0.697841** | **-0.013523** |
| search run on the holdout itself (winner's curse bound) | — | 0.720265 | — |
| search run on an independent selection set, applied to holdout | — | 0.498955 | — |

Two findings, in opposite directions:

1. **The dev-argmax family is catastrophically dev-overfit: +0.4836.** E046's
   fast tier (u=1.01) and premium tier (D4 0.75/0.80) both bust the budget on
   the untouched holdout (ratio 1.307 > 1.25 and 5.735 > 4.0) and are zeroed.
2. **The frozen policy is NOT dev-inflated**: its untouched-holdout score is
   *higher* than its dev score (-0.0135), because the dev-argmax that produced
   its constants was taken under hard risk gates that pushed it far inside the
   feasible region. Dev selection still happened — but it selected a
   conservative point whose optimism is negative.

The winner's curse of the decision search itself, measured entirely inside train:
`SEARCH-on-holdout 0.720265` vs `SEARCH-on-independent-selection-set 0.498955`
= **0.221310** of pure selection optimism for a 505-candidate search at n~350.

### 3.4 Same measurement on Dev itself (group-disjoint half-split, R=20)

Selecting on one group-disjoint half of Dev with the exact
`compare_decisions.py` search and deploying on the other half
(`build/a9/selbias.log`, `build/a9/selbias_dev.json`):

```
=== decision-family search (505 cand/exp) | group-disjoint dev half-split, R=20 (123s) ===
  selected-on-D1, scored-on-D1 : 0.696876 +- 0.021400
  same config,   scored-on-D2  : 0.382336 +- 0.266302
  OPTIMISM                     : +0.314540 +- 0.276653 (median +0.336670)
  positive in 16/20;  D2 budget FAILURES per tier: {'fast': 6, 'balanced': 11, 'premium': 11}

=== model-family search (168 cand/exp) | group-disjoint dev half-split, R=20 (19s) ===
  selected-on-D1, scored-on-D1 : 0.692413 +- 0.052781
  same config,   scored-on-D2  : 0.346732 +- 0.193078
  OPTIMISM                     : +0.345681 +- 0.211776 (median +0.265246)
  positive in 19/20;  D2 budget FAILURES per tier: {'fast': 10, 'balanced': 11, 'premium': 8}
```

The in-sample value 0.696876 (n~440) sits right on top of the reported dev
leaderboard band 0.6899-0.7020 — i.e. **the reported dev numbers are
reproducible as pure in-sample search maxima.** Held out on an independent
group-disjoint half, the same configurations average 0.382336, because the
dev-tuned utilization busts the budget on 6-11 of 20 replicates per tier.

Caveat: at n~440 the budget-ratio variance is roughly `sqrt(2)x` that at n=880,
so the absolute optimism at full-dev n is smaller than +0.31. The direction, the
mechanism (budget-boundary tuning), and the ordering are unaffected, and §1.3(d)
gives the same conclusion at full n=880 (E046 - E049 paired bootstrap mean
-0.258207).

---

## 4. Nested cross-validation estimate (outer GroupKFold, inner selection)

Outer: 5 group-aware, difficulty-stratified folds of Train (no template family
crosses a fold boundary — asserted in code). Per outer fold the model is refit
on the 4 remaining folds; a further group-aware 75/25 split of that outer-train
provides an inner selection set on which the 505-candidate decision search is
run, and the selected configuration is then applied to the untouched outer test
fold. The three fixed policies are applied to the same outer-test predictions.

<!--NESTED-->

---

## 5. Findings, ranked

<!--FINDINGS-->

## 6. Artifacts

| path | content |
| --- | --- |
| `build/a9/count_dev.py` / `.json` | registry classification + dev-consultation budget + dev-cal vs train-cal gaps |
| `build/a9/sweep_curse.py` / `.json` | argmax-vs-median geometry of the 24 recorded dev sweeps |
| `build/a9/groups.py` / `groups-{train,dev}.npy` | template-family grouping |
| `build/a9/holdout_noise.py` / `.json` | sampling noise of `holdout_check.py`'s B_weighted, group leakage of `%5` |
| `build/a9/shipped_dev_preds.py` / `dev-preds.npz` | exact shipped blend4 dev predictions (verified: 0.684318181818) |
| `build/a9/provenance.py` / `.json` | re-run of `fast_design.py` / `premium_design.py` dev argmax |
| `build/a9/blendfit.py`, `nested.py` / `nested.json`, `nested.log` | group-aware inner holdout + nested GroupKFold |
| `build/a9/selbias_dev.py` / `.json`, `selbias.log` | dev group-disjoint half-split selection-bias replicates |
| `build/a9/timing.log` | member fit timings justifying the blend3 surrogate |
