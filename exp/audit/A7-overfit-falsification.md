<!--
SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
SPDX-License-Identifier: Apache-2.0
-->

# A7 — Overfitting / Statistical Falsification

Auditor track A7. Every number below was produced by a command shown in this
document and scored by the official Decimal scorer (`ossp_router.scoring`).
**Lookup is OFF everywhere in this report.** The offline experiment pipeline
has no lookup path at all, and the shipped-bundle runs load
`build/bundle-nolookup` (verified at runtime: `lookup rows in bundle: 0`).

Scripts written for this audit (all new, all under `exp/audit/`):

| script | purpose |
| --- | --- |
| `C:\portable\skt_LLM\LLMRoute\exp\audit\a7_proxy.py` | proxy pipeline: 12-head LightGBM + frozen allocator + official scorer |
| `C:\portable\skt_LLM\LLMRoute\exp\audit\a7_ablate.py` | channel ablation on the proxy |
| `C:\portable\skt_LLM\LLMRoute\exp\audit\a7_shipped_ablate.py` | channel ablation on the **shipped** 4-member blend |
| `C:\portable\skt_LLM\LLMRoute\exp\audit\a7_significance.py` | paired bootstrap CI on the ablation deltas |
| `C:\portable\skt_LLM\LLMRoute\exp\audit\a7_lift.py` | routing-decision lift of the shipped score model |
| `C:\portable\skt_LLM\LLMRoute\exp\audit\a7_util_sweep.py` | were the frozen utilization constants fit to Dev? |
| `C:\portable\skt_LLM\LLMRoute\exp\audit\a7_capacity.py` | parameter count of the shipped bundle |

Raw artifacts: `C:\portable\skt_LLM\LLMRoute\build\a7\*.json`. Nothing under
`models/final-v1`, `src/`, `container/`, `tests/`, `data/` was modified.

---

## VERDICT SUMMARY

| # | test | result | reading |
| --- | --- | --- | --- |
| 0 | reproduction of the lead's honest number | **exact to 12 dp**, incl. all budget ratios and model counts | lead's 0.684318 confirmed independently |
| 1 | label shuffle | **DOES NOT COLLAPSE** — 0.666515 vs real 0.673267, all-light 0.619318 | non-label channel identified: the **cost model + allocator**, not contamination |
| 2 | random features | **DOES NOT COLLAPSE** — 0.656297 | same; establishes the true no-information null at ≈0.6563 |
| 3 | learning curve | **saturates at 25 %** (0.672983 @ 440 rows vs 0.673267 @ 1760) | not data-limited; the model has already extracted the little signal there is |
| 4 | Train OOF vs Dev | **normalized gap +0.55 pp** (raw −0.0185) | no overfitting, and not "both high" — both are mediocre at ~29 % headroom |
| 5 | reverse validation | **symmetric** (29.2 % vs 30.7 % headroom captured) | no evidence of tuning leaking into either split |
| 6 | capacity | 609 k supervised scalars + 16.8 M SVD + 4.6 M memorized, vs 15 840 label cells | **grossly over-parameterized**; kNN + lookup memorize the corpus verbatim |
| + | utilization constants vs Dev argmax | frozen values are **NOT** the Dev argmax (they are conservative) | policy knobs were *not* fit to Dev — a clean pass |
| + | score-head ablation on the shipped blend | **+0.00159, 95 % CI [−0.00483, +0.00830], p(≤0)=0.31** | the entire 4-member score blend is statistically worthless |

**Headline.** The router is not overfit in the classical sense — the model
generalizes about as well to Dev as it does to Train, and no dev-label
contamination was found. The problem is the opposite: **there is almost
nothing being learned in the first place.** The shipped score model —
xgb-mono + irt1d + knn-k40 + lgbm, ~21 M stored numbers — contributes
`+0.0016` to the Dev weighted score, which a paired bootstrap cannot
distinguish from zero. Essentially all of the honest 0.684318 comes from
(a) the fixed floor of all-light 0.619318, (b) the cost/token model, and
(c) the budget allocator's structural cheapest-first prior.

---

## PROXY DISCLOSURE

Tests 1–5 use the **LightGBM member only** (12 heads: score ×3, log_in ×3,
log_out ×3, q90_out ×3) with the identical hyperparameters used by
`exp/build_final.py` (svd-128 features, `num_leaves=31`,
`min_data_in_leaf=20`, 300 rounds, lr 0.05, seed 42, `deterministic=True`),
followed by the **identical frozen allocation policy** (fast greedy 1.25/0.93
no-k1, balanced greedy 2.0/0.88 no-k1, premium two-stage 4.0/0.65/0.70 cap
0.1) and the **identical official scorer**. `exp/alloc_lib.py` was diffed
against the shipped `src/ossp_router/allocation.py`: the allocator bodies are
identical (only docstrings and an unused `from_pick` kwarg differ).

The proxy is 1 of the 4 blend members, so it lands 0.011 below the shipped
blend, as expected:

| pipeline | Dev final (lookup OFF) |
| --- | --- |
| shipped 4-member blend | 0.684318181818 |
| **A7 LightGBM proxy** | **0.673267045455** |

Test 6 and the two shipped-blend ablations use the **real shipped artifact**,
not the proxy.

---

## TEST 0 — INDEPENDENT REPRODUCTION OF THE HONEST NUMBER

```
$ PYTHONPATH=src ./.venv/Scripts/python.exe exp/audit/a7_shipped_ablate.py
lookup rows in bundle: 0
[14:12:10] SHIPPED-A score=blend4 cost=real  (= honest run) [dev] final=0.684318 F=0.6591 B=0.6918 P=0.7105
```

Exact comparison against the lead's `build/dev-nolookup-report.json`:

```
A7 SHIPPED-A final       0.684318181818
lead dev-nolookup final  0.684318181818
fast     A7= 0.659090909091 ratio 1.180437123011 | lead= 0.659090909091 1.180437123011 | counts A7 {'ax31-light': 466, 'ax31': 414, 'axk1-think': 0} lead {'ax31': 414, 'ax31-light': 466, 'axk1-think': 0}
balanced A7= 0.691761363636 ratio 1.677355782407 | lead= 0.691761363636 1.677355782407 | counts A7 {'ax31-light': 111, 'ax31': 769, 'axk1-think': 0} lead {'ax31': 769, 'ax31-light': 111, 'axk1-think': 0}
premium  A7= 0.710511363636 ratio 2.823888413321 | lead= 0.710511363636 2.823888413321 | counts A7 {'ax31-light': 82, 'ax31': 740, 'axk1-think': 58} lead {'ax31': 740, 'ax31-light': 82, 'axk1-think': 58}
```

Bit-for-bit identical, including budget ratios and per-model counts. Every
ablation below is therefore anchored to the real shipped artifact.

---

## REFERENCE LADDER (Dev, official scorer, lookup OFF)

This ladder is the most important context in the report, because the project
has been quoting "85.1 % of ceiling" — a ratio of two absolute scores whose
floor is not zero.

```
$ PYTHONPATH=src ./.venv/Scripts/python.exe exp/audit/a7_proxy.py alllight
$ PYTHONPATH=src ./.venv/Scripts/python.exe exp/audit/a7_shipped_ablate.py
```

| reference | Dev final | source |
| --- | --- | --- |
| all-light (do nothing) | 0.619318181818 | measured, `build/a7/alllight.json` |
| **no-information null** (const score + const cost, same allocator) | **0.641051136364** | measured, SHIPPED-D |
| random-feature control (test 2) | ≈0.656297 | measured, mean of 3 seeds |
| A7 LightGBM proxy | 0.673267045455 | measured |
| shipped blend, lookup OFF | 0.684318181818 | measured |
| hash-regex official baseline (E003) | 0.695369318182 | `exp/registry.jsonl` E003 |
| Dev oracle ceiling | 0.803939 | lead |

Derived:

* ceiling **ratio** (the quoted framing): 0.684318 / 0.803939 = **85.12 %**
* headroom captured over all-light: (0.684318 − 0.619318)/(0.803939 − 0.619318) = **35.21 %**
* headroom captured over the const/const null (0.641051): **26.56 %**
* headroom captured over the random-feature null (0.656297): **18.98 %**
* shipped blend − hash-regex baseline: **−0.011051** — with lookup off, the
  shipped router is *behind* the strongest official baseline.

**"85.1 % of the ceiling" must never be reported.** The honest statements are
"35.2 % of the available headroom over all-light", "19.0 % of the headroom
over a no-information null", and "1.1 points below the official hash-regex
baseline".

---

## HEADLINE — CHANNEL ABLATION ON THE SHIPPED BLEND

Which of the two learned channels actually earns the score? Each channel is
replaced, one at a time, by a train-marginal constant (per-model mean score /
geometric-mean cost), everything else untouched.

```
$ PYTHONPATH=src ./.venv/Scripts/python.exe exp/audit/a7_shipped_ablate.py
SHIPPED-A score=blend4 cost=real   final=0.684318181818  F=0.659091 B=0.691761 P=0.710511  ratios 1.1804/1.6774/2.8239
SHIPPED-B score=CONST  cost=real   final=0.682727272727  F=0.658523 B=0.693466 P=0.704261  ratios 1.1830/1.7333/2.5998
SHIPPED-C score=blend4 cost=CONST  final=0.667414772727  F=0.637500 B=0.673864 P=0.700852  ratios 1.0935/1.4236/2.5621
SHIPPED-D score=CONST  cost=CONST  final=0.641051136364  F=0.619318 B=0.619318 P=0.691761  ratios 1.0000/1.0000/2.1020
```

| channel removed | Dev final | loss vs A |
| --- | --- | --- |
| nothing (shipped) | 0.684318181818 | — |
| **score blend → constant** | 0.682727272727 | **−0.001591** |
| cost model → constant | 0.667414772727 | −0.016903 |
| both → constant | 0.641051136364 | −0.043267 |

Killing the **entire 4-member score blend** — xgb-mono, irt1d, knn-k40 and
the lgbm score heads, i.e. the bulk of the 21 M-number bundle — costs
**0.0016**. The balanced tier actually gets **better** without it
(0.693466 vs 0.691761).

Paired bootstrap over the 880 Dev episodes, 10 000 resamples
(`exp/audit/a7_significance.py`, allocations computed once on the full split
then held fixed):

```
"A_minus_B_score_head": {"mean": 0.0016446, "ci2.5": -0.0048310, "ci97.5": 0.0082955, "p_le_zero": 0.3094}
"A_minus_C_cost_head":  {"mean": 0.0167999, "ci2.5":  0.0069318, "ci97.5": 0.0265625, "p_le_zero": 0.0005}
```

* score head: **95 % CI straddles zero, p(≤0) = 0.31** → indistinguishable
  from no score model at all.
* cost head: **95 % CI strictly positive, p(≤0) = 0.0005** → real.

Decision churn caused by removing the score model: fast 127/880 decisions
change, balanced 84/880, premium 261/880 — the score model *does* change
decisions, it just does not change them for the better.

### Why: the score model cannot rank the routing decision

```
$ PYTHONPATH=src ./.venv/Scripts/python.exe exp/audit/a7_lift.py
```

Per-model score correlation looks respectable, but that is the shared
"prompt difficulty" factor, which is decision-irrelevant. The routing
decision depends on the **gain**:

| quantity | Pearson r | Spearman r |
| --- | --- | --- |
| score(ax31-light) | 0.494 | — |
| score(ax31) | 0.544 | — |
| score(axk1-think) | 0.492 | — |
| **gain = ax31 − ax31-light** | **0.109** | **0.091** |
| gain = axk1-think − ax31 | 0.409 | 0.316 |

Prediction spread is also roughly half the truth's
(`pred_std` 0.226 vs `true_std` 0.460 for ax31-light) — heavy shrinkage to
the mean, the signature of a model with little to say.

Fast-tier pick quality (the 414 episodes the shipped router upgrades to ax31):

| selector | fraction of picks with true gain > 0 | mean true gain |
| --- | --- | --- |
| base rate (all 880 episodes) | 0.17955 | 0.07244 |
| cheapest-first (zero score info) | 0.17874 | 0.07246 |
| **shipped router** | **0.19807** | **0.08454** |
| oracle (true gain/cost) | 0.38164 | 0.26087 |

Lift over base rate: **1.10×**. Selection signal captured relative to the
cheapest-first floor and the oracle ceiling: **6.4 %**.

For context, the true gain distribution is brutal: of 880 Dev episodes,
654 (74.3 %) gain **exactly zero** from ax31 over ax31-light, 68 (7.7 %) get
**worse**, and only 158 (18.0 %) improve. The routing problem is genuinely
hard; the model is barely above chance on it.

---

## TEST 1 — LABEL SHUFFLE

Training score labels permuted (same row permutation across the 3 model
columns, so the per-episode correlation structure survives but the
text→score link is destroyed), same pipeline retrained, same allocator,
scored on Dev with lookup OFF.

```
$ PYTHONPATH=src ./.venv/Scripts/python.exe exp/audit/a7_proxy.py shuffle --threads 4
[14:09:34] shuffle-scores seed=101 [dev] final=0.667898 F=0.6483 B=0.6739 P=0.6881
[14:12:19] shuffle-scores seed=202 [dev] final=0.665455 F=0.6460 B=0.6716 P=0.6852
[14:17:19] shuffle-scores seed=303 [dev] final=0.666193 F=0.6517 B=0.6685 P=0.6832
[14:20:36] shuffle-scores+tokens seed=101 [dev] final=0.657358 F=0.6366 B=0.6631 P=0.6793
```

Full per-tier detail (`build/a7/shuffle.json`):

```
shuffle-scores seed=101         final=0.667897727273 | F 0.648295/1.1770 | B 0.673864/1.6700 | P 0.688068/2.4968
shuffle-scores seed=202         final=0.665454545455 | F 0.646023/1.2165 | B 0.671591/1.6945 | P 0.685227/2.3058
shuffle-scores seed=303         final=0.666193181818 | F 0.651705/1.1853 | B 0.668466/1.7616 | P 0.683239/2.5915
shuffle-scores+tokens seed=101  final=0.657357954545 | F 0.636648/1.1496 | B 0.663068/1.4898 | P 0.679261/1.8336
```

| run | Dev final |
| --- | --- |
| real labels (proxy baseline) | 0.673267045455 |
| shuffled score labels, seed 101 | 0.667897727273 |
| shuffled score labels, seed 202 | 0.665454545455 |
| shuffled score labels, seed 303 | 0.666193181818 |
| **mean of 3 seeds (sd 0.001253)** | **0.666515151515** |
| shuffled score **and** token labels | 0.657357954545 |
| all-light (the expected collapse target) | 0.619318181818 |

**It does not collapse.** The shuffled-score-label pipeline retains
0.666515 − 0.619318 = **+0.047197** over all-light, i.e. **87.5 %** of the
real pipeline's entire margin over all-light (0.053949).

**The non-label channel is identified, and it is not contamination.** Test 1
as specified permutes only the *score* labels; the token/cost heads keep
their real labels, so the pipeline still knows what each episode costs. With
a near-constant score prediction the greedy allocator's `ds/dc` ranking
degenerates to `1/dc` — cheapest-first promotion — which maximizes the number
of upgrades affordable under the budget and is a genuinely strong heuristic
requiring no score information whatsoever.

The stricter variant confirms this exactly: permuting **both** the score and
token labels drops the result to **0.657358**, which is within noise of the
random-feature null of test 2 (0.656297 ± 0.002). The ladder is monotone and
fully accounted for:

```
0.673267  real labels
0.666515  score labels destroyed          (cost model intact)   -0.006752
0.657358  score + token labels destroyed  (nothing intact)      -0.009157
0.656297  random features, real labels     [test 2 null]
0.619318  all-light
```

Conclusion: **no evidence of a leak from the Dev labels.** Strong evidence
that the score labels barely matter to this pipeline — destroying them costs
0.0068, while destroying the token labels on top costs another 0.0092.

---

## TEST 2 — RANDOM-FEATURE CONTROL

Feature matrix replaced by `N(0,1)` noise of the same shape (1760×337 train,
880×337 dev), **real labels kept**, all 12 heads retrained.

```
$ PYTHONPATH=src ./.venv/Scripts/python.exe exp/audit/a7_proxy.py randfeat --threads 4
random-features seed=7 final=0.658607954545 {'fast': (0.63125, 1.1771, True), 'balanced': (0.668466, 1.5411, True), 'premium': (0.685227, 1.7916, True)}
random-features seed=8 final=0.654943181818 {'fast': (0.63892, 1.1078, True), 'balanced': (0.655114, 1.5173, True), 'premium': (0.676136, 1.9072, True)}
random-features seed=9 final=0.655340909091 {'fast': (0.630114, 1.2199, True), 'balanced': (0.669318, 1.5959, True), 'premium': (0.675, 1.937, True)}
```

mean **0.656297348485**, sd 0.002011. All tiers pass budget.

**It does not collapse to 0.619318 either** — it sits +0.036979 above
all-light. The reason is structural, not a leak: with both heads dead the
predictions are near-constant plus jitter, and the allocator still spends the
budget on *some* upgrades, each of which has positive expected gain
(base-rate mean gain +0.0724).

The important consequence: **≈0.656297, not 0.619318, is the correct
no-information null for this allocator.** Measured against it:

| pipeline | Dev final | gain over the 0.656297 null | share of null→oracle headroom |
| --- | --- | --- | --- |
| A7 LightGBM proxy | 0.673267045455 | +0.016970 | **11.49 %** |
| shipped 4-member blend | 0.684318181818 | +0.028021 | **18.98 %** |

(The `score=CONST, cost=CONST` run gives 0.641051 rather than 0.6563 because
perfectly tied predictions make the greedy allocator's group-promotion
degenerate: all 880 rows form one group whose total cost exceeds the cap, so
nothing is promoted in fast/balanced. The random-feature number is the
better-behaved null.)

---

## TEST 3 — LEARNING CURVE

Same pipeline trained on random 10/25/50/100 % subsets of Train (SVD refit on
each subset), scored on the full Dev split.

```
$ PYTHONPATH=src ./.venv/Scripts/python.exe exp/audit/a7_proxy.py curve --threads 4
[14:10:41] curve frac=0.10 n=176  [dev] final=0.410028 F=0.6557 B=0.6747 P=0.6920
[14:11:28] curve frac=0.25 n=440  [dev] final=0.672983 F=0.6523 B=0.6770 P=0.6966
[14:15:39] curve frac=0.50 n=880  [dev] final=0.676392 F=0.6582 B=0.6759 P=0.7011
[14:19:41] curve frac=1.00 n=1760 [dev] final=0.673267 F=0.6543 B=0.6759 P=0.6960
```

Full detail with budget ratios and pass flags (`build/a7/curve.json`):

```
curve frac=0.10 n=176   final=0.410028409091 | F 0.655682/1.3374 False | B 0.674716/1.8237 True | P 0.692045/3.0175 True
curve frac=0.25 n=440   final=0.672982954545 | F 0.652273/1.1813 True  | B 0.676989/1.6603 True | P 0.696591/3.1522 True
curve frac=0.50 n=880   final=0.676392045455 | F 0.658239/1.2239 True  | B 0.675852/1.7071 True | P 0.701136/3.4186 True
curve frac=1.00 n=1760  final=0.673267045455 | F 0.654261/1.1840 True  | B 0.675852/1.6388 True | P 0.696023/2.7155 True
```

| train fraction | n rows | Dev final | headroom captured | note |
| --- | --- | --- | --- | --- |
| 10 % | 176 | 0.410028409091 | — | **fast tier ratio 1.3374 > 1.25 → budget FAIL, tier zeroed** |
| 25 % | 440 | 0.672982954545 | 29.07 % | all tiers pass |
| 50 % | 880 | **0.676392045455** | **30.91 %** | best of the four |
| 100 % | 1760 | 0.673267045455 | 29.22 % | reproduces the independent `baseline` run to 12 dp |

**Saturating immediately — the "suspicious" branch of this test.** Going from
440 to 1760 training rows moves the Dev score by **+0.000284**; going from 880
to 1760 moves it by **−0.003125** (i.e. doubling the data makes it slightly
worse, well inside seed noise). The curve is
flat because, as tests 1/2 and the ablation show, the only thing being
learned is the cost/token model, which needs very little data — and the score
model, which more data does not rescue.

The 10 % point is the informative failure: at 176 training rows the cost
model is bad enough that the **fast tier blows the 1.25× budget and is zeroed
by the scorer**, dragging the weighted final to 0.410028 despite healthy
per-tier quality scores. That is the real fragility in this system: budget
feasibility depends entirely on the token-count regression.

---

## TEST 4 — TRAIN OOF vs DEV GAP

Train-side numbers are 5-fold OOF (`fold_id = arange(N) % 5`, validation fold
never seen in training — the same protocol as `exp/models/lgbm.py`), then run
through the identical frozen allocator and scored on the Train split. The
SVD is fit once on the whole Train split (unsupervised, no labels) exactly as
the production pipeline does; this is a mild optimism in the OOF number and
is noted rather than removed.

```
$ PYTHONPATH=src ./.venv/Scripts/python.exe exp/audit/a7_proxy.py gap --threads 4
[14:11:53] oof train fold 0 done
[14:16:17] oof train fold 1 done
[14:19:34] oof train fold 2 done
[14:20:48] oof train fold 3 done
[14:21:25] lgbm-proxy OOF on train [train] final=0.654815 F=0.6325 B=0.6581 P=0.6813
```

```
lgbm-proxy OOF on train  final=0.654815340909 | F 0.632528/1.1355 True | B 0.658097/1.7151 True | P 0.681250/2.5095 True
lgbm-proxy train->dev    final=0.673267045455 | F 0.654261/1.1840 True | B 0.675852/1.6388 True | P 0.696023/2.7155 True
```

Raw and normalized comparison. The two splits have **different floors and
different ceilings** (all-light Train 0.597301 vs Dev 0.619318; oracle Train
0.790507 vs Dev 0.803939), so the raw difference is not interpretable on its
own:

| quantity | Train (5-fold OOF) | Dev (fit on full Train) |
| --- | --- | --- |
| final | 0.654815340909 | 0.673267045455 |
| all-light floor | 0.597301136364 | 0.619318181818 |
| oracle ceiling | 0.790507 | 0.803939 |
| lift over own all-light | +0.057514 | +0.053949 |
| **headroom captured** | **29.77 %** | **29.22 %** |

* raw gap (Train OOF − Dev) = **−0.018452** — *negative*, entirely explained
  by the lower Train all-light floor.
* normalized gap = **+0.55 percentage points**.

**PASS, decisively. Gap 0.0055 normalized (or −0.018 raw) ≪ the 0.05 overfit
threshold.** Nor does the "both contaminated" branch apply: both sides sit at
~29–30 % of headroom, i.e. both are mediocre in the same way, not both
suspiciously high. This is the profile of a model that is under-fitting the
problem, not memorizing it.

---

## TEST 5 — REVERSE VALIDATION

Same pipeline trained on Dev (880 episodes, SVD refit on Dev) and evaluated
on Train (1760 episodes).

```
$ PYTHONPATH=src ./.venv/Scripts/python.exe exp/audit/a7_proxy.py reverse --threads 4
[14:11:45] lgbm-proxy dev(880)->train(1760) [train] final=0.656591 F=0.6362 B=0.6574 P=0.6830
```

| direction | train n | eval split | final | all-light floor | oracle ceiling | headroom captured |
| --- | --- | --- | --- | --- | --- | --- |
| forward Train→Dev | 1760 | dev | 0.673267045455 | 0.619318181818 | 0.803939 | **29.22 %** |
| reverse Dev→Train | 880 | train | 0.656590909091 | 0.597301136364 | 0.790507 | **30.69 %** |

**Symmetric — healthy.** The reverse direction, trained on *half* the data,
captures a marginally *larger* share of its split's headroom. There is no
sign that Dev tuning leaked into the Dev-side number. The size control is the
learning-curve 50 % point (also 880 training rows → 0.676392 on Dev,
**30.91 %** headroom), which matches the reverse direction (30.69 %) almost
exactly — so the reverse result is not an artifact of the smaller training
set either.

### Supplementary — were the frozen policy constants fit to Dev?

The registry has 49 entries and `exp/eval_preds.py` sweeps a 56-point
utilization grid, so selection bias on the policy knobs is a live risk. It
was tested directly against the shipped Dev predictions:

```
$ PYTHONPATH=src ./.venv/Scripts/python.exe exp/audit/a7_util_sweep.py
fast: frozen u=0.93 score=0.659091 ratio=1.1804 | dev argmax u=1.01 score=0.672443 | frozen==argmax False gap=0.013352
balanced: frozen u=0.88 score=0.691761 ratio=1.6774 | dev argmax u=0.89 score=0.691761 | frozen==argmax False gap=0.000000
premium.k1_utilization: frozen u=0.65 score=0.710511 | dev argmax u=1.13 score=0.730114 | frozen==argmax False gap=0.019602
premium.fill_utilization: frozen u=0.7 score=0.710511 | dev argmax u=0.68 score=0.710511 | frozen==argmax False gap=0.000000
```

**Clean pass.** None of the four frozen constants sits at the Dev argmax;
they are deliberately conservative and leave 0.0134 (fast) and 0.0196
(premium k1) of Dev score unclaimed. The policy knobs were chosen for budget
safety, not fitted to Dev.

---

## TEST 6 — CAPACITY vs DATA

Parameter count of the **shipped** `models/final-v1`. Counting convention:
LightGBM tree with L leaves → `3L−2` learned scalars (L leaf values +
(L−1) thresholds + (L−1) split feature indices); XGBoost →
`leaves × size_leaf_vector + 2 × internal`; IRT → `W + a + b`; SVD →
`components`; kNN/lookup counted as stored (memorized) scalars.

```
$ PYTHONPATH=src ./.venv/Scripts/python.exe exp/audit/a7_capacity.py
{
 "supervised_learned_scalars": 17386704,
 "unsupervised_svd_scalars": 16777216,
 "memorized_scalars": 4610740,
 "grand_total": 21997444,
 "train_episodes": 1760,
 "supervised_label_cells_1760x3x3": 15840,
 "params_per_label_cell": 1097.65,
 "params_per_label_cell_excl_svd": 38.48
}
lightgbm: {"n_boosters": 12, "total_trees": 3600, "total_leaves": 111042, "learned_scalars": 325926}
xgboost: {"n_boosters": 7, "total_trees": 2800, "total_leaves": 53454, "keep_cols_stored": 117575, "learned_scalars": 283219}
irt: {"shapes": {"W": [1, 337], "a": [3, 1], "b": [3], "scaler_mean": [81], "scaler_scale": [81]}, "learned_scalars": 343, "scaler_scalars": 162}
svd: {"per_view": {"svd-word.npz": {"shape": [128, 65536], "params": 8388608}, "svd-char.npz": {"shape": [128, 65536], "params": 8388608}}, "learned_scalars": 16777216}
knn: {"index_shape": [1760, 131072], "index_nnz": 2249927, "outcome_shapes": {...}, "stored_scalars": 4510420, "note": "non-parametric: the entire train split is memorized verbatim"}
lookup: {"rows": 2640, "shapes": {"key": [2640, 32], "scores": [2640, 3], "costs": [2640, 3]}, "stored_scalars": 100320, "note": "sha256(prompt) -> realized score/cost; pure memorization"}
```

Denominator: 1760 episodes × 3 models × 3 targets (score, in_tokens,
out_tokens) = **15 840 supervised label cells**.

| component | learned/stored scalars | ratio to 15 840 label cells | flag |
| --- | --- | --- | --- |
| SVD word+char (unsupervised, train-fit) | 16 777 216 | **1059×** | **FLAG** — 2 × (128 × 65536) projections fit on 1760 documents |
| LightGBM, 12 boosters × 300 trees | 325 926 (3600 trees, **111 042 leaves**) | 20.6× | **FLAG** — 63 leaves per training episode |
| XGBoost, 7 boosters × 400 trees | 283 219 (2800 trees, 53 454 leaves) | 17.9× | **FLAG** — incl. 117 575 stored kept-column indices |
| IRT 1-d | 343 (+162 scaler) | 0.02× | fine — the only right-sized member |
| kNN k=40 index | 4 510 420 (2 249 927 nnz over 1760×131072) | 285× | **FLAG** — memorizes the full Train split verbatim |
| lookup table | 100 320 (2640 rows) | 6.3× | by design pure memorization (rule-allowed) |
| **supervised total (excl. SVD)** | **609 488** | **38.5×** | |
| **grand total** | **21 997 444** | **1389×** | |

**Every learned component except IRT dwarfs the data.** 164 496 total tree
leaves are fit to 1760 episodes — 93 leaves per episode. The kNN member is
not a model at all, it is the training set. The SVD alone is 16.8 M numbers
estimated from 1760 documents.

The reassuring counterweight is Test 5: despite this capacity, Train↔Dev
performance is symmetric, so the capacity is not being converted into
memorization of Train in a way that hurts Dev. It is simply not being
converted into anything — consistent with the ablation showing the score
model contributes 0.0016. And IRT, with 343 parameters, is not measurably
worse than the 21 M-parameter stack.

---

## WHAT THIS TRACK DID **NOT** FIND

State plainly, so nobody over-reads this report:

* **No Dev-label contamination in the offline pipeline.** Shuffled labels do
  not reproduce the real score; reverse validation is symmetric; the
  utilization constants are not at the Dev argmax.
* **No classical Train/Dev overfitting.** Train 5-fold OOF captures 29.77 %
  of its split's headroom, Dev captures 29.22 % — a +0.55 pp gap, and the
  raw gap is *negative* (−0.018452).
* The lookup table is rule-allowed and separately handled by track A2; it was
  disabled for every number here.

## WHAT THIS TRACK **DID** FIND

1. The honest lookup-OFF Dev score 0.684318 is **1.1 points below** the
   official strongest baseline (hash-regex 0.695369).
2. "85.1 % of ceiling" is a floor-less ratio and overstates the result. The
   honest figures are **35.2 %** of the headroom over all-light, or **19.0 %**
   of the headroom over the measured no-information null (0.656297).
3. The entire 4-member score blend contributes **+0.0016 (95 % CI
   [−0.0048, +0.0083], p(≤0)=0.31)** — statistically indistinguishable from
   deleting it. Only the cost/token model earns anything
   (+0.0169, p(≤0)=0.0005).
4. The score model's routing-relevant discrimination is
   **Pearson r = 0.109** on the ax31-vs-light gain, a **1.10× lift** over the
   base rate, capturing **6.4 %** of the available selection signal.
5. The bundle carries ~22 M stored numbers against 15 840 supervised label
   cells, of which 4.6 M are verbatim memorization of the public splits.
