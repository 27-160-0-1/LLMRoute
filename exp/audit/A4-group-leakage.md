<!--
SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
SPDX-License-Identifier: Apache-2.0
-->

# A4 — Near-duplicate / group leakage audit

Scope: `data/materialized/train/inputs.json` (1760) + `data/materialized/dev/inputs.json` (880).
All numbers below were produced by scripts under `build/` in this session; every claim is
paired with the command that produced it. Nothing under `models/final-v1/`, `src/`,
`container/`, `tests/`, `data/` was modified.

Scripts written (all under `build/`, per the write-scope constraint):

| script | purpose |
| --- | --- |
| `build/a4_groups.py` | reproduce the reported family / prefix statistics from scratch |
| `build/a4_probe.py` | identify which masking key yields the "~33%" figure |
| `build/a4_groupkey.py` | build the canonical group key + per-dev max-Jaccard to train |
| `build/a4_subset_score.py` | exact-Decimal subset scorer (validated against official reports) |
| `build/a4_baseline_sel.py` | emit hash-regex / all-light dev selections in the same layout |
| `build/a4_inspect.py` | characterise what the near-duplicate episodes actually are |
| `build/a4_cv.py` | THE CENTRAL EXPERIMENT: random KFold vs GroupKFold |

---

## 1. Reproduction of the reported statistics

Command:

```
$env:PYTHONPATH='src'; .venv/Scripts/python.exe build/a4_groups.py
.venv/Scripts/python.exe build/a4_probe.py
.venv/Scripts/python.exe build/a4_groupkey.py
```

### 1.1 Exact normalized duplicates — CONFIRMED

Normalization = whitespace collapsed only.

```
"exact_norm_dup_train_dev": 0,
"exact_norm_dup_within_train": 0,
"exact_norm_dup_within_dev": 0,
```

There is **no exact duplication anywhere** — not across splits and not within a split.

### 1.2 Template families — the "~33%" claim is CONFIRMED, but only for one specific key

The claim "~33% of Train sits in digit-masked template families of size >= 2" is **not**
true of a full-prompt digit mask. It is true of a digit-masked **60-character prefix**.
`build/a4_probe.py` sweeps candidate keys:

```
key                                    fams   >=2   pct     max  dev-share
digits->#                              1740     36   2.05%     3    20
digits->#, 1-letter vars->@            1737     41   2.33%     3    20
digits->#, vars->@, Caps-><N>          1737     41   2.33%     3    20
digit-mask prefix 30c                  1375    707  40.17%    24   129
digit-mask prefix 40c                  1420    641  36.42%    24    86
digit-mask prefix 60c                  1455    583  33.12%    24    65   <-- the "~33%"
digit-mask prefix 80c                  1469    555  31.53%    24    56
digit-mask prefix 120c                 1517    459  26.08%    24    51
digit-mask first 3 words               1052   1062  60.34%    59   321
digit-mask first 5 words               1291    826  46.93%    24   189
digit-mask first 8 words               1425    635  36.08%    24    89
digit-mask first 12 words              1476    541  30.74%    24    63
```

**The 33% figure is prefix-length-dependent and is therefore soft.** The same data supports
2.05% (exact full-prompt template), 11.42% (Jaccard >= 0.8), 33.1% (60c prefix) or 60.3%
(first 3 words). Any headline that quotes "33%" without naming the key is not reproducible.
For everything downstream I use the 60c-prefix key (plus 1-letter-variable masking) because
it is the one that reproduces the reported number.

Canonical key (`build/a4_groupkey.py`, `grp_union` = union-find over full-prompt digit-mask
∪ 60c-prefix digit-mask ∪ Jaccard>=0.8 edges). The union turns out to be **identical** to
the 60c-prefix key alone — the other two are strict subsets:

```
grp_union / grp_prefix60:
  n_train_groups        1448
  train_in_group_ge2     595   (33.8068 %)
  max_group               24
  size_hist   {1: 1165, 2: 275, 3: 7, 24: 1}      (1165 + 550 + 21 + 24 = 1760 OK)
  dev_near                66
  dev_novel              814
```

Family-size histogram in words: **1165 singletons, 275 pairs, 7 triples, and exactly one
24-episode family** (the Korean age-calculation template — confirmed by inspection, see §1.5).

### 1.3 Near-duplicate clusters (char 5-gram, exact Jaccard >= 0.8)

I used **exact** Jaccard over hashed char-5gram sets (2^20 bins) rather than a MinHash
estimate, so there are no false negatives from signature noise.

```
"nd_jaccard_0.8": {
  "n_pairs": 389,
  "n_train_clusters": 1648,
  "train_in_cluster_ge2": 201,
  "train_pct_in_cluster_ge2": 11.4205,
  "train_cluster_size_hist": {"1": 1559, "2": 87, "3": 1, "24": 1},
  "train_max_cluster_size": 24,
  "dev_sharing_train_cluster": 18,
  "cross_split_pairs": 104
}
```

104 of the 389 near-duplicate pairs straddle the train/dev boundary.

### 1.4 Dev prefix overlap — CONFIRMED EXACTLY

Raw (unmasked, whitespace-collapsed) shared prefix with **some** train episode:

```
"dev_shares_40char_prefix_with_train": 65,
"dev_shares_60char_prefix_with_train": 47,     <-- reported 47, reproduced exactly
"dev_shares_80char_prefix_with_train": 38,
"dev_shares_100char_prefix_with_train": 35,
```

### 1.5 Per-dev-episode similarity to the nearest train episode

```
dev max-Jaccard-to-train quantiles:
  p05 0.0719  p25 0.1054  p50 0.1540  p75 0.3858
  p90 0.5198  p95 0.6713  p99 0.8720  max 0.9555

dev episodes with maxjac >= 0.3 : 275
                        >= 0.4 : 206
                        >= 0.5 : 107
                        >= 0.6 :  56
                        >= 0.7 :  40
                        >= 0.8 :  18
```

The largest family, verified by inspection (`build/a4_inspect.py`), is the Korean
age-calculation template: **24 train + 9 dev episodes** differing only in the birth year and
the ordinal ("2000년 12월 10일" vs "1998년 12월 10일", "23번째" vs "22번째"). The rest of the
size-3 families are DeepMind-Mathematics arithmetic stencils:
`What is the tens digit of <N>?`, `What is the millions digit of <N>?`,
`Calculate -<N> divided by <N>.`, `What is the <var>'th term of <N>, <N>, ...?`.

---

## 2. THE CENTRAL EXPERIMENT — random KFold vs GroupKFold

*(filled in below once `build/a4_cv.py` completes)*

---

## 3. dev-near vs dev-novel under the FROZEN policy, lookup OFF

### 3.1 Scorer validation (mandatory before any subset number is quoted)

`build/a4_subset_score.py` re-implements `src/ossp_router/scoring.py` in exact `Decimal`
(quality = Σ score / n, ROUND_HALF_EVEN at 12 dp). It was validated on the **full** dev set
against three independently produced official reports:

| selections | my scorer, FULL | official | match |
| --- | --- | --- | --- |
| `build/dev-nolookup` | `0.684318182` | `build/dev-nolookup-report.json` → `0.684318181818` | yes |
| `build/dev-final` (lookup ON) | `0.760284091` | `build/dev-final-report.json` → `0.760284090909` | yes |
| `build/dev-hashregex` | `0.695369318` | official hash-regex baseline `0.695369` | yes |
| `build/dev-alllight` | `0.619318182` | reference all-light `0.619318` | yes |

Budget note: budget pass/fail is a **global** property of the full 880-episode submission and
was already satisfied (`budget_passed: true` on all three tiers). Subset "quality_score" is
therefore the correct restricted quantity; the `cost_ratio_on_subset` column is reported for
information only and is not a pass/fail gate on a subset.

### 3.2 Results (lookup OFF, `build/bundle-nolookup` selections)

```
subset                                     n     fast      bal     prem     weighted    light   oracle   hdrm%
FULL                                     880  0.65909  0.69176  0.71051  0.684318182  0.61932  0.87983   24.95
dev-near(group_union)                     66  0.51515  0.53030  0.58333  0.540151515  0.50758  0.84091    9.77
dev-novel(group_union)                   814  0.67076  0.70485  0.72082  0.696007371  0.62838  0.88299   26.56
dev-near(maxjac>=0.5)                    107  0.52804  0.56075  0.59346  0.557476636  0.51869  0.77570   15.09
dev-novel(maxjac<0.5)                    773  0.67723  0.70990  0.72671  0.701875809  0.63325  0.89424   26.29
dev-near(maxjac>=0.4)                    206  0.58252  0.62379  0.64078  0.612378641  0.55825  0.83738   19.39
dev-novel(maxjac<0.4)                    674  0.68249  0.71254  0.73182  0.706305638  0.63798  0.89280   26.81
maxjac_quintile_1 (least similar)        176  0.79403  0.82528  0.83097  0.814488636  0.74858  0.94886   32.91
maxjac_quintile_2                        176  0.71307  0.76562  0.76562  0.744602273  0.67898  0.86790   34.74
maxjac_quintile_3                        176  0.60653  0.61790  0.64347  0.621022727  0.56392  0.86364   19.05
maxjac_quintile_4                        176  0.57955  0.62216  0.66761  0.618750000  0.52273  0.88352   26.61
maxjac_quintile_5 (most similar)         176  0.60227  0.62784  0.64489  0.622727273  0.58239  0.83523   15.96
```

`light` = all-light on that subset, `oracle` = budget-ignoring best-model-per-episode on that
subset, `hdrm%` = (router − light) / (oracle − light), i.e. the fraction of the achievable
headroom the router actually captured. This normalisation matters because the two subsets
have very different intrinsic difficulty.

**The direction is the opposite of the leakage hypothesis.**

* dev-novel = **0.696007371** > FULL 0.684318182 > dev-near = **0.540151515**
* gap (novel − near) = **+0.155855856**
* gap (novel − headline) = **+0.011689189**

The router is *worse* on the near-duplicate subset, not better, in both raw score (0.540 vs
0.696) and headroom captured (9.77% vs 26.56%).

### 3.3 Same subsets, other policies (apples-to-apples)

| subset | n | all-light | hash-regex | ours lookup OFF | ours lookup ON |
| --- | ---: | ---: | ---: | ---: | ---: |
| FULL | 880 | 0.619318182 | 0.695369318 | 0.684318182 | 0.760284091 |
| dev-near (group_union) | 66 | 0.507575758 | 0.603030303 | **0.540151515** | 0.627272727 |
| dev-novel (group_union) | 814 | 0.628378378 | 0.702856265 | **0.696007371** | 0.771068796 |
| dev-near (maxjac>=0.5) | 107 | 0.518691589 | 0.595327103 | 0.557476636 | 0.643457944 |
| dev-novel (maxjac<0.5) | 773 | 0.633247089 | 0.709217335 | 0.701875809 | 0.776455369 |

Two things fall out:

1. On the **novel** subset our lookup-OFF policy is still **below** the strongest official
   baseline: 0.696007 vs hash-regex 0.702856, a deficit of **0.006849**. On the full set the
   deficit is 0.011051. So restricting to novel episodes narrows the gap but does not close it.
2. On the **near** subset our lookup-OFF policy is dramatically worse than hash-regex
   (0.540152 vs 0.603030, deficit **0.062879**) — it under-performs a simple baseline
   precisely where train contains lookalikes.

### 3.4 Why the near subset is hard, and why the router loses there

`build/a4_inspect.py`:

```
dev-near   n=  66 meanlen=   283.0 {'korean': 28, 'other': 26, 'code': 12}
dev-novel  n= 814 meanlen=  4536.7 {'other': 412, 'code': 107, 'korean': 153, 'mcq': 142}

dev-near   per-model mean score ax31-light=0.5076 ax31=0.5303 axk1-think=0.7879  all-3-identical=0.439
dev-novel  per-model mean score ax31-light=0.6284 ax31=0.7049 axk1-think=0.8295  all-3-identical=0.593
```

The near-duplicate episodes are **short** (283 chars mean vs 4537) template-math and short
Korean prompts. On them `axk1-think` is worth **+0.280** over light — the single largest
model-choice payoff anywhere in dev. The frozen lookup-OFF policy routes **0 / 0 / 58** of
880 episodes to `axk1-think` in fast / balanced / premium, and its length-driven features
score short prompts as easy. It therefore leaves almost all of that headroom on the table
(9.77% captured). This is a **missed opportunity created by the group structure, not an
inflation caused by it.**

---

## 4. Verdict on the 0.684318 headline

*(finalised below once §2 completes)*
