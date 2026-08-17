<!--
SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
SPDX-License-Identifier: Apache-2.0
-->

# A1 — Scoring Pipeline Integrity

Auditor: Track A1 subagent. Date: 2026-08-17. Repo: `C:/portable/skt_LLM/LLMRoute` @ `5baef67`.

**VERDICT: PASS**, with two documented caveats (one reproducibility gap in the experiment
registry, one reporting gotcha in `budget_ratio`). Nothing found that inflates any reported score.

Everything below was executed. Scripts live in `C:/portable/skt_LLM/LLMRoute/build/audit-a1/`:

| script | purpose |
| --- | --- |
| `xcheck.py` | registry vs private harness vs official CLI, 3-way, 12-decimal exact |
| `counts_and_denominator.py` | episode-count integrity + missing-episode denominator attack |
| `decimal_boundary.py` | adversarial Decimal-vs-float64 budget boundary cases |
| `reverse_validate.py` | known-answer reverse validation through the official CLI |
| `shipped_and_basefile.py` | re-score shipped reports; `inputs-base.json` swap probe |
| `float_margin.py` | headroom between the float selection path and the Decimal verdict |
| `registry_arith.py` | internal arithmetic consistency of all 49 registry rows |

Command prefix for every run: `$env:PYTHONPATH='src'; ./.venv/Scripts/python.exe <script>`

---

## 1. Harness vs official scorer — 12-decimal cross-check

### 1a. Structural finding (read first)

`exp/harness.py:136` is **not an independent scorer**. `evaluate()` ends in:

```python
from ossp_router.scoring import score_submissions   # harness.py:37
...
return score_submissions(inputs, outcomes, submissions, POLICY)   # harness.py:136
```

Every registry-writing path (`exp/eval_preds.py:146`, `exp/compare_decisions.py:89`,
`exp/final_policy.py:137`, `exp/freeze_final.py:71`, `exp/retro_register.py:43`) goes
`evaluate(...)` → `registry_lib.metrics_from_report(...)`. So *all 49 registry rows carry
official-scorer numbers by construction.* There is no private scoring formula to diverge.

The genuinely independent private path is `exp/eval_preds.py:47 fast_eval()` — a numpy float64
mean/ratio used inside the utilization sweeps to **select** a configuration. That is what
actually needed checking, and it is checked in §1c and §7.

### 1b. Three-way comparison — result

For each experiment the assignment was rebuilt from its stored `exp/E0xx/detail.npz`
(`picks_dev`, shape `(3, 880)`), then scored three ways:

- **registry** — the number recorded in `exp/registry.jsonl`
- **harness** — `exp/harness.evaluate("dev", assign)` in-process
- **CLI** — submissions written to JSON, then `python -m ossp_router.cli self-check`
  (full protocol re-parse from disk, official code path)

The literal top 5 registry rows by `weighted_final` are **E004, E041, E039, E044, E040**.
Only E004 and E039 are re-derivable (see §2). The other three have no stored artifacts, so the
set was completed with the next-highest re-derivable rows: **E014, E017, E011, E010**.
All 25 re-derivable rows were then checked, not just five.

```
$ ./.venv/Scripts/python.exe build/audit-a1/xcheck.py E004 E039 E014 E017 E011 E010

E004 budget-oracle (dev)
  registry_final=0.803693181818  harness_final=0.803693181818  cli_final=0.803693181818
  fast      score reg/harness/cli = 0.759375 / 0.759375 / 0.759375
  fast      ratio reg/harness/cli = 1.249121708721 / 1.249121708721 / 1.249121708721
  fast      float(numpy) score/ratio = 0.759375000000 / 1.249121708721   pass=True
  balanced  score reg/harness/cli = 0.807386363636 / 0.807386363636 / 0.807386363636
  balanced  ratio reg/harness/cli = 1.988687399633 / 1.988687399633 / 1.988687399633
  balanced  float(numpy) score/ratio = 0.807386363636 / 1.988687399633   pass=True
  premium   score reg/harness/cli = 0.859090909091 / 0.859090909091 / 0.859090909091
  premium   ratio reg/harness/cli = 3.994989149656 / 3.994989149656 / 3.994989149656
  premium   float(numpy) score/ratio = 0.859090909091 / 3.994989149656   pass=True
  VERDICT: AGREE (12dp exact)

E039 xgb-mono+lagrangian
  registry_final=0.701505681818  harness_final=0.701505681818  cli_final=0.701505681818
  fast      score reg/harness/cli = 0.673295454545 / 0.673295454545 / 0.673295454545
  fast      ratio reg/harness/cli = 1.246978817408 / 1.246978817408 / 1.246978817408
  balanced  score reg/harness/cli = 0.697727272727 / 0.697727272727 / 0.697727272727
  balanced  ratio reg/harness/cli = 1.789707813297 / 1.789707813297 / 1.789707813297
  premium   score reg/harness/cli = 0.742897727273 / 0.742897727273 / 0.742897727273
  premium   ratio reg/harness/cli = 3.990814093944 / 3.990814093944 / 3.990814093944
  VERDICT: AGREE (12dp exact)

E014 irt1d+lagrangian
  registry_final=0.694829545455  harness_final=0.694829545455  cli_final=0.694829545455
  fast      score  0.665340909091 / 0.665340909091 / 0.665340909091
  fast      ratio  1.236250206554 / 1.236250206554 / 1.236250206554
  balanced  score  0.697159090909 / 0.697159090909 / 0.697159090909
  balanced  ratio  1.963204578964 / 1.963204578964 / 1.963204578964
  premium   score  0.731818181818 / 0.731818181818 / 0.731818181818
  premium   ratio  3.983409010715 / 3.983409010715 / 3.983409010715
  VERDICT: AGREE (12dp exact)

E017 irt2d+lagrangian
  registry_final=0.69375  harness_final=0.69375  cli_final=0.69375
  fast      score  0.667329545455 / 0.667329545455 / 0.667329545455
  fast      ratio  1.246419567319 / 1.246419567319 / 1.246419567319
  balanced  score  0.691193181818 / 0.691193181818 / 0.691193181818
  balanced  ratio  1.952033875942 / 1.952033875942 / 1.952033875942
  premium   score  0.731534090909 / 0.731534090909 / 0.731534090909
  premium   ratio  3.915564586249 / 3.915564586249 / 3.915564586249
  VERDICT: AGREE (12dp exact)

E011 knn-k20+greedy
  registry_final=0.691875  harness_final=0.691875  cli_final=0.691875
  fast      score  0.664772727273 / 0.664772727273 / 0.664772727273
  fast      ratio  1.24687695313 / 1.24687695313 / 1.24687695313
  balanced  score  0.694034090909 / 0.694034090909 / 0.694034090909
  balanced  ratio  1.926081884034 / 1.926081884034 / 1.926081884034
  premium   score  0.725852272727 / 0.725852272727 / 0.725852272727
  premium   ratio  3.937880570444 / 3.937880570444 / 3.937880570444
  VERDICT: AGREE (12dp exact)

E010 knn-k20+lagrangian
  registry_final=0.690511363636  harness_final=0.690511363636  cli_final=0.690511363636
  fast      score  0.664772727273 / 0.664772727273 / 0.664772727273
  fast      ratio  1.234790856314 / 1.234790856314 / 1.234790856314
  balanced  score  0.691761363636 / 0.691761363636 / 0.691761363636
  balanced  ratio  1.96945732647 / 1.96945732647 / 1.96945732647
  premium   score  0.723579545455 / 0.723579545455 / 0.723579545455
  premium   ratio  3.9513743873 / 3.9513743873 / 3.9513743873
  VERDICT: AGREE (12dp exact)
```

Full sweep over every re-derivable row:

```
$ ./.venv/Scripts/python.exe build/audit-a1/xcheck.py E004 E039 E014 E017 E011 E010 E005 E006 \
    E007 E008 E009 E016 E018 E021 E022 E023 E024 E027 E030 E031 E032 E033 E038 E042 E043 \
    | Select-String "VERDICT" | Group-Object Line

Count Name
----- ----
   25   VERDICT: AGREE (12dp exact)
```

**Zero divergences.** 25 experiments × 3 tiers × (score + cost ratio) + 25 final scores
= 175 quantities, all identical across registry / harness / official CLI at 12 decimals.

### 1c. The FINAL shipped policy (E049) — independently re-scored

E049 has no `detail.npz`, but the shipped lookup-disabled bundle output in `build/dev-nolookup/`
reproduces it. Re-scored from disk with the official CLI:

```
$ ./.venv/Scripts/python.exe build/audit-a1/shipped_and_basefile.py

  dev-final      fresh_final=0.760284090909  published(dev-final-report.json)=0.760284090909  byte_identical_report=True
     fast      q=0.738636363636 ratio=1.157056214093 pass=True
     balanced  q=0.742045454545 ratio=1.204906954308 pass=True
     premium   q=0.807386363636 ratio=2.08673316736 pass=True
  dev-nolookup   fresh_final=0.684318181818  published(dev-nolookup-report.json)=0.684318181818  byte_identical_report=True
     fast      q=0.659090909091 ratio=1.180437123011 pass=True
     balanced  q=0.691761363636 ratio=1.677355782407 pass=True
     premium   q=0.710511363636 ratio=2.823888413321 pass=True
  ctr-dev-all    fresh_final=0.760284090909  published(ctr-dev-report.json)=0.760284090909  byte_identical_report=True
```

Registry E049 records fast `0.659090909091` / `1.180437123011`, balanced `0.691761363636` /
`1.677355782407`, premium `0.710511363636` / `2.823888413321`, final `0.684318181818`.
**All six tier numbers and the final match the fresh CLI re-score exactly.** The lead's honest
lookup-OFF figure of 0.684318 is confirmed against the official scorer, not merely to 4 decimals
but to all 12.

Host report and container report are byte-identical dicts:

```
  host final_score      = 0.760284090909
  container final_score = 0.760284090909
  full report dicts equal = True
```

---

## 2. CAVEAT — 24 of 49 registry rows cannot be independently re-derived

`build/preds/` does not exist (`build/` is gitignored, `.gitignore:22`), and the
`decision`-family rows written by `exp/compare_decisions.py` never persist their picks —
`registry_lib.register(...)` is called there with no `artifacts` and no `detail.npz` dump
(contrast `exp/eval_preds.py:177-190`, which does write `detail.npz`).

```
$ ls build/preds
ls: cannot access '.../build/preds': No such file or directory

$ ./.venv/Scripts/python.exe build/audit-a1/registry_arith.py
registry rows checked: 49
weighted-sum mismatches (> 2e-12): 0
budget_pass flag vs stored cost_ratio mismatches: 0

rows re-derivable from stored artifacts: 25
rows NOT re-derivable (no picks, no pred set): 24
    E041 decision 0.701960227273 blend6x-cost3+best-decision
    E044 decision 0.701363636364 blend4-cost3+best-decision
    E040 decision 0.700994318182 blend7-cost3+best-decision
    E046 decision 0.700909090909 blend4-cost2b+best-decision
    E036 decision 0.700113636364 blend6-cost2+best-decision
    ...
```

Three of the literal top 5 (**E041 0.701960, E044 0.701364, E040 0.700994**) fall in this set.
Their numbers are **UNVERIFIED** — not contradicted, just not checkable.

Mitigating evidence: all 49 rows are internally arithmetically consistent
(`final == 0.4·fast + 0.3·balanced + 0.3·premium`, busted tiers contributing 0, zero mismatches
above 2e-12), and every `budget_pass` flag agrees with its own stored `cost_ratio`. And the code
path that produced them provably calls the official scorer. But an auditor cannot rerun them.

**These rows must not be cited as headline results.** The defensible numbers are the ones that
re-derive: E049 = 0.684318181818 (lookup OFF) and the shipped 0.760284090909 (lookup ON,
memorization — see the lead's finding 2).

---

## 3. Episode counts — 880 / 1760, exactly once each

```
$ ./.venv/Scripts/python.exe build/audit-a1/counts_and_denominator.py

PART 1: shipped submission dirs -- count / dup / missing / extra
  dev-final/fast.json      n=880 unique=880 expected=880 dups=[] missing=[] extra=[] order_matches_input=True
  dev-final/balanced.json  n=880 unique=880 expected=880 dups=[] missing=[] extra=[] order_matches_input=True
  dev-final/premium.json   n=880 unique=880 expected=880 dups=[] missing=[] extra=[] order_matches_input=True
  dev-nolookup/fast.json   n=880 unique=880 expected=880 dups=[] missing=[] extra=[] order_matches_input=True
  dev-nolookup/balanced.json n=880 unique=880 expected=880 dups=[] missing=[] extra=[] order_matches_input=True
  dev-nolookup/premium.json  n=880 unique=880 expected=880 dups=[] missing=[] extra=[] order_matches_input=True
  ctr-dev-all/fast.json    n=880 unique=880 expected=880 dups=[] missing=[] extra=[] order_matches_input=True
  ctr-dev-all/balanced.json n=880 unique=880 expected=880 dups=[] missing=[] extra=[] order_matches_input=True
  ctr-dev-all/premium.json  n=880 unique=880 expected=880 dups=[] missing=[] extra=[] order_matches_input=True
```

Train side (built by the official CLI, then scored):

```
$ python -m ossp_router.cli always-light --input data/materialized/train/inputs.json --output-dir <...>
  rc=0  decisions per tier: 1760
   fast      n= 1760 counts={'ax31': 0, 'ax31-light': 1760, 'axk1-think': 0}
```

### 3a. Proof that a missing episode CANNOT shrink the denominator

Two independent guards, both in `src/ossp_router/scoring.py`.

**Guard 1 — the denominator is the input count, never the decision count.**

```python
# scoring.py:157
quality_score = quality_total / Decimal(len(inputs.episodes))
# scoring.py:236
final_score = final_weighted_points / Decimal(len(inputs.episodes))
```

Static confirmation:

```
PART 2: denominator source in scoring.py
  scoring.py:157: quality_score = quality_total / Decimal(len(inputs.episodes))
  scoring.py:171: "num_episodes": len(inputs.episodes),
  scoring.py:236: final_score = final_weighted_points / Decimal(len(inputs.episodes))
  -> occurrences of 'len(decisions)' in scoring.py: 0
  -> occurrences of 'len(inputs.episodes)' in scoring.py: 3
```

`len(decisions)` appears **zero** times. Dropping a decision could only ever *lower* the numerator,
never the denominator — so a dropped hard episode would hurt, not help.

**Guard 2 — you never get that far; the set comparison rejects first.**

```python
# scoring.py:103-116
def _decision_index(inputs, submission):
    expected = {episode.episode_id for episode in inputs.episodes}
    decisions = {d.episode_id: d.model_id for d in submission.decisions}
    missing = sorted(expected - set(decisions))
    extra = sorted(set(decisions) - expected)
    if missing or extra:
        raise ScoringError(f"{submission.tier} decision 범위 오류: 누락={missing}, 초과={extra}")
```

Live attacks against the real 880-episode dev submission:

```
PART 3: adversarial submissions -- can a MISSING episode shrink the denominator?
  baseline (untampered)   fast quality_score=0.738636363636  num_episodes=880  final=0.760284090909

  (a) DROP decision dev-0001 from fast -> n=879
      REJECTED [ScoringError]: fast decision 범위 오류: 누락=['dev-0001'], 초과=[]

  (a2) DROP the worst-scoring episode dev-0001 (light score=0) -> n=879
      REJECTED [ScoringError]: fast decision 범위 오류: 누락=['dev-0001'], 초과=[]

  (b) DUPLICATE decision dev-0001 -> n=881
      REJECTED [ProtocolError]: 중복 decision episode_id: dev-0001

  (c) ADD extra episode 'zzz-not-a-real-episode' -> n=881
      REJECTED [ScoringError]: fast decision 범위 오류: 누락=[], 초과=['zzz-not-a-real-episode']

  (d) SWAP one real episode for a bogus one (count stays 880)
      REJECTED [ScoringError]: fast decision 범위 오류: 누락=['dev-0001'], 초과=['zzz-not-a-real-episode']

  (e) OMIT the premium tier submission entirely
      REJECTED [ScoringError]: submission은 tier별로 정확히 하나씩 필요합니다: ['fast', 'balanced', 'premium']

  (f) REMOVE one outcome row from the outcomes matrix
      REJECTED [ScoringError]: outcome 행렬이 완전하지 않습니다: 누락=[('dev-0001', 'ax31'), ...]
```

Case (d) is the important one: a **count-preserving** substitution is still caught, because the
guard compares *sets of ids*, not lengths. Duplicate ids die even earlier, in
`protocol.py:434-436`, at parse time.

Same behaviour through the official CLI on the train split:

```
  R4b: truncate train fast.json to 1759 decisions and re-run the OFFICIAL CLI
  fast.json now has 1759 decisions
$ python -m ossp_router.cli self-check --input .../train/inputs.json --outcomes .../train/outcomes.json ...
  rc=2  오류: fast decision 범위 오류: 누락=['train-1760'], 초과=[]
  report written? False
```

### 3b. Denominator-swap via the wrong input file is also blocked

`data/dev/inputs-base.json` has 868 episodes vs 880 materialized. Scoring the real submissions
against the smaller file is rejected — the outcomes matrix no longer matches:

```
  data\dev\inputs-base.json: 868 episodes
  data\materialized\dev\inputs.json: 880 episodes
  data\train\inputs-base.json: 1736 episodes
  data\materialized\train\inputs.json: 1760 episodes

  self-check(inputs-base.json, dev-final submissions) rc=2
  stderr: 오류: outcome 행렬이 완전하지 않습니다: 누락=[], 초과=[('dev-0008', 'ax31'), ('dev-0008', 'ax31-light'), ...]
  report written? False
```

---

## 4. Decimal (not float) budget comparison

`src/ossp_router/scoring.py:129-158`, quoted verbatim from the file:

```python
    with localcontext() as context:
        context.prec = SCORING_PRECISION          # 160
        context.rounding = ROUND_HALF_EVEN
        total_cost = Decimal("0")
        light_baseline_cost = Decimal("0")
        quality_total = Decimal("0")
        ...
        for episode in inputs.episodes:
            ...
            total_cost += _cost(selected, policy)
            light_baseline_cost += _cost(light, policy)
            quality_total += selected.score
        ...
        tier_policy = policy.tiers[submission.tier]
        budget_limit = light_baseline_cost * tier_policy.budget_multiplier
        budget_ratio = total_cost / light_baseline_cost
        budget_passed = total_cost <= budget_limit
        quality_score = quality_total / Decimal(len(inputs.episodes))
        tier_score = quality_score if budget_passed else Decimal("0")
```

Every operand is `Decimal`. `_cost` (`scoring.py:49-56`) builds from `Decimal` rates and
`Decimal(input_tokens)`; `protocol.py:209-235 _decimal()` parses money as **strings** with a
regex, never through `float`; `protocol.py:135-141 loads_json` uses `parse_float=Decimal`, so
no JSON number ever transits a binary float. `SCORING_PRECISION = 160` with
`ROUND_HALF_EVEN`. `budget_passed` compares the **unrounded** accumulations.

### 4a. Adversarial cases where float64 and Decimal disagree

Three synthetic datasets built under `build/audit-a1/adv/`, each scored by the **official CLI**
with a custom `--policy`.

```
$ ./.venv/Scripts/python.exe build/audit-a1/decimal_boundary.py

CASE A -- selected cost sits ONE part in 1e30 ABOVE the cap
  2 episodes; light cost 1.0 each -> light total 2, fast cap = 2 * 1.25 = 2.5

  fast = all ax31  (at-limit)             [exact tie, must PASS]
    exact Decimal selected total = 2.50   exact Decimal cap = 2.5
    float64 selected total       = 2.5    float64 cap = 2.5
    -> a float64 scorer would say budget_passed = True
    OFFICIAL total_cost   = 2.5
    OFFICIAL budget_limit = 2.5
    OFFICIAL budget_passed= True   tier_score=0.6   over_budget_zero_applied=False

  fast = all axk1-think  (hair-over)      [rate 1.250000000000000000000000000001]
    exact Decimal selected total = 2.500000000000000000000000000002
    exact Decimal cap            = 2.5
    float64 selected total       = 2.5    float64 cap = 2.5
    -> a float64 scorer would say budget_passed = True      <-- FALSE PASS
    OFFICIAL total_cost   = 2.500000000000000000000000000002
    OFFICIAL budget_limit = 2.5
    OFFICIAL budget_passed= False  tier_score=0   over_budget_zero_applied=True
```

The verdict **flips**: float64 cannot represent the 1e-30 overrun (it rounds to exactly 2.5) and
would have handed out 0.6; the official Decimal scorer correctly zeroes the tier.

```
CASE B -- light costs that float64 cannot sum exactly (0.1 x 10)
  10 episodes; light cost 0.1 each -> Decimal light total 1.0, float64 0.9999999999999999
  fast cap = 1.25 exactly; selected = 0.125 x 10 = 1.25 exactly -> EXACT TIE, must PASS

    float64 selected total = 1.25   float64 cap = 1.2499999999999998
    -> a float64 scorer would say budget_passed = False  (FALSE ZERO)
    OFFICIAL total_cost   = 1.25
    OFFICIAL budget_limit = 1.25
    OFFICIAL budget_passed= True   tier_score=0.6

CASE C -- one part in 1e30 BELOW the cap must still PASS
    OFFICIAL total_cost   = 2.499999999999999999999999999998
    OFFICIAL budget_limit = 2.5
    OFFICIAL budget_passed= True   tier_score=0.9
```

Both error directions are covered: float would have **falsely passed** an over-budget submission
(A) and **falsely zeroed** a legal one (B). The official scorer gets all four boundary cases
right, including the exact-equality tie, which `<=` admits by design.

### 4b. GOTCHA — the reported `budget_ratio` is NOT what decides pass/fail

`budget_ratio` goes through `_score_text` (`scoring.py:41-46`), which quantizes to
`SCORE_DECIMAL_PLACES = 12` with `ROUND_HALF_EVEN`. `budget_passed` does not. Consequence,
straight from the CASE A hair-over report:

```
S4: rounded budget_ratio vs the real pass/fail comparison
  reported budget_ratio = 1.25  (12dp, ROUND_HALF_EVEN)
  tier budget_multiplier= 1.25
  reported ratio <= multiplier ?  True
  ACTUAL budget_passed  = False   <-- decided on UNROUNDED Decimals
  total_cost=2.500000000000000000000000000002  budget_limit=2.5
```

A report can show `budget_ratio` exactly equal to the multiplier and still be `budget_passed:
false`. **Never infer the verdict from the printed ratio — read `budget_passed`.** This is
correct behaviour in the scorer, but it is a trap for any downstream tooling or write-up that
re-derives pass/fail from the rounded field. Our own tooling does not do this
(`registry_lib.metrics_from_report:71` copies `budget_passed` directly), and the check in §2
found 0/49 rows where the stored flag disagrees with the stored ratio.

---

## 5. Reverse validation — known answers through the official scorer

```
$ ./.venv/Scripts/python.exe build/audit-a1/reverse_validate.py

R1: all ax31-light on dev  -- built by the OFFICIAL CLI 'always-light'
$ python -m ossp_router.cli always-light --input data/materialized/dev/inputs.json --output-dir build/audit-a1/reverse/dev-all-light
  rc=0   decisions per tier: 880
$ python -m ossp_router.cli self-check --input ... --outcomes ... --submissions ... --report ...
  rc=0

  --- all ax31-light (dev) ---
   fast      n=  880 counts={'ax31': 0, 'ax31-light': 880, 'axk1-think': 0}
             total_cost=4.381428   budget_limit=5.476785
             budget_ratio=1  multiplier=1.25
             budget_passed=True  quality_score=0.619318181818  tier_score=0.619318181818
   balanced  n=  880 total_cost=4.381428  budget_limit=8.762856
             budget_ratio=1  multiplier=2
             budget_passed=True  quality_score=0.619318181818  tier_score=0.619318181818
   premium   n=  880 total_cost=4.381428  budget_limit=17.525712
             budget_ratio=1  multiplier=4
             budget_passed=True  quality_score=0.619318181818  tier_score=0.619318181818
   FINAL SCORE = 0.619318181818   (expected 0.619318181818)
   MATCH: True
```

Cost ratio is exactly `1` on all three tiers (the selected set *is* the light baseline), so every
cap passes, and the final equals the published all-light reference **0.619318181818** exactly.

```
R2: all axk1-think on dev  -- must BUST every tier -> tier_score 0 -> final 0
  decisions per tier: 880
  --- all axk1-think (dev) ---
   fast      n=  880 counts={'ax31': 0, 'ax31-light': 0, 'axk1-think': 880}
             total_cost=104.25636221   budget_limit=5.476785
             budget_ratio=23.795064579402  multiplier=1.25
             budget_passed=False  quality_score=0.826420454545  tier_score=0  over_budget_zero_applied=True
   balanced  budget_limit=8.762856   ratio=23.795064579402  multiplier=2
             budget_passed=False  quality_score=0.826420454545  tier_score=0  over_budget_zero_applied=True
   premium   budget_limit=17.525712  ratio=23.795064579402  multiplier=4
             budget_passed=False  quality_score=0.826420454545  tier_score=0  over_budget_zero_applied=True
   FINAL SCORE = 0   (expected 0)
   MATCH: True
```

Note the discipline this demonstrates: raw quality would have been **0.826420454545** — far above
every routed result we have (best lookup-OFF tier is 0.710511) — and the scorer still returns
**0** on all three tiers because the budget busts 23.8× the light baseline. The over-budget zero
is real and total.

```
R3: all ax31 (mid model) on dev -- third sanity point, partial bust
   fast      total_cost=9.209955088  budget_limit=5.476785   ratio=2.102044148164  multiplier=1.25
             budget_passed=False  quality_score=0.691761363636  tier_score=0
   balanced  budget_limit=8.762856   ratio=2.102044148164  multiplier=2
             budget_passed=False  quality_score=0.691761363636  tier_score=0
   premium   budget_limit=17.525712  ratio=2.102044148164  multiplier=4
             budget_passed=True   quality_score=0.691761363636  tier_score=0.691761363636
   FINAL SCORE = 0.207528409091
```

R3 is a useful independent consistency probe: `0.3 × 0.691761363636 = 0.2075284090908`, and only
the premium tier survives (2.1020 ≤ 4 but > 2 and > 1.25). The scorer applies the per-tier cap
independently and weights correctly. It also reproduces a number already in the registry —
E002's premium score `0.691761363636` and ratio `2.102044148164` are exactly the all-ax31 figures,
confirming the published prompt-heuristic baseline's premium tier is plain all-ax31.

```
R4: TRAIN split all-light
   fast/balanced/premium  n=1760  total_cost=8.603786  ratio=1
   quality_score=0.597301136364 on all three tiers
   FINAL SCORE = 0.597301136364
```

---

## 6. Float selection path — headroom vs the Decimal verdict

`exp/eval_preds.py:52` accepts a configuration when `ratio <= multiplier + 1e-12`. That
tolerance could in principle admit a config the Decimal scorer rejects. Measured across all 25
re-derivable experiments × 3 tiers:

```
$ ./.venv/Scripts/python.exe build/audit-a1/float_margin.py

max |float ratio - Decimal ratio| over 25 experiments x 3 tiers = 0.000e+00
tightest budget headroom (multiplier - ratio)            = 4.734787e-04  at E038/premium
fast_eval acceptance tolerance in exp/eval_preds.py:52   = 1e-12
headroom / tolerance                                     = 4.735e+08 x
```

The float ratio agrees with the Decimal ratio to all 12 printed decimals in 75/75 cases, and the
tightest configuration clears its cap by 4.7×10⁸ times the tolerance. The 1e-12 slack never
changed a verdict. It remains a latent hazard only for a hypothetical future config that lands
within 1e-12 of a cap; the shipped policy (E049) sits at ratios 1.180 / 1.677 / 2.824, i.e. 5.6% /
16% / 29% under its caps, nowhere near.

---

## 7. Findings summary

| # | Sev | Finding | Status |
| --- | --- | --- | --- |
| 1 | info | `exp/harness.evaluate` *is* `ossp_router.scoring.score_submissions`; all 49 registry rows are official-scorer numbers by construction | PASS |
| 2 | info | 25 re-derivable experiments agree registry / harness / official CLI to 12 decimals, scores and cost ratios, 175/175 quantities | PASS |
| 3 | **medium** | 24/49 registry rows (incl. 3 of the literal top 5: E041, E044, E040) have no stored picks or pred sets — `build/preds/` is gone and `compare_decisions.py` never dumps `detail.npz`. Numbers are internally consistent but **not independently verifiable** | UNVERIFIED — do not headline |
| 4 | info | E049 (0.684318181818 lookup-OFF) re-scores exactly from `build/dev-nolookup/`; shipped 0.760284090909 re-scores exactly and host == container byte-for-byte | PASS |
| 5 | info | 880/1760 decisions, exactly once, no dup/missing/extra, order matches input, in all 9 shipped tier files | PASS |
| 6 | info | Denominator is `len(inputs.episodes)` (3 sites); `len(decisions)` appears 0 times. Drop / dup / extra / count-preserving swap / missing tier / holed outcome matrix all rejected | PASS |
| 7 | info | `inputs-base.json` (868/1736) cannot be substituted — outcome-matrix completeness check rejects it | PASS |
| 8 | info | Budget comparison is exact `Decimal` at prec 160, ROUND_HALF_EVEN; money parsed from strings, `parse_float=Decimal`. Adversarial ±1e-30 cases flip the verdict away from what float64 would say, in both directions | PASS |
| 9 | **medium** | Reported `budget_ratio` is rounded to 12dp while `budget_passed` uses unrounded Decimals — a report can show ratio == multiplier with `budget_passed: false`. Never re-derive the verdict from the printed ratio | Correct behaviour; reporting trap |
| 10 | low | `exp/eval_preds.py:52` selection tolerance `+1e-12` could admit a config the Decimal scorer rejects; measured tightest headroom is 4.7e8× the tolerance, so it never bit | PASS (latent) |
| 11 | info | Reverse validation: all-light dev = 0.619318181818 exact; all-k1 dev busts every tier (ratio 23.795) → final 0 exact; all-ax31 dev = 0.207528409091 (premium only) | PASS |

### Recommended actions

1. **Do not cite E041 / E044 / E040 / E046 or any `decision`-family row as a result.** State
   0.684318181818 (E049, lookup OFF, re-derivable) and 0.760284090909 (shipped, lookup ON,
   memorization) as the only defensible dev numbers.
2. Patch `exp/compare_decisions.py` to dump `detail.npz` the way `exp/eval_preds.py:177-190`
   already does, so future `decision`-family rows are re-derivable. (Out of scope for this audit —
   `exp/` writes were limited to `exp/audit/`.)
3. In any write-up, never re-derive budget pass/fail from the printed `budget_ratio`; quote
   `budget_passed` / `over_budget_zero_applied`.
