<!--
SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
SPDX-License-Identifier: Apache-2.0
-->

# A3 — DATASET IDENTITY AUDIT

Auditor track: A3. Date: 2026-08-17. Repo: `C:/portable/skt_LLM/LLMRoute` @ `5baef67`.
Python: `C:/portable/skt_LLM/LLMRoute/.venv/Scripts/python.exe`, `PYTHONPATH=src`.

**VERDICT: PASS.** All five sub-checks verified with executed commands and pasted output.
No training or evaluation path reads `inputs-base.json`. No model member is fit on dev.
One material caveat (not a rule violation, but must not be reported as generalization) is
recorded in Finding 6.

Scripts used (created under `build/`, per the write allowlist):

- `C:\portable\skt_LLM\LLMRoute\build\a3_verify.py`   → output `build\a3_verify_out.txt`
- `C:\portable\skt_LLM\LLMRoute\build\a3_verify2.py`  → output `build\a3_verify2_out.txt`

Command form (both):

```powershell
$env:PYTHONPATH='src'; & C:/portable/skt_LLM/LLMRoute/.venv/Scripts/python.exe C:/portable/skt_LLM/LLMRoute/build/a3_verify.py
$env:PYTHONPATH='src'; & C:/portable/skt_LLM/LLMRoute/.venv/Scripts/python.exe C:/portable/skt_LLM/LLMRoute/build/a3_verify2.py
```

---

## 1. Episode counts — CONFIRMED

```
==============================================================================
SECTION 1 : EPISODE COUNTS
==============================================================================
  materialized/train/inputs.json     episodes=  1760  expected=  1760  OK
  materialized/dev/inputs.json       episodes=   880  expected=   880  OK
  train/inputs-base.json             episodes=  1736  expected=  1736  OK
  dev/inputs-base.json               episodes=   868  expected=   868  OK
  train/outcomes.json                episodes=  1760
  dev/outcomes.json                  episodes=   880
  materialized/train/inputs.json     split='train' challenge_id='ossp-2026-llm-router-challenge' schema=1
  materialized/dev/inputs.json       split='dev' challenge_id='ossp-2026-llm-router-challenge' schema=1
  train/inputs-base.json             split='train' challenge_id='ossp-2026-llm-router-challenge' schema=1
  dev/inputs-base.json               split='dev' challenge_id='ossp-2026-llm-router-challenge' schema=1
```

All four counts are exactly as stated in the brief. `split` / `challenge_id` / `schema_version`
fields are internally consistent. Note `outcomes.json` uses key `episodes` (one row per episode,
with a nested `models` dict), not a flat `outcomes` list.

---

## 2. SHA-256 vs `data/public-data.v1.json` — ALL 8 MATCH

`source_fetch_selection` in the manifest resolves to `data/{split}/aime-selection.json`.

```
==============================================================================
SECTION 2 : SHA-256 vs data/public-data.v1.json
==============================================================================
  [train] materialized_inputs    MATCH
          manifest = 029a0fb1f70432a05b837a1291d86d42278bb202d808a6a12911b0dae8628ac4
          on-disk  = 029a0fb1f70432a05b837a1291d86d42278bb202d808a6a12911b0dae8628ac4
          file     = data/materialized/train/inputs.json  bytes=7865187
  [train] inputs_base            MATCH
          manifest = 71abfb65bcdaa2bae14c1a405cfa3e218f53a79294b97b1dba106aeaab528613
          on-disk  = 71abfb65bcdaa2bae14c1a405cfa3e218f53a79294b97b1dba106aeaab528613
          file     = data/train/inputs-base.json  bytes=7855233
  [train] outcomes               MATCH
          manifest = 97a5a787086b3e1d9fa9c7945518543540e527ea248df4a4760de581b612a4ba
          on-disk  = 97a5a787086b3e1d9fa9c7945518543540e527ea248df4a4760de581b612a4ba
          file     = data/train/outcomes.json  bytes=926348
  [train] source_fetch_selection MATCH
          manifest = 97a9b0c9fd55b0faf18d11aedb5071d25d3dc4ab206425d6cdc9881f69ceb3c3
          on-disk  = 97a9b0c9fd55b0faf18d11aedb5071d25d3dc4ab206425d6cdc9881f69ceb3c3
          file     = data/train/aime-selection.json  bytes=6109
  [dev  ] materialized_inputs    MATCH
          manifest = 5920f9ea9e3da147aa546659054feb08afb7e11a0e4db6967b293ff79b759abc
          on-disk  = 5920f9ea9e3da147aa546659054feb08afb7e11a0e4db6967b293ff79b759abc
          file     = data/materialized/dev/inputs.json  bytes=3915215
  [dev  ] inputs_base            MATCH
          manifest = f8fee2d272fd4ce383a38986fa5213f221a205d725c3e9202c7aa84800915b25
          on-disk  = f8fee2d272fd4ce383a38986fa5213f221a205d725c3e9202c7aa84800915b25
          file     = data/dev/inputs-base.json  bytes=3908459
  [dev  ] outcomes               MATCH
          manifest = acb7c5ed522c4e1b65e9ab14b3fe9458fcba32eb3d9de8d3f53e24b8904d2e66
          on-disk  = acb7c5ed522c4e1b65e9ab14b3fe9458fcba32eb3d9de8d3f53e24b8904d2e66
          file     = data/dev/outcomes.json  bytes=461433
  [dev  ] source_fetch_selection MATCH
          manifest = 4a73c7428e0d795df2e2b71cc2d267158e9ab52f070a5b0013aca7f62d46f859
          on-disk  = 4a73c7428e0d795df2e2b71cc2d267158e9ab52f070a5b0013aca7f62d46f859
          file     = data/dev/aime-selection.json  bytes=3063
  ALL SHA-256 MATCH: True
  manifest counts:
    train: manifest total=1760 redistributable=1736 source_fetch=24
    train: on-disk materialized=1760 base=1736 diff=24 aime-selection entries=24
    train: total==materialized -> True; redistributable==base -> True; base+source_fetch==total -> True
    dev: manifest total=880 redistributable=868 source_fetch=12
    dev: on-disk materialized=880 base=868 diff=12 aime-selection entries=12
    dev: total==materialized -> True; redistributable==base -> True; base+source_fetch==total -> True
```

Arithmetic closes exactly: 1736 + 24 = 1760, 868 + 12 = 880.
The materialized files on disk are byte-identical to the published registry, so the AIME merge
was performed by the sanctioned materializer and not hand-edited.

---

## 3. `episode_id` set identity and prompt-text overlap — CLEAN

```
==============================================================================
SECTION 3 : EPISODE_ID SET IDENTITY
==============================================================================
  train: inputs episodes=1760 unique_ids=1760 dup=0
  train: outcomes rows=1760 unique_episode_ids=1760 dup_rows=0
  train: distinct model-key tuples per episode = [('ax31', 'ax31-light', 'axk1-think')]
  train: unique (episode_id,model_id) pairs=5280
  train: inputs\outcomes (missing outcomes) = 0 []
  train: outcomes\inputs (extra outcomes)   = 0 []
  train: EXACT 1:1 -> True
  train: base ids subset of materialized -> True; materialized-only ids = 24
  train: materialized-only (AIME) sample = ['train-0035', 'train-0044', 'train-0045', 'train-0177', 'train-0193', 'train-0207']
  train: aime-selection ids == materialized-only ids -> True (n_sel=24)
  train: AIME prompt_sha256 mismatches = 0 []
  train: inputs/outcomes row order identical -> True
  dev: inputs episodes=880 unique_ids=880 dup=0
  dev: outcomes rows=880 unique_episode_ids=880 dup_rows=0
  dev: distinct model-key tuples per episode = [('ax31', 'ax31-light', 'axk1-think')]
  dev: unique (episode_id,model_id) pairs=2640
  dev: inputs\outcomes (missing outcomes) = 0 []
  dev: outcomes\inputs (extra outcomes)   = 0 []
  dev: EXACT 1:1 -> True
  dev: base ids subset of materialized -> True; materialized-only ids = 12
  dev: materialized-only (AIME) sample = ['dev-0008', 'dev-0031', 'dev-0059', 'dev-0140', 'dev-0149', 'dev-0240']
  dev: aime-selection ids == materialized-only ids -> True (n_sel=12)
  dev: AIME prompt_sha256 mismatches = 0 []
  dev: inputs/outcomes row order identical -> True
  TRAIN n DEV episode_id overlap = 0 []

==============================================================================
SECTION 4 : PROMPT TEXT OVERLAP TRAIN vs DEV
==============================================================================
  train texts: 1760 rows, 1760 distinct raw
  dev   texts: 880 rows, 880 distinct raw
  RAW exact overlap (train n dev) = 0
  NORMALIZED (NFKC+strip+lower) overlap = 0
  within-train normalized duplicate texts = 0
  within-dev   normalized duplicate texts = 0
  sha256(text) distinct: train=1760 dev=880 overlap=0 union=2640
```

Findings:

- `inputs` ↔ `outcomes` `episode_id` sets are exactly 1:1 in both splits (no extras, no missing,
  no duplicate rows). Row order is also identical, so index-aligned array construction in
  `harness.load_split` is safe.
- Train and dev `episode_id` sets are disjoint (overlap 0).
- Zero prompt-text overlap between train and dev, both raw-exact and after
  NFKC + strip + lowercase normalization. There are also **zero duplicate prompts within**
  either split, which is why `sha256(text)` yields exactly 2640 distinct keys — this is the
  reason the lookup table can hit 100% with 2640 rows.
- The 24 train / 12 dev episodes present only in the materialized file are exactly the
  `aime-selection.json` id sets, and every one of their prompt texts hashes to the
  `prompt_sha256` recorded in the selection file (0 mismatches). The AIME merge is verified
  content-correct, not just count-correct.

---

## 4. CRITICAL — which inputs file does training actually read?

**Answer: `data/materialized/{split}/inputs.json` (1760 / 880), always. `inputs-base.json` is
never read by any training or evaluation path.**

Single point of resolution, `exp/harness.py` lines 53–59 (verbatim):

```python
def load_split(split: str) -> dict:
    """Load inputs + outcomes for a split with aligned per-episode arrays."""

    if split in _CACHE:
        return _CACHE[split]
    inputs = load_input(ROOT / "data" / "materialized" / split / "inputs.json")
    outcomes = load_outcomes(ROOT / "data" / split / "outcomes.json")
```

with `ROOT = Path(__file__).resolve().parents[1]` (`exp/harness.py:29`), i.e. the repo root.
The path is a hard-coded literal — there is no environment variable, CLI flag, or config key
that can redirect it.

`exp/precompute.py` — the only producer of `build/feats/*`, which every model consumes — goes
through that function and nothing else (lines 18, 25–27):

```python
from harness import ROOT, load_split
...
    for split in ("train", "dev"):
        table = load_split(split)
        texts = table["texts"]
```

`exp/feat_lib.py` performs **no file I/O at all** — it receives `texts` in memory:

```
$ grep -n "open(\|json\.\|np.load\|Path(\|ROOT\|data/" exp/feat_lib.py
(no output)
```

`exp/build_final.py` reads only `build/feats/*` (`load_feats`, lines 44–48) plus one
`harness.load_split` call for the lookup table (line 275), so it inherits the same path.

Exhaustive call-site sweep — every `load_input` / `load_outcomes` call in `exp/`:

```
$ grep -rn "load_input(\|load_outcomes(" exp --include=*.py
exp/harness.py:58:    inputs = load_input(ROOT / "data" / "materialized" / split / "inputs.json")
exp/harness.py:59:    outcomes = load_outcomes(ROOT / "data" / split / "outcomes.json")
exp/selftest.py:204:    inputs = load_input(root / "data/toy/inputs.json")
exp/selftest.py:205:    outcomes = load_outcomes(root / "data/toy/outcomes.json")
```

(`selftest.py` uses the 3-episode `data/toy/` fixture for a scorer smoke test; it is not a
training or scoring path for the real splits.)

Repo-wide sweep for any reference to the base file:

```
$ grep -rn "inputs-base|inputs_base" C:\portable\skt_LLM\LLMRoute
data\train\README.md:8:- `inputs-base.json`: 재배포 가능한 prompt 1,736개
tools\materialize_public_data.py:84:    base = _load_json(ROOT / "data" / split / "inputs-base.json")
REUSE.toml:33:  "data/dev/inputs-base.json",
REUSE.toml:34:  "data/train/inputs-base.json",
THIRD_PARTY_NOTICES.md:9:`data/train/inputs-base.json` and `data/dev/inputs-base.json`. Those files are
tests\test_materialize_public_data.py:24:                "inputs_base": ROOT / "data" / split / "inputs-base.json",
data\public-data.v1.json:12:        "inputs_base": "f8fee2d272fd4ce383a38986fa5213f221a205d725c3e9202c7aa84800915b25",
data\public-data.v1.json:25:        "inputs_base": "71abfb65bcdaa2bae14c1a405cfa3e218f53a79294b97b1dba106aeaab528613",
data\dev\README.md:8:- `inputs-base.json`: 재배포 가능한 prompt 868개
```

The only code reader is `tools/materialize_public_data.py` (the materializer that *produces*
the merged file) and its unit test. **Zero experiments are invalidated on this axis.**

Corroborating evidence from the actual on-disk artifacts (Section 5 output, below): every
feature matrix and every fitted member has a leading dimension of 1760 (train) or 880 (dev),
never 1736 or 868.

---

## 5. Is dev ever concatenated into training? — NO for every fitted member

`exp/build_final.py` loads train and dev features separately (lines 78–79) and every `.fit` /
`train` call takes a `*_tr` array. Fit-site inventory (`grep -n` on `exp/build_final.py`):

| Member | Fit call (line) | Fit on |
| --- | --- | --- |
| SVD word | `TruncatedSVD(...).fit(word_tr)` (88) | train |
| SVD char | `TruncatedSVD(...).fit(char_tr)` (89) | train |
| LGBM ×12 heads | `lgb.Dataset(X_svd_tr, label=y)` → `lgb.train` (108–109) | train |
| XGB keep-cols | `keep = np.where(np.diff(X_sp_tr.tocsc().indptr) > 0)[0]` (147) | train |
| XGB ×7 heads | `xgb.QuantileDMatrix(X_sp_tr, ...)` → `xgb.train` (157, 165) | train |
| Scaler (IRT) | `StandardScaler().fit(dense_tr)` (197) | train |
| IRT1d | `irt_mod.fit_irt_linear(X_irt_tr, scores_tr, 1, 0.01)` (204) | train |
| kNN index | `sp.save_npz(OUT/"knn"/"index.npz", x_tr_n)` (215, 218) | train |
| kNN outcomes | `scores=scores_tr, logcost=logcost_tr` (219–223) | train |
| **lookup** | `for split in ("train", "dev")` (280) | **train + dev** |

`X_svd_dev`, `X_sp_dev`, `X_irt_dev`, `x_dev_n` appear **only** on the right-hand side of
`.predict(...)` / similarity calls, i.e. for the reproduce-vs-snapshot `check()` assertions.
No `np.vstack` / `np.concatenate` of a train and a dev array exists anywhere in `exp/`
(the `augment/train_aug.py` and `models/*.py` hits are within-train augmentation and
within-model reshapes, verified individually).

### Shipped-artifact verification (does not trust the source, loads the bundle)

```
==============================================================================
A) kNN index == L2-normalized hstack(word_train, char_train) ?
==============================================================================
  shipped index shape      = (1760, 131072)  nnz=2249927
  rebuilt TRAIN-only shape = (1760, 131072)  nnz=2249927
  rebuilt DEV-only   shape = (880, 131072)  nnz=1125260
  train+dev union would be = (2640, 131072)
  shape match (train-only) = True
  elementwise max |shipped - rebuilt_train| = 0.000e+00  (diff nnz=0)
  IDENTICAL TO TRAIN MATRIX -> True

  dev-row membership probe: max cosine(dev_i, index) == 1.0 would mean dev is in index
  dev->index max cosine: min=0.221903 mean=0.633110 max=0.997603
  #dev rows with max cosine > 0.999999 (i.e. present in index) = 0
  train row i vs index row i cosine: min=1.000000000000 max=1.000000000000

==============================================================================
B) knn/outcomes.npz rows == TRAIN targets ?
==============================================================================
  shipped scores (1760, 3) vs train targets (1760, 3): max abs diff = 0.000e+00
  shipped logcost (1760, 3) vs log(train costs): max abs diff = 0.000e+00
  (dev targets shape would be (880, 3) - NOT present)

==============================================================================
D) xgb keep-cols derived from train only?
==============================================================================
  shipped keep-cols n = 117575
  train-only nonzero cols n = 117575  equal to shipped -> True
  dev-only  nonzero cols n = 108655
  train|dev union      n = 121282  equal to shipped -> False

==============================================================================
F) SVD components: train-only fit?
==============================================================================
  svd-word: shipped (128, 65536)  max|abs diff| vs TRAIN-fit=0.000e+00  vs TRAIN+DEV-fit=5.953e-01
  svd-char: shipped (128, 65536)  max|abs diff| vs TRAIN-fit=0.000e+00  vs TRAIN+DEV-fit=2.396e-01
```

The kNN index is **(1760, 131072) as specified** and is bit-identical (max abs diff `0.000e+00`,
zero differing nonzeros) to the train-only matrix. The membership probe is the stronger form of
the claim: not one dev row appears anywhere in the index (max cosine over all 880 dev rows is
0.997603 < 1, achieved by a genuinely similar-but-distinct neighbour), while every train row
matches its own index row at cosine 1.000000000000.

`keep-cols` (117575) equals the train-only nonzero-column set exactly and is *not* the
train∪dev union (121282) — a dev-contaminated build would have produced the larger number.
SVD components reproduce a train-only fit to 0.000e+00 and differ from a train+dev fit by
O(0.1–0.6), which rules out silent union fitting.

IRT scaler (refined check, since a near-constant column made the first pass ambiguous):

```
$ python -c "... StandardScaler().fit(d_tr) vs .fit(vstack([d_tr,d_dev])) ..."
vs TRAIN-fit  StandardScaler: max|mean diff|=0.000e+00  max|scale diff|=0.000e+00
vs T+D-fit    StandardScaler: max|mean diff|=1.184e-02  max|scale diff|=9.865e-01
cols where my naive std recon differed: [3] raw train std there: [3.1197267e-14] sklearn scale: [1.]
```

The shipped scaler is bit-exact to a train-only `StandardScaler` and demonstrably not a
train+dev one. (The lone 1.0 discrepancy in the first pass was my hand-rolled `np.std`
reconstruction on dense column 3, whose train std is 3.12e-14 and which sklearn clamps to 1.0.)

Supporting shapes:

```
  build/feats dense-train=(1760, 81) word-train=(1760, 65536) char-train=(1760, 65536)
  build/feats targets-train: scores(1760, 3), costs(1760, 3), in_tokens(1760, 3), out_tokens(1760, 3), gens(1760, 3)
  build/feats dense-dev=(880, 81) word-dev=(880, 65536) char-dev=(880, 65536)
  build/feats targets-dev: scores(880, 3), costs(880, 3), in_tokens(880, 3), out_tokens(880, 3), gens(880, 3)
  models/final-v1/knn/index.npz: shape=(1760, 131072) nnz=2249927 dtype=float64 format=csr
  build/bundle-nolookup/knn/index.npz: shape=(1760, 131072) nnz=2249927 dtype=float64 format=csr
  models/final-v1/lookup.npz: key(2640, 32), scores(2640, 3), costs(2640, 3)
  build/bundle-nolookup/lookup.npz: key(0, 32), scores(0, 3), costs(0, 3)
```

`build/final-model/knn/index.npz` is ABSENT (that staging dir has been cleared); the audit
therefore verified the *shipped* `models/final-v1/` bundle directly, which is the stronger target.

---

## 6. Finding — the ONE place dev enters a shipped artifact (severity: HIGH, rule-allowed)

```
==============================================================================
C) lookup.npz key composition: which split(s) are memorized?
==============================================================================
  lookup keys: 2640 rows, 2640 distinct
  train: 1760/1760 prompt hashes present in lookup  (100.0%)
  dev: 880/880 prompt hashes present in lookup  (100.0%)
  lookup keys NOT explained by train+dev public prompts = 0
  train+dev prompt hashes NOT in lookup                 = 0
```

`models/final-v1/lookup.npz` is exactly the 1760 train + 880 dev public prompts, no more and no
less (both residual sets are 0). This independently reconfirms the lead's finding at the
artifact level. Dev is **not fit into any model**, but its realized scores and costs **are
shipped verbatim** keyed by `sha256(prompt)`.

Consequence for reporting: the lookup-ON Dev figure **0.760284090909** is memorization recall,
not generalization. The generalization number is the lookup-OFF **0.684318**. This is
rule-allowed under `docs/CHALLENGE_RULES.md` but must never be presented as a generalization
result. A3 does not challenge the lead on this; it corroborates it.

## 7. Finding — dev-informed hyperparameter selection (severity: MEDIUM, disclose)

No fitting on dev, but dev *is* consulted when selecting the per-tier utilization constants that
end up frozen in `policy.json`. `exp/final_policy.py` docstring (lines 4–12) is explicit and
honest about it:

```
"""Final per-tier utilization selection under a triple risk gate.

Gates (all must pass at the chosen utilization):
  bootstrap P(over) on dev        < 1.0% (premium < 0.5%)
  category-shift P(over) (+-50%)  < 5.0%
  CLT P(over) from train residuals< 1.0%
Utilization = max grid point passing all gates; reported with dev official
score for transparency. Dev is used for measurement, not for picking the
utilization that maximizes score.
"""
```

`exp/freeze_final.py:main` likewise runs `final_assignment("dev")` and `evaluate("dev", ...)`.
Mitigations already in place: blend weights are searched on a train-internal 5-fold holdout with
dev untouched (`exp/optimize_blend.py:4` — *"Blend-weight search scored on the train-internal
holdout (dev untouched)"*; `main` calls `load_split("train")` at line 59), kNN `k` is chosen by
train OOF MSE (`exp/models/knn.py:201` — `best_k = min(KS, key=lambda k: results[k]["score_mse_mean"])`
over the OOF dict), and `exp/holdout_check.py` provides a leak-free pseudo-dev rank-stability gate.

Net effect: the utilization constants are chosen by a *risk* criterion rather than a score
criterion, so the optimism is small — but the lookup-OFF dev number 0.684318 is still a
lightly dev-tuned figure, not a pristine held-out one. Disclose it as such.

---

## Summary table

| # | Check | Result |
| --- | --- | --- |
| 1 | materialized train=1760, dev=880; base train=1736, dev=868 | PASS |
| 2 | 8/8 SHA-256 match `public-data.v1.json`; counts close (1736+24=1760, 868+12=880) | PASS |
| 3 | inputs↔outcomes ids exactly 1:1 both splits; train∩dev ids = 0 | PASS |
| 3b | train/dev prompt overlap: 0 raw, 0 normalized; 0 intra-split dupes | PASS |
| 3c | AIME-only ids == aime-selection ids; 0 `prompt_sha256` mismatches | PASS |
| 4 | Training reads materialized only; `inputs-base.json` read by no exp/ code | PASS |
| 5 | kNN index (1760, 131072), bit-identical to train matrix; 0 dev rows present | PASS |
| 5b | SVD / scaler / keep-cols / IRT / LGBM / XGB all train-only, verified on shipped bundle | PASS |
| 6 | `lookup.npz` = train 1760 + dev 880, 100% coverage — memorization, rule-allowed | DISCLOSE |
| 7 | Utilization constants selected with dev in the loop (risk gate, not score max) | DISCLOSE |

No experiment is invalidated by a dataset-identity defect. No file under `models/final-v1/**`,
`src/**`, `container/**`, `tests/**`, or `data/**` was modified; all audit artifacts live in
`build/` and `exp/audit/`.
