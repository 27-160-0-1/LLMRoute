<!--
SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
SPDX-License-Identifier: Apache-2.0
-->

# TRACK A2 — Feature / inference-path leakage audit

Auditor: subagent A2. Date: 2026-08-17. Repo: `C:/portable/skt_LLM/LLMRoute` @ `5baef67` (branch `main`).
Python: `C:/portable/skt_LLM/LLMRoute/.venv/Scripts/python.exe`, `PYTHONPATH=src`.

**VERDICT: PASS — no illegitimate leakage on the inference path.**
Every artifact read at inference is either a fitted model parameter, a train-only
kNN memory (1760 rows), or the rule-explicitly-permitted prompt-hash lookup table.
Zero references to `data/*/outcomes.json` at runtime. The lookup table IS
memorization of public Dev outcomes and is rule-legal, but it makes the
lookup-ON Dev number unusable as a generalization claim.

Scripts used (all under `build/`, none touching read-only paths):
`build/a2_callgraph.py`, `build/a2_artifacts.py`, `build/a2_lookup_verify.py`,
`build/a2_runtime_audit.py`, `build/a2_train_runs.sh`, `build/a2_report_summary.py`,
`build/a2_decision_diff.py`, `build/a2_feature_probe.py`, `build/a2_registry_probe.py`.

---

## 1. Call graph from `container/entrypoint.py` — what is actually reachable

`container/entrypoint.py` is 13 lines and does exactly one thing:

```
container/entrypoint.py:9   from ossp_router.final_router import main
container/entrypoint.py:13  raise SystemExit(main())
```

### 1a. Static import closure

Command:

```
$env:PYTHONPATH='src'; & .venv/Scripts/python.exe build/a2_callgraph.py
```

Output (AST-based, verbatim):

```
ENTRYPOINT imports: ['final_router']

REACHABLE modules (import closure, incl. package __init__):
    __init__.py
    allocation.py
    features_v2.py
    final_router.py
    heuristic.py
    protocol.py

UNREACHABLE modules present in src/ (dead at inference):
    cli.py
    image_evidence.py
    operator_helper.py
    orchestrator.py
    public_runtime.py
    runtime.py
    scoring.py
    tiebreak_latency.py

IMPORT EDGES:
    __init__.py -> protocol.py
    final_router.py -> allocation.py
    final_router.py -> features_v2.py
    final_router.py -> heuristic.py
    final_router.py -> protocol.py
    heuristic.py -> protocol.py
```

The reachable set is exactly the 5 files the task named, plus the package
`__init__.py` (which only re-exports 4 constants from `protocol`).

### 1b. Runtime proof (not just static)

`build/a2_runtime_audit.py` installs `sys.addaudithook`, then executes the real
entrypoint function `final_router.main(...)` on `data/materialized/dev/inputs.json`,
tier `fast`, bundle `models/final-v1`. Verbatim output:

```
OK: fast 제출 파일을 생성했습니다.
return code: 0

=== ossp_router modules ACTUALLY imported at runtime ===
    ossp_router
    ossp_router.allocation
    ossp_router.features_v2
    ossp_router.final_router
    ossp_router.heuristic
    ossp_router.protocol
    ossp_router.resources

=== files opened under the repo, EXCLUDING .py/.pyc source loads ===
    data/materialized/dev/inputs.json
    src/ossp_router/resources/routing-policy.v1.json
    .venv/Lib/site-packages/sklearn/utils/_repr_html/estimator.css
    .venv/Lib/site-packages/sklearn/utils/_repr_html/params.css
    .venv/Lib/site-packages/sklearn/utils/_repr_html/features.css
    .venv/Lib/site-packages/lightgbm/VERSION.txt
    .venv/Lib/site-packages/xgboost/VERSION
    models/final-v1/policy.json
    models/final-v1/svd-word.npz
    models/final-v1/svd-char.npz
    models/final-v1/irt.npz
    models/final-v1/lgbm/score-0.txt  ... lgbm/q90-2.txt   (12 files)
    models/final-v1/xgb/keep-cols.npy
    models/final-v1/knn/index.npz
    models/final-v1/knn/outcomes.npz
    models/final-v1/lookup.npz
    build/a2-runtime-audit/.fast.json.tmp-29288

=== any opened path containing 'outcome' / 'data/' ? ===
    data/materialized/dev/inputs.json
    models/final-v1/knn/outcomes.npz

=== 'scoring'/'runtime'/'orchestrator'/'cli' imported? ===
    ossp_router.scoring                imported = False
    ossp_router.runtime                imported = False
    ossp_router.orchestrator           imported = False
    ossp_router.cli                    imported = False
    ossp_router.public_runtime         imported = False
    ossp_router.image_evidence         imported = False
    ossp_router.operator_helper        imported = False
    ossp_router.tiebreak_latency       imported = False
```

**The only two "outcome-ish" paths opened are the router's own input file and the
bundled train-only kNN memory. `data/dev/outcomes.json` and `data/train/outcomes.json`
are never opened.** (`sklearn` appears only because the host venv's LightGBM imports
it opportunistically; sklearn is not installed in the submission image.)

### 1c. The image ships dead modules but never executes them

```
docker run --rm --network none --entrypoint python3 ossp-router:arm64final \
  -c "import os;[print(os.path.join(r,f)) for r,d,fs in os.walk('/opt/router') for f in sorted(fs)]"
```

The image contains `/opt/router/ossp_router/{cli,image_evidence,operator_helper,
orchestrator,runtime,scoring}.py` — all confirmed **never imported** by 1b. It
contains **no `data/` directory and no `outcomes.json`**. Bundle files inside the
image hash-match the repo manifest:

```
lookup.npz        9764202bcdb1886fa61259e88a0e1d59194342b8ffb14e1dbcd76b8baea36863
knn/outcomes.npz  c2fd336f398b5b6309bd9d88ee72c59a03b29be2390bd76aade79e23abec3b70
lookup key shape (2640, 32) uint8
knn outcomes scores shape (1760, 3)
```

Both digests equal `models/final-v1/manifest.json`.

---

## 2. Static grep of the inference path — every hit classified

Grep command:

```
for f in final_router features_v2 allocation protocol heuristic; do
  grep -n -i -E "outcome|score|input_tokens|output_tokens|answer|label|benchmark|gold|target|aime|belebele|cruxeval" src/ossp_router/$f.py
done
```

Classification key: **(a)** genuine leakage · **(b)** predicted quantity with a
confusingly similar name · **(c)** unreachable / dead · **(P)** real use of realized
outcomes that is *explicitly rule-permitted* (see §4).

### `src/ossp_router/final_router.py`

| line | text | class |
|---|---|---|
| 4, 9 | `"""Final prompt-only router: 4-member score blend..."` / `-> blended scores` | (b) docstring, refers to *predicted* score |
| 11 | `#  4. exact-match lookup (public Train/Dev outcomes, rule-allowed) overrides` | **(P)** honest self-description of the lookup table |
| 78 | `for name in [f"{kind}-{j}" for kind in ("score","lin","lout","q90") ...]` | (b) LightGBM model **filenames** |
| 81–82 | `self.xgb_score = xgb.Booster(); ...load_model(.../"score-multi.json")` | (b) model params |
| 90–94 | `knn_out = np.load(b/"knn"/"outcomes.npz"); self.knn_scores = knn_out["scores"]` | **(P)** train-only realized outcomes, 1760 rows — proven in §3 |
| 95–98 | `lookup = np.load(b/"lookup.npz"); self.lookup_scores/…_costs` | **(P)** train+dev realized outcomes keyed by SHA-256(prompt) |
| 116–121 | `lgbm_score`, `lgbm_lin`, `lgbm_lout`, `lgbm_q90 = ...predict(x_svd)` | (b) predictions |
| 132 | `xgb_score = np.clip(self.xgb_score.predict(dmat), 0.0, 1.0)` | (b) prediction |
| 146 | `irt_score = 1.0 / (1.0 + np.exp(-z))` | (b) prediction |
| 164–167 | `knn_score = np.where(ok[:,None], (w[:,:,None]*self.knn_scores[idx]).sum(...)…)` | **(P)** neighbour-averaged *train* outcomes (legitimate training data) |
| 170–171 | `blend_score = np.clip((xgb_score+irt_score+knn_score+…)/4.0, 0, 1)` | (b) prediction |
| 179–195 | `if len(self.lookup_key): … blend_score[hit] = self.lookup_scores[pos[hit]]` | **(P)** the override — memorization, rule-legal |
| 200–229 | `def allocate(self, tier, score, cost_mean, cost_q90)` etc. | (b) predicted arrays |

Nothing in this file reads `input_tokens`/`output_tokens`/`answer`/`label`/`gold`
from the *runtime input*. The strings never appear.

### `src/ossp_router/features_v2.py`

Exactly **one** hit in the whole file:

```
src/ossp_router/features_v2.py:63
    "aime_style": re.compile(r"\b(?:AIME|integer answer|answer is an integer)\b", re.I),
```

Class **(b)+(c)**. It is a regex applied to `text[:40000]` — prompt content only,
available at routing time; the rules state prompt→regex transforms are content-based
routing. Empirically it is also **dead**:

```
$env:PYTHONPATH='src'; & .venv/Scripts/python.exe build/a2_feature_probe.py
...
[train] aime_style regex matches 0/1760 episodes (0.00%)
[dev]   aime_style regex matches 0/880 episodes (0.00%)
```

The feature is the constant `log1p(0) = 0.0` for all 2640 public episodes. It cannot
carry any signal, benchmark-identity or otherwise. `belebele` / `cruxeval` / `gold` /
`label` / `benchmark` do not appear anywhere in `src/`.

Layout check (same script):

```
DENSE_DIM = 81  (46 reserved + 35 regex counters)
hand-written dense slots filled: 22   zero-padding slots 22..45: True
```

24 of the 46 reserved dense slots are hard-coded zero padding — dead weight, not leakage.

### `src/ossp_router/allocation.py`

All hits are the parameter/local name `score_pred` (lines 19, 26, 33, 35, 61, 69, 83,
105, 114, 116, 118, 124, 132). Class **(b)** — it is the predicted-score array passed
in by `final_router.allocate`. No file I/O, no realized outcomes.

### `src/ossp_router/protocol.py`

| line | text | class |
|---|---|---|
| 23–24 | `SCORE_DECIMAL_PLACES = 12` / `SCORE_ROUNDING = "ROUND_HALF_EVEN"` | (c) consumed only by `scoring.py`, which is never imported (§1b) |
| 59–65 | `class Outcome: score / num_generations / input_tokens / output_tokens` | (c) dataclass, never instantiated on the inference path |
| 68–73 | `class OutcomeBatch` | (c) same |
| 326–398 | `def parse_outcomes(...)` — the whole outcomes parser | (c) **unreachable** |
| 609–610 | `def load_outcomes(path)` | (c) **unreachable** |
| all `label=` hits (153–522) | error-message field names in the validator | (b) not an ML label |

Proof of unreachability, two independent ways:

```
grep -rn "parse_outcomes\|load_outcomes\|OutcomeBatch\|Outcome(" \
  src/ossp_router/final_router.py src/ossp_router/allocation.py \
  src/ossp_router/features_v2.py src/ossp_router/heuristic.py src/ossp_router/__init__.py
EXIT=1 (1 means no hits)
```

and the AST call graph lists `protocol.py::parse_outcomes` and
`protocol.py::load_outcomes` under *"functions in reachable modules NEVER called from
entrypoint"*. `parse_input` (protocol.py:257–323) accepts **only** the keys
`episode_id` + exactly one of `prompt`/`messages` (`_exact_keys`, line 269–274) — an
outcomes-bearing input file would be *rejected*, not silently read.

### `src/ossp_router/heuristic.py`

All hits are `complexity_score` / the local `score` int of the deliberately weak
reference baseline (lines 97–134) — class **(b)**, and that baseline is not used by
`final_router` (only `episode_text` at line 65 and `write_submission_atomic` at line
176 are imported, per `final_router.py:33`).

`episode_text` is the sole gateway from input to features:

```
src/ossp_router/heuristic.py:65-71
def episode_text(episode: Episode) -> str:
    """Return only the prompt or message content available at routing time."""
    if episode.prompt is not None:
        return episode.prompt
    assert episode.messages is not None
    return "\n".join(message.content for message in episode.messages)
```

### ID / order / split independence (sanity, cross-track)

```
grep -n "episode_id\|split\|challenge_id" src/ossp_router/final_router.py \
  src/ossp_router/features_v2.py src/ossp_router/allocation.py

final_router.py:232      challenge_id=inputs.challenge_id,     <- echoed into output only
final_router.py:234      split=inputs.split,                   <- echoed into output only
final_router.py:237      Decision(episode.episode_id, MODEL_IDS[j])   <- output pairing only
features_v2.py:87        lines = text.split("\n")              <- str.split, unrelated
```

`episode_id`, `split`, `challenge_id` never enter `predict_batch` or `allocate`.

---

## 3. Artifact contents — opened, not just listed

Command: `$env:PYTHONPATH='src'; & .venv/Scripts/python.exe build/a2_artifacts.py`

### 3.1 `models/final-v1/lookup.npz` — **the memorization table**

```
keys in archive: ['key', 'scores', 'costs']
  key        shape=(2640, 32) dtype=uint8   nbytes=84480
  scores     shape=(2640, 3)  dtype=float64 nbytes=63360
  costs      shape=(2640, 3)  dtype=float64 nbytes=63360

key row 0 (hex): 004bbf488e05ac2e436bbc90451e2b7cbaf2a0453f6f2058bb2e87051f6fa5fa
key row 1 (hex): 0091d50981f15c512944ca59233f249c548cae02465a8105202006361e49125d
key row len bytes = 32 (SHA-256 digest = 32 bytes)
all rows exactly 32 bytes: True
unique keys: 2640
keys sorted ascending (searchsorted precondition): True
scores stats: min=0.000000 max=1.000000 mean=0.701389
costs  stats: min=0.00016400 max=3.43076396 mean=0.04345616
scores row0: [1. 1. 1.]   costs row0: [0.001816 0.00346739 0.01319565]
```

**`key` is a SHA-256 digest and prompt text is NOT recoverable.** Raw-byte scan of the
whole 137 774-byte file:

```
printable ASCII runs >=12 chars in lookup.npz (excluding npy headers): 2
    b'6Ay|!8/;-kb{b'      <- random compressed bytes
    b'scores.npyPK'       <- zip member name
substring b'ep-'      present: False        substring b'episode'  present: False
substring b'train-'   present: False        substring b'dev-'     present: False
substring b'aime'     present: False        substring b'belebele' present: False
substring b'cruxeval' present: False
```

Three numeric arrays only. **No `episode_id` is stored anywhere** — not in
`lookup.npz`, not in `knn/outcomes.npz` (checked the same way, all probes `False`).
The digest is a one-way membership oracle: holding a candidate prompt you can test
whether it was public, but you cannot invert a digest to text. Since all 2640 prompts
are already public, this is not an information disclosure.

Semantics proven against the ground truth (`build/a2_lookup_verify.py`), recomputing
`sha256(episode_text(e))` and the exact cost formula
`input_tokens*rate_in/1e6 + output_tokens*rate_out/1e6`:

```
lookup rows: 2640  unique digests: 2640

[train] episodes=1760  sha256(prompt) hits in lookup = 1760/1760  (100.00%)  rows touched = 1760
   lookup 'scores' == realized outcomes score exactly: 1760/1760   max|err|=0.000e+00
   lookup 'costs'  == realized token cost (<1e-12):    1760/1760   max|err|=4.441e-16

[dev]   episodes=880   sha256(prompt) hits in lookup = 880/880   (100.00%)  rows touched = 880
   lookup 'scores' == realized outcomes score exactly: 880/880    max|err|=0.000e+00
   lookup 'costs'  == realized token cost (<1e-12):    880/880    max|err|=4.441e-16

distinct sha256(prompt) over train+dev: 2640
lookup rows NOT explained by any public train/dev prompt: 0
public prompts NOT in lookup: 0
```

So: `key[i] = SHA-256(prompt_i)`, `scores[i]` = the **exact realized** per-model score
from `data/{train,dev}/outcomes.json`, `costs[i]` = the **exact realized** per-model
token cost. 2640 = 1760 train + 880 dev, no extra rows, no missing rows.
**Dev outcomes are in this table.** Confirms the lead's 100 % hit-rate measurement.

Build provenance (`exp/build_final.py:273-294`):

```
    # ---------------- lookup table ----------------
    log("lookup table (train+dev)")
    for split in ("train", "dev"):
        ...
            hashes.append(hashlib.sha256(text.encode("utf-8")).digest())
```

### 3.2 `models/final-v1/irt.npz`

```
  W              shape=(1, 337)   dtype=float64  min=-0.271008 max=+0.261899
  a              shape=(3, 1)     dtype=float64  min=-8.00954  max=-5.00978
  b              shape=(3,)       dtype=float64  min=+1.01433  max=+2.00943
  scaler_mean    shape=(81,)      dtype=float64  min=+0        max=+14.6845
  scaler_scale   shape=(81,)      dtype=float64  min=+0.0165176 max=+9.2011
```

1-D IRT: `W` projects the 337-dim feature vector (81 dense + 128 word-SVD +
128 char-SVD, matching `DENSE_DIM=81`) to a scalar ability; `a` (3 discriminations)
and `b` (3 difficulties) give one logistic per model. `scaler_mean/scale` are the
StandardScaler statistics for the 81 dense features. **Model parameters only — no
per-episode rows.** (`final_router.py:70-74, 143-146`.)

### 3.3 `models/final-v1/svd-word.npz`, `svd-char.npz`

```
svd-word.npz  components  shape=(128, 65536) dtype=float64  min=-0.450268 max=+0.661926
svd-char.npz  components  shape=(128, 65536) dtype=float64  min=-0.33796  max=+0.389844
```

TruncatedSVD right-singular vectors over the 65 536-bin hashed word / char n-gram
spaces. Fit on **train only** — `exp/build_final.py:88-89`:

```
    svd_w = TruncatedSVD(n_components=128, random_state=SEED).fit(word_tr)
    svd_c = TruncatedSVD(n_components=128, random_state=SEED).fit(char_tr)
```

Basis vectors, no per-episode identity, no outcomes. Legitimate.

### 3.4 `models/final-v1/knn/index.npz` — the kNN memory (features side)

```
  format=csr shape=(1760, 131072) dtype=float64 nnz=2249927
  rows = training episodes ; cols = 65536 word-hash + 65536 char-hash = 131072
  row L2 norms: min=1.000000 max=1.000000 (unit-normalised rows -> cosine via dot)
```

**1760 rows = train episodes only.** Each row is one training prompt's L2-normalised
hashed n-gram vector — i.e. the index does store a lossy fingerprint of *training
prompts* (public, allowed as a "검색 색인" per the rules), but it contains **zero dev
rows**.

### 3.5 `models/final-v1/knn/outcomes.npz` — the artifact to scrutinise

```
  scores       shape=(1760, 3)  dtype=float64  min=+0       max=+1
  logcost      shape=(1760, 3)  dtype=float64  min=-8.70352 max=+1.2303
  fb_score     shape=(3,)       dtype=float64  min=+0.597301 max=+0.811648
  fb_logcost   shape=(3,)       dtype=float64  min=-6.22042 max=-2.88939

  ROW COUNT of scores = 1760
  train episodes = 1760 ; train+dev = 2640
  -> equals 1760 (train only)? True
  -> equals 2640 (train+dev)? False
  index rows == outcome rows: True
  substring b'ep-' in raw bytes: False
  substring b'episode' in raw bytes: False
  substring b'dev' in raw bytes: False
```

**PROVEN: 1760 rows, not 2640. Dev outcomes are NOT in the kNN memory.**
`index.npz` and `outcomes.npz` are row-aligned (1760 == 1760), so neighbour *i* in the
index maps to training outcome *i*. `fb_score`/`fb_logcost` are the 3-vector global
fallbacks used when every similarity is ≤ 0 (`final_router.py:164-168`).

**Is kNN-neighbour outcome averaging leakage for DEV episodes? No.** For a dev
episode the router computes cosine similarity against 1760 *training* prompts and
averages *training* realized scores with weight `clip(sim,0,∞)**3`
(`final_router.py:149-168`). The dev episode's own outcome is not in the table and
cannot be retrieved. This is textbook instance-based learning on training data —
identical in kind to a fitted GBM, just non-parametric. It is legitimate, and unlike
the lookup table it does **not** invalidate the lookup-OFF Dev measurement.

### 3.6 `models/final-v1/xgb/keep-cols.npy`

```
  shape=(117575,) dtype=int64 min=0 max=131152 unique=117575 sorted=True
  first 10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

A strictly-increasing column-index vector selecting 117 575 of the 131 153 possible
sparse columns (81 dense + 65 536 word + 65 536 char = 131 153; max index 131 152 =
131 153−1). These are the columns non-zero in the training matrix. **Indices only —
no values, no labels, no episode identity.** Used at `final_router.py:130`.

### 3.7 `models/final-v1/lgbm/*.txt` and `xgb/*.json` — model params only

```
  lin-0.txt    bytes= 938801 first_line='tree'  Tree= count=300   leak-substring hits: []
  lin-1.txt    bytes= 938480 first_line='tree'  Tree= count=300   leak-substring hits: []
  lin-2.txt    bytes= 937306 first_line='tree'  Tree= count=300   leak-substring hits: []
  lout-0.txt   bytes= 909529 first_line='tree'  Tree= count=300   leak-substring hits: []
  lout-1.txt   bytes= 914958 first_line='tree'  Tree= count=300   leak-substring hits: []
  lout-2.txt   bytes= 912633 first_line='tree'  Tree= count=300   leak-substring hits: []
  q90-0.txt    bytes= 882881 first_line='tree'  Tree= count=300   leak-substring hits: []
  q90-1.txt    bytes= 885425 first_line='tree'  Tree= count=300   leak-substring hits: []
  q90-2.txt    bytes= 906228 first_line='tree'  Tree= count=300   leak-substring hits: []
  score-0.txt  bytes= 923186 first_line='tree'  Tree= count=300   leak-substring hits: []
  score-1.txt  bytes= 925489 first_line='tree'  Tree= count=300   leak-substring hits: []
  score-2.txt  bytes= 929207 first_line='tree'  Tree= count=300   leak-substring hits: []

  lin-0.json       top-level keys=['learner','version']  leak-substring hits: []
  ... (all 7 xgb json identical shape)
  score-multi.json top-level keys=['learner','version']  leak-substring hits: []
      learner attrs=['attributes','feature_names','feature_types',
                     'gradient_booster','learner_model_param','objective']
```

Standard LightGBM text dumps (300 trees each) and XGBoost UBJSON-style dumps. No
`episode`, `prompt`, `aime`, `belebele`, `cruxeval`, or `outcome` substring in any of
them. **Model parameters only.** All learners fit on train (`exp/build_final.py:108`
`lgb.Dataset(X_svd_tr, ...)`, line 173/179/182 `xgb_fit(..._tr)`, line 197
`StandardScaler().fit(dense_tr)`).

---

## 4. Is the lookup table permitted? — rule text quoted

`docs/CHALLENGE_RULES.md` §"사용할 수 있는 정보", lines 113–116:

> 공개 자료에서 만든 분류기, 회귀 계수, 어휘·IDF, 토크나이저, **조회표**, 검색
> 색인과 캐시를 제출 이미지에 포함할 수 있습니다. **정확한 프롬프트나 프롬프트
> 해시를 사용하는 공개 자료 조회도 허용합니다.**

("Classifiers, regression coefficients, vocabulary/IDF, tokenizers, **lookup tables**,
search indexes and caches built from public data may be included in the submission
image. **Public-data lookup using the exact prompt or the prompt hash is also
permitted.**")

Reinforced at lines 118–121:

> 공식 평가 실행 시 문항별 모델 평가 결과나 실제 비용은 라우터에 전달하지
> 않습니다. … 입력 프롬프트를 해시·n-gram·정규식·임베딩 등으로 변환하는 것도
> 내용 기반 라우팅입니다.

And line 14 of the summary: `공개 Train/Dev의 프롬프트·평가 결과와 공개 비용 정책은
학습과 최적화에 사용할 수 있습니다.`

The prohibited-strategies list (lines 126–134) contains nothing matching this: the
router does not call the three models, does not retry, does not branch on
`challenge_id`/`split`/`episode_id`/input order (§2 last block), uses no non-public
data, and makes no network call.

**Ruling: the SHA-256-prompt lookup table over public Train/Dev outcomes is
EXPLICITLY PERMITTED. It is not a rule violation. It is, however, pure memorization,
and on the private evaluation set its hit rate will be ~0 %, so it contributes
nothing there.**

---

## 5. Quantifying the lookup effect — TRAIN and DEV, lookup ON vs OFF

`build/bundle-nolookup` is byte-identical to `models/final-v1` except `lookup.npz`
(verified with `sha256sum` on all 28 files): 27/28 SAME, `lookup.npz` DIFF. The
replacement holds `key (0,32) uint8 / scores (0,3) / costs (0,3)` — empty, so
`final_router.py:179` `if len(self.lookup_key):` short-circuits and `hit` is all-False.
The lookup-OFF measurement therefore isolates exactly one variable.

Runs (`build/a2_train_runs.sh`, then `ossp_router.cli self-check`):

```
.venv/Scripts/python.exe -m ossp_router.final_router \
  --input data/materialized/train/inputs.json --tier {fast,balanced,premium} \
  --output build/train-nolookup/{tier}.json --bundle build/bundle-nolookup
.venv/Scripts/python.exe -m ossp_router.cli self-check \
  --input data/materialized/train/inputs.json --outcomes data/train/outcomes.json \
  --submissions build/train-nolookup --report build/train-nolookup-report.json
```

(and the same with `--bundle models/final-v1` into `build/train-final/`).

### 5.1 Headline table

| split | lookup | final score | ceiling | achieved | source |
|---|---|---|---|---|---|
| **train** 1760 | **OFF** | **0.729971590909** | 0.790507 | **92.34 %** | `build/train-nolookup-report.json` (NEW) |
| **train** 1760 | ON | 0.745269886364 | 0.790507 | 94.28 % | `build/train-final-report.json` (NEW) |
| **dev** 880 | **OFF** | **0.684318181818** | 0.803939 | **85.12 %** | `build/dev-nolookup-report.json` |
| **dev** 880 | ON | 0.760284090909 | 0.803939 | 94.57 % | `build/dev-final-report.json` |
| dev 880 | ON (container) | 0.760284090909 | 0.803939 | 94.57 % | `build/ctr-dev-report.json` |

Lookup delta: **train +0.015298**, **dev +0.075966** — the dev gain is **4.97×** the
train gain. Exactly the signature of memorization: on train the fitted members already
reproduce the labels they were trained on, so the exact table adds little; on dev the
members are out-of-sample and the table supplies the answer key.

### 5.2 Per-tier (verbatim from `build/a2_report_summary.py`)

```
### build/train-nolookup-report.json   split=train  final_score=0.729971590909   ceiling=0.790507  achieved=92.34%
    fast      quality=  0.709517045455 ceiling=0.741723 achieved= 95.66%  budget_passed=True ratio=1.099161469265  counts={'ax31': 754, 'ax31-light': 1006, 'axk1-think': 0}
    balanced  quality=       0.7234375 ceiling=0.793305 achieved= 91.19%  budget_passed=True ratio=1.681512554822  counts={'ax31': 1549, 'ax31-light': 211, 'axk1-think': 0}
    premium   quality=  0.763778409091 ceiling=0.852754 achieved= 89.57%  budget_passed=True ratio=2.403322836482  counts={'ax31': 1477, 'ax31-light': 160, 'axk1-think': 123}

### build/train-final-report.json      split=train  final_score=0.745269886364   ceiling=0.790507  achieved=94.28%
    fast      quality=  0.718039772727 ceiling=0.741723 achieved= 96.81%  budget_passed=True ratio=1.145851451559  counts={'ax31': 301, 'ax31-light': 1459, 'axk1-think': 0}
    balanced  quality=  0.723579545455 ceiling=0.793305 achieved= 91.21%  budget_passed=True ratio=1.260370288266  counts={'ax31': 322, 'ax31-light': 1438, 'axk1-think': 0}
    premium   quality=  0.803267045455 ceiling=0.852754 achieved= 94.20%  budget_passed=True ratio=2.377732878642  counts={'ax31': 294, 'ax31-light': 1284, 'axk1-think': 182}
```

All six runs pass budget. Note the *decision* signature of memorization: with the
lookup ON the router routes far more episodes to `ax31-light` (train balanced
1438 vs 211 light) because it knows exactly which prompts the light model already
gets right.

### 5.3 Decision churn caused by the lookup (`build/a2_decision_diff.py`)

```
===== dev : decision diff lookup-ON vs lookup-OFF =====
  fast      n=880  changed=402 (45.68%)
  balanced  n=880  changed=637 (72.39%)
  premium   n=880  changed=681 (77.39%)

===== train : decision diff lookup-ON vs lookup-OFF =====
  fast      n=1760  changed=503 (28.58%)
  balanced  n=1760  changed=1229 (69.83%)
  premium   n=1760  changed=1228 (69.77%)
```

**45–77 % of all Dev routing decisions are produced by table lookup rather than by the
learned model.** On the private set, 0 % will be.

---

## 6. What must be reported as generalization performance

**Report `0.684318181818` (Dev, lookup-OFF) as the generalization number.**
Per-tier: fast `0.659090909091`, balanced `0.691761363636`, premium `0.710511363636`.
85.12 % of the Dev oracle ceiling 0.803939. Reproduces registry `E049`
(`"weighted_final": "0.684318181818"`) exactly.

This is the right number because every learned member is fit on **train only** —
verified in `exp/build_final.py`: SVD `.fit(word_tr)/.fit(char_tr)` (88–89), LGBM
`lgb.Dataset(X_svd_tr, label=y)` (108), XGB `xgb_fit(..._tr)` (173/179/182), IRT
`StandardScaler().fit(dense_tr)` (197), kNN index+outcomes 1760 train rows (§3.4/3.5).
So with the table disabled, Dev is genuinely held out for the *model*.

**Numbers that must NOT be presented as generalization:**

- `0.760284090909` (Dev, lookup ON) — 100 % of Dev prompts hit the memorized table
  (§3.1); this is a memorization score. Rule-legal, must be labelled
  "lookup-ON / includes exact public-outcome recall".
- `0.745269886364` (Train, lookup ON) — memorization on top of in-sample fit.
- `0.729971590909` (Train, lookup OFF) — **in-sample**: the members were fit on these
  1760 episodes. Its purpose is the optimism diagnostic below, nothing else.

**Optimism gap:** train-OFF 0.729972 − dev-OFF 0.684318 = **+0.045653** (train is
6.7 % relatively higher). Modest, i.e. the members are not catastrophically overfit —
but the gap is real and the private-set expectation should sit at or below 0.684318.

**Honest caveat the report must also carry.** `0.684318` is a *selection-optimistic*
held-out estimate, not a pristine one. `build/a2_registry_probe.py`:

```
registry entries: 49
entries carrying a weighted_final metric: 49
top 5 by weighted_final: E004 0.803693 / E041 0.701960 / E039 0.701506 / E044 0.701364 / E040 0.700994
E049: [('E049', '0.684318181818')]
```

All 49 experiments were scored on **Dev**, and `models/final-v1/policy.json` carries
`"registry_ref": "E049"` with its frozen hyperparameters (`fast.utilization 0.93`,
`balanced.utilization 0.88`, `premium.k1_utilization 0.65`, `fill_utilization 0.70`,
`k1_cost_cap 0.1`). Those constants were chosen by looking at Dev. The *weights* are
train-only; the *policy* is Dev-selected. Phrase it as: **"Dev held-out score with the
public-outcome lookup disabled, 0.684318; policy hyperparameters were selected on Dev
across 49 experiments, so treat this as an upper-ish estimate."**

**Competitive context (must be stated, not buried):** `0.684318` is **below** the
official strongest baseline hash-regex `0.695369` (−0.011051) and above all-light
`0.619318`. Without the memorization table the router does not beat the strongest
public baseline on held-out data. On the private evaluation set the lookup contributes
nothing, so `0.695369` — not `0.760284` — is the bar this submission must be judged
against.

---

## 7. Findings summary

| # | severity | finding |
|---|---|---|
| 1 | info (PASS) | Inference import closure is exactly `{final_router, features_v2, allocation, protocol, heuristic, __init__}`; 8 other `src/` modules unreachable; proven statically **and** at runtime via `sys.addaudithook`. |
| 2 | info (PASS) | No runtime read of `data/*/outcomes.json`. Only files opened: the input JSON, the bundled policy, and 21 bundle artifacts. Image contains no `data/` directory. |
| 3 | info (PASS) | `protocol.parse_outcomes` / `load_outcomes` / `Outcome` / `OutcomeBatch` are dead code on the inference path; `parse_input` `_exact_keys` would *reject* an outcomes-bearing input file. |
| 4 | info (PASS) | Only benchmark-named feature is `features_v2.py:63 "aime_style"`; it is a prompt-content regex and matches **0/2640** public episodes — constant zero, carries no signal. |
| 5 | info (PASS) | `knn/outcomes.npz` is **1760 rows, not 2640** — train-only. kNN neighbour averaging for a Dev episode retrieves only training outcomes. Legitimate. |
| 6 | info (PASS) | `lookup.npz key` is 2640×32 uint8 = SHA-256 digests, sorted, unique; no printable text, no `episode_id`, no split marker in the raw bytes. Prompt text not recoverable. |
| 7 | **high (reporting)** | `lookup.npz` memorizes **all 880 Dev** realized scores/costs (exact match, max err 0.0 / 4.4e-16). Dev hit rate 880/880. Rule-**permitted** (CHALLENGE_RULES.md:113-116) but the lookup-ON Dev score 0.760284 is memorization, not generalization. |
| 8 | **high (reporting)** | Lookup changes **45.68 % / 72.39 % / 77.39 %** of Dev fast/balanced/premium decisions. Dev gain +0.075966 vs train gain +0.015298 (4.97×) — quantitative memorization signature. |
| 9 | medium (reporting) | NEW: Train lookup-OFF = **0.729971590909** (92.34 % of train ceiling 0.790507); Train lookup-ON = 0.745269886364 (94.28 %). Train−Dev optimism gap with lookup off = **+0.045653**. |
| 10 | medium (reporting) | `0.684318` is Dev-selection-optimistic: all 49 registry experiments were scored on Dev and `policy.json` freezes `registry_ref: E049`. Must be disclosed. |
| 11 | low (hygiene) | The image ships 6 unreachable modules (`cli, image_evidence, operator_helper, orchestrator, runtime, scoring`) and 24/46 dense feature slots are hard-coded zero padding (`features_v2.py:121`). No leakage; attack surface / dead weight only. |
