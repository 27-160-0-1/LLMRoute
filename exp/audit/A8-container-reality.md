<!--
SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
SPDX-License-Identifier: Apache-2.0
-->

# A8 — Container Reality Check

**Image:** `ossp-router:arm64final` (linux/arm64, digest-pinned base `python:3.11-slim-bookworm@sha256:2e32f7d3…`)
**Host:** Windows 11 Pro 26200, Intel i9-13900H (20 logical CPUs), 32 GiB RAM, Docker 29.7.2, overlayfs.
**arm64 runs execute under QEMU binfmt emulation** (host is x64).

> **Headline:** Score and image-boundary claims **PASS and are reproducible byte-for-byte**.
> **Timing claims are UNVERIFIABLE on this host** — the box was pinned at 96–100 % CPU by concurrent
> audit tracks, and measured wall-clock for the *same* tier varied by **3.9×** (67.3 s → 259.4 s).
> A second, deeper issue: the 90 s cap applies to the **2,640-episode** Train+Dev workload, **not** the
> 880-episode Dev workload the lead timed. See §2.

---

## 1. Dev re-run under the official resource profile — PASS

Command actually executed (per tier, `$t` ∈ fast/balanced/premium):

```powershell
docker run --rm --platform linux/arm64 `
  --network none --read-only --tmpfs /tmp:size=256m `
  --cpus 2 --memory 2g --memory-swap 2g --pids-limit 32 `
  --ipc none --cgroupns private --ulimit core=0:0 `
  -v "C:\portable\skt_LLM\LLMRoute\data\materialized\dev:/challenge/input:ro" `
  -v "C:\portable\skt_LLM\LLMRoute\build\a8-dev-$t:/challenge/output" `
  ossp-router:arm64final `
  --input /challenge/input/inputs.json --tier $t --output /challenge/output/submission.json
```

Output:

```
OK: fast 제출 파일을 생성했습니다.
TIER=fast exit=0 wall=67.3s
OK: balanced 제출 파일을 생성했습니다.
TIER=balanced exit=0 wall=137.9s
OK: premium 제출 파일을 생성했습니다.
TIER=premium exit=0 wall=259.4s
```

### 1.1 Submissions are byte-identical to the lead's container run

```
fast      a8=13E170FD024BE640B893C04E31746D1A1D626A36CD291D7C7D2C8F3FEFBF08AB
fast      lead=13E170FD024BE640B893C04E31746D1A1D626A36CD291D7C7D2C8F3FEFBF08AB  MATCH=True
balanced  a8=CBD98A26C4E558FF6910455865BCEA2330C4E91F3324C37F43988FC400DFBA91
balanced  lead=CBD98A26C4E558FF6910455865BCEA2330C4E91F3324C37F43988FC400DFBA91  MATCH=True
premium   a8=CB8A0B5B3AD2877FC0C9D4FC8E94050BD7D8245FFB6185A5A22F71F654B17907
premium   lead=CB8A0B5B3AD2877FC0C9D4FC8E94050BD7D8245FFB6185A5A22F71F654B17907  MATCH=True
```

### 1.2 Scored in a separate process with the official scorer — 0.760284090909 CONFIRMED

```
PYTHONPATH=src python -m ossp_router.cli self-check \
  --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json \
  --submissions build/a8-dev-all --report build/a8-dev-report.json
```

```
A8 final_score       = 0.760284090909
policy_sha256        = 7c892c423da5fa762e7e1a93b9fa071be51e259b65d2b63a5ba434c4342d7a8e
split                = dev  tier_weights = {'balanced': '0.3', 'fast': '0.4', 'premium': '0.3'}
  fast      score=0.738636363636 cost=5.069558494 limit=5.476785 passed=True counts={'ax31': 152, 'ax31-light': 728, 'axk1-think': 0}
  balanced  score=0.742045454545 cost=5.279213067 limit=8.762856 passed=True counts={'ax31': 158, 'ax31-light': 722, 'axk1-think': 0}
  premium   score=0.807386363636 cost=9.142871128 limit=17.525712 passed=True counts={'ax31': 140, 'ax31-light': 668, 'axk1-think': 72}
LEAD ctr final_score = 0.760284090909
HOST dev final_score = 0.760284090909
ALL THREE IDENTICAL  = True
```

All three tiers pass budget. **Verdict: PASS** — container run reproduces host score to 12 decimals.

### 1.3 Episode ID / order audit — PASS

```
input episodes = 880  unique = 880
fast      order_identical=True  set_identical=True  n=880 extra_top_fields=none decision_fields=['episode_id', 'model_id'] models_subset_ok=True tier_field=fast split=dev
balanced  order_identical=True  set_identical=True  n=880 extra_top_fields=none decision_fields=['episode_id', 'model_id'] models_subset_ok=True tier_field=balanced split=dev
premium   order_identical=True  set_identical=True  n=880 extra_top_fields=none decision_fields=['episode_id', 'model_id'] models_subset_ok=True tier_field=premium split=dev
```

Exactly the 6 permitted top-level fields; decisions carry only `episode_id` + `model_id`; order preserved.

---

## 2. Timing — the lead's numbers do not reproduce, and the wrong workload was timed

### 2.1 The lead's Dev timings are not reproducible on this host

| Tier | Lead (Dev 880) | A8 run #1 (fwd order) | A8 run #2 (reverse order) |
| --- | ---: | ---: | ---: |
| fast | 59.9 s | **67.3 s** | **192.9 s** |
| balanced | 67.0 s | **137.9 s** | **213.0 s** |
| premium | 68.2 s | **68.2 → 259.4 s** | **227.9 s** |

Reverse-order run output:

```
TIER=premium exit=0 wall=227.9s
TIER=balanced exit=0 wall=213s
TIER=fast exit=0 wall=192.9s
```

Running the tiers in reverse order produced ~190–230 s for **every** tier, including `fast`, which took
67.3 s when it ran first. **The spread tracks execution position, not tier.** Cause:

```
=== CURRENT CPU LOAD (%) ===
96
=== TOP 8 CPU PROCESSES ===
Name                    CPU         WS
----                    ---         --
claude         17823.171875  295186432
claude         17195.921875  894423040
remoting_host   13850.34375  194420736
Taskmgr             7323.75  154275840
claude            4817.9375  161050624
python          3380.984375 1963003904
Docker Desktop  2382.765625  127991808
```

The host was saturated by concurrent audit tracks. **Any wall-clock number measured on this host during
this window — the lead's included — is a contention artifact, not a property of the image.**

### 2.2 Fixed-cost decomposition (measured)

```
BASELINE_STARTUP rep=1 exit=0 wall=2.08s     (docker run … --entrypoint python3 … -c "pass")
BASELINE_STARTUP rep=2 exit=0 wall=2.25s
BASELINE_STARTUP rep=3 exit=0 wall=2.12s
BASELINE_IMPORTS rep=1 exit=0 wall=19.21s    (… -c "import numpy,scipy,lightgbm,xgboost")
BASELINE_IMPORTS rep=2 exit=0 wall=23.38s
```

QEMU container startup is cheap (~2.1 s). **Importing numpy/scipy/lightgbm/xgboost under QEMU costs
~19–23 s** and is a fixed per-run cost independent of episode count.

### 2.3 The 90 s cap applies to 2,640 episodes, not 880

`docs/RUNTIME.md` §로컬 검증 (line 164) and `docs/runtime-benchmark.md` are explicit:

- `runtime-benchmark.md`: `문항 수: 2,640` — official reference measurements use the **combined
  Train 1,760 + Dev 880** workload.
- `RUNTIME.md`: `최종 이미지는 공개 Train/Dev 전체로 세 등급의 90초 한도를 미리 확인할 수 있습니다.`
- `tools/check_runtime.py` calls `load_public_runtime_workload(train_path=…, dev_path=…)` and prints a
  single combined episode count — i.e. **one container run per tier over all 2,640 episodes.**

The lead timed **Dev only (880)** — 1/3 of the gated workload. Those numbers cannot clear the cap even
if they had been stable.

COMBINED_RESULTS_PLACEHOLDER

### 2.4 Native Apple Silicon calibration

Official native-ARM container reference (`runtime-benchmark.md`, 2,640 episodes, Colima, 5 reps):

| Baseline | container elapsed median/max (s) |
| --- | ---: |
| always-light | 0.2121 / 0.3167 |
| prompt-heuristic | 1.6659 / 1.6838 |
| feature-budget | 3.8301 / 3.8594 |
| hash-regex | 7.3152 / 7.5791 |

Official observed maxima: `memory.peak` 76.66 MiB, `pids.peak` 6.

**QEMU is an upper bound, not an estimate.** This host's `python3 -c "pass"` under QEMU takes 2.1 s,
whereas native ARM `always-light` — which additionally parses the 11.23 MiB input and writes a
submission — takes 0.21 s. That implies a QEMU penalty of **roughly 10×** on this workload class, and
QEMU penalises the vectorised BLAS/tree-inference inner loops far more than it penalises I/O.

---

## 3. Filesystem forensics — no evaluation data in the image

### 3.1 Largest files in the image

```
docker run --rm --entrypoint sh ossp-router:arm64final -c "find / -xdev -type f -printf '%s %p\n' | sort -rn | head -80"
```

```
475497136 /usr/local/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2
234832113 /usr/local/lib/python3.11/site-packages/xgboost/lib/libxgboost.so
 64973998 /opt/router/model-bundle/svd-word.npz
 52538910 /opt/router/model-bundle/svd-char.npz
 24847545 /usr/local/lib/python3.11/site-packages/scipy.libs/libscipy_openblas-c5a9b014.so
 22982609 /usr/local/lib/python3.11/site-packages/numpy.libs/libscipy_openblas64_-0f683016.so
  8085632 /usr/local/lib/python3.11/site-packages/lightgbm/lib/lib_lightgbm.so
  5538168 /opt/router/model-bundle/knn/index.npz
```

Every large file is either a Python wheel payload or a `models/final-v1` artifact. **No episode data.**

### 3.2 Absence proofs

```
=== [1] ABSENCE PROOF: evaluation-data filename patterns anywhere in image ===
pattern 'outcomes' (excluding stdlib/site-packages): 1
/opt/router/model-bundle/knn/outcomes.npz
pattern 'gold' (excluding stdlib/site-packages): 2
/etc/ssl/certs/SwissSign_Gold_CA_-_G2.pem
/etc/ssl/certs/NetLock_Arany_=Class_Gold=_Főtanúsítvány.pem
pattern 'answer' (excluding stdlib/site-packages): 0
pattern 'label' (excluding stdlib/site-packages): 2
/usr/sbin/swaplabel
/usr/sbin/e2label
pattern 'truth' (excluding stdlib/site-packages): 0
pattern 'solution' (excluding stdlib/site-packages): 0
pattern 'target' (excluding stdlib/site-packages): 0
pattern 'reference' (excluding stdlib/site-packages): 1
/etc/apt/preferences.d

=== [2] ABSENCE PROOF: exact challenge data paths ===
ABSENT : /opt/router/data
ABSENT : /data
ABSENT : /challenge/input
ABSENT : /challenge/output
ABSENT : /opt/router/data/materialized
ABSENT : /opt/router/data/train
ABSENT : /opt/router/data/dev
ABSENT : /opt/router/model-bundle/outcomes.json

=== [3] any file literally named outcomes.json / inputs.json in whole image ===
(empty above == none found)

=== [7] identity ===
uid=65532 gid=65532 groups=65532
```

`answers`, `truth`, `solution`, `target` → zero hits. `gold`/`label`/`reference` hits are CA
certificates, `e2fsprogs` binaries and an APT config dir — unrelated. `/challenge/input` and
`/challenge/output` do **not** exist in the image; they are created only by the bind mounts at run time.

### 3.3 The only data-like payload is `models/final-v1` — CONFIRMED, with one nuance

`/opt/router` contains exactly 43 files: `entrypoint.py` (388 B) + 14 `ossp_router/*` source files
(276,542 B) + **28 model-bundle files (142,528,551 B)**. The 28 bundle files match host
`models/final-v1` one-for-one by name and byte length (28 files, 142,512,167 B of file content).

**Nuance the phrase "no evaluation data" must not paper over.** Two bundle artifacts are
outcome-derived, and one of them covers Dev:

```
=== knn/outcomes.npz characterization ===
  key=scores         shape=(1760, 3)        dtype=float64      # Train only
  key=logcost        shape=(1760, 3)        dtype=float64
  key=fb_score       shape=(3,)             dtype=float64
  key=fb_logcost     shape=(3,)             dtype=float64

=== lookup.npz characterization (memorization table) ===
  key=key            shape=(2640, 32)       dtype=uint8        # SHA-256 digests
  key=scores         shape=(2640, 3)        dtype=float64
  key=costs          shape=(2640, 3)        dtype=float64
```

- `knn/outcomes.npz` — 1,760 rows = **Train only**. Ordinary supervised artifact.
- `lookup.npz` — 2,640 rows = Train 1,760 + **Dev 880**, keyed by 32-byte prompt hash. This is the
  memorization table, and it *does* embed realized Dev score/cost inside the image.

This is **rule-allowed**. `docs/CHALLENGE_RULES.md` L112-115 permits `조회표` (lookup tables) and
`정확한 프롬프트나 프롬프트 해시를 사용하는 공개 자료 조회`. `final_router.py` documents it openly:
`4. exact-match lookup (public Train/Dev outcomes, rule-allowed) overrides predictions with realized values`.

**Correct statement of the finding:** the image contains **no hidden-evaluation data and no
gold answers**, and nothing that could leak the private test set. It *does* contain realized
outcomes for the 880 **public Dev** prompts. Accordingly the 0.760284090909 Dev figure in §1 is a
memorization readout, not generalization — consistent with the lead's finding #2 and the honest
lookup-OFF figure of 0.684318.

---

## 4. Image size vs caps — PASS

Caps (`docs/RUNTIME.md` L119-120): OCI compressed layer total **1 GiB**, merged rootfs apparent size **2 GiB**.

**Method A — compressed layers.** `docker save | gzip -c | wc -c`:

```
720141040

real	1m9.024s
```

`docker save | tar -tv` shows the blobs are *already* distribution-compressed, so the gzip pass is a
no-op and the blob sum is the true compressed total:

```
545269036 blobs/sha256/12264a287e61…   (pip install layer — nvidia nccl + xgboost)
129023677 blobs/sha256/feaf2480b94b…   (model-bundle layer)
 28117202 blobs/sha256/0f5d7465a5bb…
 15956619 blobs/sha256/ada6f4d8d947…
  3368451 blobs/sha256/7db0e5f619af…
```

Blob sum = **721,803,337 B = 0.672 GiB**, matching `docker image inspect .Size` exactly.

**Method B — merged rootfs apparent size**, `du -sb /` inside the container:

```
rootfs apparent bytes = 1232779715 (1.148 GiB)
--- top dirs ---
1.1G	/usr
137M	/opt
6.8M	/var
1.4M	/etc
```

| Measure | Value | Cap | Headroom | Verdict |
| --- | ---: | ---: | ---: | --- |
| OCI compressed layer total | 0.672 GiB | 1.00 GiB | 33 % | PASS |
| Merged rootfs apparent size | 1.148 GiB | 2.00 GiB | 43 % | PASS |

**Two observations for the lead:**

1. `docs/REPRODUCE.md` §10 reports `Size=721803819 — 겉보기 크기 721.8 MB(한도 2 GiB)`, i.e. it checks
   the **compressed** number against the **rootfs** cap. The real rootfs apparent size is **1.148 GiB**,
   1.7× the quoted figure. Both still pass, but the documented check is measuring the wrong quantity.
2. `nvidia/nccl/lib/libnccl.so.2` is **475 MB** — 66 % of the compressed image — pulled in as an
   `xgboost` dependency. No GPU is ever passed to the container. Dropping it would cut the compressed
   image to roughly 0.23 GiB and the rootfs to ~0.7 GiB.

---

## 5. VOLUME / USER / output-volume hygiene — PASS

```
Os=linux Arch=arm64 User=65532:65532 Volumes=map[] Size=721803337
NO VOLUME DECLARATION
Entrypoint=[python3 /opt/router/entrypoint.py] Cmd=[] WorkingDir=/opt/router
8 layers
```

Runtime identity confirmed from inside the container: `uid=65532 gid=65532 groups=65532`.

Output volume contents after each Dev run:

```
tier=fast  entry_count=1  names=[submission.json]
tier=balanced  entry_count=1  names=[submission.json]
tier=premium  entry_count=1  names=[submission.json]
```

Exactly one file per tier, 66,116–66,223 B (cap 4 MiB, 64 inodes). No temp/lock/partial files left behind.

MEMORY_PLACEHOLDER

---

## 6. `tools/check_runtime.py` — UNVERIFIED on this host (POSIX-only)

```
PYTHONPATH=src .venv/Scripts/python.exe tools/check_runtime.py --image ossp-router:arm64final --report build/a8-check-runtime.json
```

```
Traceback (most recent call last):
  File "C:\portable\skt_LLM\LLMRoute\tools\check_runtime.py", line 21, in <module>
    from ossp_router.orchestrator import input_batch_to_dict
  File "C:\portable\skt_LLM\LLMRoute\src\ossp_router\orchestrator.py", line 38, in <module>
    from .runtime import (
  File "C:\portable\skt_LLM\LLMRoute\src\ossp_router\runtime.py", line 9, in <module>
    import fcntl
ModuleNotFoundError: No module named 'fcntl'
```

`src/ossp_router/runtime.py:9` imports `fcntl` at module scope (used for `flock` at L2176/L2194).
`fcntl` ships only on POSIX, so the official checker **cannot run on Windows**. Reported as
**UNVERIFIED-on-this-host**, not as a pass and not as a defect.

**Exact command a Linux/macOS operator must run** (from repo root, Docker with `linux/arm64` support):

```bash
PYTHONPATH=src python3 tools/check_runtime.py \
  --image ossp-router:arm64final \
  --repetitions 3 \
  --report build/runtime-check-report.json
```

Defaults it will pick up (`tools/check_runtime.py` L46-48):
`--train-input data/materialized/train/inputs.json`, `--dev-input data/materialized/dev/inputs.json`,
`--registry data/public-data.v1.json` — all three present in this repo. This is the **only** check that
can settle the 90 s question, and it must be run on **native arm64**, not under QEMU.

---
