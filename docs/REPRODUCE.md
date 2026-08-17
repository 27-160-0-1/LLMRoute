<!--
SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
SPDX-License-Identifier: Apache-2.0
-->

# 재현 절차 — 단계별 명령

이 문서의 모든 명령은 **2026-08-17에 Windows 11 + PowerShell 5.1 +
Python 3.11.9 + Docker 29.7.2 환경에서 실제로 실행하여 검증**했다. 각
단계에 실측 소요 시간과 기대 출력을 적었다.

- PowerShell 5.1에는 `&&`, `||`, 삼항 연산자가 **없다**. 명령을 이어붙이려면
  `;` 또는 `if ($?) { ... }`를 쓴다.
- `PYTHONPATH=src cmd` 형태의 인라인 환경변수 접두사는 PowerShell에서
  동작하지 않는다. `$env:PYTHONPATH='src'`로 먼저 설정한다.
- bash(Git Bash / macOS / Linux) 대안을 각 단계에 함께 적었다.

> **Step 0 (권장)** — Windows에서 작업한다면 반드시 먼저 읽을 것:
> [부록 A. Windows CRLF 함정](#부록-a-windows-crlf-함정). 이 저장소를
> `core.autocrlf=true`(Git for Windows 기본값)로 clone하면 라우터가
> **아예 기동하지 않는다.**

---

## 0. 사전 요건 확인

```powershell
python --version; git --version; docker --version; docker buildx version
```

```bash
python3 --version && git --version && docker --version && docker buildx version
```

**검증 결과:** `Python 3.11.9` / `git 2.55.0.windows.2` /
`Docker 29.7.2` / `buildx v0.36.0`. Python 3.10 이상, Docker는 8~11단계에만
필요하다.

---

## 1. 줄바꿈 정규화 (Windows 전용, 필수)

`.gitattributes`가 저장소에 있는지 확인하고, 작업 트리를 git blob과
바이트 단위로 일치시킨다.

```powershell
Get-Content .gitattributes -TotalCount 3
git rm --cached -r . -q
git reset --hard
git status --short
```

```bash
cat .gitattributes | head -3
git rm --cached -r . -q && git reset --hard && git status --short
```

**소요:** 약 5초. **기대:** `git status`가 비어 있거나 추적하지 않는
파일만 표시.

**검증 명령** — 추적 파일 225개가 전부 git blob과 일치해야 한다.

```powershell
python -c "import pathlib,subprocess; n=0; t=0; [ (globals().__setitem__('t',t+1), globals().__setitem__('n', n + (0 if pathlib.Path(r).read_bytes()==subprocess.run(['git','cat-file','-p','HEAD:'+r],capture_output=True).stdout else 1))) for r in [l.decode() for l in subprocess.run(['git','ls-files','-z'],capture_output=True).stdout.split(b'\x00') if l] if pathlib.Path(r).is_file() ]; print('tracked',t,'differing',n)"
```

**검증 결과:** `tracked 225 differing 0`.

---

## 2. 공개 데이터 재현용 가상환경

```powershell
python -m venv .venv-data
.\.venv-data\Scripts\python.exe -m pip install --upgrade pip
.\.venv-data\Scripts\python.exe -m pip install -r data\sources\requirements-materialize-public-data.txt
```

```bash
python3 -m venv .venv-data
.venv-data/bin/pip install --upgrade pip
.venv-data/bin/pip install -r data/sources/requirements-materialize-public-data.txt
```

**소요:** 약 40초 · **네트워크 필요** · 설치: `pyarrow==23.0.1`.

---

## 3. 공개 Train/Dev 실체화

```powershell
.\.venv-data\Scripts\python.exe tools\materialize_public_data.py
```

```bash
.venv-data/bin/python tools/materialize_public_data.py
```

**소요:** 약 30초 · **네트워크 필요**(SHA-256으로 고정된 HuggingFace 출처).

**검증 결과:**

```
train: 1760 episodes: .../data/materialized/train/inputs.json
dev: 880 episodes: .../data/materialized/dev/inputs.json
```

생성 경로는 git 비추적이며 컨테이너 이미지에도 포함되지 않는다.

---

## 4. 런타임 가상환경 (학습 환경과 동일 핀)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: "numpy==2.0.2" "scipy==1.17.1" "lightgbm==4.7.0" "xgboost==3.2.0"
```

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --only-binary=:all: numpy==2.0.2 scipy==1.17.1 lightgbm==4.7.0 xgboost==3.2.0
```

**소요:** 약 2분 · **네트워크 필요** · 다운로드 약 156 MB(xgboost 휠이 101 MB).

**버전 정합성 확인:**

```powershell
.\.venv\Scripts\python.exe -c "import numpy,scipy,lightgbm,xgboost; print(numpy.__version__, scipy.__version__, lightgbm.__version__, xgboost.__version__)"
```

**검증 결과:** `2.0.2 1.17.1 4.7.0 3.2.0` — Dockerfile의 핀과 정확히 일치.
Python 3.11에 대해 네 패키지 모두 cp311 휠이 존재하며 의존성 충돌이 없음을
확인했다.

---

## 5. 모델 번들 무결성 확인

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -c "import hashlib,json,pathlib; b=pathlib.Path('models/final-v1'); m=json.loads((b/'manifest.json').read_text(encoding='utf-8')); f={str(p.relative_to(b)).replace(chr(92),'/') for p in b.rglob('*') if p.is_file() and p.name!='manifest.json'}; bad=[k for k in sorted(set(m)&f) if hashlib.sha256((b/k).read_bytes()).hexdigest()!=m[k]]; print('entries',len(m),'files',len(f),'missing',sorted(set(m)-f),'unlisted',sorted(f-set(m)),'mismatch',len(bad))"
```

**검증 결과:** `entries 27 files 27 missing [] unlisted [] mismatch 0`.

---

## 6. 단위 테스트

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

**소요:** Windows 약 11초 / Linux 약 222초.

**검증 결과와 해석은 [부록 B](#부록-b-테스트-실패-분류)를 반드시 참고할 것.**
Windows에서는 POSIX 전용 API(`fcntl`, `os.mkfifo`, 심볼릭 링크 권한)와 콘솔
인코딩 때문에 11개 오류가 추가로 발생하며, 이는 플랫폼 인공물이다.
플랫폼 독립적인 실패는 **6개**이고 전부 설명된다.

Linux에서 확인하려면:

```powershell
docker run --rm -v "${PWD}:/w" -w /w python:3.11-slim-bookworm bash -c "pip install -q numpy==2.0.2 scipy==1.17.1 lightgbm==4.7.0 xgboost==3.2.0 && apt-get update -qq && apt-get install -y -qq libgomp1 && PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'"
```

**검증 결과:** `Ran 266 tests ... FAILED (failures=6, skipped=11)` — 오류 0개.

---

## 7. toy 스모크 테스트

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe baselines\always_light.py --input data\toy\inputs.json --output-dir build\toy-submission
.\.venv\Scripts\python.exe -m ossp_router.cli self-check --input data\toy\inputs.json --outcomes data\toy\outcomes.json --submissions build\toy-submission --report build\toy-report.json
```

```bash
PYTHONPATH=src .venv/bin/python baselines/always_light.py --input data/toy/inputs.json --output-dir build/toy-submission
PYTHONPATH=src .venv/bin/python -m ossp_router.cli self-check --input data/toy/inputs.json --outcomes data/toy/outcomes.json --submissions build/toy-submission --report build/toy-report.json
```

**소요:** 약 2초. **기대:** `OK: 검증 보고서를 ...에 기록했습니다.`

---

## 8. 최종 라우터 — Dev 880 세 등급 + 공식 채점

```powershell
$env:PYTHONPATH='src'
foreach ($t in @('fast','balanced','premium')) {
  .\.venv\Scripts\python.exe -m ossp_router.final_router --input data\materialized\dev\inputs.json --tier $t --output build\dev-final\$t.json
}
.\.venv\Scripts\python.exe -m ossp_router.cli self-check --input data\materialized\dev\inputs.json --outcomes data\dev\outcomes.json --submissions build\dev-final --report build\dev-final-report.json
```

```bash
for t in fast balanced premium; do
  PYTHONPATH=src .venv/bin/python -m ossp_router.final_router \
    --input data/materialized/dev/inputs.json --tier "$t" --output "build/dev-final/$t.json"
done
PYTHONPATH=src .venv/bin/python -m ossp_router.cli self-check \
  --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json \
  --submissions build/dev-final --report build/dev-final-report.json
```

**소요:** 등급당 7.3–11.1초, 채점 약 3초.

**점수 확인:**

```powershell
.\.venv\Scripts\python.exe -c "import json; r=json.load(open('build/dev-final-report.json',encoding='utf-8')); [print('%-9s score=%s ratio=%s passed=%s near=%s' % (t, r['tiers'][t]['tier_score'], r['tiers'][t]['budget_ratio'], r['tiers'][t]['budget_passed'], r['tiers'][t]['near_budget'])) for t in ('fast','balanced','premium')]; print('FINAL', r['final_score'])"
```

**검증 결과** — `exp/final-report.md`의 0.7603과 12자리까지 일치:

| 등급 | tier_score | 비용 비율 | 한도 | 통과 | near_budget |
| --- | ---: | ---: | ---: | --- | --- |
| fast | 0.738636363636 | 1.157056214093 | 1.25 | true | false |
| balanced | 0.742045454545 | 1.204906954308 | 2.0 | true | false |
| premium | 0.807386363636 | 2.086733167360 | 4.0 | true | false |
| **최종** | **0.760284090909** | | | | |

> Dev는 정확 매칭 조회표에 전부 포함되므로 이 값은 **공개 Dev 상한**이다.
> 조회표를 끈 동결 정책의 기대값은 0.6843이다(기술 보고서 §4.3.1).

---

## 9. 결정성 감사 (규칙 준수)

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m unittest tests.test_final_router -v
```

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_final_router -v
```

**검증 결과:** 5개 테스트 전부 통과 — 입력 역순 / `episode_id` 개명 /
`split`·`challenge_id` 변경 / 반복 실행 / 단일 문항 배치에서 프롬프트별
선택이 동일.

---

## 10. `linux/arm64` 이미지 빌드

```powershell
docker buildx build --pull --platform linux/arm64 --file container\Dockerfile --tag ossp-router:arm64check --load .
```

```bash
docker buildx build --pull --platform linux/arm64 --file container/Dockerfile --tag ossp-router:arm64check --load .
```

**소요:** 첫 빌드 약 3분(x64 호스트에서는 QEMU 에뮬레이션) · **네트워크 필요**.

**이미지 사전검증** — 접수 단계에서 거부되는 조건을 미리 확인한다
([ENFORCEMENT.md](ENFORCEMENT.md) §이미지 사전검증 실패).

```powershell
docker image inspect ossp-router:arm64check --format "Os={{.Os}} Arch={{.Architecture}} User={{.Config.User}} Volumes={{.Config.Volumes}} Size={{.Size}}"
```

**검증 결과:** `Os=linux Arch=arm64 User=65532:65532 Volumes=map[]
Size=721803819` — `VOLUME` 선언 없음, 겉보기 크기 721.8 MB(한도 2 GiB).

---

## 11. 공식 자원 한도로 컨테이너 실행

```powershell
foreach ($t in @('fast','balanced','premium')) {
  $o = "build\ctr-dev-$t"
  New-Item -ItemType Directory -Force -Path $o | Out-Null
  $sw = [Diagnostics.Stopwatch]::StartNew()
  docker run --rm --platform linux/arm64 `
    --network none --read-only --tmpfs /tmp:size=256m `
    --cpus 2 --memory 2g --memory-swap 2g --pids-limit 32 `
    --ipc none --cgroupns private --ulimit core=0:0 `
    -v "${PWD}/data/materialized/dev:/challenge/input:ro" `
    -v "${PWD}/$o`:/challenge/output" `
    ossp-router:arm64check `
    --input /challenge/input/inputs.json --tier $t --output /challenge/output/submission.json
  "TIER=$t exit=$LASTEXITCODE wall=$([math]::Round($sw.Elapsed.TotalSeconds,1))s"
}
```

```bash
for t in fast balanced premium; do
  o="build/ctr-dev-$t"; mkdir -p "$o"
  /usr/bin/time -f "TIER=$t wall=%es" docker run --rm --platform linux/arm64 \
    --network none --read-only --tmpfs /tmp:size=256m \
    --cpus 2 --memory 2g --memory-swap 2g --pids-limit 32 \
    --ipc none --cgroupns private --ulimit core=0:0 \
    -v "$PWD/data/materialized/dev:/challenge/input:ro" \
    -v "$PWD/$o:/challenge/output" \
    ossp-router:arm64check \
    --input /challenge/input/inputs.json --tier "$t" --output /challenge/output/submission.json
done
```

**검증 결과** (x64 호스트에서 QEMU 에뮬레이션, Dev 880):

| 등급 | 종료 코드 | 벽시계 | 한도 |
| --- | ---: | ---: | ---: |
| fast | 0 | 59.9 s | 90 s |
| balanced | 0 | 67.0 s | 90 s |
| premium | 0 | 68.2 s | 90 s |

공식 환경은 **네이티브 Apple Silicon**이므로 위 수치는 상한이다. 출력
볼륨에는 `submission.json` 하나만 생성되며, 필드는 허용된 6개
(`schema_version`, `challenge_id`, `policy_id`, `split`, `tier`,
`decisions`)뿐이다.

---

## 12. 기술 제출 정보 파일

`submission-ossp-skt.json`은 **아직 저장소에 없다.** 커밋 SHA와 이미지
다이제스트가 확정된 뒤에야 작성할 수 있다.

```powershell
Copy-Item submission-ossp-skt.template.json submission-ossp-skt.json
# 편집: repository_url / commit_sha(40자) / image_digest(registry/repo@sha256:64자)
python tools\validate_technical_submission.py
```

```bash
cp submission-ossp-skt.template.json submission-ossp-skt.json
$EDITOR submission-ossp-skt.json
python3 tools/validate_technical_submission.py
```

**현재 상태:** 파일 부재로 검증기가 종료 코드 2를 반환한다(정상). 제출
직전 순서는 [SUBMISSION.md](SUBMISSION.md) §기술 제출 정보 파일을 따른다 —
**코드 커밋 → 이미지 빌드·push → 다이제스트 확인 → JSON만 별도 커밋.**

---

## 부록 A. Windows CRLF 함정

**증상.** `core.autocrlf=true`(Git for Windows 기본값)인 환경에서 clone하면
`models/final-v1/lgbm/*.txt` 12개 LightGBM 모델 파일이 CRLF로 변환된다.
`.gitattributes`가 없으면 git은 이를 텍스트로 판단한다.

**영향.** LightGBM 4.7.0이 해당 파일을 파싱하지 못한다.

```
[LightGBM] [Fatal] Model format error, expect a tree here.
```

라우터가 기동조차 하지 못하므로, 그 체크아웃에서 빌드한 `linux/arm64`
이미지는 **세 등급 모두 실행 실패 → 최종 점수 0점**이 된다.

**실측 근거.**

| 파일 | 작업 트리 | git blob | CRLF 개수 |
| --- | ---: | ---: | ---: |
| `lgbm/score-0.txt` (변환 후) | 929,308 B | 923,186 B | 6,122 |

같은 파일을 LF로 되돌리면 `num_trees=300, num_feature=337`로 정상
로드된다. 영향 범위는 정확히 12개 LightGBM `.txt` 파일이며,
`.npz`/`.npy`/XGBoost `.json`은 git이 이진 또는 단일 행으로 취급해
변환되지 않는다.

**대책.** 저장소 루트의 `.gitattributes`가 `models/final-v1/** -text`로
번들 전체를 바이트 보존 대상으로 지정한다. 이미 잘못 체크아웃한 트리는
§1의 명령으로 복구한다.

---

## 부록 B. 테스트 실패 분류

`docs/RUNTIME.md`는 표준 라이브러리 테스트를 항상 실행하도록 안내한다.
현재 상태를 있는 그대로 기록한다.

### B.1 플랫폼 인공물 (Windows 전용, Linux에서는 통과)

Linux 컨테이너에서 266개 테스트를 돌린 결과 **오류(error) 0개**다. Windows
에서 발생하는 11개 오류와 2개 실패는 전부 아래 원인이다.

| 원인 | 영향받는 테스트 |
| --- | --- |
| `fcntl` 미존재 (POSIX 전용) | `test_runtime`, `test_orchestrator`, `test_retry_policy`, `test_tiebreak_latency`, `test_benchmark_runtime`, `test_check_runtime` 모듈 로드 실패 |
| `os.mkfifo` 미존재 · 심볼릭 링크 권한(WinError 1314) | `test_operator_helper` 4건 |
| 콘솔 인코딩(cp949)으로 한글 stdout/stderr 캡처 실패 | `test_cli` 3건 |
| 휠 빌드 도구 부재 | `test_wheel_console_script_uses_bundled_policy_outside_checkout` |

### B.2 플랫폼 독립 실패 6건 — 전부 설명됨

| # | 테스트 | 원인 | 판정 |
| --- | --- | --- | --- |
| 1 | `test_commentable_files_have_spdx_tags` | 참가자 문서 4개에 SPDX 헤더 누락 | **수정 완료** (2026-08-17) |
| 2 | `test_phase_c_policy_does_not_change_frozen_v1` | 기반 이미지를 alpine → `python:3.11-slim-bookworm`으로 교체 | **의도적 · 규칙 허용** |
| 3 | `test_dockerfile_pins_multi_platform_base_and_nonroot_user` | 위와 동일한 정규식(alpine 핀 기대) | **의도적 · 규칙 허용** |
| 4 | `test_public_tree_has_no_internal_paths_secrets_or_model_artifacts` | 학습 산출물 39개 커밋 | **의도적 · 규칙 허용** |
| 5 | `test_sensitive_and_materialized_paths_are_ignored` | `.dockerignore`가 `models/final-v1`을 빌드 컨텍스트에 포함하도록 수정 | **의도적 · 필수** |
| 6 | `test_reference_images_remove_unused_packaging_tools` | `pip uninstall`을 별도 `RUN`이 아니라 `&&` 체인에 병합 | **동작 동일 · 무해** |

**#2·#3의 근거.** [RUNTIME.md](RUNTIME.md) §이미지 빌드와 제출은
"반드시 이 기반 이미지를 사용할 필요는 없습니다"라고 명시한다. alpine은
musl 기반이라 scipy/lightgbm/xgboost의 manylinux **aarch64 휠이 없어**
소스 빌드가 필요하고, 이는 90초 실행 한도와 무관하게 이미지 빌드
재현성을 크게 해친다. Debian glibc 기반으로 교체한 이유가 이것이다.

> **다만 테스트를 붉은 채로 두는 것은 별개 문제다.**
> [RUNTIME.md](RUNTIME.md) §로컬 검증은 "표준 라이브러리 테스트는 항상
> 실행합니다"라고 규정하고,
> [APPLE_SILICON_MEASUREMENT.md](APPLE_SILICON_MEASUREMENT.md)는 측정
> 절차를 `tests.test_runtime`이 `OK`로 끝나는 것에 걸어 둔다. 기반 이미지
> **선택**은 규칙이 허용하지만, alpine 시절 문자열을 그대로 들고 있는
> 세 개의 단언(`test_phase_c_policy_does_not_change_frozen_v1`,
> `test_dockerfile_pins_multi_platform_base_and_nonroot_user`,
> `test_reference_images_remove_unused_packaging_tools`)은 남은 정리
> 대상이다. 제출 전에 둘 중 하나를 택할 것:
>
> 1. **권장** — 세 단언을 실제로 채택한 기반 이미지·다이제스트에 맞게
>    갱신한다. 그러면 `test_repository_policy`에서 남는 실패는 학습 산출물
>    커밋(#4)과 `.dockerignore`(#5) 두 개뿐이고, 둘 다 참가자 fork에서는
>    성립할 수 없는 원본 저장소 불변식이다.
> 2. 갱신하지 않는다면 이 문서를 결과보고서에서 인용하여 다섯 실패가 전부
>    의도된 것임을 명시한다.

**#4의 근거.** [CHALLENGE_RULES.md](CHALLENGE_RULES.md) §사용할 수 있는
정보가 "공개 자료에서 만든 분류기, 회귀 계수, 어휘·IDF, 토크나이저,
조회표, 검색 색인과 캐시를 제출 이미지에 포함할 수 있습니다"라고 명시적으로
허용한다. 이 테스트는 **원본 과제 저장소가 학습 산출물을 담지 않는다**는
불변식을 검사하는 것이며, 참가자 fork에는 성립할 수 없다.

### B.3 남은 준수 항목 (제출 전 처리 필요)

테스트로는 잡히지 않지만 [ENFORCEMENT.md](ENFORCEMENT.md) §포함 파일의
권리와 고지에 따라 **접수 검증에서 보완 요구 사유**가 되는 항목이다. 이
조항은 "라이선스 근거나 필수 고지가 빠진 제출은 ... 보완 전에는 평가 대상으로
받아들이지 않습니다"라고 규정한다.

| # | 항목 | 근거 조항 | 조치 |
| --- | --- | --- | --- |
| 1 | `submission-ossp-skt.json` 미작성 | [SUBMISSION.md](SUBMISSION.md) §기술 제출 정보 파일 | §12 절차 수행 (커밋 SHA·이미지 다이제스트 확정 후) |
| 2 | `container/BASE_IMAGE.md`가 **alpine** 기반 이미지를 기록 — 실제 Dockerfile은 `python:3.11-slim-bookworm` | [RUNTIME.md](RUNTIME.md) §이미지 빌드와 제출 ("기반 이미지 출처와 라이선스는 BASE_IMAGE.md에 기록") | 실제 기반 이미지·다이제스트·라이선스로 문서 갱신 |
| 3 | `models/final-v1`(142.5 MB)이 `REUSE.toml` 주석 경로에 없음 | [CHALLENGE_RULES.md](CHALLENGE_RULES.md) §사용할 수 있는 정보 ("이미지에 포함한 모든 파일은 제출 저장소에서 출처와 라이선스를 확인할 수 있어야") | `REUSE.toml`에 `models/**` 항목 추가 (SPDX: Apache-2.0, 유래: 공개 Train/Dev) |
| 4 | `THIRD_PARTY_NOTICES.md`에 이미지 동봉 런타임 4종 미기재 | [RUNTIME.md](RUNTIME.md) §이미지 빌드와 제출 ("버전과 라이선스를 기록") | numpy 2.0.2 (BSD-3-Clause), scipy 1.17.1 (BSD-3-Clause), lightgbm 4.7.0 (**MIT**), xgboost 3.2.0 (Apache-2.0), libgomp1 (GPL-3.0 + GCC Runtime Library Exception) 절 추가 |
| 5 | `container/Dockerfile:9` 주석이 lightgbm을 BSD-3-Clause로 표기 | 위와 동일 | **MIT**로 정정 (`exp/final-report.md`에는 이미 MIT로 올바르게 기록됨) |
| 6 | 이미지 레지스트리 push 후 다이제스트 기록 | [SUBMISSION.md](SUBMISSION.md) §컨테이너 이미지 | §10 이후 수행 |

**참고 — 2026-08-17에 처리 완료한 항목**

| 항목 | 처리 |
| --- | --- |
| CRLF로 인한 LightGBM 기동 실패 | `.gitattributes` 추가 + 작업 트리 정규화 (부록 A) |
| `models/final-v1/manifest.json` 26개 항목 중 13개 해시 불일치, 117.5 MB 미등재 | 재생성 — 27/27 일치 |
| 참가자 문서 4개 SPDX 헤더 누락 | 헤더 추가, 해당 테스트 통과 |
| `setup.cfg`의 `router-run`이 약한 baseline(`heuristic:main`)을 가리킴 | `final_router:main`으로 정정. 이미지는 `ENTRYPOINT`로 `entrypoint.py`를 직접 호출하므로 **평가 대상 실행 경로는 원래부터 정상**이었고, [ENFORCEMENT.md](ENFORCEMENT.md) §공정성 위반의 "제출 소스와 실행 프로그램의 불일치" 감사 플래그만 제거한 것이다 |
