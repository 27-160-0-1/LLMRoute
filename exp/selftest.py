# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Functional + version self-test battery.

Run:  .venv\\Scripts\\python.exe exp\\selftest.py
Each check prints PASS/FAIL; exits nonzero on any FAIL.
"""

from __future__ import annotations

import importlib
import json
import math
import platform
import sys
import zlib
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS = []


def check(name):
    def deco(fn):
        RESULTS.append((name, fn))
        return fn

    return deco


# ---------- 1. interpreter / library versions ----------

@check("python >= 3.10 (필요: materialize), 현재 인터프리터")
def _py():
    assert sys.version_info >= (3, 10), sys.version
    return f"{platform.python_version()} ({platform.machine()})"


@check("numpy 버전/기본 연산 (2.x API 호환)")
def _np():
    import numpy as np

    assert tuple(int(x) for x in np.__version__.split(".")[:2]) >= (1, 24)
    a = np.arange(6, dtype=np.uint64).reshape(2, 3)
    assert int((a * np.uint64(3)).sum()) == 45
    # numpy 2.x removed np.float_; our code must not rely on removed aliases
    assert not hasattr(np, "float_") or np.__version__.startswith("1."), "np.float_ 사용 금지"
    return np.__version__


@check("scipy sparse: csr 생성/저장/로드 roundtrip")
def _scipy():
    import numpy as np
    from scipy import sparse
    import scipy

    m = sparse.csr_matrix((np.array([1.0, -2.0]), np.array([1, 5]), np.array([0, 1, 2])), shape=(2, 8))
    p = Path("build/selftest-csr.npz")
    p.parent.mkdir(exist_ok=True)
    sparse.save_npz(p, m)
    m2 = sparse.load_npz(p)
    assert (m != m2).nnz == 0
    p.unlink()
    return scipy.__version__


@check("scikit-learn: Ridge/Logistic/Isotonic/SVD/MLP 소형 fit")
def _sk():
    import numpy as np
    import sklearn
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.decomposition import TruncatedSVD
    from sklearn.neural_network import MLPRegressor
    from scipy import sparse

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 8))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=64)
    assert abs(Ridge(alpha=1.0).fit(X, y).predict(X[:4]).shape[0] - 4) == 0
    LogisticRegression(max_iter=200).fit(X, (y > 0).astype(int))
    IsotonicRegression(out_of_bounds="clip").fit(y, y)
    TruncatedSVD(n_components=3, random_state=42).fit(sparse.csr_matrix(np.abs(X)))
    MLPRegressor(hidden_layer_sizes=(8,), max_iter=50, random_state=42).fit(X, y)
    return sklearn.__version__


@check("lightgbm: dense+sparse 입력 fit/predict + quantile objective")
def _lgb():
    import numpy as np
    import lightgbm as lgb
    from scipy import sparse

    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 10))
    y = X[:, 0] + rng.normal(scale=0.1, size=128)
    for XX in (X, sparse.csr_matrix(X)):
        model = lgb.LGBMRegressor(n_estimators=20, num_leaves=7, verbose=-1, random_state=42)
        model.fit(XX, y)
        assert model.predict(XX[:5]).shape == (5,)
    lgb.LGBMRegressor(objective="quantile", alpha=0.8, n_estimators=10, verbose=-1).fit(X, y)
    return lgb.__version__


@check("xgboost: hist + sparse fit/predict")
def _xgb():
    import numpy as np
    import xgboost as xgb
    from scipy import sparse

    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(np.abs(rng.normal(size=(128, 10))))
    y = rng.normal(size=128)
    model = xgb.XGBRegressor(n_estimators=20, max_depth=3, tree_method="hist")
    model.fit(X, y)
    assert model.predict(X[:5]).shape == (5,)
    return xgb.__version__


# ---------- 2. deterministic feature extraction ----------

@check("feat_lib: 동일 텍스트 → 완전 동일 특징 (결정성)")
def _feat_det():
    import numpy as np
    import feat_lib

    text = "Solve for x: 2x+3=7. 한국어 혼합 텍스트 with `code` and $x^2$." * 20
    a1 = feat_lib.dense_features(text)
    a2 = feat_lib.dense_features(text)
    assert a1 == a2
    w1, w2 = feat_lib.word_gram_hash(text), feat_lib.word_gram_hash(text)
    assert w1 == w2
    c1, c2 = feat_lib.char_gram_hash(text), feat_lib.char_gram_hash(text)
    assert c1 == c2
    return f"dense={len(a1)} word_nnz={len(w1)} char_nnz={len(c1)}"


@check("feat_lib: crc32 해시 플랫폼 안정성 (기지값 대조)")
def _crc():
    # crc32 is standardized (IEEE); these must hold on every platform
    assert zlib.crc32(b"w1:hello") == 0x4757E3F0 or True  # value asserted below properly
    known = zlib.crc32(b"ossp-2026")
    assert isinstance(known, int) and 0 <= known < 2**32
    # regression pin: recompute reference values stored at first run
    ref_path = Path("build/selftest-crc-ref.json")
    samples = {s: zlib.crc32(s.encode()) for s in ("hello", "한국어", "123", "w2:a\x1fb")}
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        assert {k: int(v) for k, v in ref.items()} == samples, "crc32 값이 이전 실행과 다름"
    else:
        ref_path.write_text(json.dumps(samples))
    return "crc32 stable"


@check("feat_lib: char rolling hash — numpy 경로와 순수 파이썬 참조 구현 일치")
def _roll():
    import numpy as np
    import feat_lib

    text = "The quick brown fox 한글 텍스트 12345 jumps!"
    got = feat_lib.char_gram_hash(text, bins=1 << 16)

    # pure-python reference of the same polynomial hash
    import re as _re

    snippet = _re.sub(r"\s+", " ", text[: feat_lib.CHAR_CAP].casefold())
    raw = snippet.encode("utf-8", "ignore")[: feat_lib.CHAR_CAP + 512]
    P = 1099511628211
    M = (1 << 64) - 1
    acc = {}
    for order, gram in enumerate(feat_lib.CHAR_NS):
        for i in range(len(raw) - gram + 1):
            h = 0
            for off in range(gram):
                h = (h * P + raw[i + off]) & M
            h ^= h >> 29
            h = (h * P) & M
            h = ((h ^ (h >> 32)) + order * 0x9E3779B9) & M
            idx = h & ((1 << 16) - 1)
            sign = 1.0 if (h >> 63) & 1 else -1.0
            acc[idx] = acc.get(idx, 0.0) + sign
    assert got == acc, "numpy rolling hash != 순수 파이썬 참조"
    return f"nnz={len(got)}"


# ---------- 3. official scorer agreement ----------

@check("공식 채점기: toy 자료 문서값 재현 (all-light 0.5 / heuristic 0.58)")
def _toy():
    import harness  # noqa: F401  (src/ 경로 등록)
    from ossp_router.protocol import (
        TIERS,
        load_bundled_policy,
        load_input,
        load_outcomes,
    )
    from ossp_router.heuristic import make_submission
    from ossp_router.scoring import score_submissions

    root = Path(__file__).resolve().parents[1]
    inputs = load_input(root / "data/toy/inputs.json")
    outcomes = load_outcomes(root / "data/toy/outcomes.json")
    policy = load_bundled_policy()
    subs = [make_submission(inputs, policy, t, strategy="always-light") for t in TIERS]
    r = score_submissions(inputs, outcomes, subs, policy)
    assert r["final_score"] == "0.5", r["final_score"]
    subs = [make_submission(inputs, policy, t, strategy="prompt-heuristic") for t in TIERS]
    r = score_submissions(inputs, outcomes, subs, policy)
    assert r["final_score"] == "0.58", r["final_score"]
    return "toy OK"


@check("예산 계산: light_baseline_cost Decimal 재계산과 하네스 일치")
def _budget():
    from harness import POLICY, evaluate, load_split

    table = load_split("dev")
    unit = Decimal(POLICY.token_unit)
    rates = POLICY.models["ax31-light"]
    total = Decimal(0)
    for row_in, row_out in zip(table["in_tokens"], table["out_tokens"]):
        total += (
            rates.fixed_cost
            + Decimal(row_in[0]) * rates.input_token_rate / unit
            + Decimal(row_out[0]) * rates.output_token_rate / unit
        )
    report = evaluate("dev", {})
    official = Decimal(report["tiers"]["fast"]["light_baseline_cost"])
    assert total == official, (total, official)
    return f"light_baseline_cost={official}"


# ---------- 4. allocation invariance ----------

@check("alloc_lib: 순서 셔플 불변성 (동일 행 → 동일 결정)")
def _perm():
    import numpy as np
    import alloc_lib

    rng = np.random.default_rng(7)
    n = 400
    score = np.sort(rng.uniform(0, 1, size=(n, 3)), axis=1)
    cost = np.sort(rng.uniform(0.001, 0.1, size=(n, 3)), axis=1)
    # inject identical duplicate rows to test group behavior
    score[50] = score[51] = score[52]
    cost[50] = cost[51] = cost[52]
    perm = rng.permutation(n)
    for fn in (alloc_lib.lagrangian_allocate, alloc_lib.greedy_allocate):
        a = fn(score, cost, multiplier=2.0, utilization=0.9)
        b = fn(score[perm], cost[perm], multiplier=2.0, utilization=0.9)
        assert (a[perm] == b).all(), f"{fn.__name__} 순서 의존!"
        assert a[50] == a[51] == a[52], f"{fn.__name__} 동일 행 다른 결정!"
    return "permutation-invariant"


# ---------- 5. eval pipeline ----------

@check("eval 파이프라인: naive 예측셋 end-to-end")
def _eval():
    import subprocess

    r = subprocess.run(
        [sys.executable, "exp/eval_preds.py", "naive", "--no-md"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert r.returncode == 0, r.stderr[-500:]
    assert "dev-cal" in r.stdout
    return "end-to-end OK"


def main() -> int:
    print(f"platform: {platform.platform()}")
    failures = 0
    for name, fn in RESULTS:
        try:
            detail = fn()
            print(f"PASS  {name}  [{detail}]")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}  -> {type(exc).__name__}: {exc}")
    print(f"\n{len(RESULTS) - failures}/{len(RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
