# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

# python:3.11-slim-bookworm 다중 플랫폼 인덱스 다이제스트 고정 (Debian glibc:
# numpy/scipy/lightgbm/xgboost manylinux aarch64 휠 설치 가능).
FROM python:3.11-slim-bookworm@sha256:2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91

# 런타임 추론 의존성 (BSD-3-Clause: numpy/scipy/lightgbm, Apache-2.0: xgboost).
# 버전은 학습 환경과 동일하게 고정. 설치 후 pip 제거(실행 중 다운로드 없음).
RUN python3 -m pip install --no-cache-dir \
        numpy==2.0.2 \
        scipy==1.17.1 \
        lightgbm==4.7.0 \
        xgboost==3.2.0 \
    && apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip uninstall --yes pip setuptools wheel

ARG SOURCE_MANIFEST_SHA256=unbound

LABEL io.sktelecom.ossp.source-manifest-sha256="${SOURCE_MANIFEST_SHA256}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/router \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp \
    OSSP_MODEL_BUNDLE=/opt/router/model-bundle \
    OMP_NUM_THREADS=2

COPY --chown=65532:65532 src container/entrypoint.py /opt/router/
COPY --chown=65532:65532 models/final-v1 /opt/router/model-bundle

WORKDIR /opt/router

USER 65532:65532

ENTRYPOINT ["python3", "/opt/router/entrypoint.py"]
