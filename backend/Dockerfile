# syntax=docker/dockerfile:1

ARG INSTALL_EASYEDIT=true
# Match the EasyEdit revision used by the workbench integration.
ARG EASYEDIT_REVISION=14cea8245f06715684592ab55184939b99d70784

FROM python:3.10-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    EASYEDIT_ROOT=/opt/EasyEdit \
    DATA_DIR=/app/data \
    HF_HUB_CACHE=/hf-cache

WORKDIR /app

# EasyEdit eagerly imports OpenCV even for text-only editing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS dependencies

COPY --from=ghcr.io/astral-sh/uv:0.12.20 /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && uv venv --python /usr/local/bin/python /opt/venv

ARG INSTALL_EASYEDIT
ARG EASYEDIT_REVISION

# Keep code, hyperparameters and licensing, without Git history or demo assets.
RUN mkdir -p "$EASYEDIT_ROOT" \
    && case "$INSTALL_EASYEDIT" in \
      true) \
        git init /tmp/easyedit \
        && git -C /tmp/easyedit remote add origin https://github.com/zjunlp/EasyEdit.git \
        && git -C /tmp/easyedit fetch --depth=1 origin "$EASYEDIT_REVISION" \
        && git -C /tmp/easyedit checkout --detach FETCH_HEAD \
        && cp -a /tmp/easyedit/easyeditor /tmp/easyedit/hparams \
            /tmp/easyedit/LICENSE /tmp/easyedit/requirements.txt "$EASYEDIT_ROOT/" \
        && rm -rf /tmp/easyedit ;; \
      false) ;; \
      *) echo 'INSTALL_EASYEDIT must be true or false' >&2; exit 1 ;; \
    esac

# Resolve both sets of requirements together; app-only changes reuse this layer.
# uv.lock currently covers the workbench, not EasyEdit's full environment.
COPY pyproject.toml /build/pyproject.toml
RUN if [ "$INSTALL_EASYEDIT" = "true" ]; then \
      uv pip install --no-cache --python /opt/venv/bin/python --extra ml \
        -r /build/pyproject.toml -r "$EASYEDIT_ROOT/requirements.txt"; \
    else \
      uv pip install --no-cache --python /opt/venv/bin/python --extra ml \
        -r /build/pyproject.toml; \
    fi \
    && uv pip check --python /opt/venv/bin/python

FROM base AS runtime

COPY --from=dependencies /opt/venv /opt/venv
COPY --from=dependencies /opt/EasyEdit /opt/EasyEdit
COPY app ./app

ARG INSTALL_EASYEDIT
RUN mkdir -p /app/data /hf-cache \
    && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c 'from app.main import app; print("Backend import OK")' \
    && if [ "$INSTALL_EASYEDIT" = "true" ]; then \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c \
        'from app.ml.editing.easyedit_runtime import _load_easyedit; _load_easyedit("ROMEHyperParams"); print("EasyEdit import OK")'; \
    fi

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
