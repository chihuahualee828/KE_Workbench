FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ARG INSTALL_EASYEDIT=false
ENV EASYEDIT_ROOT=/opt/EasyEdit

RUN if [ "$INSTALL_EASYEDIT" = "true" ]; then \
      git clone --depth=1 https://github.com/zjunlp/EasyEdit.git "$EASYEDIT_ROOT" \
      && pip install --no-cache-dir -r "$EASYEDIT_ROOT/requirements.txt"; \
    fi

COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir -e ".[ml]"

EXPOSE 8000