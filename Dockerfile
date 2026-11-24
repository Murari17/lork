FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    DADO_DEVICE=cuda \
    DADO_HOST=0.0.0.0 \
    DADO_PORT=8000 \
    DADO_CHECKPOINT_ROOT=/data/checkpoints

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 dado \
    && mkdir -p /app /cache/huggingface /data/checkpoints \
    && chown -R dado:dado /app /cache/huggingface /data/checkpoints

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python3.11 -m pip install --no-cache-dir --upgrade pip \
    && python3.11 -m pip install --no-cache-dir ".[server]"

USER dado
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/v1/healthz || exit 1

CMD ["sh", "-c", "python3.11 -m uvicorn dado.server.app:create_app --factory --host ${DADO_HOST} --port ${DADO_PORT}"]
