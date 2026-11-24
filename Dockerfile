FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    LORAFORGE_DEVICE=cuda \
    LORAFORGE_HOST=0.0.0.0 \
    LORAFORGE_PORT=8000 \
    LORAFORGE_CHECKPOINT_ROOT=/data/checkpoints

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 loraforge \
    && mkdir -p /app /cache/huggingface /data/checkpoints \
    && chown -R loraforge:loraforge /app /cache/huggingface /data/checkpoints

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python3.11 -m pip install --no-cache-dir --upgrade pip \
    && python3.11 -m pip install --no-cache-dir ".[server]"

USER loraforge
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/v1/healthz || exit 1

CMD ["sh", "-c", "python3.11 -m uvicorn loraforge.server.app:create_app --factory --host ${LORAFORGE_HOST} --port ${LORAFORGE_PORT}"]
