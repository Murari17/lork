# Dado Architecture

## Overview

`dado` is an open-source LoRA fine-tuning execution engine built around low-level training primitives. The caller owns the training loop, dataset construction, and loss shaping; `dado` owns model execution, gradient accumulation, optimizer stepping, sampling, and state persistence.

---

## Package Structure

```text
dado/
├── pyproject.toml                   # Split client deps vs [server] & [dev] extras, release metadata
├── LICENSE                          # Apache-2.0
├── CHANGELOG.md                     # Release history
├── Dockerfile & docker-compose.yml  # CUDA 12.4 runtime container (unbuilt locally without Docker)
├── .github/workflows/ci.yml         # CI: lint+type, fast no-torch tests, full CPU tests, wheel check
├── AGENTS.md                        # Project spec & hard rules 1-11
├── ARCHITECTURE.md                  # Architecture & phase documentation
├── README.md                        # Quickstart, benchmarks, examples, limits
├── docs/
│   ├── api.md                       # HTTP /v1 endpoint reference with curl examples
│   ├── retries.md                   # Client retry & exception mapping table
│   ├── benchmark.md                 # Benchmark methodology, 8.4e-5 deviation analysis, p50/p95 latencies & payload sizes
│   ├── benchmark.json               # Measured 20-step HTTP vs direct PEFT+AdamW benchmark
│   └── loss_curves.png              # Loss curve plot
├── examples/
│   ├── sft.py                       # ~30-line SFT example via ServiceClient
│   ├── grpo_toy.py                  # Deterministic GRPO RL loop via sample(seed=...) + advantage loss
│   └── colab_quickstart.ipynb       # Colab GPU notebook
├── scripts/
│   ├── smoke_http.py                # Real-uvicorn 20-step HTTP SFT smoke script
│   ├── gpu_check.py                 # CUDA bf16 verification script
│   └── benchmark.py                 # Live HTTP vs plain PEFT+AdamW benchmark
├── src/dado/
│   ├── __init__.py                  # Eager wire exports; lazy __getattr__ for LoraTrainer
│   ├── py.typed                     # PEP 561 type marker
│   ├── wire.py                      # Pure-Pydantic wire schemas (no torch/transformers/peft)
│   ├── schemas.py                   # Torch-backed tensor validators + wire re-exports
│   ├── trainer.py                   # LoraTrainer core execution class
│   ├── client/
│   │   ├── __init__.py              # Public Python SDK exports
│   │   ├── py.typed                 # PEP 561 marker for client package
│   │   ├── exceptions.py            # Typed SDK exception hierarchy
│   │   ├── futures.py               # Long-polling APIFuture[T]
│   │   └── client.py                # ServiceClient, TrainingClient, batch validator, retries
│   └── server/
│       ├── __init__.py              # Server package exports
│       ├── models.py                # ServerSettings & limit-checked HTTP wrappers
│       ├── futures.py               # Bounded FutureStore & long-poll FutureRecord
│       ├── sessions.py              # Per-session FIFO worker threads & TTL SessionManager
│       └── app.py                   # FastAPI application factory, body limit, auth, logging, /v1 routes
└── tests/
    ├── conftest.py                  # Shared model/device fixtures + HF_HUB_OFFLINE=1
    ├── test_trainer.py              # Phase 1 + dtype/GPU verification suite
    ├── test_server.py               # Phase 2 + safety defaults verification suite
    └── test_client.py               # Phase 3 live-server SDK & examples verification suite
```

---

## Phase 1: `LoraTrainer` Core

- **Input Validation (`src/dado/schemas.py` & `src/dado/wire.py`)**: Pure-Pydantic configuration and wire schemas live in `wire.py` with zero heavy ML imports. `schemas.py` validates `torch.Tensor` batches and precomputed advantage tensors.
- **Execution (`src/dado/trainer.py`)**:
  - Loads causal LM in the resolved `dtype`, freezes base weights, injects LoRA adapters (`autocast_adapter_dtype=True`), and keeps trainable LoRA parameters and `AdamW` optimizer state in `torch.float32`.
  - Computes per-token log-probabilities and loss in `torch.float32`, backpropagates unnormalized weighted token losses, and divides accumulated gradients by total accumulated window weight at `optim_step`.
  - Tracks `_applied_steps` for idempotent `optim_step`, supports KV-cached `sample()` with optional per-call `torch.Generator` seeding (`SampleParams.seed`), and persists/restores weights + `AdamW` state via `save_state()` / `load_state()`.

---

## Phase 2: Async FastAPI HTTP Server

- **Non-blocking event loop**: Mutating endpoints return `202 Accepted` (`{"future_id", "session_id"}`); `LoraTrainer` initialization and execution run strictly on per-session FIFO worker threads (`SessionWorker`).
- **Fault isolation & idempotency**: Worker exceptions map to `status="error"` futures without poisoning subsequent ops; `SessionWorker` caches `optim_step` results by `step_id` across subsequent failures.
- **Security & Limits**: `Bearer` auth via `hmac.compare_digest`, env-configurable `ServerSettings` limits (`422` on over-limit), symlink-safe checkpoint containment (`^[A-Za-z0-9_-]{1,64}$`), bounded `FutureStore`, idle TTL eviction (`404`), and `max_sessions` default `2` (`~2 GB` RAM per `fp32` `0.5B` session until shared base weights).

---

## Phase 3: Lightweight Python SDK & Packaging Split

- **Zero-ML Client Packaging**: `pip install dado` installs only `httpx`, `pydantic`, and lightweight transitive deps. `import dado` and `import dado.client` never import `torch`, `transformers`, or `peft`.
- **Correctness-Critical Retry Semantics**: `optim_step` (reusing identical `step_id`), `GET /v1/futures/{id}`, and `DELETE` retry on pre-send errors, `429`, and `5xx`. Non-idempotent submissions (`forward_backward`, `sample`, `save_state`, `load_state`, `create_session`) retry **only** on pre-send transport failures, `429`, or `503` with our structured `ErrorEnvelope`; any other post-send failure (including `502`/`500`/`504`/drops) raises `AmbiguousSubmissionError` without retrying.

---

## Phase 4: Launch Hardening

- **`dtype` & CUDA Mixed Precision**: `TrainerInitConfig` and `CreateSessionRequest` accept `dtype: Literal["float32", "bfloat16", "float16"] | None`. Defaults to `"bfloat16"` on CUDA devices supporting bf16 and `"float32"` on CPU/MPS; rejects `"float16"` on CPU. Base weights use the selected dtype while LoRA adapters, `AdamW` state, and loss computation remain `float32`.
- **Server Safety Defaults**: Refuses startup without `DADO_API_KEYS` (no default key), defaults `host` to `127.0.0.1`, enables zero CORS headers, and enforces `max_request_body_bytes` (default `10 MiB`) in raw ASGI middleware `_BodySizeLimitMiddleware` on both `Content-Length` and chunked/streaming bodies before JSON parsing (`413`). `forward_backward` supports `return_logprobs: bool = True` to omit per-token logprobs when unused.
- **Release & Tooling**: `Apache-2.0` `LICENSE`, `CHANGELOG.md`, `dist/` wheel + sdist build, `scripts/gpu_check.py`, `scripts/benchmark.py`, `Dockerfile` + `docker-compose.yml`, `.github/workflows/ci.yml`, `docs/api.md`, `docs/retries.md`, `docs/benchmark.md`, and `examples/colab_quickstart.ipynb`.
