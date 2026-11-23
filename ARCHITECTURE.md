# loraforge Architecture

## Overview

`loraforge` is an open-source LoRA fine-tuning execution engine built around low-level training primitives. The caller owns the training loop, dataset construction, and loss shaping; `loraforge` owns model execution, gradient accumulation, optimizer stepping, sampling, and state persistence.

---

## Package Structure

```text
loraforge/
├── pyproject.toml                   # Split client deps (httpx, pydantic) vs [server] & [dev] extras
├── AGENTS.md                        # Project spec & hard rules 1-11
├── ARCHITECTURE.md                  # Architecture & phase documentation
├── README.md                        # Quickstart guide
├── examples/
│   ├── sft.py                       # ~30-line SFT example via ServiceClient
│   └── grpo_toy.py                  # Toy GRPO RL loop via sample + advantage loss_fn
├── scripts/
│   └── smoke_http.py                # Real-uvicorn 20-step HTTP SFT smoke script
├── src/loraforge/
│   ├── __init__.py                  # Eager wire exports; lazy __getattr__ for LoraTrainer
│   ├── py.typed                     # PEP 561 type marker
│   ├── wire.py                      # Pure-Pydantic wire schemas (no torch/transformers/peft)
│   ├── schemas.py                   # Torch-backed tensor validators + wire re-exports
│   ├── trainer.py                   # LoraTrainer core execution class
│   ├── client/
│   │   ├── __init__.py              # Public Python SDK exports
│   │   ├── exceptions.py            # Typed SDK exception hierarchy
│   │   ├── futures.py               # Long-polling APIFuture[T]
│   │   └── client.py                # ServiceClient, TrainingClient, batch validator, retries
│   └── server/
│       ├── __init__.py              # Server package exports
│       ├── models.py                # ServerSettings & limit-checked HTTP wrappers
│       ├── futures.py               # Bounded FutureStore & long-poll FutureRecord
│       ├── sessions.py              # Per-session FIFO worker threads & TTL SessionManager
│       └── app.py                   # FastAPI application factory, auth, logging, /v1 routes
└── tests/
    ├── conftest.py                  # Shared model/device fixtures + HF_HUB_OFFLINE=1
    ├── test_trainer.py              # Phase 1 verification suite
    ├── test_server.py               # Phase 2 async HTTP server verification suite
    └── test_client.py               # Phase 3 live-server SDK & examples verification suite
```

---

## Phase 1: `LoraTrainer` Core

### 1. Input Validation (`src/loraforge/schemas.py` & `src/loraforge/wire.py`)
- Pure-Pydantic configuration and wire schemas live in `src/loraforge/wire.py` with zero heavy ML imports (`TrainerInitConfig`, `OptimParams`, `SampleParams`, `WireBatch`, etc.).
- `src/loraforge/schemas.py` adds `torch.Tensor` validation (`ForwardBackwardBatch`, `AdvantageLossInput`, `validate_loss_fn`) and re-exports the shared `wire.py` models.

### 2. `LoraTrainer` (`src/loraforge/trainer.py`)
- Loads causal LM in `torch.float32`, freezes all base parameters, injects LoRA adapters, and optimizes only `lora_` parameters with `AdamW`.
- Backpropagates unnormalized weighted token losses and divides accumulated gradients by total accumulated window weight at `optim_step`.
- Tracks `_applied_steps` for idempotent `optim_step`, KV-cached `sample()`, and `save_state()` / `load_state()` for bit-identical resume.

---

## Phase 2: Async FastAPI HTTP Server

- **Non-blocking event loop**: Mutating endpoints return `202 Accepted` (`{"future_id", "session_id"}`); `LoraTrainer` initialization and execution run strictly on per-session FIFO worker threads (`SessionWorker`).
- **Fault isolation & idempotency**: Worker exceptions map to `status="error"` futures without poisoning subsequent ops; `SessionWorker` caches `optim_step` results by `step_id` across subsequent failures.
- **Security & Limits**: `Bearer` auth via `hmac.compare_digest`, env-configurable `ServerSettings` limits (`422` on over-limit, never truncating), symlink-safe checkpoint containment (`^[A-Za-z0-9_-]{1,64}$`), bounded `FutureStore`, idle TTL eviction (`404`), and `max_sessions` default `2` (`~2 GB` RAM per `fp32` `0.5B` session until Phase 4 shared base weights).

---

## Phase 3: Lightweight Python SDK & Packaging Split

- **Zero-ML Client Packaging**: `pip install loraforge` installs only `httpx`, `pydantic`, and their lightweight transitive dependencies. `import loraforge` and `import loraforge.client` never import `torch`, `transformers`, or `peft` (`LoraTrainer` and `ForwardBackwardBatch` are imported lazily via `loraforge.__getattr__`).
- **Client API (`src/loraforge/client/`)**:
  - `ServiceClient`: manages authenticated `httpx.Client`, session creation/deletion, and retry/error mapping.
  - `TrainingClient`: context manager deleting the session on `__exit__`; submits pipelined async operations returning `APIFuture[T]` (`forward_backward`, `optim_step`, `sample`, `save_state`, `load_state`).
  - `APIFuture[T]`: implements `done()` and `result(timeout=None)` looping on `GET /v1/futures/{id}?wait=25` until completion or raising `LoraforgeTimeoutError` without cancelling the remote op.
- **Correctness-Critical Retry Semantics**:
  - Pre-send transport errors (`ConnectError`, `ConnectTimeout`, `PoolTimeout`) and `429`/`5xx` responses retry with exponential backoff + jitter, honoring numeric `Retry-After` headers.
  - `optim_step` generates `step_id` (`uuid4`) once per call and retries safely using the identical `step_id`.
  - Non-idempotent submissions (`forward_backward`, `sample`, `save_state`, `load_state`) **never** auto-retry after post-send transport failures (`ReadTimeout`, `ReadError`, `WriteError`, `RemoteProtocolError`), instead raising `AmbiguousSubmissionError` carrying the underlying exception.
  - Future `GET` polling is always safely retried.
- **Typed Exceptions**: `AuthError` (`401`), `ValidationError` (`400`/`403`/`422`/client-side batch shape), `SessionNotFoundError` (`404`), `ServerBusyError` (`429`/`503`), `RemoteOpError`, `AmbiguousSubmissionError`, `LoraforgeTimeoutError`.
