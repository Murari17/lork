# loraforge Architecture

## Overview

`loraforge` is an open-source LoRA fine-tuning execution engine built around low-level training primitives. The caller owns the training loop, dataset construction, and loss shaping; `loraforge` owns model execution, gradient accumulation, optimizer stepping, sampling, and state persistence.

---

## Package Structure

```text
loraforge/
├── pyproject.toml                   # Pinned dependencies (Python >=3.11), ruff, mypy --strict, pytest config
├── AGENTS.md                        # Project spec & hard rules 1-11
├── ARCHITECTURE.md                  # Architecture & phase documentation
├── scripts/
│   └── smoke_http.py                # Real-uvicorn 20-step HTTP SFT smoke script
├── src/loraforge/
│   ├── __init__.py                  # Public exports (LoraTrainer, schemas)
│   ├── py.typed                     # PEP 561 type marker
│   ├── schemas.py                   # Pydantic v2 input validation models
│   ├── trainer.py                   # LoraTrainer core execution class
│   └── server/
│       ├── __init__.py              # Server package exports
│       ├── models.py                # ServerSettings & HTTP models (reusing schemas.*)
│       ├── futures.py               # Bounded FutureStore & long-poll FutureRecord
│       ├── sessions.py              # Per-session FIFO worker threads & TTL SessionManager
│       └── app.py                   # FastAPI application factory, auth, logging, /v1 routes
└── tests/
    ├── conftest.py                  # Shared model/device fixtures (Qwen/Qwen2.5-0.5B)
    ├── test_trainer.py              # Phase 1 verification suite
    └── test_server.py               # Phase 2 async HTTP server verification suite
```

---

## Phase 1: `LoraTrainer` Core

### 1. Input Validation (`src/loraforge/schemas.py`)
All external inputs to `LoraTrainer` are validated at the boundary using Pydantic v2 (`ConfigDict(extra="forbid")`):
- `TrainerInitConfig`: validates `base_model`, `rank > 0`, `alpha > 0`, non-empty `target_modules`, and `device` (`cpu`, `mps`, `cuda`, `cuda:N`).
- `ForwardBackwardBatch`: validates `input_ids`, `target_ids`, and `weights` as matching 2D `(B, T)` tensors (or nested lists coerced to `int64`/`float32`), enforcing finite non-negative `weights` with positive total weight.
- `validate_loss_fn`: accepts `"cross_entropy"` or a finite 2D advantage tensor of shape `(B, T)`.
- `OptimStepRequest` / `OptimParams`: validates non-empty `step_id`, `lr > 0`, `betas` strictly in `(0, 1)`, `weight_decay >= 0`, and `eps > 0` (default `1e-8`).
- `SampleRequest` / `SampleParams`: validates non-empty `prompts`, `max_tokens > 0`, and `temperature >= 0`.
- `StatePathInput`: validates checkpoint directory path.

### 2. `LoraTrainer` (`src/loraforge/trainer.py`)
- **Model & Adapter Setup**: loads the causal LM in `torch.float32`, freezes all base parameters (`requires_grad=False`), injects LoRA adapters (`lora_dropout=0.0`, `bias="none"`, `TaskType.CAUSAL_LM`), verifies only `lora_` parameters have `requires_grad=True`, and initializes `torch.optim.AdamW`.
- **Forward/Backward & Gradient Accumulation**: computes per-token log-probabilities, backpropagates the unnormalized weighted sum $S_m = \sum_{b,t} \ell^{(m)}_{b,t}$, accumulates $W_m = \sum_{b,t} w^{(m)}_{b,t}$ into `_accum_weight_sum`, and scales accumulated gradients by $1 / \sum_m W_m$ at `optim_step`.
- **Idempotent `optim_step`**: tracks `_applied_steps: dict[str, int]` so replaying `step_id` returns the stored `{"step": n}` without re-applying.
- **Sampling & State Persistence**: KV-cached autoregressive generation in `sample()`; `save_state()` / `load_state()` persist LoRA adapter weights (`safetensors`) and `AdamW` optimizer state, step ledger, pending gradients, and CPU RNG state.

---

## Phase 2: Async FastAPI HTTP Server

### Concurrency & Execution Model
- **Non-blocking event loop**: Every mutating/model endpoint (`POST /v1/sessions`, `forward_backward`, `optim_step`, `sample`, `save_state`, `load_state`) enqueues work and returns `202 Accepted` with `{"future_id": ..., "session_id": ...}` immediately. Model loading, forward/backward passes, optimizer steps, and generation never run on the asyncio event loop.
- **One dedicated worker thread per session**: `SessionWorker` owns a single FIFO `queue.Queue` drained sequentially by one dedicated thread. Operations on a session execute strictly in submission order and never concurrently, while separate sessions run independently.
- **Fault isolation & idempotency**: Exceptions inside `LoraTrainer` are caught in the worker loop, recorded as `status="error"` on the `FutureRecord` (`{code, message}`), and never poison subsequent queued ops or crash the worker. `SessionWorker` caches completed `optim_step` results by `step_id`, so replaying `optim_step` returns the stored result even after intervening failed ops.
- **Bounded Futures & Long-Polling**: `FutureStore` retains futures with TTL and max-capacity eviction. `GET /v1/futures/{future_id}?wait=<sec>` (max `30s`) uses `asyncio.to_thread(record.wait_done, wait)` to wake immediately upon completion without blocking the event loop.

### Validation, Security & Lifecycle
- **Wire Models (`src/loraforge/server/models.py`)**: Reuses `schemas.OptimParams` (including `eps`), `schemas.SampleParams`, and `schemas.TrainerInitConfig` directly. Enforces env-configurable `ServerSettings` limits (`max_batch_size`, `max_seq_len`, `max_sample_tokens`, `max_prompts`) returning `422` on over-limit inputs without truncation.
- **Authentication**: Static API keys loaded from `LORAFORGE_API_KEYS` validated via `hmac.compare_digest` on `Authorization: Bearer <key>`, returning `401` on missing/invalid keys. Keys, prompts, and token IDs are never logged.
- **Structured Logging**: Middleware attaches `X-Request-ID` and emits JSON logs containing `request_id`, `method`, `path`, `status_code`, and `duration_ms`.
- **Checkpoint Path Safety**: `save_state` / `load_state` restrict names to `^[A-Za-z0-9_-]{1,64}$`, resolve under a per-session directory inside `checkpoint_root`, and reject traversal or symlinks pointing outside the session checkpoint directory.
- **Session Limits & Shutdown**: Enforces `max_sessions` (default `2`, configurable via `LORAFORGE_MAX_SESSIONS`; returns `429` at capacity), sweeps idle sessions past `session_idle_ttl_seconds` (`404` on subsequent access), frees `LoraTrainer` references + empties CUDA/MPS caches on close/eviction, and drains all session workers on server shutdown (logging `server_shutdown_drained`). Because Phase 2 instantiates an independent `LoraTrainer` and `fp32` base model per session (`~2 GB` RAM per `Qwen/Qwen2.5-0.5B` session), `max_sessions` defaults to `2` until Phase 4 introduces shared base-model weights across adapters.
