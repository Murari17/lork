# Changelog

All notable changes to `dado` are documented in this file.

## [0.1.0] - Phase 4 (Launch Hardening)

### Added
- `dtype` (`float32`, `bfloat16`, `float16`) and CUDA support in `LoraTrainer`, `TrainerInitConfig`, `CreateSessionRequest`, and `ServiceClient.create_training_client` (default `bfloat16` on supported CUDA GPUs, `float32` on CPU/MPS; explicit error for `float16` on CPU). Base weights use the selected dtype while LoRA adapters, `AdamW` optimizer state, and loss computation remain `float32`.
- `SampleParams.seed` for deterministic per-call sampling via `torch.Generator`.
- Server hardening: raw request body size limit (`max_request_body_bytes`, `413` before JSON parsing), `127.0.0.1` default host, no CORS, and strict `DADO_API_KEYS` requirement.
- Packaging metadata (`LICENSE`, `CHANGELOG.md`, `client/py.typed`), `scripts/gpu_check.py`, `scripts/benchmark.py`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `docs/api.md`, `docs/retries.md`, and `examples/colab_quickstart.ipynb`.

### Earlier Phases
- **Phase 3**: Lightweight `dado.client` SDK (`pip install dado` without `torch`/`transformers`/`peft`), pure-Pydantic `dado.wire`, idempotent `optim_step` retries, `AmbiguousSubmissionError` on non-idempotent post-send failures, `examples/sft.py`, and `examples/grpo_toy.py`.
- **Phase 2**: Async FastAPI server under `/v1` with per-session FIFO worker threads, bounded `FutureStore` with `?wait=` long-polling, Bearer `hmac.compare_digest` auth, idle TTL eviction, and symlink-safe checkpointing.
- **Phase 1**: `LoraTrainer` core (`forward_backward`, idempotent `optim_step`, `sample`, `save_state`/`load_state`).
