# Changelog

All notable changes to `lork` are documented in this file.

## [0.2.0] - Phase 5 (Sampling Controls, Adapter Export, CLI & Packaging)

### Added
- `SampleParams.top_p` (nucleus filtering) and `SampleParams.stop` (up to 8 non-empty stop strings), plus `SampleResult.stop_reasons` (`"length"`, `"eos"`, `"stop"`).
- Standalone PEFT LoRA adapter export (`LoraTrainer.export_adapter`, `POST /v1/sessions/{session_id}/export_adapter`, `TrainingClient.export_adapter`) with symlink-safe `exports/` path containment.
- Zero-ML `lork` CLI (`src/lork/cli.py`, `[project.scripts]`): `lork --version` and `lork serve`.
- Bounded compatible dependency ranges in `pyproject.toml`, `.github/workflows/publish.yml` for tag-based PyPI Trusted Publishing, and upgraded `examples/colab_quickstart.ipynb` sentiment JSON mini-project.

## [0.1.0] - Phase 4 (Launch Hardening)

### Added
- `dtype` (`float32`, `bfloat16`, `float16`) and CUDA support in `LoraTrainer`, `TrainerInitConfig`, `CreateSessionRequest`, and `ServiceClient.create_training_client` (default `bfloat16` on supported CUDA GPUs, `float32` on CPU/MPS; explicit error for `float16` on CPU). Base weights use the selected dtype while LoRA adapters, `AdamW` optimizer state, and loss computation remain `float32`.
- `SampleParams.seed` for deterministic per-call sampling via `torch.Generator`.
- Server hardening: raw request body size limit (`max_request_body_bytes`, `413` before JSON parsing), `127.0.0.1` default host, no CORS, and strict `LORK_API_KEYS` requirement.
- Packaging metadata (`LICENSE`, `CHANGELOG.md`, `client/py.typed`), `scripts/gpu_check.py`, `scripts/benchmark.py`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `docs/api.md`, `docs/retries.md`, and `examples/colab_quickstart.ipynb`.

### Earlier Phases
- **Phase 3**: Lightweight `lork.client` SDK (`pip install lork` without `torch`/`transformers`/`peft`), pure-Pydantic `lork.wire`, idempotent `optim_step` retries, `AmbiguousSubmissionError` on non-idempotent post-send failures, `examples/sft.py`, and `examples/grpo_toy.py`.
- **Phase 2**: Async FastAPI server under `/v1` with per-session FIFO worker threads, bounded `FutureStore` with `?wait=` long-polling, Bearer `hmac.compare_digest` auth, idle TTL eviction, and symlink-safe checkpointing.
- **Phase 1**: `LoraTrainer` core (`forward_backward`, idempotent `optim_step`, `sample`, `save_state`/`load_state`).
