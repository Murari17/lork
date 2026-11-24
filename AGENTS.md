# Lork: agent rules

PROJECT: Open-source LoRA fine-tuning service. Low-level primitives; the
USER owns the training loop, data, and loss. Server owns execution only.

API (all async/future-based):
- create_session(base_model, rank, alpha, target_modules) -> session_id
- forward_backward(session_id, batch{input_ids, target_ids, weights}, loss_fn)
    -> {loss, logprobs}. Accumulates grads. Does NOT update weights.
- optim_step(session_id, step_id, {lr, betas, weight_decay, eps}) -> {step}
    Applies + zeros grads. MUST be idempotent on step_id (replay returns
    the same result, never double-applies).
- sample(session_id, prompts, {max_tokens, temperature, top_p, stop, seed}) -> completions + logprobs + stop_reasons
- save_state / load_state(session_id, name): LoRA weights + optimizer state
- export_adapter(session_id, name): standalone PEFT adapter files (adapter_config.json + adapter_model.safetensors)
- delete_session(session_id)

STACK: Python 3.11+, PyTorch, transformers, peft, FastAPI, Pydantic v2,
pytest, ruff, mypy --strict. Specify bounded compatible version ranges in pyproject.toml.

HARD RULES
1. Work in PHASES. Stop at the end of each phase and wait for approval.
2. Before coding a phase: write a plan in an Implementation Plan artifact
   listing files, interfaces, and the tests that define "done".
3. Write tests FIRST for each function, then implement.
4. A phase is done only when you have RUN, and shown the output of:
   `ruff check . && mypy --strict src && pytest -x -q`. All green.
   Never claim something works without running it.
5. No placeholder code, no TODO stubs, no mocked core logic. If you can't
   implement something, say so and stop.
6. Never invent library APIs. If unsure of a signature, read the installed
   package source or docs before using it.
7. Validate every external input with Pydantic. Explicit errors, no bare
   except, no silent fallbacks.
8. No features outside the current phase. No refactors outside it either.
9. Keep ARCHITECTURE.md updated at the end of every phase.
10. Dev model: Qwen/Qwen2.5-0.5B. Must run on CPU/MPS (tests) and CUDA.
11. Never weaken, loosen, delete, or restructure a test to make it pass
    without stating it explicitly BEFORE the edit, with the diff and the
    reason. Prefer fixing code. If the test is wrong, say why.
