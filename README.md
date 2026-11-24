# Lork

Open-source, self-hosted LoRA fine-tuning execution service built around low-level primitives. **You own the training loop, dataset, and loss function; the server owns execution only.**

## Why Lork?

Most fine-tuning frameworks wrap data loading, trainer loops, and loss definitions inside opinionated abstractions that break as soon as you need custom RL/GRPO objectives, non-uniform per-token weighting, or custom gradient accumulation schedules. `lork` exposes only the core execution primitives over an async future-based HTTP API and a lightweight Python SDK (`httpx` + `pydantic`, no `torch` required on the client).

## 60-Second Quickstart

### 1. Install

Lightweight Python SDK only (no `torch`, `transformers`, or `peft`):
```bash
pip install lork
```

Server + trainer runtime (`torch`, `transformers`, `peft`, `fastapi`, `uvicorn`):
```bash
pip install "lork[server]"
```

### 2. Start the Server

```bash
export LORK_API_KEYS="dev-secret-key"
export LORK_DEVICE="cpu"   # or cuda / mps
lork serve --host 127.0.0.1 --port 8000
```

### 3. Run Training via the SDK (5 Lines)

```python
from lork.client import OptimParams, ServiceClient

with ServiceClient("http://127.0.0.1:8000", "dev-secret-key") as client, \
     client.create_training_client("Qwen/Qwen2.5-0.5B", 8, 16.0, ["q_proj", "v_proj"]) as tc:
    loss = tc.forward_backward({"input_ids": [[1, 2, 3]], "target_ids": [[2, 3, 4]], "weights": [[1.0, 1.0, 1.0]]}).result().loss
    step = tc.optim_step(OptimParams(lr=5e-4)).result().step
    print(f"step={step} loss={loss:.4f}")
```

## Primitives

| Primitive | Description |
| :--- | :--- |
| `create_training_client(base_model, rank, alpha, target_modules, device, dtype)` | Initializes a dedicated session worker thread with frozen base weights and `fp32` LoRA adapters. |
| `forward_backward(batch, loss_fn, return_logprobs=True)` | Accumulates weighted gradients across micro-batches without stepping weights; supports `"cross_entropy"` or precomputed per-token advantages, and can omit `logprobs` (`return_logprobs=False`) to shrink wire payloads. |
| `optim_step(OptimParams(...))` | Normalizes accumulated gradients by window weight, steps `AdamW`, zeros gradients. Idempotent on SDK-generated `step_id`. |
| `sample(prompts, SampleParams(...))` | Autoregressive generation returning completions, token IDs, per-token log-probabilities, and `stop_reasons` (`"length"`, `"eos"`, `"stop"`); supports `temperature`, `top_p`, `stop` sequences, and per-call `seed`. |
| `save_state(name)` / `load_state(name)` | Persists and restores LoRA weights (`safetensors`) and full `AdamW` optimizer state. |
| `export_adapter(name)` | Exports standalone PEFT adapter files (`adapter_config.json` + `adapter_model.safetensors`) ready for `PeftModel.from_pretrained` or vLLM. |
| `close()` / context manager | Deletes the remote session, frees trainer memory, and empties device caches. |

## Examples

Run against a running server (`LORK_BASE_URL` and `LORK_API_KEY`):

```bash
# Supervised fine-tuning (~30 lines, Qwen/Qwen2.5-0.5B)
LORK_BASE_URL=http://127.0.0.1:8000 LORK_API_KEY=dev-secret-key python examples/sft.py

# Toy GRPO RL task (sample -> reward -> advantages -> forward_backward -> optim_step)
LORK_BASE_URL=http://127.0.0.1:8000 LORK_API_KEY=dev-secret-key python examples/grpo_toy.py 42
```

### Measured 5-Seed Toy GRPO Results (`examples/grpo_toy.py`, `Qwen/Qwen2.5-0.5B`, CPU `fp32`, 12 steps, `GROUP_SIZE=8`)

> **Toy task & saturation note**: `examples/grpo_toy.py` is a synthetic toy RL task (`"Reply with only one word (apple or banana):"` with binary reward for `"banana"`). `initial_mean_reward` equals `step_rewards[0]` (the pre-update step-1 group). Once the policy reaches `mean_reward == 1.000` (all 8 samples in the group receive identical reward `1.0`), per-group advantages are identically zero; those **saturated** steps are skipped (`skipped_steps`) rather than running a zero-gradient update.

| Base Seed | Initial Mean Reward (Step 1) | Final Mean Reward (Post-Step 12) | Skipped Steps (Saturation) | Per-Step Mean Rewards (`step 1..12`) |
| :--- | :--- | :--- | :--- | :--- |
| `42` | `0.125` | `1.000` | `9` | `0.125, 0.500, 0.875, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000` |
| `43` | `0.125` | `1.000` | `9` | `0.125, 0.750, 0.750, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000` |
| `44` | `0.125` | `1.000` | `9` | `0.125, 0.875, 0.750, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000` |
| `45` | `0.125` | `1.000` | `8` | `0.125, 0.500, 0.750, 1.000, 1.000, 1.000, 0.875, 1.000, 1.000, 1.000, 1.000, 1.000` |
| `46` | `0.125` | `1.000` | `9` | `0.125, 0.750, 0.750, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000` |

## Benchmark (SDK over HTTP vs. Plain `PEFT + AdamW`)

Measured via [scripts/benchmark.py](scripts/benchmark.py) on CPU `float32` (`Qwen/Qwen2.5-0.5B`, 20 steps, `batch_size=2`, `seq_len=12`, `480` total tokens — full breakdown in [docs/benchmark.md](docs/benchmark.md), raw data in [docs/benchmark.json](docs/benchmark.json), plot in [docs/loss_curves.png](docs/loss_curves.png)):
- **Plain `PEFT + AdamW`**: `4.6406 s` (`103.43 tok/s`), `fb` p50/p95 `221.41 / 246.21 ms`, `optim` p50/p95 `3.68 / 4.25 ms`
- **`lork` SDK (`return_logprobs=True`)**: `4.7333 s` (`101.41 tok/s`, `+1.96%` overhead), `fb` p50/p95 `224.29 / 240.98 ms` (`648.0 B`), `optim` p50/p95 `7.21 / 7.91 ms` (`101.5 B`), max loss deviation `8.39e-05`
- **`lork` SDK (`return_logprobs=False`)**: `4.7447 s` (`101.17 tok/s`, `+2.19%` overhead), `fb` p50/p95 `224.36 / 238.23 ms` (`118.0 B`), `optim` p50/p95 `7.27 / 8.31 ms` (`101.5 B`), max loss deviation `8.39e-05`

## Docker (CUDA Runtime)

```bash
export LORK_API_KEYS="your-secret-key"
docker compose up --build
```

> Note: `Dockerfile` targets `nvidia/cuda:12.4.1-runtime-ubuntu22.04` as a non-root user (`lork`) with `/cache/huggingface` and `/data/checkpoints` volumes.

## Honest Limits

- **Self-hosted / single-tenant**: Static Bearer API keys (`LORK_API_KEYS`) with no multi-tenant isolation or user billing.
- **No shared base model**: Each active session loads its own base model instance (`~2 GB` RAM per `fp32` `Qwen/Qwen2.5-0.5B` session; `LORK_MAX_SESSIONS` defaults to `2`).
- **Single-node execution**: One worker thread per session on a single server process (no distributed tensor/data parallelism).

## Docs

- [HTTP API Reference](docs/api.md)
- [Client Retry & Failure Semantics](docs/retries.md)
- [Benchmark Breakdown](docs/benchmark.md)
- [Architecture](ARCHITECTURE.md)
- [GPU Check Script](scripts/gpu_check.py)
- [Colab Quickstart Notebook](examples/colab_quickstart.ipynb)
