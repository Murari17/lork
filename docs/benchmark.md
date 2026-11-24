# Benchmark: `dado` HTTP SDK vs. Plain `PEFT + AdamW`

Measured via [scripts/benchmark.py](../scripts/benchmark.py) on CPU (`float32`, `Qwen/Qwen2.5-0.5B`, `rank=8`, `alpha=16.0`, `target_modules=["q_proj", "v_proj"]`, 20 steps, `batch_size=2`, `seq_len=12`, `480` total tokens — raw data in [benchmark.json](benchmark.json), curve plot in [loss_curves.png](loss_curves.png)).

---

## Throughput, Latency, and Payload Comparison

| Configuration | Total Time | Throughput | HTTP Overhead | `forward_backward` p50 / p95 | `optim_step` p50 / p95 | `fb` / `optim` Response Bytes | Max Loss Deviation vs. Plain |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Plain `PEFT + AdamW` (in-process) | `4.6406 s` | `103.43 tok/s` | — | `221.41 ms` / `246.21 ms` | `3.68 ms` / `4.25 ms` | — | `0.0` (baseline) |
| `dado` SDK (`return_logprobs=True`) | `4.7333 s` | `101.41 tok/s` | `+1.96%` | `224.29 ms` / `240.98 ms` | `7.21 ms` / `7.91 ms` | `648.0 B` / `101.5 B` | `8.39e-05` |
| `dado` SDK (`return_logprobs=False`) | `4.7447 s` | `101.17 tok/s` | `+2.19%` | `224.36 ms` / `238.23 ms` | `7.27 ms` / `8.31 ms` | `118.0 B` / `101.5 B` | `8.39e-05` |

- On a `2 x 12` batch, omitting `logprobs` (`return_logprobs=False`) reduces the `forward_backward` future response payload by **5.5×** (`648 B` → `118 B`), which matters more at larger `(B, T)` shapes (e.g., `B=16, T=1024` JSON floats) over non-loopback links; on `2 x 12` loopback CPU, total HTTP overhead is `~2%` (`~3 ms` per `forward_backward` + `~3.5 ms` per `optim_step`) in both modes.

---

## Cause of the `8.39e-05` Max Per-Step Loss Deviation

In the 20-step benchmark, `max_per_step_loss_deviation` between the plain `PEFT + AdamW` loop and `dado` is `8.392333984375e-05` (peaking at step 8), whereas the gradient-accumulation test asserts `< 1e-5`. Two controlled experiments isolated the exact source:

1. **HTTP / JSON roundtrip contributes `0.0` deviation**: Running `LoraTrainer` directly in-process vs. running `dado` SDK over HTTP with the same seed produced **`0.0` max loss difference** across all 20 steps.
2. **Pre-scale vs. post-scale backward (`8.392333984375e-05`)**:
   - The plain `PEFT + AdamW` loop differentiates the mean loss directly: `(loss_sum / 24.0).backward()`, scaling token gradients by `1 / 24` *before* backpropagating through the transformer layers.
   - `LoraTrainer` differentiates the unnormalized sum `loss_sum.backward()` during `forward_backward` and scales the accumulated gradients by `1.0 / total_window_weight` once at `optim_step`.
   - Switching the plain `PEFT + AdamW` loop to post-scale `loss_sum.backward()` + `grad.mul_(1.0 / 24.0)` produced the exact **`8.392333984375e-05`** difference against pre-scale and **`0.0`** against `LoraTrainer`.

Because unnormalized accumulation followed by window-weight division at `optim_step` is required for exact gradient accumulation across arbitrary micro-batch sizes and token masks (`2 x N == 1 x 2N`), `LoraTrainer` preserves post-scale normalization by design. Under cold-start `AdamW(eps=1e-8)` in `float32`, the resulting `1-ULP` gradient rounding (`~1e-7`) produces up to `8.39e-05` trajectory divergence over 20 steps relative to pre-scaled `PEFT + AdamW`.
