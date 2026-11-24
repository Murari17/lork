"""CUDA/bfloat16 GPU verification script for loraforge (run on Colab/Modal with a GPU)."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from loraforge.trainer import LoraTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
TARGET_MODULES = ["q_proj", "v_proj"]


def _make_batch(
    tokenizer_name: str, batch_size: int = 2, seq_len: int = 16
) -> dict[str, torch.Tensor]:
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    encoded = tok(
        "The quick brown fox jumps over the lazy dog near the quiet river bank.",
        return_tensors="pt",
    )["input_ids"]
    while encoded.shape[1] < seq_len + 1:
        encoded = torch.cat([encoded, encoded], dim=1)
    tokens = encoded[:, : seq_len + 1]
    input_ids = tokens[:, :-1].repeat(batch_size, 1)
    target_ids = tokens[:, 1:].repeat(batch_size, 1)
    weights = torch.ones_like(input_ids, dtype=torch.float32)
    return {"input_ids": input_ids, "target_ids": target_ids, "weights": weights}


def main() -> None:
    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is not available. Run scripts/gpu_check.py on a GPU machine.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}, bf16_supported={torch.cuda.is_bf16_supported()}")
    batch = _make_batch(MODEL_NAME, batch_size=2, seq_len=16)
    tokens_per_step = int(batch["input_ids"].numel())

    # 1. 50-step CUDA bfloat16 SFT loss drop + throughput + peak VRAM.
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    bf16_trainer = LoraTrainer(
        base_model=MODEL_NAME,
        rank=8,
        alpha=16.0,
        target_modules=TARGET_MODULES,
        device="cuda",
        dtype="bfloat16",
    )
    assert bf16_trainer.dtype_name == "bfloat16"
    assert {p.dtype for p in bf16_trainer.model.parameters() if p.requires_grad} == {torch.float32}

    bf16_losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(50):
        out = bf16_trainer.forward_backward(batch, loss_fn="cross_entropy")
        bf16_losses.append(float(out["loss"]))
        bf16_trainer.optim_step(f"gpu-bf16-{step}", {"lr": 5e-4})
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    tokens_per_sec = (50 * tokens_per_step) / elapsed
    print(
        f"50-step bf16 SFT: initial_loss={bf16_losses[0]:.4f} final_loss={bf16_losses[-1]:.4f} "
        f"peak_vram_mb={peak_mb:.1f} tokens_per_sec={tokens_per_sec:.1f}"
    )
    assert bf16_losses[-1] <= 0.5 * bf16_losses[0], (
        f"Expected >=50% bf16 loss drop, got {bf16_losses[0]:.4f} -> {bf16_losses[-1]:.4f}"
    )

    # 2. 10-step fp32 vs bf16 loss curve deviation.
    torch.manual_seed(77)
    fp32_t = LoraTrainer(MODEL_NAME, 4, 8.0, TARGET_MODULES, device="cuda", dtype="float32")
    torch.manual_seed(77)
    bf16_t = LoraTrainer(MODEL_NAME, 4, 8.0, TARGET_MODULES, device="cuda", dtype="bfloat16")
    fp32_curve: list[float] = []
    bf16_curve: list[float] = []
    for step in range(10):
        fp32_curve.append(float(fp32_t.forward_backward(batch)["loss"]))
        fp32_t.optim_step(f"cmp-{step}", {"lr": 5e-4})
        bf16_curve.append(float(bf16_t.forward_backward(batch)["loss"]))
        bf16_t.optim_step(f"cmp-{step}", {"lr": 5e-4})
    max_curve_dev = max(abs(a - b) for a, b in zip(fp32_curve, bf16_curve, strict=True))
    print(f"10-step fp32 vs bf16 max loss deviation: {max_curve_dev:.6f}")

    # 3. Save/load roundtrip on CUDA.
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt = Path(tmp_dir) / "cuda_ckpt"
        bf16_trainer.save_state(ckpt)
        uninterrupted_next = float(bf16_trainer.forward_backward(batch)["loss"])

        resumed = LoraTrainer(MODEL_NAME, 8, 16.0, TARGET_MODULES, device="cuda", dtype="bfloat16")
        resumed.load_state(ckpt)
        resumed_next = float(resumed.forward_backward(batch)["loss"])
        assert resumed_next == uninterrupted_next, (
            f"CUDA save/load mismatch: {resumed_next} != {uninterrupted_next}"
        )
        print(f"CUDA save/load roundtrip verified: next_loss={resumed_next:.6f}")

    # 4. Sample check on CUDA.
    samp = bf16_trainer.sample(
        ["Hello, CUDA world:"], {"max_tokens": 8, "temperature": 1.0, "seed": 42}
    )
    assert len(samp["completions"]) == 1 and len(samp["token_ids"][0]) >= 1
    print(f"CUDA sample() ok: {samp['completions'][0]!r}")
    print("GPU_CHECK_PASSED")


if __name__ == "__main__":
    main()
