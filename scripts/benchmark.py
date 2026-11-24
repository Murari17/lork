"""Benchmark loraforge SDK-over-HTTP vs plain PEFT+AdamW SFT loop."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import matplotlib
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from loraforge.client import OptimParams, ServiceClient

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
TARGET_MODULES = ["q_proj", "v_proj"]
API_KEY = "benchmark-secret-key"
NUM_STEPS = 20


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_batch(batch_size: int = 2, seq_len: int = 12) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded = tok(
        "The quick brown fox jumps over the lazy dog near the quiet river bank.",
        return_tensors="pt",
    )["input_ids"]
    while encoded.shape[1] < seq_len + 1:
        encoded = torch.cat([encoded, encoded], dim=1)
    tokens = encoded[:, : seq_len + 1]
    return {
        "input_ids": tokens[:, :-1].repeat(batch_size, 1).tolist(),
        "target_ids": tokens[:, 1:].repeat(batch_size, 1).tolist(),
        "weights": [[1.0] * seq_len for _ in range(batch_size)],
    }


def _run_plain_peft(batch: dict[str, Any]) -> tuple[list[float], float]:
    torch.manual_seed(42)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    for param in base.parameters():
        param.requires_grad_(False)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-4, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)

    input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
    target_ids = torch.tensor(batch["target_ids"], dtype=torch.long)
    weights = torch.tensor(batch["weights"], dtype=torch.float32)

    losses: list[float] = []
    t0 = time.perf_counter()
    for _ in range(NUM_STEPS):
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids).logits.float()
        log_probs = F.log_softmax(logits, dim=-1)
        tok_lp = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        loss = (-tok_lp * weights).sum() / weights.sum()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    elapsed = time.perf_counter() - t0
    return losses, elapsed


def _run_http_sdk(batch: dict[str, Any]) -> tuple[list[float], float]:
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory() as tmp_dir:
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["LORAFORGE_API_KEYS"] = API_KEY
        env["LORAFORGE_CHECKPOINT_ROOT"] = tmp_dir
        env["LORAFORGE_DEVICE"] = "cpu"
        # Seed server process identically at startup for deterministic comparison.
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import torch, uvicorn; torch.manual_seed(42); "
                    f"uvicorn.run('loraforge.server.app:create_app', factory=True, "
                    f"host='127.0.0.1', port={port}, log_level='warning')"
                ),
            ],
            env=env,
        )
        try:
            with httpx.Client(base_url=base_url, timeout=2.0) as probe:
                deadline = time.monotonic() + 20.0
                while time.monotonic() < deadline:
                    try:
                        if probe.get("/v1/healthz").status_code == 200:
                            break
                    except httpx.RequestError:
                        pass
                    time.sleep(0.1)

            with (
                ServiceClient(base_url=base_url, api_key=API_KEY) as client,
                client.create_training_client(
                    base_model=MODEL_NAME,
                    rank=8,
                    alpha=16.0,
                    target_modules=TARGET_MODULES,
                    device="cpu",
                ) as tc,
            ):
                losses: list[float] = []
                optim = OptimParams(lr=5e-4, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)
                t0 = time.perf_counter()
                for _ in range(NUM_STEPS):
                    fb = tc.forward_backward(batch, loss_fn="cross_entropy").result()
                    tc.optim_step(optim).result()
                    losses.append(float(fb.loss))
                elapsed = time.perf_counter() - t0
                return losses, elapsed
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait(timeout=5.0)


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    batch = _make_batch(batch_size=2, seq_len=12)
    total_tokens = NUM_STEPS * len(batch["input_ids"]) * len(batch["input_ids"][0])

    direct_losses, direct_elapsed = _run_plain_peft(batch)
    http_losses, http_elapsed = _run_http_sdk(batch)

    per_step_diffs = [abs(a - b) for a, b in zip(direct_losses, http_losses, strict=True)]
    max_loss_dev = max(per_step_diffs)
    direct_tps = total_tokens / direct_elapsed
    http_tps = total_tokens / http_elapsed
    http_overhead_pct = ((direct_tps - http_tps) / direct_tps) * 100.0

    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": MODEL_NAME,
        "device": "cpu",
        "dtype": "float32",
        "steps": NUM_STEPS,
        "batch_size": 2,
        "seq_len": 12,
        "total_tokens": total_tokens,
        "direct_elapsed_sec": round(direct_elapsed, 4),
        "http_elapsed_sec": round(http_elapsed, 4),
        "direct_tokens_per_sec": round(direct_tps, 2),
        "http_tokens_per_sec": round(http_tps, 2),
        "http_overhead_pct": round(http_overhead_pct, 2),
        "max_per_step_loss_deviation": max_loss_dev,
        "direct_losses": direct_losses,
        "http_losses": http_losses,
    }
    (docs_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    steps = list(range(1, NUM_STEPS + 1))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, direct_losses, label="Plain PEFT + AdamW", linewidth=2.2, marker="o")
    ax.plot(
        steps,
        http_losses,
        label="loraforge SDK over HTTP",
        linewidth=1.8,
        linestyle="--",
        marker="x",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"loraforge HTTP vs Plain PEFT+AdamW ({MODEL_NAME}, CPU fp32)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(docs_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    print(json.dumps({k: v for k, v in report.items() if "losses" not in k}, indent=2))


if __name__ == "__main__":
    main()
