"""Benchmark dado SDK-over-HTTP vs plain PEFT+AdamW SFT loop."""

from __future__ import annotations

import json
import math
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

from dado.client import OptimParams, ServiceClient

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


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return float(ordered[idx])


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


def _run_plain_peft(batch: dict[str, Any]) -> dict[str, Any]:
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
    fb_lat_ms: list[float] = []
    optim_lat_ms: list[float] = []
    t0 = time.perf_counter()
    for _ in range(NUM_STEPS):
        t_fb = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids).logits.float()
        log_probs = F.log_softmax(logits, dim=-1)
        tok_lp = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        loss = (-tok_lp * weights).sum() / weights.sum()
        loss.backward()
        fb_lat_ms.append((time.perf_counter() - t_fb) * 1000.0)

        t_opt = time.perf_counter()
        opt.step()
        optim_lat_ms.append((time.perf_counter() - t_opt) * 1000.0)
        losses.append(float(loss.item()))
    elapsed = time.perf_counter() - t0
    return {
        "losses": losses,
        "elapsed": elapsed,
        "fb_p50_ms": round(_percentile(fb_lat_ms, 50.0), 2),
        "fb_p95_ms": round(_percentile(fb_lat_ms, 95.0), 2),
        "optim_p50_ms": round(_percentile(optim_lat_ms, 50.0), 2),
        "optim_p95_ms": round(_percentile(optim_lat_ms, 95.0), 2),
    }


def _run_http_sdk(batch: dict[str, Any], *, return_logprobs: bool) -> dict[str, Any]:
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory() as tmp_dir:
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["DADO_API_KEYS"] = API_KEY
        env["DADO_CHECKPOINT_ROOT"] = tmp_dir
        env["DADO_DEVICE"] = "cpu"
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import torch, uvicorn; torch.manual_seed(42); "
                    f"uvicorn.run('dado.server.app:create_app', factory=True, "
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

            payload_bytes_by_fid: dict[str, int] = {}

            def _record_response_size(resp: httpx.Response) -> None:
                if resp.request.method == "GET" and "/v1/futures/" in resp.request.url.path:
                    resp.read()
                    fid = resp.request.url.path.rsplit("/", 1)[-1]
                    payload_bytes_by_fid[fid] = len(resp.content)

            with ServiceClient(base_url=base_url, api_key=API_KEY) as client:
                client._http.event_hooks["response"].append(_record_response_size)
                with client.create_training_client(
                    base_model=MODEL_NAME,
                    rank=8,
                    alpha=16.0,
                    target_modules=TARGET_MODULES,
                    device="cpu",
                ) as tc:
                    losses: list[float] = []
                    fb_lat_ms: list[float] = []
                    optim_lat_ms: list[float] = []
                    fb_sizes: list[int] = []
                    optim_sizes: list[int] = []
                    optim = OptimParams(
                        lr=5e-4, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8
                    )
                    t0 = time.perf_counter()
                    for _ in range(NUM_STEPS):
                        t_fb = time.perf_counter()
                        fb_fut = tc.forward_backward(
                            batch,
                            loss_fn="cross_entropy",
                            return_logprobs=return_logprobs,
                        )
                        fb = fb_fut.result()
                        fb_lat_ms.append((time.perf_counter() - t_fb) * 1000.0)
                        fb_sizes.append(payload_bytes_by_fid[fb_fut.future_id])

                        t_opt = time.perf_counter()
                        opt_fut = tc.optim_step(optim)
                        opt_fut.result()
                        optim_lat_ms.append((time.perf_counter() - t_opt) * 1000.0)
                        optim_sizes.append(payload_bytes_by_fid[opt_fut.future_id])

                        losses.append(float(fb.loss))
                    elapsed = time.perf_counter() - t0
                    return {
                        "losses": losses,
                        "elapsed": elapsed,
                        "fb_p50_ms": round(_percentile(fb_lat_ms, 50.0), 2),
                        "fb_p95_ms": round(_percentile(fb_lat_ms, 95.0), 2),
                        "optim_p50_ms": round(_percentile(optim_lat_ms, 50.0), 2),
                        "optim_p95_ms": round(_percentile(optim_lat_ms, 95.0), 2),
                        "fb_response_bytes": round(sum(fb_sizes) / len(fb_sizes), 1),
                        "optim_response_bytes": round(sum(optim_sizes) / len(optim_sizes), 1),
                    }
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

    direct = _run_plain_peft(batch)
    http_lp = _run_http_sdk(batch, return_logprobs=True)
    http_no_lp = _run_http_sdk(batch, return_logprobs=False)

    direct_tps = total_tokens / direct["elapsed"]
    http_lp_tps = total_tokens / http_lp["elapsed"]
    http_no_lp_tps = total_tokens / http_no_lp["elapsed"]

    max_dev_lp = max(
        abs(a - b) for a, b in zip(direct["losses"], http_lp["losses"], strict=True)
    )
    max_dev_no_lp = max(
        abs(a - b) for a, b in zip(direct["losses"], http_no_lp["losses"], strict=True)
    )

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
        "direct_elapsed_sec": round(direct["elapsed"], 4),
        "direct_tokens_per_sec": round(direct_tps, 2),
        "direct_fb_p50_ms": direct["fb_p50_ms"],
        "direct_fb_p95_ms": direct["fb_p95_ms"],
        "direct_optim_p50_ms": direct["optim_p50_ms"],
        "direct_optim_p95_ms": direct["optim_p95_ms"],
        "http_with_logprobs": {
            "elapsed_sec": round(http_lp["elapsed"], 4),
            "tokens_per_sec": round(http_lp_tps, 2),
            "http_overhead_pct": round(((direct_tps - http_lp_tps) / direct_tps) * 100.0, 2),
            "max_per_step_loss_deviation": max_dev_lp,
            "fb_p50_ms": http_lp["fb_p50_ms"],
            "fb_p95_ms": http_lp["fb_p95_ms"],
            "optim_p50_ms": http_lp["optim_p50_ms"],
            "optim_p95_ms": http_lp["optim_p95_ms"],
            "fb_response_bytes": http_lp["fb_response_bytes"],
            "optim_response_bytes": http_lp["optim_response_bytes"],
        },
        "http_without_logprobs": {
            "elapsed_sec": round(http_no_lp["elapsed"], 4),
            "tokens_per_sec": round(http_no_lp_tps, 2),
            "http_overhead_pct": round(((direct_tps - http_no_lp_tps) / direct_tps) * 100.0, 2),
            "max_per_step_loss_deviation": max_dev_no_lp,
            "fb_p50_ms": http_no_lp["fb_p50_ms"],
            "fb_p95_ms": http_no_lp["fb_p95_ms"],
            "optim_p50_ms": http_no_lp["optim_p50_ms"],
            "optim_p95_ms": http_no_lp["optim_p95_ms"],
            "fb_response_bytes": http_no_lp["fb_response_bytes"],
            "optim_response_bytes": http_no_lp["optim_response_bytes"],
        },
        "direct_losses": direct["losses"],
        "http_losses": http_lp["losses"],
        "http_no_lp_losses": http_no_lp["losses"],
    }
    (docs_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    steps = list(range(1, NUM_STEPS + 1))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, direct["losses"], label="Plain PEFT + AdamW", linewidth=2.2, marker="o")
    ax.plot(
        steps,
        http_lp["losses"],
        label="Dado SDK (return_logprobs=True)",
        linewidth=1.8,
        linestyle="--",
        marker="x",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"Dado HTTP vs Plain PEFT+AdamW ({MODEL_NAME}, CPU fp32)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(docs_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    summary = {k: v for k, v in report.items() if not k.endswith("_losses")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
