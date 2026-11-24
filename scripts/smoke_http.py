"""End-to-end smoke test launching a real uvicorn server and running 20-step SFT over HTTP."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
API_KEY = "smoke-secret-key"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(client: httpx.Client, proc: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"uvicorn exited prematurely with code {proc.returncode}")
        try:
            resp = client.get("/v1/healthz")
            if resp.status_code == 200:
                return
        except httpx.RequestError:
            pass
        time.sleep(0.1)
    raise TimeoutError("Timed out waiting for uvicorn /v1/healthz")


def _wait_future(client: httpx.Client, future_id: str, headers: dict[str, str]) -> dict[str, Any]:
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        resp = client.get(f"/v1/futures/{future_id}", params={"wait": 30.0}, headers=headers)
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        if payload["status"] == "done":
            result: dict[str, Any] = payload["result"]
            return result
        if payload["status"] == "error":
            raise RuntimeError(f"Future {future_id} failed: {payload['error']}")
    raise TimeoutError(f"Timed out waiting for future {future_id}")


def main() -> None:
    # Avoid tqdm creating a multiprocessing SemLock in the client process.
    tqdm.set_lock(threading.RLock())

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded = tokenizer(
        "The quick brown fox jumps over the lazy dog near the river bank.",
        return_tensors="pt",
    )["input_ids"]
    seq_len = 10
    while encoded.shape[1] < seq_len + 1:
        encoded = torch.cat([encoded, encoded], dim=1)
    tokens = encoded[:, : seq_len + 1]
    batch = {
        "input_ids": tokens[:, :-1].repeat(2, 1).tolist(),
        "target_ids": tokens[:, 1:].repeat(2, 1).tolist(),
        "weights": [[1.0] * seq_len for _ in range(2)],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        log_path = tmp_path / "uvicorn.log"
        env = os.environ.copy()
        env["LORK_API_KEYS"] = API_KEY
        env["LORK_CHECKPOINT_ROOT"] = str(tmp_path / "checkpoints")
        env["LORK_DEVICE"] = "cpu"

        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "lork.server.app:create_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--log-level",
                    "info",
                ],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                with httpx.Client(base_url=base_url, timeout=60.0) as client:
                    _wait_for_health(client, proc)

                    create_resp = client.post(
                        "/v1/sessions",
                        json={
                            "base_model": MODEL_NAME,
                            "rank": 8,
                            "alpha": 16.0,
                            "target_modules": ["q_proj", "v_proj"],
                            "device": "cpu",
                        },
                        headers=headers,
                    )
                    create_resp.raise_for_status()
                    init_res = _wait_future(client, create_resp.json()["future_id"], headers)
                    session_id = init_res["session_id"]
                    print(f"Session created: {session_id}")

                    losses: list[float] = []
                    for step in range(20):
                        fb_resp = client.post(
                            f"/v1/sessions/{session_id}/forward_backward",
                            json={"batch": batch, "loss_fn": "cross_entropy"},
                            headers=headers,
                        )
                        fb_resp.raise_for_status()
                        fb_res = _wait_future(client, fb_resp.json()["future_id"], headers)
                        loss = float(fb_res["loss"])
                        losses.append(loss)

                        opt_resp = client.post(
                            f"/v1/sessions/{session_id}/optim_step",
                            json={
                                "step_id": f"smoke-step-{step}",
                                "optim_params": {
                                    "lr": 5e-4,
                                    "betas": [0.9, 0.999],
                                    "weight_decay": 0.0,
                                    "eps": 1e-8,
                                },
                            },
                            headers=headers,
                        )
                        opt_resp.raise_for_status()
                        opt_res = _wait_future(client, opt_resp.json()["future_id"], headers)
                        print(f"step={opt_res['step']:02d} loss={loss:.6f}")

                    assert losses[-1] < losses[0], (
                        f"Expected loss to decrease: start={losses[0]:.6f}, end={losses[-1]:.6f}"
                    )
                    print(
                        f"Smoke test passed: initial_loss={losses[0]:.6f} "
                        f"final_loss={losses[-1]:.6f}"
                    )

                    del_resp = client.delete(f"/v1/sessions/{session_id}", headers=headers)
                    del_resp.raise_for_status()
            finally:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGINT)
                    try:
                        proc.wait(timeout=15.0)
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                        proc.wait(timeout=5.0)

        server_logs = log_path.read_text(encoding="utf-8")
        drain_lines = [
            line for line in server_logs.splitlines() if "server_shutdown_drained" in line
        ]
        assert drain_lines, f"Expected server_shutdown_drained in logs:\n{server_logs}"
        assert "leaked semaphore" not in server_logs, (
            f"Unexpected semaphore leak warning:\n{server_logs}"
        )
        print(f"Graceful shutdown verified (exit={proc.returncode}): {drain_lines[-1]}")


if __name__ == "__main__":
    main()
