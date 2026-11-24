"""Minimal ~30-line SFT example using the dado Python SDK."""

from __future__ import annotations

import os

from transformers import AutoTokenizer

from dado.client import OptimParams, ServiceClient

MODEL = "Qwen/Qwen2.5-0.5B"
BASE_URL = os.environ.get("DADO_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("DADO_API_KEY", "dev-secret-key")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("dado lets you own your LoRA training loop cleanly.", return_tensors="pt")[
        "input_ids"
    ][0].tolist()
    batch = {
        "input_ids": [ids[:-1], ids[:-1]],
        "target_ids": [ids[1:], ids[1:]],
        "weights": [[1.0] * (len(ids) - 1)] * 2,
    }
    with (
        ServiceClient(BASE_URL, API_KEY) as client,
        client.create_training_client(MODEL, 8, 16.0, ["q_proj", "v_proj"]) as tc,
    ):
        for step in range(15):
            loss = tc.forward_backward(batch).result().loss
            tc.optim_step(OptimParams(lr=5e-4)).result()
            print(f"step={step + 1:02d} loss={loss:.4f}")


if __name__ == "__main__":
    main()
