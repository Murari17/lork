# loraforge

Open-source LoRA fine-tuning service with low-level training primitives. You own the training loop, dataset, and loss; the server owns execution only.

## Install

Lightweight Python SDK only (`httpx` + `pydantic`, no `torch` / `transformers` / `peft`):

```bash
pip install loraforge
```

Server + trainer runtime (`torch`, `transformers`, `peft`, `fastapi`, `uvicorn`):

```bash
pip install "loraforge[server]"
```

## Start the Server

> **Memory note**: In the current phase, each active session holds an independent `fp32` `LoraTrainer` instance (`~2 GB` RAM per `Qwen/Qwen2.5-0.5B` session). `LORAFORGE_MAX_SESSIONS` defaults to `2`.

```bash
export LORAFORGE_API_KEYS="dev-secret-key"
export LORAFORGE_DEVICE="cpu"   # or cuda / mps
uvicorn loraforge.server.app:create_app --factory --host 127.0.0.1 --port 8000
```

## 5-Line Client Example

```python
from loraforge.client import OptimParams, ServiceClient

with ServiceClient("http://127.0.0.1:8000", "dev-secret-key") as client, \
     client.create_training_client("Qwen/Qwen2.5-0.5B", 8, 16.0, ["q_proj", "v_proj"]) as tc:
    loss = tc.forward_backward({"input_ids": [[1, 2, 3]], "target_ids": [[2, 3, 4]], "weights": [[1.0, 1.0, 1.0]]}).result().loss
    step = tc.optim_step(OptimParams(lr=5e-4)).result().step
    print(f"step={step} loss={loss:.4f}")
```

## Run the Examples

Against a running server (`LORAFORGE_BASE_URL` and `LORAFORGE_API_KEY`):

```bash
LORAFORGE_BASE_URL=http://127.0.0.1:8000 LORAFORGE_API_KEY=dev-secret-key python examples/sft.py
LORAFORGE_BASE_URL=http://127.0.0.1:8000 LORAFORGE_API_KEY=dev-secret-key python examples/grpo_toy.py
```
