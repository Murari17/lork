# HTTP API Reference (`/v1`)

All endpoints except `GET /v1/healthz` require `Authorization: Bearer <api_key>` matching `LORAFORGE_API_KEYS`. Errors return `{"error": {"code": "<code >", "message": "<message>"}}`.

---

## `GET /v1/healthz`

Returns `200 OK` immediately without touching session workers.

```bash
curl -s http://127.0.0.1:8000/v1/healthz
```

Response:
```json
{"status": "ok"}
```

---

## `POST /v1/sessions`

Asynchronously initializes a `LoraTrainer` session on a dedicated worker thread. Returns `202 Accepted`.

```bash
curl -s -X POST http://127.0.0.1:8000/v1/sessions \
  -H "Authorization: Bearer dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "Qwen/Qwen2.5-0.5B",
    "rank": 8,
    "alpha": 16.0,
    "target_modules": ["q_proj", "v_proj"],
    "device": "cpu",
    "dtype": "float32"
  }'
```

Response (`202`):
```json
{"future_id": "fut_...", "session_id": "sess_..."}
```

---

## `DELETE /v1/sessions/{session_id}`

Closes the session worker, frees `LoraTrainer` memory, and empties device caches.

```bash
curl -s -X DELETE http://127.0.0.1:8000/v1/sessions/sess_... \
  -H "Authorization: Bearer dev-secret-key"
```

Response (`200`):
```json
{"deleted": "sess_..."}
```

---

## `POST /v1/sessions/{session_id}/forward_backward`

Runs forward and backward pass on a `(B, T)` batch, accumulating gradients without stepping weights. Rejects batches exceeding `LORAFORGE_MAX_BATCH_SIZE` / `LORAFORGE_MAX_SEQ_LEN` with `422`.

```bash
curl -s -X POST http://127.0.0.1:8000/v1/sessions/sess_.../forward_backward \
  -H "Authorization: Bearer dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "batch": {
      "input_ids": [[1, 2, 3, 4]],
      "target_ids": [[2, 3, 4, 5]],
      "weights": [[1.0, 1.0, 1.0, 1.0]]
    },
    "loss_fn": "cross_entropy"
  }'
```

Future result:
```json
{"loss": 6.1234, "logprobs": [[-6.1, -6.2, -5.9, -6.3]]}
```

---

## `POST /v1/sessions/{session_id}/optim_step`

Normalizes accumulated gradients by total window weight, runs `AdamW.step()`, and zeros gradients. **Idempotent on `step_id`**: replays return the stored `{"step": n}` without double-applying, even after later failed operations.

```bash
curl -s -X POST http://127.0.0.1:8000/v1/sessions/sess_.../optim_step \
  -H "Authorization: Bearer dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "step_id": "step-0001",
    "optim_params": {"lr": 5e-4, "betas": [0.9, 0.999], "weight_decay": 0.0, "eps": 1e-8}
  }'
```

Future result:
```json
{"step": 1}
```

---

## `POST /v1/sessions/{session_id}/sample`

Autoregressively generates completions and per-token log-probabilities. Supports optional `seed` for deterministic per-call sampling.

```bash
curl -s -X POST http://127.0.0.1:8000/v1/sessions/sess_.../sample \
  -H "Authorization: Bearer dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["Hello, my name is"],
    "params": {"max_tokens": 8, "temperature": 1.0, "seed": 42}
  }'
```

Future result:
```json
{"completions": ["..."], "token_ids": [[...]], "logprobs": [[...]]}
```

---

## `POST /v1/sessions/{session_id}/save_state` & `load_state`

Saves or restores LoRA weights (`safetensors`) and `AdamW` optimizer state. Checkpoint `name` must match `^[A-Za-z0-9_-]{1,64}$` and stay within the per-session checkpoint directory (symlinks escaping the directory are rejected).

```bash
curl -s -X POST http://127.0.0.1:8000/v1/sessions/sess_.../save_state \
  -H "Authorization: Bearer dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "ckpt_01"}'
```

---

## `GET /v1/futures/{future_id}?wait=<seconds>`

Long-polls an async operation (`0 <= wait <= 30`), waking immediately when the worker finishes.

```bash
curl -s "http://127.0.0.1:8000/v1/futures/fut_...?wait=25" \
  -H "Authorization: Bearer dev-secret-key"
```

Response (`200`):
```json
{"future_id": "fut_...", "status": "done", "result": {"step": 1}, "error": null}
```
