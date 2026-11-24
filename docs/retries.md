# Client Retry & Failure Semantics

The Python SDK (`loraforge.client.ServiceClient`) distinguishes operations that are safe to retry from non-idempotent operations that must **never** be silently retried after bytes may have reached the server.

| Operation | Pre-send Errors (`ConnectError`, `ConnectTimeout`, `PoolTimeout`) | `429 Too Many Requests` | `503` with Structured `ErrorEnvelope` | `500` / `502` / `504` or Unstructured `503` | Post-send Transport Drop (`ReadTimeout`, `ReadError`, `WriteError`, `RemoteProtocolError`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `optim_step` (reuses identical `step_id`) | Retries (exp. backoff + jitter) | Retries (honors `Retry-After`) | Retries (honors `Retry-After`) | Retries (exp. backoff + jitter) | Retries (exp. backoff + jitter) |
| `GET /v1/futures/{id}`, `DELETE /v1/sessions/{id}` | Retries (exp. backoff + jitter) | Retries (honors `Retry-After`) | Retries (honors `Retry-After`) | Retries (exp. backoff + jitter) | Retries (exp. backoff + jitter) |
| `forward_backward`, `sample`, `save_state`, `load_state`, `create_session` | Retries (exp. backoff + jitter) | Retries (honors `Retry-After`) | Retries (honors `Retry-After`) | Raises `AmbiguousSubmissionError` (NO retry) | Raises `AmbiguousSubmissionError` (NO retry) |

## Typed Exceptions

| HTTP / Failure Condition | SDK Exception |
| :--- | :--- |
| `401 Unauthorized` | `AuthError` |
| `400`, `403`, `422`, or client-side shape error | `ValidationError` |
| `404 Not Found` (unknown or TTL-evicted session/future) | `SessionNotFoundError` |
| `429` or `503` after retries exhausted | `ServerBusyError` |
| Future resolved with `status="error"` | `RemoteOpError` (carries `code` and `remote_message`) |
| Post-send failure on non-idempotent `POST` | `AmbiguousSubmissionError` (carries `cause`, never retried) |
| `future.result(timeout=...)` elapsed | `LoraforgeTimeoutError` (does not cancel remote op) |
