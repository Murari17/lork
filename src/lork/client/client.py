"""Synchronous Python SDK for lork with safe retries and zero heavy ML imports."""

from __future__ import annotations

import logging
import math
import random
import time
import uuid
from typing import Any, Literal

import httpx
from pydantic import ValidationError as PydanticValidationError

from lork.client.exceptions import (
    AmbiguousSubmissionError,
    AuthError,
    LorkError,
    ServerBusyError,
    SessionNotFoundError,
    ValidationError,
)
from lork.client.futures import APIFuture
from lork.wire import (
    CreateSessionRequest,
    ErrorEnvelope,
    ExportAdapterResult,
    ForwardBackwardResult,
    ForwardBackwardWireRequest,
    FutureStatusResponse,
    FutureSubmittedResponse,
    OptimParams,
    OptimStepResult,
    OptimStepWireRequest,
    SampleParams,
    SampleResult,
    SampleWireRequest,
    SessionCreatedResult,
    StateResult,
    StateWireRequest,
    WireBatch,
)

logger = logging.getLogger("lork.client")

_PRE_SEND_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)


def validate_batch(batch: dict[str, Any] | WireBatch) -> WireBatch:
    """Validate nested-list batch shapes client-side before sending over HTTP."""
    if isinstance(batch, WireBatch):
        return batch
    try:
        return WireBatch.model_validate(batch)
    except PydanticValidationError as exc:
        raise ValidationError(str(exc)) from exc


def _parse_error_body(response: httpx.Response) -> tuple[str, str]:
    try:
        envelope = ErrorEnvelope.model_validate(response.json())
        return envelope.error.code, envelope.error.message
    except (ValueError, PydanticValidationError):
        return f"http_{response.status_code}", response.text or response.reason_phrase


def _has_structured_error_envelope(response: httpx.Response) -> bool:
    try:
        ErrorEnvelope.model_validate(response.json())
    except (ValueError, PydanticValidationError):
        return False
    return True


def _parse_retry_after(header_val: str | None) -> float | None:
    if header_val is None:
        return None
    try:
        parsed = float(header_val.strip())
    except ValueError:
        return None
    return max(0.0, parsed)


class TrainingClient:
    """Session-scoped training client wrapping one remote LoRA session."""

    def __init__(self, service_client: ServiceClient, session_id: str) -> None:
        self._client = service_client
        self.session_id = session_id
        self._closed = False

    def __enter__(self) -> TrainingClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Delete the remote session, freeing server resources."""
        if self._closed:
            return
        self._closed = True
        self._client.delete_session(self.session_id)

    def forward_backward(
        self,
        batch: dict[str, Any] | WireBatch,
        loss_fn: str | list[list[float]] = "cross_entropy",
        return_logprobs: bool = True,
    ) -> APIFuture[ForwardBackwardResult]:
        """Submit forward_backward asynchronously; never retries ambiguous failures."""
        validated_batch = validate_batch(batch)
        try:
            wire_req = ForwardBackwardWireRequest.model_validate(
                {
                    "batch": validated_batch,
                    "loss_fn": loss_fn,
                    "return_logprobs": return_logprobs,
                }
            )
        except PydanticValidationError as exc:
            raise ValidationError(str(exc)) from exc

        submitted = self._client._post_op(
            f"/v1/sessions/{self.session_id}/forward_backward",
            payload=wire_req.model_dump(),
            idempotent=False,
        )
        return APIFuture(
            client=self._client,
            future_id=submitted.future_id,
            parser=ForwardBackwardResult.model_validate,
        )

    def optim_step(
        self,
        optim_params: OptimParams | dict[str, Any],
    ) -> APIFuture[OptimStepResult]:
        """Submit optim_step with an SDK-generated UUID step_id (safe to retry)."""
        step_id = f"step_{uuid.uuid4().hex}"
        try:
            wire_req = OptimStepWireRequest.model_validate(
                {"step_id": step_id, "optim_params": optim_params}
            )
        except PydanticValidationError as exc:
            raise ValidationError(str(exc)) from exc

        submitted = self._client._post_op(
            f"/v1/sessions/{self.session_id}/optim_step",
            payload=wire_req.model_dump(),
            idempotent=True,
        )
        return APIFuture(
            client=self._client,
            future_id=submitted.future_id,
            parser=OptimStepResult.model_validate,
        )

    def sample(
        self,
        prompts: list[str],
        params: SampleParams | dict[str, Any],
    ) -> APIFuture[SampleResult]:
        """Submit autoregressive sampling asynchronously."""
        try:
            wire_req = SampleWireRequest.model_validate(
                {"prompts": prompts, "params": params}
            )
        except PydanticValidationError as exc:
            raise ValidationError(str(exc)) from exc

        submitted = self._client._post_op(
            f"/v1/sessions/{self.session_id}/sample",
            payload=wire_req.model_dump(),
            idempotent=False,
        )
        return APIFuture(
            client=self._client,
            future_id=submitted.future_id,
            parser=SampleResult.model_validate,
        )

    def save_state(self, name: str) -> APIFuture[StateResult]:
        """Submit save_state asynchronously."""
        try:
            wire_req = StateWireRequest.model_validate({"name": name})
        except PydanticValidationError as exc:
            raise ValidationError(str(exc)) from exc

        submitted = self._client._post_op(
            f"/v1/sessions/{self.session_id}/save_state",
            payload=wire_req.model_dump(),
            idempotent=False,
        )
        return APIFuture(
            client=self._client,
            future_id=submitted.future_id,
            parser=StateResult.model_validate,
        )

    def load_state(self, name: str) -> APIFuture[StateResult]:
        """Submit load_state asynchronously."""
        try:
            wire_req = StateWireRequest.model_validate({"name": name})
        except PydanticValidationError as exc:
            raise ValidationError(str(exc)) from exc

        submitted = self._client._post_op(
            f"/v1/sessions/{self.session_id}/load_state",
            payload=wire_req.model_dump(),
            idempotent=False,
        )
        return APIFuture(
            client=self._client,
            future_id=submitted.future_id,
            parser=StateResult.model_validate,
        )

    def export_adapter(self, name: str) -> APIFuture[ExportAdapterResult]:
        """Export standalone PEFT LoRA adapter files asynchronously."""
        try:
            wire_req = StateWireRequest.model_validate({"name": name})
        except PydanticValidationError as exc:
            raise ValidationError(str(exc)) from exc

        submitted = self._client._post_op(
            f"/v1/sessions/{self.session_id}/export_adapter",
            payload=wire_req.model_dump(),
            idempotent=False,
        )
        return APIFuture(
            client=self._client,
            future_id=submitted.future_id,
            parser=ExportAdapterResult.model_validate,
        )


class ServiceClient:
    """Top-level HTTP client for connecting to a lork server."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 0.1,
    ) -> None:
        if not base_url.strip():
            raise ValidationError("base_url must not be empty")
        if not api_key.strip():
            raise ValidationError("api_key must not be empty")
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValidationError(f"timeout must be finite and > 0, got {timeout}")
        if max_retries < 0:
            raise ValidationError("max_retries must be >= 0")
        if not math.isfinite(backoff_base) or backoff_base < 0.0:
            raise ValidationError(f"backoff_base must be finite and >= 0, got {backoff_base}")
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def __enter__(self) -> ServiceClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def create_training_client(
        self,
        base_model: str,
        rank: int,
        alpha: float,
        target_modules: list[str],
        device: str | None = None,
        dtype: Literal["float32", "bfloat16", "float16"] | None = None,
    ) -> TrainingClient:
        """Create a remote training session, await initialization, and return TrainingClient."""
        try:
            req = CreateSessionRequest(
                base_model=base_model,
                rank=rank,
                alpha=alpha,
                target_modules=target_modules,
                device=device,
                dtype=dtype,
            )
            req.to_trainer_config(default_device=device or "cpu")
        except PydanticValidationError as exc:
            raise ValidationError(str(exc)) from exc

        submitted = self._post_op(
            "/v1/sessions",
            payload=req.model_dump(),
            idempotent=False,
        )
        fut: APIFuture[SessionCreatedResult] = APIFuture(
            client=self,
            future_id=submitted.future_id,
            parser=SessionCreatedResult.model_validate,
        )
        created = fut.result()
        return TrainingClient(service_client=self, session_id=created.session_id)

    def delete_session(self, session_id: str) -> None:
        """Delete an existing remote session."""
        self._request(
            "DELETE",
            f"/v1/sessions/{session_id}",
            idempotent=True,
        )

    def _post_op(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        idempotent: bool,
    ) -> FutureSubmittedResponse:
        data = self._request("POST", path, json_body=payload, idempotent=idempotent)
        return FutureSubmittedResponse.model_validate(data)

    def _get_future(self, future_id: str, wait: float = 0.0) -> FutureStatusResponse:
        req_timeout = max(self._timeout, wait + 10.0)
        data = self._request(
            "GET",
            f"/v1/futures/{future_id}",
            params={"wait": wait},
            idempotent=True,
            timeout_override=req_timeout,
        )
        return FutureStatusResponse.model_validate(data)

    def _sleep_for_retry(self, attempt: int, retry_after: float | None = None) -> None:
        if retry_after is not None:
            delay = retry_after
        else:
            jitter = random.uniform(0.8, 1.2)
            delay = self._backoff_base * (2**attempt) * jitter
        time.sleep(delay)

    def _raise_for_http_error(self, response: httpx.Response) -> None:
        code, message = _parse_error_body(response)
        status_code = response.status_code
        if status_code == 401:
            raise AuthError(f"{code}: {message}")
        if status_code in (400, 403, 422):
            raise ValidationError(f"{code}: {message}")
        if status_code == 404:
            raise SessionNotFoundError(f"{code}: {message}")
        if status_code in (429, 503):
            raise ServerBusyError(f"{code}: {message}")
        raise LorkError(f"HTTP {status_code} ({code}): {message}")

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        idempotent: bool,
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        for attempt in range(self._max_retries + 1):
            try:
                response = self._http.request(
                    method,
                    path,
                    json=json_body,
                    params=params,
                    timeout=timeout_override if timeout_override is not None else self._timeout,
                )
            except _PRE_SEND_EXCEPTIONS as exc:
                logger.debug("pre_send_error method=%s path=%s attempt=%d", method, path, attempt)
                if attempt < self._max_retries:
                    self._sleep_for_retry(attempt)
                    continue
                raise LorkError(f"Connection failed for {method} {path}: {exc}") from exc
            except httpx.HTTPError as exc:
                logger.debug("transport_error method=%s path=%s attempt=%d", method, path, attempt)
                if not idempotent:
                    raise AmbiguousSubmissionError(
                        f"Ambiguous failure on non-idempotent {method} {path}: {exc}",
                        cause=exc,
                    ) from exc
                if attempt < self._max_retries:
                    self._sleep_for_retry(attempt)
                    continue
                raise LorkError(
                    f"Request failed after retries for {method} {path}: {exc}"
                ) from exc

            logger.debug(
                "http_response method=%s path=%s status=%d attempt=%d",
                method,
                path,
                response.status_code,
                attempt,
            )

            status_code = response.status_code
            if status_code in (429, 500, 502, 503, 504):
                can_retry_status = idempotent or status_code == 429 or (
                    status_code == 503 and _has_structured_error_envelope(response)
                )
                if not can_retry_status:
                    raise AmbiguousSubmissionError(
                        f"Ambiguous HTTP {status_code} on non-idempotent {method} {path}",
                        cause=LorkError(f"HTTP {status_code}: {response.text}"),
                    )
                if attempt < self._max_retries:
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    self._sleep_for_retry(attempt, retry_after=retry_after)
                    continue
                self._raise_for_http_error(response)

            if status_code >= 400:
                self._raise_for_http_error(response)

            payload: dict[str, Any] = response.json()
            return payload

        raise LorkError(f"Exhausted retries for {method} {path}")
