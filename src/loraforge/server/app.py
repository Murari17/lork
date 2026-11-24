"""FastAPI application factory and /v1 endpoints for loraforge."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from tqdm import tqdm

from loraforge.server.futures import FutureStore
from loraforge.server.models import (
    CreateSessionRequest,
    ErrorDetail,
    ErrorEnvelope,
    ForwardBackwardHttpRequest,
    FutureStatusResponse,
    FutureSubmittedResponse,
    OptimStepHttpRequest,
    SampleHttpRequest,
    ServerSettings,
    StateHttpRequest,
)
from loraforge.server.sessions import (
    InvalidCheckpointPathError,
    SessionCapacityError,
    SessionManager,
    SessionNotFoundError,
)

logger = logging.getLogger("loraforge.server")


class UnauthorizedError(Exception):
    """Raised when Bearer authentication fails."""


class _BodySizeLimitMiddleware:
    """ASGI middleware enforcing max_request_body_bytes before JSON parsing."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for header_name, header_val in scope.get("headers", ()):
            if header_name == b"content-length":
                try:
                    content_len = int(header_val.decode("latin-1"))
                except ValueError:
                    break
                if content_len > self.max_bytes:
                    resp = _error_response(
                        413,
                        "request_too_large",
                        f"Request body {content_len} bytes exceeds limit={self.max_bytes}",
                    )
                    await resp(scope, receive, send)
                    return

        chunks: list[bytes] = []
        received_bytes = 0
        more_body = True
        while more_body:
            message = await receive()
            msg_type = message["type"]
            if msg_type == "http.disconnect":
                return
            if msg_type != "http.request":
                continue
            body_chunk = message.get("body", b"")
            received_bytes += len(body_chunk)
            if received_bytes > self.max_bytes:
                resp = _error_response(
                    413,
                    "request_too_large",
                    f"Request body exceeds max_request_body_bytes={self.max_bytes}",
                )
                await resp(scope, receive, send)
                return
            if body_chunk:
                chunks.append(body_chunk)
            more_body = bool(message.get("more_body", False))

        buffered = b"".join(chunks)
        sent_to_app = False

        async def replay_receive() -> Message:
            nonlocal sent_to_app
            if not sent_to_app:
                sent_to_app = True
                return {"type": "http.request", "body": buffered, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    payload = ErrorEnvelope(error=ErrorDetail(code=code, message=message))
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _verify_bearer_token(
    provided_header: str | None,
    allowed_keys: tuple[str, ...],
) -> None:
    if not provided_header:
        raise UnauthorizedError("Missing Authorization header")
    scheme, _, token = provided_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise UnauthorizedError("Authorization header must use Bearer scheme")
    token_bytes = token.encode("utf-8")
    matched = False
    for expected in allowed_keys:
        if hmac.compare_digest(token_bytes, expected.encode("utf-8")):
            matched = True
    if not matched:
        raise UnauthorizedError("Invalid API key")


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    # Uvicorn invokes this factory directly (no separate server CLI entrypoint), and tqdm
    # stores its default write lock globally on TqdmDefaultWriteLock.mp_lock. Replacing it
    # here with a process-local threading.RLock prevents transformers weight-loading bars
    # from creating a multiprocessing POSIX semaphore that leaks on shutdown.
    tqdm.set_lock(threading.RLock())
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    cfg = settings if settings is not None else ServerSettings.from_env()
    future_store = FutureStore(
        max_futures=cfg.max_futures,
        ttl_seconds=cfg.future_ttl_seconds,
    )
    session_manager = SessionManager(settings=cfg, future_store=future_store)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        stop_event = asyncio.Event()

        async def _eviction_loop() -> None:
            interval = max(0.05, min(5.0, cfg.session_idle_ttl_seconds / 2.0))
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                except TimeoutError:
                    await asyncio.to_thread(session_manager.evict_idle_sessions)

        evictor = asyncio.create_task(_eviction_loop())
        try:
            yield
        finally:
            stop_event.set()
            evictor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await evictor
            drained = await asyncio.to_thread(session_manager.shutdown)
            logger.info(
                json.dumps(
                    {
                        "event": "server_shutdown_drained",
                        "drained_sessions": drained,
                    }
                )
            )

    app = FastAPI(title="loraforge", version="0.1.0", lifespan=lifespan)
    app.add_middleware(_BodySizeLimitMiddleware, max_bytes=cfg.max_request_body_bytes)

    @app.middleware("http")
    async def structured_logging_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = round((time.monotonic() - start) * 1000.0, 2)
        response.headers["X-Request-ID"] = request_id
        # Never log Authorization headers, API keys, prompts, or token IDs.
        log_record = {
            "event": "http_request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        }
        logger.info(json.dumps(log_record))
        return response

    @app.exception_handler(UnauthorizedError)
    async def _handle_unauthorized(_request: Request, exc: UnauthorizedError) -> JSONResponse:
        return _error_response(status.HTTP_401_UNAUTHORIZED, "unauthorized", str(exc))

    @app.exception_handler(SessionNotFoundError)
    async def _handle_not_found(_request: Request, exc: SessionNotFoundError) -> JSONResponse:
        return _error_response(status.HTTP_404_NOT_FOUND, "session_not_found", str(exc))

    @app.exception_handler(SessionCapacityError)
    async def _handle_capacity(_request: Request, exc: SessionCapacityError) -> JSONResponse:
        return _error_response(
            status.HTTP_429_TOO_MANY_REQUESTS, "session_capacity_exceeded", str(exc)
        )

    @app.exception_handler(InvalidCheckpointPathError)
    async def _handle_bad_ckpt_path(
        _request: Request, exc: InvalidCheckpointPathError
    ) -> JSONResponse:
        return _error_response(status.HTTP_403_FORBIDDEN, "invalid_checkpoint_path", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "validation_error",
            str(exc),
        )

    @app.exception_handler(ValueError)
    async def _handle_value_error(_request: Request, exc: ValueError) -> JSONResponse:
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "validation_error",
            str(exc),
        )

    def require_auth(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        _verify_bearer_token(authorization, cfg.api_keys)

    @app.get("/v1/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/v1/sessions",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_auth)],
        response_model=FutureSubmittedResponse,
    )
    async def create_session(payload: CreateSessionRequest) -> FutureSubmittedResponse:
        trainer_cfg = payload.to_trainer_config(cfg.default_device)
        session_id, future = session_manager.create_session(trainer_cfg)
        return FutureSubmittedResponse(future_id=future.future_id, session_id=session_id)

    @app.delete(
        "/v1/sessions/{session_id}",
        dependencies=[Depends(require_auth)],
    )
    async def delete_session(session_id: str) -> dict[str, str]:
        await asyncio.to_thread(session_manager.delete_session, session_id)
        return {"deleted": session_id}

    @app.post(
        "/v1/sessions/{session_id}/forward_backward",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_auth)],
        response_model=FutureSubmittedResponse,
    )
    async def forward_backward(
        session_id: str,
        payload: ForwardBackwardHttpRequest,
    ) -> FutureSubmittedResponse:
        payload.check_limits(cfg)
        worker = session_manager.get_session(session_id)
        future = worker.submit_forward_backward(
            batch=payload.batch.model_dump(),
            loss_fn=payload.loss_fn,
            return_logprobs=payload.return_logprobs,
        )
        return FutureSubmittedResponse(future_id=future.future_id, session_id=session_id)

    @app.post(
        "/v1/sessions/{session_id}/optim_step",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_auth)],
        response_model=FutureSubmittedResponse,
    )
    async def optim_step(
        session_id: str,
        payload: OptimStepHttpRequest,
    ) -> FutureSubmittedResponse:
        worker = session_manager.get_session(session_id)
        future = worker.submit_optim_step(
            step_id=payload.step_id,
            optim_params=payload.optim_params.model_dump(),
        )
        return FutureSubmittedResponse(future_id=future.future_id, session_id=session_id)

    @app.post(
        "/v1/sessions/{session_id}/sample",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_auth)],
        response_model=FutureSubmittedResponse,
    )
    async def sample(
        session_id: str,
        payload: SampleHttpRequest,
    ) -> FutureSubmittedResponse:
        payload.check_limits(cfg)
        worker = session_manager.get_session(session_id)
        future = worker.submit_sample(
            prompts=payload.prompts,
            params=payload.params.model_dump(),
        )
        return FutureSubmittedResponse(future_id=future.future_id, session_id=session_id)

    @app.post(
        "/v1/sessions/{session_id}/save_state",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_auth)],
        response_model=FutureSubmittedResponse,
    )
    async def save_state(
        session_id: str,
        payload: StateHttpRequest,
    ) -> FutureSubmittedResponse:
        worker = session_manager.get_session(session_id)
        future = worker.submit_save_state(payload.name)
        return FutureSubmittedResponse(future_id=future.future_id, session_id=session_id)

    @app.post(
        "/v1/sessions/{session_id}/load_state",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_auth)],
        response_model=FutureSubmittedResponse,
    )
    async def load_state(
        session_id: str,
        payload: StateHttpRequest,
    ) -> FutureSubmittedResponse:
        worker = session_manager.get_session(session_id)
        future = worker.submit_load_state(payload.name)
        return FutureSubmittedResponse(future_id=future.future_id, session_id=session_id)

    @app.get(
        "/v1/futures/{future_id}",
        dependencies=[Depends(require_auth)],
        response_model=FutureStatusResponse,
    )
    async def get_future(
        future_id: str,
        wait: Annotated[float, Query(ge=0.0, le=30.0)] = 0.0,
    ) -> FutureStatusResponse | JSONResponse:
        record = future_store.get(future_id)
        if record is None:
            return _error_response(
                status.HTTP_404_NOT_FOUND,
                "future_not_found",
                f"Future {future_id!r} not found",
            )
        if wait > 0.0:
            await asyncio.to_thread(record.wait_done, wait)
        return record.snapshot()

    return app
