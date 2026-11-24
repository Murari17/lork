"""Lightweight synchronous Python SDK for dado."""

from dado.client.client import ServiceClient, TrainingClient, validate_batch
from dado.client.exceptions import (
    AmbiguousSubmissionError,
    AuthError,
    DadoError,
    DadoTimeoutError,
    RemoteOpError,
    ServerBusyError,
    SessionNotFoundError,
    ValidationError,
)
from dado.client.futures import APIFuture
from dado.wire import (
    ForwardBackwardResult,
    OptimParams,
    OptimStepResult,
    SampleParams,
    SampleResult,
    StateResult,
    WireBatch,
)

__all__ = [
    "APIFuture",
    "AmbiguousSubmissionError",
    "AuthError",
    "ForwardBackwardResult",
    "DadoError",
    "DadoTimeoutError",
    "OptimParams",
    "OptimStepResult",
    "RemoteOpError",
    "SampleParams",
    "SampleResult",
    "ServerBusyError",
    "ServiceClient",
    "SessionNotFoundError",
    "StateResult",
    "TrainingClient",
    "ValidationError",
    "WireBatch",
    "validate_batch",
]
