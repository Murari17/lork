"""Typed exception hierarchy for the lork Python SDK."""

from __future__ import annotations


class LorkError(Exception):
    """Base exception for all lork SDK errors."""


class AuthError(LorkError):
    """Raised when authentication fails (HTTP 401)."""


class ValidationError(LorkError):
    """Raised when request parameters or batch shapes are invalid (HTTP 422/403 or client-side)."""


class SessionNotFoundError(LorkError):
    """Raised when a session or future does not exist or was evicted (HTTP 404)."""


class ServerBusyError(LorkError):
    """Raised when the server is at capacity or rate-limited (HTTP 429/503)."""


class RemoteOpError(LorkError):
    """Raised when an async server operation fails during execution."""

    def __init__(self, code: str, remote_message: str) -> None:
        self.code = code
        self.remote_message = remote_message
        super().__init__(f"{code}: {remote_message}")


class AmbiguousSubmissionError(LorkError):
    """Raised when a non-idempotent POST fails after bytes may have reached the server."""

    def __init__(self, message: str, cause: BaseException) -> None:
        self.cause = cause
        super().__init__(message)


class LorkTimeoutError(LorkError):
    """Raised when waiting for an APIFuture exceeds the caller's timeout."""
