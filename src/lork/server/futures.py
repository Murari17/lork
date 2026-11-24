"""Thread-safe bounded future store with long-poll wait support."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Literal

from lork.server.models import ErrorDetail, FutureStatusResponse


class FutureRecord:
    """Single async operation future tracked by FutureStore."""

    def __init__(self, future_id: str, session_id: str, created_at: float) -> None:
        self.future_id: str = future_id
        self.session_id: str = session_id
        self.created_at: float = created_at
        self.completed_at: float | None = None
        self.status: Literal["pending", "done", "error"] = "pending"
        self.result: dict[str, Any] | None = None
        self.error: ErrorDetail | None = None
        self._lock = threading.Lock()
        self._done_event = threading.Event()

    def set_done(self, result: dict[str, Any]) -> None:
        with self._lock:
            self.status = "done"
            self.result = result
            self.error = None
            self.completed_at = time.monotonic()
            self._done_event.set()

    def set_error(self, code: str, message: str) -> None:
        with self._lock:
            self.status = "error"
            self.result = None
            self.error = ErrorDetail(code=code, message=message)
            self.completed_at = time.monotonic()
            self._done_event.set()

    def wait_done(self, timeout_seconds: float) -> bool:
        """Block the calling worker thread until done or timeout expires."""
        return self._done_event.wait(timeout=timeout_seconds)

    def snapshot(self) -> FutureStatusResponse:
        with self._lock:
            return FutureStatusResponse(
                future_id=self.future_id,
                status=self.status,
                result=self.result,
                error=self.error,
            )


class FutureStore:
    """Bounded in-memory future registry with TTL and capacity eviction."""

    def __init__(self, max_futures: int, ttl_seconds: float) -> None:
        self._max_futures = max_futures
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._records: dict[str, FutureRecord] = {}

    def create(self, session_id: str) -> FutureRecord:
        now = time.monotonic()
        future_id = f"fut_{uuid.uuid4().hex}"
        record = FutureRecord(future_id=future_id, session_id=session_id, created_at=now)
        with self._lock:
            self._evict_locked(now)
            self._records[future_id] = record
        return record

    def get(self, future_id: str) -> FutureRecord | None:
        now = time.monotonic()
        with self._lock:
            self._evict_locked(now)
            return self._records.get(future_id)

    def _evict_locked(self, now: float) -> None:
        expired_ids = [
            fid
            for fid, rec in self._records.items()
            if rec.completed_at is not None and (now - rec.completed_at) > self._ttl_seconds
        ]
        for fid in expired_ids:
            del self._records[fid]

        if len(self._records) >= self._max_futures:
            completed = sorted(
                (
                    (rec.completed_at or rec.created_at, fid)
                    for fid, rec in self._records.items()
                    if rec.status != "pending"
                ),
                key=lambda item: item[0],
            )
            excess = len(self._records) - self._max_futures + 1
            for _, fid in completed[:excess]:
                del self._records[fid]
