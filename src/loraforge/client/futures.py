"""Long-polling future wrapper for async server operations."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from loraforge.client.exceptions import LoraforgeTimeoutError, RemoteOpError

if TYPE_CHECKING:
    from loraforge.client.client import ServiceClient
    from loraforge.wire import FutureStatusResponse

T = TypeVar("T")


class APIFuture(Generic[T]):
    """Handle to an in-flight asynchronous server operation."""

    def __init__(
        self,
        client: ServiceClient,
        future_id: str,
        parser: Callable[[dict[str, Any]], T],
    ) -> None:
        self._client = client
        self.future_id = future_id
        self._parser = parser
        self._done = False
        self._result: T | None = None
        self._error: RemoteOpError | None = None

    def _apply_status(self, status_resp: FutureStatusResponse) -> bool:
        if status_resp.status == "done":
            self._result = self._parser(status_resp.result or {})
            self._done = True
            return True
        if status_resp.status == "error":
            err = status_resp.error
            code = err.code if err is not None else "remote_error"
            msg = err.message if err is not None else "Remote operation failed"
            self._error = RemoteOpError(code=code, remote_message=msg)
            self._done = True
            return True
        return False

    def done(self) -> bool:
        """Return True if the remote operation has completed (success or error)."""
        if self._done:
            return True
        status_resp = self._client._get_future(self.future_id, wait=0.0)
        return self._apply_status(status_resp)

    def result(
        self,
        timeout: float | None = None,
        poll_wait: float = 25.0,
    ) -> T:
        """Long-poll until the operation finishes, or raise LoraforgeTimeoutError."""
        if self._done:
            if self._error is not None:
                raise self._error
            assert self._result is not None
            return self._result

        deadline = None if timeout is None else (time.monotonic() + timeout)
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise LoraforgeTimeoutError(
                        f"Timed out waiting for future {self.future_id}"
                    )
                wait_seconds = max(0.0, min(poll_wait, 30.0, remaining))
            else:
                wait_seconds = max(0.0, min(poll_wait, 30.0))

            status_resp = self._client._get_future(self.future_id, wait=wait_seconds)
            if self._apply_status(status_resp):
                if self._error is not None:
                    raise self._error
                assert self._result is not None
                return self._result
