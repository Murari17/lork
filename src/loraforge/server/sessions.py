"""Dedicated per-session worker threads, FIFO execution, and TTL lifecycle."""

from __future__ import annotations

import gc
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from loraforge.schemas import TrainerInitConfig
from loraforge.server.futures import FutureRecord, FutureStore
from loraforge.server.models import ServerSettings
from loraforge.trainer import LoraTrainer

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SessionNotFoundError(Exception):
    """Raised when a session ID does not exist or was evicted."""


class SessionCapacityError(Exception):
    """Raised when active sessions are at configured max_sessions capacity."""


class InvalidCheckpointPathError(Exception):
    """Raised when checkpoint path validation or symlink containment fails."""


def resolve_safe_checkpoint_path(checkpoint_root: Path, session_id: str, name: str) -> Path:
    """Resolve checkpoint path inside per-session directory and reject traversal/symlinks."""
    if not _SAFE_NAME_RE.match(name):
        raise InvalidCheckpointPathError(
            f"Checkpoint name {name!r} violates pattern ^[A-Za-z0-9_-]{{1,64}}$"
        )

    root_resolved = checkpoint_root.resolve()
    session_dir = root_resolved / session_id
    if session_dir.is_symlink():
        raise InvalidCheckpointPathError("Session checkpoint root must not be a symlink")
    session_dir.mkdir(parents=True, exist_ok=True)
    session_resolved = session_dir.resolve()
    if not session_resolved.is_relative_to(root_resolved):
        raise InvalidCheckpointPathError("Session checkpoint directory escaped checkpoint root")

    target = session_resolved / name
    if target.is_symlink():
        target_resolved = target.resolve()
        if not target_resolved.is_relative_to(session_resolved):
            raise InvalidCheckpointPathError(
                f"Checkpoint symlink {name!r} points outside session directory"
            )
        raise InvalidCheckpointPathError(f"Checkpoint path {name!r} must not be a symlink")

    target_resolved = target.resolve()
    if target_resolved.parent != session_resolved or not target_resolved.is_relative_to(
        session_resolved
    ):
        raise InvalidCheckpointPathError(
            f"Checkpoint path {name!r} resolved outside session directory"
        )

    if target.exists() and target.is_dir():
        for child in target.rglob("*"):
            if child.is_symlink():
                child_resolved = child.resolve()
                if not child_resolved.is_relative_to(session_resolved):
                    raise InvalidCheckpointPathError(
                        "Checkpoint directory contains symlink pointing outside session directory"
                    )

    return target_resolved


def _empty_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class SessionWorker:
    """Owns a single LoraTrainer and executes queued ops strictly FIFO on one thread."""

    def __init__(
        self,
        session_id: str,
        init_config: TrainerInitConfig,
        checkpoint_root: Path,
        future_store: FutureStore,
    ) -> None:
        self.session_id: str = session_id
        self._init_config: TrainerInitConfig = init_config
        self._checkpoint_root: Path = checkpoint_root
        self._future_store: FutureStore = future_store

        self._queue: queue.Queue[tuple[FutureRecord, Callable[[], dict[str, Any]]] | None] = (
            queue.Queue()
        )
        self._trainer: LoraTrainer | None = None
        self._step_results: dict[str, dict[str, int]] = {}
        self._state_lock = threading.Lock()
        self._busy: bool = False
        self._closed: bool = False
        self._last_active: float = time.monotonic()

        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"loraforge-session-{session_id[:12]}",
            daemon=True,
        )
        self._thread.start()

    def submit_init(self) -> FutureRecord:
        def _init_op() -> dict[str, Any]:
            self._trainer = LoraTrainer(
                base_model=self._init_config.base_model,
                rank=self._init_config.rank,
                alpha=self._init_config.alpha,
                target_modules=self._init_config.target_modules,
                device=self._init_config.device,
                dtype=self._init_config.dtype,
            )
            return {"session_id": self.session_id}

        return self.submit(_init_op)

    def submit(self, op: Callable[[], dict[str, Any]]) -> FutureRecord:
        with self._state_lock:
            if self._closed:
                raise SessionNotFoundError(f"Session {self.session_id} is closed")
            self._last_active = time.monotonic()
            record = self._future_store.create(self.session_id)
            self._queue.put((record, op))
            return record

    def submit_forward_backward(
        self,
        batch: dict[str, Any],
        loss_fn: str | list[list[float]],
    ) -> FutureRecord:
        def _op() -> dict[str, Any]:
            trainer = self._require_trainer()
            result = trainer.forward_backward(batch=batch, loss_fn=loss_fn)
            logprobs_tensor: torch.Tensor = result["logprobs"]
            return {
                "loss": float(result["loss"]),
                "logprobs": logprobs_tensor.cpu().tolist(),
            }

        return self.submit(_op)

    def submit_optim_step(
        self,
        step_id: str,
        optim_params: dict[str, Any],
    ) -> FutureRecord:
        def _op() -> dict[str, Any]:
            if step_id in self._step_results:
                return dict(self._step_results[step_id])
            trainer = self._require_trainer()
            result = trainer.optim_step(step_id=step_id, optim_params=optim_params)
            stored = {"step": int(result["step"])}
            self._step_results[step_id] = stored
            return dict(stored)

        return self.submit(_op)

    def submit_sample(
        self,
        prompts: list[str],
        params: dict[str, Any],
    ) -> FutureRecord:
        def _op() -> dict[str, Any]:
            trainer = self._require_trainer()
            return trainer.sample(prompts=prompts, params=params)

        return self.submit(_op)

    def submit_save_state(self, name: str) -> FutureRecord:
        # Validate eagerly on request path before queueing, and re-verify in worker.
        resolve_safe_checkpoint_path(self._checkpoint_root, self.session_id, name)

        def _op() -> dict[str, Any]:
            safe_path = resolve_safe_checkpoint_path(
                self._checkpoint_root, self.session_id, name
            )
            trainer = self._require_trainer()
            trainer.save_state(safe_path)
            return {"name": name}

        return self.submit(_op)

    def submit_load_state(self, name: str) -> FutureRecord:
        resolve_safe_checkpoint_path(self._checkpoint_root, self.session_id, name)

        def _op() -> dict[str, Any]:
            safe_path = resolve_safe_checkpoint_path(
                self._checkpoint_root, self.session_id, name
            )
            trainer = self._require_trainer()
            trainer.load_state(safe_path)
            return {"name": name}

        return self.submit(_op)

    def is_idle_expired(self, now: float, ttl_seconds: float) -> bool:
        with self._state_lock:
            if self._closed or self._busy or not self._queue.empty():
                return False
            return (now - self._last_active) > ttl_seconds

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=30.0)
        self._trainer = None
        _empty_device_cache()

    def _require_trainer(self) -> LoraTrainer:
        if self._trainer is None:
            raise RuntimeError(f"Session {self.session_id} trainer is not initialized")
        return self._trainer

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            record, op = item
            with self._state_lock:
                self._busy = True
                self._last_active = time.monotonic()
            try:
                result = op()
                record.set_done(result)
            except Exception as exc:
                record.set_error(code=type(exc).__name__, message=str(exc))
            finally:
                with self._state_lock:
                    self._busy = False
                    self._last_active = time.monotonic()
                self._queue.task_done()


class SessionManager:
    """Manages active SessionWorkers, limits, idle TTL eviction, and clean shutdown."""

    def __init__(self, settings: ServerSettings, future_store: FutureStore) -> None:
        self._settings = settings
        self._future_store = future_store
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionWorker] = {}

    def create_session(self, init_config: TrainerInitConfig) -> tuple[str, FutureRecord]:
        self.evict_idle_sessions()
        with self._lock:
            if len(self._sessions) >= self._settings.max_sessions:
                raise SessionCapacityError(
                    f"Maximum active sessions ({self._settings.max_sessions}) reached"
                )
            session_id = f"sess_{uuid.uuid4().hex}"
            worker = SessionWorker(
                session_id=session_id,
                init_config=init_config,
                checkpoint_root=self._settings.checkpoint_root,
                future_store=self._future_store,
            )
            self._sessions[session_id] = worker
        future = worker.submit_init()
        return session_id, future

    def get_session(self, session_id: str) -> SessionWorker:
        self.evict_idle_sessions()
        with self._lock:
            worker = self._sessions.get(session_id)
            if worker is None:
                raise SessionNotFoundError(f"Session {session_id!r} not found")
            return worker

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            worker = self._sessions.pop(session_id, None)
        if worker is None:
            raise SessionNotFoundError(f"Session {session_id!r} not found")
        worker.close()

    def evict_idle_sessions(self) -> int:
        now = time.monotonic()
        to_close: list[SessionWorker] = []
        with self._lock:
            expired_ids = [
                sid
                for sid, worker in self._sessions.items()
                if worker.is_idle_expired(now, self._settings.session_idle_ttl_seconds)
            ]
            for sid in expired_ids:
                to_close.append(self._sessions.pop(sid))
        for worker in to_close:
            worker.close()
        return len(to_close)

    def shutdown(self) -> int:
        with self._lock:
            workers = list(self._sessions.values())
            self._sessions.clear()
        for worker in workers:
            worker.close()
        return len(workers)
