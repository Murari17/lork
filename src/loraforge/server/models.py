"""Pydantic v2 settings and HTTP request/response models for loraforge server."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from loraforge.wire import (
    CreateSessionRequest,
    ErrorDetail,
    ErrorEnvelope,
    ForwardBackwardWireRequest,
    FutureStatusResponse,
    FutureSubmittedResponse,
    OptimParams,
    OptimStepWireRequest,
    SampleParams,
    SampleWireRequest,
    StateWireRequest,
    TrainerInitConfig,
    WireBatch,
)


class ServerSettings(BaseModel):
    """Environment-configurable settings and limits for the HTTP server."""

    model_config = ConfigDict(extra="forbid")

    api_keys: tuple[str, ...] = Field(..., min_length=1)
    max_batch_size: int = Field(default=32, gt=0)
    max_seq_len: int = Field(default=2048, gt=0)
    max_sample_tokens: int = Field(default=512, gt=0)
    max_prompts: int = Field(default=32, gt=0)
    max_sessions: int = Field(default=2, gt=0)
    session_idle_ttl_seconds: float = Field(default=1800.0, gt=0.0)
    max_futures: int = Field(default=1000, gt=0)
    future_ttl_seconds: float = Field(default=3600.0, gt=0.0)
    checkpoint_root: Path = Path("checkpoints")
    default_device: str = Field(default="cpu", pattern=r"^(cpu|mps|cuda(:\d+)?)$")

    @field_validator("api_keys")
    @classmethod
    def _validate_api_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(k.strip() for k in value if k.strip())
        if not cleaned:
            raise ValueError("api_keys must contain at least one non-empty key")
        return cleaned

    @classmethod
    def from_env(cls) -> ServerSettings:
        """Load ServerSettings from LORAFORGE_* environment variables."""
        raw_keys = os.environ.get("LORAFORGE_API_KEYS", "")
        keys = tuple(part.strip() for part in raw_keys.split(",") if part.strip())
        if not keys:
            raise ValueError(
                "LORAFORGE_API_KEYS environment variable must specify at least one API key"
            )
        return cls(
            api_keys=keys,
            max_batch_size=int(os.environ.get("LORAFORGE_MAX_BATCH_SIZE", "32")),
            max_seq_len=int(os.environ.get("LORAFORGE_MAX_SEQ_LEN", "2048")),
            max_sample_tokens=int(os.environ.get("LORAFORGE_MAX_SAMPLE_TOKENS", "512")),
            max_prompts=int(os.environ.get("LORAFORGE_MAX_PROMPTS", "32")),
            max_sessions=int(os.environ.get("LORAFORGE_MAX_SESSIONS", "2")),
            session_idle_ttl_seconds=float(
                os.environ.get("LORAFORGE_SESSION_IDLE_TTL_SECONDS", "1800.0")
            ),
            max_futures=int(os.environ.get("LORAFORGE_MAX_FUTURES", "1000")),
            future_ttl_seconds=float(
                os.environ.get("LORAFORGE_FUTURE_TTL_SECONDS", "3600.0")
            ),
            checkpoint_root=Path(os.environ.get("LORAFORGE_CHECKPOINT_ROOT", "checkpoints")),
            default_device=os.environ.get("LORAFORGE_DEVICE", "cpu"),
        )


class ForwardBackwardHttpRequest(ForwardBackwardWireRequest):
    """HTTP payload for /v1/sessions/{session_id}/forward_backward with server limits."""

    def check_limits(self, settings: ServerSettings) -> None:
        """Reject oversized batches with explicit ValueError (no truncation)."""
        batch_size, seq_len = self.batch.shape
        if batch_size > settings.max_batch_size:
            raise ValueError(
                f"batch size {batch_size} exceeds max_batch_size={settings.max_batch_size}"
            )
        if seq_len > settings.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len={settings.max_seq_len}"
            )


class OptimStepHttpRequest(OptimStepWireRequest):
    """HTTP payload for /v1/sessions/{session_id}/optim_step reusing wire.OptimParams."""


class SampleHttpRequest(SampleWireRequest):
    """HTTP payload for /v1/sessions/{session_id}/sample reusing wire.SampleParams."""

    def check_limits(self, settings: ServerSettings) -> None:
        """Reject oversized sample requests with explicit ValueError."""
        if len(self.prompts) > settings.max_prompts:
            raise ValueError(
                f"prompt count {len(self.prompts)} exceeds max_prompts={settings.max_prompts}"
            )
        if self.params.max_tokens > settings.max_sample_tokens:
            raise ValueError(
                f"max_tokens {self.params.max_tokens} exceeds "
                f"max_sample_tokens={settings.max_sample_tokens}"
            )


class StateHttpRequest(StateWireRequest):
    """HTTP payload for save_state / load_state with strict checkpoint name rules."""


__all__ = [
    "CreateSessionRequest",
    "ErrorDetail",
    "ErrorEnvelope",
    "ForwardBackwardHttpRequest",
    "FutureStatusResponse",
    "FutureSubmittedResponse",
    "OptimParams",
    "OptimStepHttpRequest",
    "SampleHttpRequest",
    "SampleParams",
    "ServerSettings",
    "StateHttpRequest",
    "TrainerInitConfig",
    "WireBatch",
]
