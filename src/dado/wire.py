"""Pure-Pydantic wire schemas shared by server and client without heavy ML imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _validate_rect_2d_list(rows: list[list[Any]], name: str) -> tuple[int, int]:
    if not rows:
        raise ValueError(f"{name} must not be empty")
    seq_len = len(rows[0])
    if seq_len == 0:
        raise ValueError(f"{name} rows must not be empty")
    for idx, row in enumerate(rows):
        if len(row) != seq_len:
            raise ValueError(
                f"{name} must be rectangular; row {idx} has length {len(row)} != {seq_len}"
            )
    return len(rows), seq_len


class TrainerInitConfig(BaseModel):
    """Validated configuration for initializing a LoraTrainer."""

    model_config = ConfigDict(extra="forbid")

    base_model: str = Field(..., min_length=1)
    rank: int = Field(..., gt=0)
    alpha: float = Field(..., gt=0.0)
    target_modules: list[str] = Field(..., min_length=1)
    device: str = Field(default="cpu", pattern=r"^(cpu|mps|cuda(:\d+)?)$")
    dtype: Literal["float32", "bfloat16", "float16"] | None = None

    @field_validator("base_model")
    @classmethod
    def _validate_base_model(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("base_model must not be empty or whitespace")
        return stripped

    @field_validator("target_modules")
    @classmethod
    def _validate_target_modules(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            stripped = item.strip()
            if not stripped:
                raise ValueError("target_modules entries must not be empty")
            cleaned.append(stripped)
        return cleaned


class OptimParams(BaseModel):
    """Validated AdamW hyperparameters for optim_step."""

    model_config = ConfigDict(extra="forbid")

    lr: float = Field(..., gt=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = Field(default=0.0, ge=0.0)
    eps: float = Field(default=1e-8, gt=0.0)

    @field_validator("betas")
    @classmethod
    def _validate_betas(cls, value: tuple[float, float]) -> tuple[float, float]:
        b1, b2 = value
        if not (0.0 < b1 < 1.0 and 0.0 < b2 < 1.0):
            raise ValueError(f"betas must each be strictly between 0 and 1, got {value}")
        return value


class OptimStepRequest(BaseModel):
    """Validated inputs to optim_step."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(..., min_length=1)
    optim_params: OptimParams

    @field_validator("step_id")
    @classmethod
    def _validate_step_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("step_id must not be empty or whitespace")
        return stripped


class SampleParams(BaseModel):
    """Validated decoding parameters for sample."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(..., gt=0)
    temperature: float = Field(default=1.0, ge=0.0)
    seed: int | None = None


class SampleRequest(BaseModel):
    """Validated inputs to sample."""

    model_config = ConfigDict(extra="forbid")

    prompts: list[str] = Field(..., min_length=1)
    params: SampleParams


class StatePathInput(BaseModel):
    """Validated filesystem path for save_state / load_state."""

    model_config = ConfigDict(extra="forbid")

    path: Path

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: Any) -> Path:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("path must not be empty")
            return Path(value)
        if isinstance(value, Path):
            if not str(value).strip():
                raise ValueError("path must not be empty")
            return value
        raise ValueError("path must be a str or pathlib.Path")


class CreateSessionRequest(BaseModel):
    """HTTP payload for creating a LoRA training session."""

    model_config = ConfigDict(extra="forbid")

    base_model: str = Field(..., min_length=1)
    rank: int = Field(..., gt=0)
    alpha: float = Field(..., gt=0.0)
    target_modules: list[str] = Field(..., min_length=1)
    device: str | None = None
    dtype: Literal["float32", "bfloat16", "float16"] | None = None

    def to_trainer_config(self, default_device: str) -> TrainerInitConfig:
        """Validate session initialization parameters via shared TrainerInitConfig."""
        return TrainerInitConfig(
            base_model=self.base_model,
            rank=self.rank,
            alpha=self.alpha,
            target_modules=self.target_modules,
            device=self.device if self.device is not None else default_device,
            dtype=self.dtype,
        )


class WireBatch(BaseModel):
    """JSON wire representation of a forward_backward token batch."""

    model_config = ConfigDict(extra="forbid")

    input_ids: list[list[int]] = Field(..., min_length=1)
    target_ids: list[list[int]] = Field(..., min_length=1)
    weights: list[list[float]] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_shapes(self) -> WireBatch:
        b_in, t_in = _validate_rect_2d_list(self.input_ids, "input_ids")
        b_tgt, t_tgt = _validate_rect_2d_list(self.target_ids, "target_ids")
        b_w, t_w = _validate_rect_2d_list(self.weights, "weights")
        if (b_tgt, t_tgt) != (b_in, t_in):
            raise ValueError(
                f"target_ids shape {(b_tgt, t_tgt)} does not match input_ids {(b_in, t_in)}"
            )
        if (b_w, t_w) != (b_in, t_in):
            raise ValueError(
                f"weights shape {(b_w, t_w)} does not match input_ids {(b_in, t_in)}"
            )
        total_weight = 0.0
        for row in self.weights:
            for val in row:
                if val < 0.0:
                    raise ValueError("weights must be non-negative")
                total_weight += val
        if total_weight <= 0.0:
            raise ValueError("weights must sum to a positive value")
        return self

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.input_ids), len(self.input_ids[0])


class ForwardBackwardWireRequest(BaseModel):
    """Wire payload for /v1/sessions/{session_id}/forward_backward."""

    model_config = ConfigDict(extra="forbid")

    batch: WireBatch
    loss_fn: Literal["cross_entropy"] | list[list[float]] = "cross_entropy"
    return_logprobs: bool = True

    @model_validator(mode="after")
    def _validate_loss_shape(self) -> ForwardBackwardWireRequest:
        if isinstance(self.loss_fn, list):
            adv_shape = _validate_rect_2d_list(self.loss_fn, "loss_fn")
            if adv_shape != self.batch.shape:
                raise ValueError(
                    f"loss_fn advantage shape {adv_shape} does not match "
                    f"batch shape {self.batch.shape}"
                )
        return self


class OptimStepWireRequest(OptimStepRequest):
    """Wire payload for /v1/sessions/{session_id}/optim_step."""


class SampleWireRequest(SampleRequest):
    """Wire payload for /v1/sessions/{session_id}/sample."""


class StateWireRequest(BaseModel):
    """Wire payload for save_state / load_state with strict checkpoint name rules."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,64}$")


class ErrorDetail(BaseModel):
    """Structured error detail payload."""

    code: str
    message: str


class ErrorEnvelope(BaseModel):
    """Top-level structured error response body."""

    error: ErrorDetail


class FutureSubmittedResponse(BaseModel):
    """202 Accepted response for async operations."""

    future_id: str
    session_id: str | None = None


class FutureStatusResponse(BaseModel):
    """Response for GET /v1/futures/{future_id}."""

    future_id: str
    status: Literal["pending", "done", "error"]
    result: dict[str, Any] | None = None
    error: ErrorDetail | None = None


class SessionCreatedResult(BaseModel):
    """Typed result for session creation future."""

    session_id: str


class ForwardBackwardResult(BaseModel):
    """Typed result for forward_backward future."""

    loss: float
    logprobs: list[list[float]] | None = None


class OptimStepResult(BaseModel):
    """Typed result for optim_step future."""

    step: int


class SampleResult(BaseModel):
    """Typed result for sample future."""

    completions: list[str]
    token_ids: list[list[int]]
    logprobs: list[list[float]]


class StateResult(BaseModel):
    """Typed result for save_state / load_state future."""

    name: str
