"""Pydantic v2 tensor and trainer validation schemas for dado."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import torch
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    model_validator,
)

from dado.wire import (
    OptimParams,
    OptimStepRequest,
    SampleParams,
    SampleRequest,
    StatePathInput,
    TrainerInitConfig,
)


def _to_int64_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
            raise ValueError("Expected integer tensor")
        return value.to(dtype=torch.long)
    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError("Expected non-empty 2D integer list")
        tensor = torch.tensor(value)
        if tensor.is_floating_point() or tensor.is_complex() or tensor.dtype == torch.bool:
            raise ValueError("Expected integer values")
        return tensor.to(dtype=torch.long)
    raise ValueError("Expected torch.Tensor or nested list of integers")


def _to_float32_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.is_complex() or value.dtype == torch.bool:
            raise ValueError("Expected real numeric tensor")
        return value.to(dtype=torch.float32)
    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError("Expected non-empty 2D numeric list")
        tensor = torch.tensor(value)
        if tensor.is_complex() or tensor.dtype == torch.bool:
            raise ValueError("Expected real numeric values")
        return tensor.to(dtype=torch.float32)
    raise ValueError("Expected torch.Tensor or nested list of numbers")


Int64Tensor2D = Annotated[torch.Tensor, BeforeValidator(_to_int64_tensor)]
Float32Tensor2D = Annotated[torch.Tensor, BeforeValidator(_to_float32_tensor)]


class ForwardBackwardBatch(BaseModel):
    """Validated token batch for forward_backward."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    input_ids: Int64Tensor2D
    target_ids: Int64Tensor2D
    weights: Float32Tensor2D

    @model_validator(mode="after")
    def _validate_shapes_and_values(self) -> ForwardBackwardBatch:
        if self.input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got shape {tuple(self.input_ids.shape)}")
        if self.input_ids.shape[0] == 0 or self.input_ids.shape[1] == 0:
            raise ValueError(
                f"input_ids must be non-empty, got shape {tuple(self.input_ids.shape)}"
            )
        if self.target_ids.shape != self.input_ids.shape:
            raise ValueError(
                f"target_ids shape {tuple(self.target_ids.shape)} does not match "
                f"input_ids shape {tuple(self.input_ids.shape)}"
            )
        if self.weights.shape != self.input_ids.shape:
            raise ValueError(
                f"weights shape {tuple(self.weights.shape)} does not match "
                f"input_ids shape {tuple(self.input_ids.shape)}"
            )
        if not bool(torch.isfinite(self.weights).all().item()):
            raise ValueError("weights must contain only finite values")
        if bool((self.weights < 0.0).any().item()):
            raise ValueError("weights must be non-negative")
        if float(self.weights.sum().item()) <= 0.0:
            raise ValueError("weights must sum to a positive value")
        return self


class AdvantageLossInput(BaseModel):
    """Validated precomputed per-token advantages tensor."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    advantages: Float32Tensor2D
    expected_shape: tuple[int, int]

    @model_validator(mode="after")
    def _validate_advantages(self) -> AdvantageLossInput:
        if self.advantages.ndim != 2:
            raise ValueError(
                f"advantages must be 2D, got shape {tuple(self.advantages.shape)}"
            )
        if tuple(self.advantages.shape) != self.expected_shape:
            raise ValueError(
                f"advantages shape {tuple(self.advantages.shape)} does not match "
                f"batch shape {self.expected_shape}"
            )
        if not bool(torch.isfinite(self.advantages).all().item()):
            raise ValueError("advantages must contain only finite values")
        return self


def validate_loss_fn(
    loss_fn: str | torch.Tensor | list[list[float]],
    expected_shape: tuple[int, int],
) -> Literal["cross_entropy"] | torch.Tensor:
    """Validate loss_fn as either 'cross_entropy' or a per-token advantage tensor."""
    if isinstance(loss_fn, str):
        if loss_fn != "cross_entropy":
            raise ValueError(
                f"Unsupported loss_fn {loss_fn!r}; expected 'cross_entropy' or advantage tensor"
            )
        return "cross_entropy"
    validated = AdvantageLossInput.model_validate(
        {"advantages": loss_fn, "expected_shape": expected_shape}
    )
    return validated.advantages


__all__ = [
    "AdvantageLossInput",
    "Float32Tensor2D",
    "ForwardBackwardBatch",
    "Int64Tensor2D",
    "OptimParams",
    "OptimStepRequest",
    "SampleParams",
    "SampleRequest",
    "StatePathInput",
    "TrainerInitConfig",
    "validate_loss_fn",
]
