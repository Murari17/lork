"""dado: Open-source low-level LoRA fine-tuning primitives and client SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dado.wire import (
    OptimParams,
    SampleParams,
    TrainerInitConfig,
    WireBatch,
)

if TYPE_CHECKING:
    from dado.schemas import ForwardBackwardBatch
    from dado.trainer import LoraTrainer

__all__ = [
    "ForwardBackwardBatch",
    "LoraTrainer",
    "OptimParams",
    "SampleParams",
    "TrainerInitConfig",
    "WireBatch",
]


def __getattr__(name: str) -> Any:
    if name == "LoraTrainer":
        from dado.trainer import LoraTrainer

        return LoraTrainer
    if name == "ForwardBackwardBatch":
        from dado.schemas import ForwardBackwardBatch

        return ForwardBackwardBatch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
