"""lork: Open-source low-level LoRA fine-tuning primitives and client SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lork.wire import (
    ExportAdapterResult,
    OptimParams,
    SampleParams,
    TrainerInitConfig,
    WireBatch,
)

__version__ = "0.2.0"

if TYPE_CHECKING:
    from lork.schemas import ForwardBackwardBatch
    from lork.trainer import LoraTrainer

__all__ = [
    "ExportAdapterResult",
    "ForwardBackwardBatch",
    "LoraTrainer",
    "OptimParams",
    "SampleParams",
    "TrainerInitConfig",
    "WireBatch",
    "__version__",
]


def __getattr__(name: str) -> Any:
    if name == "LoraTrainer":
        from lork.trainer import LoraTrainer

        return LoraTrainer
    if name == "ForwardBackwardBatch":
        from lork.schemas import ForwardBackwardBatch

        return ForwardBackwardBatch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
