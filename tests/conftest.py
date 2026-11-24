"""Shared pytest fixtures for lork tests."""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
TARGET_MODULES = ["q_proj", "v_proj"]


@pytest.fixture(scope="session")
def model_name() -> str:
    return MODEL_NAME


@pytest.fixture(scope="session")
def test_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
