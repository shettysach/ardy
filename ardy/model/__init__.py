# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ARDY model package: main model class, text encoders, and loading utilities."""

from typing import TYPE_CHECKING

from .ardy_model import Ardy
from .load_model import load_model
from .loading import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_TEXT_ENCODER_URL,
    MODEL_NAMES,
    load_checkpoint_state_dict,
)

if TYPE_CHECKING:
    from .llm2vec import LLM2VecEncoder

__all__ = [
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "DEFAULT_TEXT_ENCODER_URL",
    "MODEL_NAMES",
    "Ardy",
    "LLM2VecEncoder",
    "load_checkpoint_state_dict",
    "load_model",
]


def __getattr__(name: str):
    """Load the heavyweight local text encoder only when it is requested."""
    if name == "LLM2VecEncoder":
        from .llm2vec import LLM2VecEncoder

        globals()[name] = LLM2VecEncoder
        return LLM2VecEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
