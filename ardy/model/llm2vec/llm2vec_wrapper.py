# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public LLM2Vec encoder API for ARDY text conditioning."""

from pathlib import Path

import torch

from .llm2vec import LLM2Vec


class LLM2VecEncoder:
    """Encode one prompt using an assembled AeroEx LLM2Vec checkpoint."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        llm_dim: int = 4096,
    ) -> None:
        if device != "cpu" and device != "cuda" and not device.startswith("cuda:"):
            raise ValueError("device must be 'cpu', 'cuda', or 'cuda:N'")
        self.llm_dim = llm_dim
        self.device = device
        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=str(model_path),
            peft_model_name_or_path=None,
            device_map={"": device},
        )
        # Quantized loaders own placement.  LLM2Vec's original encode path
        # normally calls ``.to(device)``; keep its tokenization/pooling intact
        # while skipping only that unsafe post-load relocation.
        self.model._device_managed = True
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def encode(self, text: str) -> torch.Tensor:
        """Return the checkpoint's pooled sentence embedding as CPU float32."""
        encoded = self.model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            device=self.device,
        )
        embedding = torch.as_tensor(encoded[0]).float().contiguous()
        if embedding.shape != (self.llm_dim,):
            raise ValueError(
                f"LLM2Vec encoder must produce [{self.llm_dim}], got {tuple(embedding.shape)}"
            )
        return embedding
