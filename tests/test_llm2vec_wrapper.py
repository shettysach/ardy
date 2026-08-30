from types import SimpleNamespace

import pytest
import torch

import ardy.model.llm2vec.llm2vec_wrapper as wrapper_module
from ardy.model.llm2vec import LLM2VecEncoder


def test_encoder_loads_assembled_checkpoint_with_requested_placement(
    monkeypatch, tmp_path
) -> None:
    received: dict[str, object] = {}

    class FakeModel:
        def eval(self):
            return self

        def parameters(self):
            return []

        def encode(self, *args, **kwargs):
            received["encode_args"] = args
            received["encode_kwargs"] = kwargs
            return torch.ones((1, 4096), dtype=torch.float64)

    monkeypatch.setattr(
        wrapper_module.LLM2Vec,
        "from_pretrained",
        lambda **kwargs: received.update(kwargs) or FakeModel(),
    )

    encoder = LLM2VecEncoder(tmp_path, device="cuda:0")
    embedding = encoder.encode("touch the box")

    assert received["base_model_name_or_path"] == str(tmp_path)
    assert received["peft_model_name_or_path"] is None
    assert received["device_map"] == {"": "cuda:0"}
    assert received["encode_args"] == (["touch the box"],)
    assert received["encode_kwargs"] == {
        "batch_size": 1,
        "show_progress_bar": False,
        "device": "cuda:0",
    }
    assert embedding.shape == (4096,)
    assert embedding.dtype is torch.float32
    assert embedding.is_contiguous()


def test_encoder_rejects_invalid_device(tmp_path) -> None:
    with pytest.raises(ValueError, match="device must"):
        LLM2VecEncoder(tmp_path, device="mps")


def test_encoder_rejects_incorrect_embedding_dimension(monkeypatch, tmp_path) -> None:
    model = SimpleNamespace(
        eval=lambda: model,
        parameters=list,
        encode=lambda *args, **kwargs: torch.ones((1, 7)),
    )
    monkeypatch.setattr(
        wrapper_module.LLM2Vec,
        "from_pretrained",
        lambda **kwargs: model,
    )

    encoder = LLM2VecEncoder(tmp_path)
    with pytest.raises(ValueError, match=r"\[4096\]"):
        encoder.encode("walk")
