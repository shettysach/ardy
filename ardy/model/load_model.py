# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load embedding-conditioned ARDY models from local checkpoints."""

from pathlib import Path

from omegaconf import OmegaConf

from .loading import DEFAULT_MODEL, get_env_var, instantiate_from_dict
from .registry import resolve_model_name


def load_model(
    modelname: str | None = None,
    *,
    device=None,
    eval_mode: bool = True,
    return_config: bool = False,
    checkpoints_dir: str | None = None,
):
    """Load a released ARDY model from a local checkpoint directory.

    ARDY is embedding-conditioned: the returned model requires ``text_feat``
    and ``text_pad_mask`` during sampling and never constructs a text encoder.

    Args:
        modelname: G1 nickname or checkpoint folder name; defaults to ``g1``.
        device: Target device for the motion model (for example, ``"cuda"``).
        eval_mode: Set the model to evaluation mode.
        return_config: Return ``(model, config)`` instead of only the model.
        checkpoints_dir: Local directory containing released checkpoint folders.
            Falls back to ``CHECKPOINTS_DIR`` and is required.
    """
    if modelname is None:
        modelname = DEFAULT_MODEL

    checkpoints_dir = checkpoints_dir or get_env_var("CHECKPOINTS_DIR")
    if not checkpoints_dir:
        raise ValueError(
            "A local checkpoint directory is required. Pass checkpoints_dir or set CHECKPOINTS_DIR."
        )

    full_name = resolve_model_name(modelname, checkpoints_dir=checkpoints_dir)
    model_path = Path(checkpoints_dir) / full_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model {full_name!r} not found under CHECKPOINTS_DIR {checkpoints_dir!r}.")

    model_config_path = model_path / "config.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"The model folder exists but config.yaml is missing: {model_config_path}")

    model_conf = OmegaConf.load(model_config_path)
    runtime_conf = OmegaConf.create({"checkpoint_dir": str(model_path)})
    model_cfg = OmegaConf.to_container(OmegaConf.merge(model_conf, runtime_conf), resolve=True)
    model_cfg.pop("checkpoint_dir", None)
    model_cfg.pop("text_encoder", None)

    model = instantiate_from_dict(model_cfg, overrides={"device": device})
    if eval_mode:
        model = model.eval()
    if return_config:
        return model, model_cfg
    return model
