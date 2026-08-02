# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate G1 motions from checkpoint-specific ARDY conditioning.

The ``--conditioning`` NPZ must contain ``text_feat`` with shape ``[B, 1,
4096]`` and boolean ``text_pad_mask`` with shape ``[B, 1]`` for the released
G1 checkpoint. ARDY deliberately does not encode prompt text itself.
"""

import argparse
import os

import numpy as np
import torch

from ardy.exports.mujoco import MujocoQposConverter
from ardy.model import DEFAULT_MODEL, load_model
from ardy.model.loading import get_env_var
from ardy.model.registry import resolve_model_name
from ardy.motion_rep.tools import length_to_mask
from ardy.tools import seed_everything, to_numpy


def parse_args():
    parser = argparse.ArgumentParser(description="Generate G1 motion from ARDY text conditioning")
    parser.add_argument(
        "--conditioning",
        required=True,
        help="NPZ file containing text_feat and text_pad_mask from the compatible encoder service.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="G1 model nickname or full checkpoint folder name.")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds (default: 5.0).")
    parser.add_argument("--num-samples", type=int, default=1, help="Batch size in the conditioning file (default: 1).")
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=None,
        help="Number of diffusion steps, at most the checkpoint's num_base_steps.",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output stem. Bare names are written under outputs/; pass a path to write elsewhere.",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=None,
        help="History frames visible to each autoregressive step; must be a token-size multiple.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible sampling.")
    parser.add_argument(
        "--cfg-weight",
        type=float,
        nargs="+",
        default=[2.0, 2.0],
        help="CFG scale: one text weight or [text_weight, constraint_weight].",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=None,
        help="Required local checkpoint root. Falls back to CHECKPOINTS_DIR.",
    )
    return parser.parse_args()


def _default_history_frames(fps: float, gen_horizon_len: int, num_frames_per_token: int) -> int:
    max_window_len = (int(10 * fps) // num_frames_per_token) * num_frames_per_token
    return ((max_window_len - gen_horizon_len) // num_frames_per_token) * num_frames_per_token


def _resolve_output_base(path: str, default_dir: str = "outputs") -> str:
    return path if os.path.dirname(path) else os.path.join(default_dir, path)


def _single_file_path(path: str, ext: str) -> str:
    if not path.endswith(ext):
        path = path.rstrip(os.sep) + ext
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def _output_dir(path: str) -> tuple[str, str]:
    folder = os.path.splitext(path)[0] if os.path.splitext(path)[1] else path
    os.makedirs(folder, exist_ok=True)
    return folder, os.path.basename(folder.rstrip(os.sep))


def _select_sample(output: dict, index: int, n_samples: int) -> dict:
    return {
        key: (value[index] if hasattr(value, "shape") and value.shape[:1] == (n_samples,) else value)
        for key, value in output.items()
    }


def load_conditioning(path: str, device: str, num_samples: int, expected_shape: tuple[int, int]):
    """Load the encoder-service conditioning payload and validate its contract."""
    with np.load(path) as data:
        missing = {"text_feat", "text_pad_mask"}.difference(data.files)
        if missing:
            raise ValueError(f"Conditioning file {path!r} is missing keys: {sorted(missing)}.")
        text_feat = data["text_feat"]
        text_pad_mask = data["text_pad_mask"]

    expected_feature_shape = (num_samples, *expected_shape)
    if text_feat.shape != expected_feature_shape:
        raise ValueError(
            f"text_feat must have shape {expected_feature_shape} for this request, got {text_feat.shape}."
        )
    if text_pad_mask.shape != expected_feature_shape[:2]:
        raise ValueError(
            f"text_pad_mask must have shape {expected_feature_shape[:2]} for this request, got {text_pad_mask.shape}."
        )
    if not np.issubdtype(text_feat.dtype, np.floating):
        raise ValueError(f"text_feat must be floating-point, got {text_feat.dtype}.")

    return (
        torch.as_tensor(text_feat, device=device),
        torch.as_tensor(text_pad_mask, device=device, dtype=torch.bool),
    )


def save_motion_npz(path: str, motion_dict: dict, fps: float) -> None:
    arrays = {key: np.asarray(value) for key, value in motion_dict.items()}
    arrays["fps"] = np.asarray(fps)
    np.savez(path, **arrays)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError(f"--num-samples must be >= 1, got {args.num_samples}.")
    if len(args.cfg_weight) == 1:
        cfg_weight = float(args.cfg_weight[0])
    elif len(args.cfg_weight) == 2:
        cfg_weight = (float(args.cfg_weight[0]), float(args.cfg_weight[1]))
    else:
        raise ValueError("--cfg-weight expects one float or two floats.")

    checkpoints_dir = args.checkpoints_dir or get_env_var("CHECKPOINTS_DIR")
    if not checkpoints_dir:
        raise ValueError("A local checkpoint directory is required; use --checkpoints-dir or CHECKPOINTS_DIR.")
    resolved_model = resolve_model_name(args.model, checkpoints_dir=checkpoints_dir)
    model = load_model(resolved_model, device=device, checkpoints_dir=checkpoints_dir)
    fps = model.motion_rep.fps
    num_frames = int(args.duration * fps)

    num_base_steps = int(model.diffusion.num_base_steps)
    diffusion_steps = args.diffusion_steps if args.diffusion_steps is not None else num_base_steps
    if not 1 <= diffusion_steps <= num_base_steps:
        raise ValueError(f"--diffusion-steps must be between 1 and {num_base_steps}; got {diffusion_steps}.")

    patch = model.num_frames_per_token
    history_frames = args.history_frames
    if history_frames is None:
        history_frames = _default_history_frames(fps, model.gen_horizon_len, patch)
    elif history_frames < patch or history_frames % patch != 0:
        raise ValueError(f"--history-frames must be a positive multiple of {patch}.")

    if args.seed is not None:
        seed_everything(args.seed)
    lengths = torch.tensor([num_frames] * args.num_samples, device=device)
    text_feat, text_pad_mask = load_conditioning(
        args.conditioning,
        device,
        args.num_samples,
        tuple(model.denoiser.llm_shape),
    )

    print(f"Using device: {device}")
    print(f"Loaded model: {resolved_model}")
    print(f"Generating {num_frames} frames at {fps} FPS from {args.conditioning}")
    with torch.no_grad():
        motion = model(
            num_frames,
            num_denoising_steps=diffusion_steps,
            pad_mask=length_to_mask(lengths),
            first_heading_angle=torch.zeros(args.num_samples, device=device),
            motion_mask=None,
            observed_motion=None,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            cfg_weight=cfg_weight,
            crop_history_length=history_frames,
        )
        output = model.motion_rep.inverse(motion, is_normalized=True)

    output = to_numpy(output)
    n_samples = int(output["posed_joints"].shape[0])
    output_base = _resolve_output_base(args.output)
    converter = MujocoQposConverter(model.skeleton)
    qpos = converter.dict_to_qpos(output, device)

    if n_samples == 1:
        npz_path = _single_file_path(output_base, ".npz")
        csv_path = _single_file_path(output_base, ".csv")
        save_motion_npz(npz_path, _select_sample(output, 0, n_samples), fps)
        converter.save_csv(qpos, csv_path)
        print(f"Saved {npz_path} and {csv_path}")
    else:
        out_dir, base_name = _output_dir(output_base)
        for index in range(n_samples):
            save_motion_npz(
                os.path.join(out_dir, f"{base_name}_{index:02d}.npz"),
                _select_sample(output, index, n_samples),
                fps,
            )
        converter.save_csv(qpos, os.path.join(out_dir, base_name + ".csv"))
        print(f"Saved {n_samples} samples under {out_dir}/")


if __name__ == "__main__":
    main()
