# ARDY G1 inference runtime

Trimmed fork of NVIDIA ARDY containing the runtime needed
for embedding-conditioned Unitree G1 motion generation. Training, datasets,
evaluation, visualization, interactive demos, text encoding, TensorRT/ONNX
export, and the MotionCorrection native extension have been removed.

The preserved inference path is:

```text
G1 checkpoint config and weights
    -> checkpoint-compatible text conditioning [B, 1, 4096]
    -> autoregressive diffusion sampling
    -> ARDY motion-representation decoding
    -> MuJoCo G1 qpos [T, 36]
```

## Setup

Requires Python 3.10 or newer, NumPy 2.4, and PyTorch 2.7 or newer. Install
a PyTorch build appropriate for the machine first, then install this repository:

```bash
pip install torch
pip install -e .
```

This fork does not understand text and contains no text encoder. Its released G1
checkpoint requires the pooled LLM2Vec representation it was trained on:
`text_feat` is floating-point `[B, 1, 4096]` and `text_pad_mask` is boolean
`[B, 1]`. Supply these from the separate encoder service as an NPZ with those
two keys. Other embedding models cannot be substituted merely by matching the
shape.

G1 checkpoints must be supplied locally; This fork does not download them at
runtime. Place the released folder below a checkpoint directory and pass
`--checkpoints-dir` (or set `CHECKPOINTS_DIR`):

```text
checkpoints/
└── ARDY-G1-RP-25FPS-Horizon52/
    ├── config.yaml
    ├── denoiser.safetensors
    ├── tokenizer.safetensors
    └── stats/
```

## Inference example

The retained command-line example consumes a compatible encoder-service NPZ and
generates an NPZ motion file and a MuJoCo qpos CSV:

```bash
python scripts/generate.py \
  --conditioning /path/to/conditioning.npz \
  --model g1 \
  --checkpoints-dir /path/to/checkpoints \
  --duration 5 \
  --output walk
```

For G1, the model runs at 25 FPS. The CSV contains 36 columns: root translation
(3), scalar-first root quaternion `(w, x, y, z)` (4), and 29 MuJoCo joint
coordinates.

Longer generations use ARDY's existing autoregressive path. `--history_frames`
controls how much generated history is supplied to each subsequent window.

## Runtime tree

- `ardy/model/`: checkpoint loading, model definitions, and
  diffusion/autoregressive sampling.
- `ardy/motion_rep/`: normalization, representation transforms, and decoding.
- `ardy/skeleton/`: shared skeleton primitives and the G1 definition.
- `ardy/exports/`: G1 ARDY-to-MuJoCo qpos conversion.
- `ardy/assets/skeletons/g1skel34/`: G1 neutral joints and the XML metadata used
  by the qpos converter.
- `scripts/generate.py`: the single retained inference example.

The checkpoint's `config.yaml` is part of the runtime interface: its Hydra
`_target_` entries construct the model classes retained here, and its `stats/`
files provide inference normalization data.

## License

Source code remains under the Apache 2.0 license in `LICENSE`. Model checkpoints
and their upstream dependencies have separate licenses; see `ATTRIBUTIONS.MD`
and the checkpoint repository.
