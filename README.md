# ARDY G1 inference runtime

This repository is a trimmed fork of NVIDIA ARDY containing the runtime needed
for text-conditioned Unitree G1 motion generation. Training, datasets,
evaluation, visualization, interactive demos, TensorRT/ONNX export, and the
MotionCorrection native extension have been removed.

The preserved inference path is:

```text
G1 checkpoint config and weights
    -> ARDY model loading and text conditioning
    -> autoregressive diffusion sampling
    -> ARDY motion-representation decoding
    -> MuJoCo G1 qpos [T, 36]
```

## Setup

ARDY requires Python 3.10 or newer, NumPy 2.4, and PyTorch 2.7 or newer. Install
a PyTorch build appropriate for the machine first, then install this repository:

```bash
pip install torch
pip install -e .
```

Local text encoding uses the gated Meta Llama 3 model through the upstream
LLM2Vec path. Request access to `meta-llama/Meta-Llama-3-8B-Instruct` and
authenticate with Hugging Face before the first run. The existing
`TEXT_ENCODER_MODE`, `TEXT_ENCODER_URL`, `TEXT_ENCODER_DEVICE`,
`TEXT_ENCODERS_DIR`, and `HUGGINGFACE_CACHE_DIR` environment variables remain
supported.

G1 checkpoints are downloaded from Hugging Face automatically. To use a local
checkpoint, place the released folder below a checkpoint directory and pass
`--checkpoints_dir` (or set `CHECKPOINTS_DIR`):

```text
checkpoints/
└── ARDY-G1-RP-25FPS-Horizon52/
    ├── config.yaml
    ├── denoiser.safetensors
    ├── tokenizer.safetensors
    └── stats/
```

## Inference example

The retained upstream command-line example generates an NPZ motion file and a
MuJoCo qpos CSV:

```bash
python scripts/generate.py \
  "A person walks forward." \
  --model g1 \
  --duration 5 \
  --output walk
```

For G1, the model runs at 25 FPS. The CSV contains 36 columns: root translation
(3), scalar-first root quaternion `(w, x, y, z)` (4), and 29 MuJoCo joint
coordinates.

Longer generations use ARDY's existing autoregressive path. `--history_frames`
controls how much generated history is supplied to each subsequent window.

## Runtime tree

- `ardy/model/`: checkpoint loading, text encoding, model definitions, and
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
