# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model registry: map nicknames to local released-model folders.

Released models are organized by skeleton and generation horizon (in frames).
``load_model`` accepts:

- the G1 nickname (``"g1"``) — resolves to its default horizon,
- a G1+horizon nickname (``"g18"`` or ``"g152"``),
- the full G1 checkpoint folder name.
"""

import os

# skeleton -> generation horizon (frames) -> released folder name
MODELS_BY_SKELETON = {
    "g1": {
        52: "ARDY-G1-RP-25FPS-Horizon52",
        8: "ARDY-G1-RP-25FPS-Horizon8",
    },
}

# Horizon the bare G1 nickname resolves to.
DEFAULT_HORIZON = {"g1": 52}

# G1 nickname -> released folder name, plus the bare default-horizon nickname.
MODELS = {
    f"{skeleton}{horizon}": folder
    for skeleton, by_horizon in MODELS_BY_SKELETON.items()
    for horizon, folder in by_horizon.items()
}
MODELS.update({skeleton: MODELS_BY_SKELETON[skeleton][DEFAULT_HORIZON[skeleton]] for skeleton in MODELS_BY_SKELETON})

DEFAULT_MODEL = "g1"
def resolve_model_name(name: str, checkpoints_dir=None) -> str:
    """Return the released folder / repo name for a nickname or full name.

    Accepts a nickname (``"g1"``, ``"g18"``, or ``"g152"``), the full G1
    folder name (case-insensitive).

    When ``checkpoints_dir`` is given, the valid model set is whatever folders live there — not just
    the released models — so a name matching a folder in it is accepted as-is (nicknames still
    resolve via the registry).
    """
    if name in MODELS:
        return MODELS[name]
    # Full folder name: match case-insensitively and return the canonical
    # casing because local folder lookups are case-sensitive.
    bare = name
    canonical = {folder.lower(): folder for folder in MODELS.values()}
    if bare.lower() in canonical:
        return canonical[bare.lower()]
    if checkpoints_dir and os.path.isdir(os.path.join(checkpoints_dir, name)):
        return name
    raise ValueError(
        f"Unknown model {name!r}. Choose a nickname {list(MODELS)} "
        f"or a full name {list(dict.fromkeys(MODELS.values()))}."
    )
