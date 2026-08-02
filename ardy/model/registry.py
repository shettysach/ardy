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
import re

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
DEFAULT_TEXT_ENCODER_URL = "http://127.0.0.1:9550/"

# --- Aliases kept for imports elsewhere (ardy.model.loading re-exports these) --
# Kept as a public alias for callers that use the upstream registry API.
MODEL_NAMES = dict(MODELS)
# a modelname is valid if it is a nickname or a full folder name
AVAILABLE_MODELS = list(MODELS) + list(dict.fromkeys(MODELS.values()))
ARDY_MODELS = list(MODELS)
TMR_MODELS: list[str] = []

# Released-style G1 folder name.
_NAME_PATTERN = re.compile(r"ardy-(g1)-.*horizon(\d+)$", re.IGNORECASE)


def parse_model_name(folder: str):
    """``(skeleton, horizon)`` parsed from a released-style folder name.

    Returns e.g. ``("g1", 52)`` for ``"ARDY-G1-RP-25FPS-Horizon52"`` (case-insensitive), or
    ``None`` when the name does not follow the released naming scheme (e.g. a local training-run
    folder).
    """
    m = _NAME_PATTERN.match(folder)
    if not m:
        return None
    return m.group(1).lower(), int(m.group(2))


def resolve_model_name(name: str, default_family=None, checkpoints_dir=None) -> str:
    """Return the released folder / repo name for a nickname or full name.

    Accepts a nickname (``"g1"``, ``"g18"``, or ``"g152"``), the full G1
    folder name (case-insensitive). ``default_family`` is ignored (kept for
    call-site compatibility).

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
