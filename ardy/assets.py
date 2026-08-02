# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS_ROOT = PACKAGE_ROOT / "assets"
SKELETONS_ROOT = ASSETS_ROOT / "skeletons"


def skeleton_asset_path(*parts: str) -> Path:
    return SKELETONS_ROOT.joinpath(*parts)
