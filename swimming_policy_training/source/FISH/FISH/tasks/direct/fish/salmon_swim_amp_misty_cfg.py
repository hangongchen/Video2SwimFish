# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Swim-to-target + AMP on the Meshy 'Misty Minnow' fish (misty_minnow_fixed USD).

Same task/env as SalmonSwimAMPCfg (old swim-to-target reward + obs, + w_amp*r_style, autoskel-style
AMP disc borrowed from SalmonAMPTankEnv) but the AGENT is the finned Meshy fish:
  - its own USD (`misty_minnow_fixed/fish_articulated.usd`, head/tail-corrected pipeline output),
  - its own per-bone hydro (8 bones; `precompute_misty_hydro.py`),
  - upright spawn (+90 deg about X so dorsoventral Y -> world Z, like the autoskel fish),
  - body_forward_sign = +1: this fish's HEAD is at +X, so the POLICY's heading_dir obs points at the
    real nose instead of the tail. The AMP feature (head-anchored via its own reversal) is untouched.
"""

from __future__ import annotations

import copy
from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from .salmon_swim_amp_cfg import SalmonSwimAMPCfg
from .salmon_swim_cfg import SALMON_SWIM_ARTICULATION_CFG

_GEN = Path(__file__).resolve().parents[6] / "SimFishLib" / "fish_asset_pipeline" / "generated_usd_dataset"
_USD = _GEN / "misty_minnow_fixed" / "fish_articulated.usd"

_ART: ArticulationCfg = copy.deepcopy(SALMON_SWIM_ARTICULATION_CFG)
_ART.spawn.usd_path = str(_USD)
# upright spawn: the pipeline exports X=length, Y=dorsoventral(tall), Z=lateral(thin); identity quat
# lies the fish on its side, so rotate +90 deg about the length (X) axis -> Y (tall) maps to world Z.
_ART.init_state = ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), rot=(0.70710678, 0.70710678, 0.0, 0.0))


@configclass
class SalmonSwimAMPMistyCfg(SalmonSwimAMPCfg):
    robot_cfg: ArticulationCfg = _ART
    per_bone_hydro_path = str(_GEN / "misty_minnow_fixed" / "per_bone_hydro.npz")
    body_length = 0.5                # Meshy fish scaled to 0.5 m (for BL/s normalization)
    body_forward_sign = 1.0          # HEAD at +X -> policy heading_dir points at the nose (AMP path untouched)
    with_deformable = True           # AMP bend feature reads the FEM soft body
    # The Meshy fish names its deformable body 'mesh_001' (not the salmon's 'deformable_salmon'), so the
    # env's deformable-search token MUST be overridden or it finds no FEM (-> "kept-active 0 soft",
    # material not bound, and the AMP bend feature reads nothing). Matches precompute_misty_hydro.py.
    deformable_prim_token = "mesh_001"
