# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the DIRECT-VIDEO BLM IMITATION experiment (no IDM): same straight-forward-swim
task/reward/action space as SalmonSwimPCACfg (all-PC, K=20, matching the BLM+RL Reach10
baseline's controller), but with a REDUCED, video-plausible observation space --
[x, y, vx, vy, psi, a_1/a_max_1, .., a_20/a_max_20] (25-dim) instead of the live sim's usual
body-frame velocities + joint state + heading/up vectors (46-dim for this task's own base
class). This is "option 2" from the obs-space-mismatch analysis: a genuinely new, reduced
env whose observation really is only what a top-view video can supply, so a policy can be
behaviorally cloned directly from real-ZeF video state/action pairs
(scripts/zef_pca_rl/build_video_blm_bc_dataset.py) with NO inverse-dynamics model, then
optionally PPO-fine-tuned on this SAME reduced obs (see salmon_swim_pca_videostate_env.py).

Because there is no navigation target in a free-swimming video, this task keeps the
straight-swim reward (forward velocity - drift - yaw - energy), NOT the Reach10 target-reach
reward -- a target-reach reward would need target_dir_b/target_dist to be OBSERVABLE, which a
video-derived state cannot provide (the fine-tuning task must be solvable from what the
policy actually sees).
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .salmon_swim_pca_cfg import SalmonSwimPCACfg


@configclass
class SalmonSwimPCAVideoStateCfg(SalmonSwimPCACfg):
    """All-PC controller (matches the BLM+RL baseline's K=20), reduced video-plausible obs."""

    pca_num_modes = 20
