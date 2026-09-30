"""Apply the trained IDM to real-fish expert curvature sequences (no action labels) to
produce pseudo-action labels for BC pre-training (Step 3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .bco_config import BCOConfig
from .idm import InverseDynamicsModel


def _decode_curvature_to_joint_pos(
    kappa: np.ndarray, calib_path: str, Mdec: torch.Tensor, device: str | torch.device,
) -> torch.Tensor:
    """kappa (T, N) -> joint_pos (T, nj) via q(t) = Mdec @ (kappa(t) - kappa_rest), the IDENTICAL
    ridge-regression curvature->joint decoder salmon_swim_pca_env.SalmonSwimPCAEnv uses (Mdec is
    read straight off the live env instance, already permuted to its OWN controlled-joint order --
    never re-derived here). kappa_rest itself isn't kept as an env attribute (only the
    already-offset q0 is), so it is reloaded directly from the SAME calibration artifact."""
    calib = np.load(calib_path)
    kappa_rest = calib["kappa_rest"].astype(np.float64)
    assert kappa.shape[-1] == kappa_rest.shape[0], (
        f"expert_curvatures has {kappa.shape[-1]} sample points per frame, but the calibration "
        f"decoder ({calib_path}) expects {kappa_rest.shape[0]} -- these must match.")
    kappa_t = torch.tensor(kappa, dtype=torch.float32, device=device)
    kappa_rest_t = torch.tensor(kappa_rest, dtype=torch.float32, device=device)
    q = (kappa_t - kappa_rest_t) @ Mdec.T.to(device)          # (T, nj)
    return q


def build_pseudo_labels(env, idm_model: InverseDynamicsModel, cfg: BCOConfig,
                         device: str | torch.device) -> dict[str, np.ndarray]:
    """Build the pseudo-labeled BC dataset from cfg.expert_curvatures.

    Args:
        env: the CONSTRUCTED (SalmonSwimJointReach10Env) gym env, used only to read its own
            obs layout / decoder constants -- never stepped here.
        idm_model: the trained InverseDynamicsModel (idm.py).
        cfg: BCOConfig (reads expert_curvatures / pseudo_action_clip).
        device: torch device.

    Returns:
        {"obs": (T-1, obs_dim) float32, "pseudo_action": (T-1, action_dim) float32}
    """
    e = env.unwrapped
    nj = int(e._control_joint_ids.shape[0])
    obs_dim = int(e._obs_dim)
    assert idm_model.obs_dim == obs_dim, (
        f"IDM was trained on obs_dim={idm_model.obs_dim}, but this env's obs_dim is {obs_dim} "
        "-- IDM and pseudo-labeling must use the SAME task/env.")
    assert idm_model.action_dim == nj

    kappa = np.load(cfg.expert_curvatures)
    assert kappa.ndim == 2, f"expert_curvatures must be (T, N), got shape {kappa.shape}"
    T = kappa.shape[0]
    assert T >= 2, "need at least 2 expert frames to form one (o_t, o_{t+1}) IDM pair"

    joint_pos = _decode_curvature_to_joint_pos(kappa, e.cfg.pca_calib_path, e._Mdec, device)  # (T, nj)
    # safety clamp: a video pose reconstructed via ridge regression can, at the extremes,
    # slightly overshoot the physical joint limits -- clip to what the sim can actually realize.
    lo = e._soft_joint_limits[0, e._control_joint_ids, 0]
    hi = e._soft_joint_limits[0, e._control_joint_ids, 1]
    joint_pos = torch.clamp(joint_pos, lo, hi)
    joint_pos = joint_pos * float(e.cfg.obs_scales.joint_pos)   # same scaling _get_observations applies

    # ---- DESIGN DECISION: zero-pad everything a curvature-only video CANNOT provide ----
    # A single free-swimming fish on video gives us body shape (curvature) per frame, decoded
    # above into the SAME joint_pos slot the live env observes. It gives us NOTHING ELSE this
    # env's observation vector contains: no root linear/angular velocity (no calibrated 3D
    # camera pose track), no joint velocity (explicitly out of scope per this pipeline's spec --
    # a finite difference of joint_pos WOULD be recoverable, but the spec calls for zero-padding
    # velocity too, so we do), and no target-relative direction/distance (there is no task target
    # in a passive video clip). Those slots are therefore zero, exactly matching this env's
    # observation ORDER (see SalmonSwimJointReach10Env._get_observations): [root_lin_vel_b(3),
    # root_ang_vel_b(3), joint_pos(nj), joint_vel(nj), heading_dir(3), up_dir(3),
    # target_dir_b(3), target_dist(1)].
    zeros3 = torch.zeros(T, 3, device=device)
    zeros1 = torch.zeros(T, 1, device=device)
    zero_jvel = torch.zeros(T, nj, device=device)
    obs = torch.cat([zeros3, zeros3, joint_pos, zero_jvel, zeros3, zeros3, zeros3, zeros1], dim=-1)
    assert obs.shape == (T, obs_dim), (obs.shape, obs_dim)

    with torch.no_grad():
        pseudo_action = idm_model(obs[:-1], obs[1:])
    pseudo_action = pseudo_action.clamp(-cfg.pseudo_action_clip, cfg.pseudo_action_clip)

    print(f"[pseudo_labeler] {T} expert frames -> {T - 1} pseudo-labeled (obs, action) pairs "
          f"(obs_dim={obs_dim}, action_dim={nj})", flush=True)
    return {
        "obs": obs[:-1].detach().cpu().numpy().astype(np.float32),
        "pseudo_action": pseudo_action.detach().cpu().numpy().astype(np.float32),
    }


def save_pseudo_labels(data: dict[str, np.ndarray], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, obs=data["obs"], pseudo_action=data["pseudo_action"])
    print(f"[pseudo_labeler] saved {data['obs'].shape[0]} pairs -> {path}", flush=True)


def load_pseudo_labels(path: str) -> dict[str, np.ndarray]:
    z = np.load(path)
    print(f"[pseudo_labeler] loaded {z['obs'].shape[0]} pairs <- {path}", flush=True)
    return {"obs": z["obs"], "pseudo_action": z["pseudo_action"]}
