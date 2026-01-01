# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""2-PC PCA-manifold PPO swim environment (see salmon_swim_pca_cfg.py for the full design).

Control pipeline:  PPO -> [da1, da2] -> bounded coefficient state a(t) -> kappa(s,t) =
mean + a@V -> calibrated curvature->joint map -> PD position targets -> FEM fish -> panel water.

Everything except the action/observation/reward/done logic is inherited from SalmonSwimEnv
(panel hydro, FEM guards, drives, reset). The target machinery the base class sets up is
inert here: no reward or termination reads it, and no marker is spawned (cfg flags).

The kappa->joint map is AFFINE and precomputed once:
    q(a) = q0 + a @ W,   q0 = M (kappa_mean - kappa_rest),  W = (M V^T)^T,
    M = (Phi^T Phi + lam I)^-1 Phi^T        (the ridge LS the playback validation used)
so the policy action literally moves the fish along the real-fish curvature manifold.
Commanded kappa is recoverable exactly as mean + a@V for logging/eval.
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils

from .salmon_swim_env import SalmonSwimEnv
from .salmon_swim_pca_cfg import SalmonSwimPCACfg


class SalmonSwimPCAEnv(SalmonSwimEnv):
    cfg: SalmonSwimPCACfg

    # ------------------------------------------------------------------ spaces
    def _configure_gym_env_spaces(self):
        # base: resolves _control_joint_ids from cfg.control_dof_suffix (the 7 ':1' joints)
        # and sizes _pos_targets/_num_actions to them -- keep that (we still command 7 joints).
        super()._configure_gym_env_spaces()
        self._num_pca = int(self.cfg.pca_num_modes)
        nj = int(self._control_joint_ids.shape[0])
        # obs: body lin vel (3) + body ang vel (3) + controlled joint pos (nj) + vel (nj)
        #      + heading dir (3) + up dir (3) + pca coeffs a (2, normalized by a_max)
        self._obs_dim = 3 + 3 + 2 * nj + 3 + 3 + self._num_pca
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(self._obs_dim,))})
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"], self.num_envs)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_pca,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.actions = torch.zeros((self.num_envs, self._num_pca), device=self.device)
        print(f"[SalmonSwimPCA] action space = {self._num_pca} PCA coefficient rates; "
              f"obs = {self._obs_dim} dims; {nj} joints driven", flush=True)

    # ------------------------------------------------------------------ init
    def __init__(self, cfg: SalmonSwimPCACfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        E = self.num_envs
        self._build_pca_mapping()
        self._a = torch.zeros(E, self._num_pca, device=dev)              # coefficient state
        self._ema_vlat = torch.zeros(E, 2, device=dev)                   # EMA of body-frame (vy, vz)
        self._ema_yaw = torch.zeros(E, device=dev)                       # body yaw-rate EMA
        self._ema_alpha = float(min(1.0, (self.cfg.sim.dt * self.cfg.decimation) / self.cfg.ema_tau_s))
        self._fwd_sign = float(getattr(self.cfg, "body_forward_sign", -1.0))

    def _build_pca_mapping(self):
        """Load the FROZEN artifacts and collapse kappa->q to an affine map (no re-derivation)."""
        c = self.cfg
        b = np.load(c.pca_basis_path)
        r = np.load(c.pca_calib_path)
        K = self._num_pca
        V = b["components"][:K].astype(np.float64)                       # (K, 20)
        mean_k = b["mean"].astype(np.float64)
        Phi = r["Phi"].astype(np.float64)                                # (20, 7) head-first joints
        k_rest = r["kappa_rest"].astype(np.float64)
        if str(getattr(c, "pca_phi_mode", "calib")) == "analytic":
            # geometric turning-angle Phi (see scripts/benchmark/analytic_phi.py): the measured Phi commands zigzag
            # joint patterns for smooth curvature targets (rank-2, cond 1e6; 2026-09-28)
            import sys as _sys; _rr = __import__("pathlib").Path(__file__).resolve().parents[6]; _sys.path.insert(0, str(_rr / "scripts" / "benchmark")); _sys.path.insert(0, str(_rr / "scripts" / "zef_playback"))
            from analytic_phi import analytic_phi as _aphi; import curvature_utils as _cu
            _geo = _cu.rest_geometry(str(c.panel_hydro_path), head_sign=int(r["head_sign"]) if "head_sign" in r.files else 1)
            Phi, _sj = _aphi(_geo, Phi.shape[1], phi_calib=Phi); k_rest = np.zeros_like(k_rest)
            print(f"[SalmonSwimPCA] ANALYTIC Phi: joints at s={np.round(_sj, 2).tolist()}, cond={np.linalg.cond(Phi):.1f}", flush=True)
        lam = float(c.pca_ridge) * float(np.mean(np.diag(Phi.T @ Phi)))
        M = np.linalg.solve(Phi.T @ Phi + lam * np.eye(Phi.shape[1]), Phi.T)   # (7, 20)
        q0 = M @ (mean_k - k_rest)                                       # (7,) head-first order
        W = (M @ V.T).T                                                  # (K, 7) head-first order

        # permute Phi's joint order (head-first names, stored in the calibration npz) onto THIS
        # env's _control_joint_ids order -- never assume the orders agree.
        calib_names = [str(n) for n in r["joint_names"]]
        phi_names = [calib_names[i] for i in r["fam1"]]                  # head-first ':1' names
        env_names = [self.robot.data.joint_names[i] for i in self._control_joint_ids]
        assert set(phi_names) == set(env_names), (
            f"calibrated joints {phi_names} != controlled joints {env_names}")
        perm = [phi_names.index(n) for n in env_names]
        q0 = q0[perm]
        W = W[:, perm]

        # data-driven bounds from the ZeF coefficient distribution
        coeffs = b["coeffs"]
        ok = np.isfinite(coeffs).all(1)
        a_data = coeffs[ok][:, :K]
        self._a_max = torch.tensor(np.percentile(np.abs(a_data), float(c.pca_amp_percentile), axis=0),
                                   device=self.device, dtype=torch.float32)
        # per-control-step (1/30 s = 2 data frames at 60 fps) coefficient change, valid pairs only
        da = coeffs[:-2, :K] - coeffs[2:, :K]
        da = da[np.isfinite(da).all(1)]
        self._da_max = torch.tensor(np.percentile(np.abs(da), float(c.pca_rate_percentile), axis=0),
                                    device=self.device, dtype=torch.float32)

        self._q0 = torch.tensor(q0, device=self.device, dtype=torch.float32)
        self._W = torch.tensor(W, device=self.device, dtype=torch.float32)
        # the raw decoder (permuted to this env's joint order), stored so OTHER curvature-space
        # controllers (e.g. the ZeF-calibrated CPG baseline) can share the IDENTICAL kappa->q map
        self._Mdec = torch.tensor(M[perm, :], device=self.device, dtype=torch.float32)   # (7, 20)
        # kappa reconstruction operators kept for logging/eval (kappa_cmd = mean + a @ V)
        self._V = torch.tensor(V, device=self.device, dtype=torch.float32)
        self._kappa_mean = torch.tensor(mean_k, device=self.device, dtype=torch.float32)
        print(f"[SalmonSwimPCA] manifold loaded: a_max={self._a_max.cpu().numpy().round(2).tolist()} "
              f"(p{c.pca_amp_percentile:.0f}|a|), da_max/step={self._da_max.cpu().numpy().round(2).tolist()} "
              f"(p{c.pca_rate_percentile:.0f}|da|); |W| max joint gain="
              f"{np.abs(W).max():.3f} rad/unit; q0 max={np.rad2deg(np.abs(q0)).max():.1f} deg", flush=True)

    # ------------------------------------------------------------------ control
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = actions.view(self.num_envs, self._num_pca)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        self.actions = actions
        # bounded-rate coefficient dynamics: temporal consistency is structural, not learned
        self._a = torch.clamp(self._a + actions * self._da_max, -self._a_max, self._a_max)
        q = self._q0.unsqueeze(0) + self._a @ self._W                     # (E, 7)
        lo = self._soft_joint_limits[:, self._control_joint_ids, 0]
        hi = self._soft_joint_limits[:, self._control_joint_ids, 1]
        self._pos_targets = torch.clamp(q, lo, hi)
        # base _apply_action does the rest each physics substep: set_joint_position_target +
        # panel hydro wrench + FEM nodal-velocity guard.

    # ------------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        d = self.robot.data
        root_quat = d.root_state_w[:, 3:7]
        fwd_b = torch.tensor([self._fwd_sign, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        heading_dir = math_utils.quat_apply(root_quat, fwd_b)
        heading_dir = heading_dir / heading_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)
        up_b = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        up_dir = math_utils.quat_apply(root_quat, up_b)
        jp = d.joint_pos[:, self._control_joint_ids] * self.cfg.obs_scales.joint_pos
        jv = d.joint_vel[:, self._control_joint_ids] * self.cfg.obs_scales.joint_vel
        obs = torch.cat((
            d.root_lin_vel_b * self.cfg.obs_scales.root_lin_vel,
            d.root_ang_vel_b * self.cfg.obs_scales.root_ang_vel,
            jp, jv, heading_dir, up_dir,
            self._a / self._a_max,                                       # normalized coeff state
        ), dim=-1)
        return {"policy": torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)}

    # ------------------------------------------------------------------ reward
    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        d = self.robot.data
        BL = self._body_length
        v_b = d.root_lin_vel_b                                           # body frame
        v_fwd_bl = self._fwd_sign * v_b[:, 0] / BL                       # signed BL/s
        yaw_rate = d.root_ang_vel_b[:, 1]                                # about body-up (local +Y)
        # EMA the oscillating quantities COMPONENT-WISE: an undulatory gait NECESSARILY wags
        # laterally and in yaw every beat, and those oscillations CANCEL in a signed average --
        # only sustained drift/turn survives the EMA. (EMA of the |magnitude| would NOT cancel
        # and would punish swimming itself -- measured -0.2/step on a clean scripted gait.)
        al = self._ema_alpha
        self._ema_vlat = (1 - al) * self._ema_vlat + al * (v_b[:, 1:3] / BL)
        self._ema_yaw = (1 - al) * self._ema_yaw + al * yaw_rate
        jv = d.joint_vel[:, self._control_joint_ids]
        energy = (jv ** 2).mean(dim=1)
        drift = self._ema_vlat.norm(dim=1)
        reward = (c.w_forward * v_fwd_bl
                  - c.w_lateral * drift
                  - c.w_yawrate * self._ema_yaw.abs()
                  - c.w_energy * energy)
        self.extras.setdefault("log", {}).update({
            "v_fwd_bl": v_fwd_bl.mean().detach(),
            "v_lat_ema_bl": drift.mean().detach(),
            "yaw_ema": self._ema_yaw.abs().mean().detach(),
            "energy_jv2": energy.mean().detach(),
            "a1_abs": self._a[:, 0].abs().mean().detach(),
            "a2_abs": self._a[:, 1].abs().mean().detach(),
            "a_sat_frac": (self._a.abs() >= 0.999 * self._a_max).float().mean().detach(),
            "reward_mean": reward.mean().detach(),
        })
        return torch.nan_to_num(reward)

    # ------------------------------------------------------------------ dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        root_state = self.robot.data.root_state_w
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        local_pos = root_state[:, 0:3] - self.scene.env_origins
        out_of_bounds = torch.linalg.norm(local_pos, dim=1) > self.cfg.root_position_limit
        joint_blowup = self.joint_vel.abs().max(dim=1).values > self.cfg.joint_velocity_limit
        nonfinite = (~torch.isfinite(root_state).all(dim=1)) \
            | (~torch.isfinite(self.joint_pos).all(dim=1)) | (~torch.isfinite(self.joint_vel).all(dim=1))
        # FEM nodal guard (same pattern as the base env; a pure-FEM blow-up misses the joints)
        self._fem_blowup = torch.zeros_like(joint_blowup)
        _fvl = getattr(self.cfg, "fem_velocity_limit", None)
        if _fvl is not None and getattr(self, "_soft_view", None) is not None:
            try:
                _nvel = self._soft_view.get_simulation_mesh_nodal_velocities()
                _nvmax = torch.nan_to_num(_nvel, nan=float("inf")).abs().flatten(1).amax(dim=1)
                self._fem_blowup = _nvmax > float(_fvl)
            except Exception:  # noqa: BLE001
                pass
        nonfinite = nonfinite | self._fem_blowup
        # throughput-print bookkeeping (base _print_throughput reads these when debug_print)
        _blew = joint_blowup | nonfinite
        self._blew_now = int(_blew.sum())
        _jvm = self.joint_vel.abs().max()
        self._jvel_max = float(_jvm) if torch.isfinite(_jvm) else float("inf")
        terminated = out_of_bounds | joint_blowup | nonfinite
        self.extras.setdefault("log", {}).update({
            "blowups": _blew.float().sum().detach(),
            "oob": out_of_bounds.float().sum().detach(),
        })
        return terminated, time_out

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        idx = slice(None) if env_ids is None else env_ids
        if hasattr(self, "_a"):                                          # base __init__ resets early
            self._a[idx] = 0.0
            self._ema_vlat[idx] = 0.0
            self._ema_yaw[idx] = 0.0
