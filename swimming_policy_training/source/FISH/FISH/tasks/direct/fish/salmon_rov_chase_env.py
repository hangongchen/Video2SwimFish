"""BlueROV2 gripper-chase task: the RL agent drives a BlueROV2 (6-DOF body wrench, thruster
abstraction); the fish is driven INTERNALLY by a FROZEN pretrained ZeF-angle PCA policy and
acts as a moving prey. Success = the ROV touches the fish with its gripper (real collisions
are enabled: fish bone convex hulls + ROV hull/jaw boxes; detection = gripper-tip within
cfg.contact_radius of the nearest fish bone, a contact-grade threshold for this body).
On success (or 15 s timeout) the episode resets: fish re-spawns via the normal reset, the
ROV teleports to a random bearing 0.9-1.4 m away, facing the fish.

Design choice vs "dual-agent RL": single learning agent + frozen opponent policy -- same
setup the user suggested; the fish policy network runs inside the env (rl_games player).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
import torch
import os as _os
import yaml

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils import configclass
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from .salmon_swim_reach10_env import SalmonSwimPCAReach10Env
from .salmon_swim_reach10_cfg import SalmonSwimPCAReach10Cfg

_REPO = Path(__file__).resolve().parents[6]


@configclass
class SalmonROVChaseCfg(SalmonSwimPCAReach10Cfg):
    episode_length_s = 45.0
    # fish behaviour: frozen ZeF-angle policy chasing its own ZeF-distribution targets
    target_angle_dist_path = str(_REPO / "data" / "zef_target_angles.npz")
    fish_ckpt = str(_REPO / "logs/rl_games/fish_random10_allPCA/2026-08-13_23-40-09/nn/"
                            "last_fish_random10_allPCA_ep_500_rew_982.2629.pth")
    # honest physics (the regime the fish policy was trained in)
    body_speed_cap_bl = 1.5
    body_yaw_cap_rad = 8.0
    # ROV
    rov_usd_path = str(_REPO / "source/FISH/FISH/tasks/direct/fish/agents/bravo_rov/bravo_rov.usd")
    require_upright = True    # basket: opening must face up for success. Net run: False.
    arm_slew = 0.0            # rad/control-step arm-target rate limit. 0 = off. ONLY for
                              # from-scratch runs (a policy trained without it collapses).
    rov_arm_joints = ["bravo_axis_g", "bravo_axis_f", "bravo_axis_e", "bravo_axis_d",
                      "bravo_axis_c", "bravo_axis_b"]
    rov_finger_joints = ["bravo_finger_jaws_rs2_300_joint", "bravo_finger_jaws_rs2_301_joint"]
    rov_f_max = 30.0          # N   (vectored thruster pairs ~2x30 N in the source model)
    rov_t_max = 5.0           # N*m
    rov_drag_lin = 8.0        # N/(m/s)      simple water damping so the ROV is controllable
    rov_drag_quad = 30.0      # N/(m/s)^2
    rov_adrag = 2.0           # N*m/(rad/s)
    rov_spawn_dist = (0.9, 1.4)
    spawn_net_under = 0.0     # >0: spawn so the net pocket is this many meters BELOW the fish
    spawn_standoff = 0.0      # >0: additionally spawn the ROV this many meters BEHIND the
                              # net-under-fish position (it must first drive in, then scoop)
    lift_height = 0.0         # >0: success also requires RAISING the netted fish this many
                              # meters (measured from the pocket height when the fish entered)
    w_lift = 0.0              # potential-based reward for raising the netted fish
    root_write_exact = False  # True: compose the reset root quat with the measured hull->root
                              # rest rotation, so the WHOLE robot spawns level regardless of the
                              # root link's rest orientation. OFF for legacy assets: their
                              # BASKET_FLIP authoring already bakes in the old mis-rotation.
    contact_radius = 0.10     # m gripper-tip to nearest bone center ~= surface touch
    draw_target_marker = False        # NO red target marker in the ROV trainings
    marker_success_sphere = False
    w_progress = 30.0         # reward weights (ROV agent)
    w_face = 1.0
    w_success = 50.0
    w_action = 0.01


class SalmonROVChaseEnv(SalmonSwimPCAReach10Env):
    cfg: SalmonROVChaseCfg

    ROV_OBS_DIM = 38   # overwritten in __init__ from the cfg joint lists
    ROV_ACT_DIM = 13   # 6 wrench + 6 arm (+1 gripper if finger joints present)

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._init_ten_target()
        self._fish_player = self._load_fish_player()
        self._rov_prev_dist = torch.zeros(self.num_envs, device=self.device)
        print(f"[ROVChase] ready: frozen fish policy + ROV agent (obs {self.ROV_OBS_DIM}, act {self.ROV_ACT_DIM})",
              flush=True)

    # ------------------------------------------------ scene
    def _spawn_env0_extras(self, env0_ns: str):
        stage = self.scene.stage
        # ROV asset under env0 (pre-clone -> replicated per env; runtime adds are NOT parsed)
        rov_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.rov_usd_path)
        rov_cfg.func(f"{env0_ns}/rov", rov_cfg, translation=(-1.2, 0.0, 1.0))
        # fish bone colliders (asset ships none; needed for REAL gripper-fish contact)
        n = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{env0_ns}/skeleton")):
            if str(prim.GetTypeName()) == "Mesh" and "bone" in str(prim.GetPath()).lower() \
                    and "mesh_node" not in str(prim.GetPath()):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
                n += 1
        # rigid-vs-FEM contact is unstable at dt=1/120 (known: the gripper test NaN'd the FEM;
        # bones are the reliable contact surface). Turn OFF the deformable's collision -- the
        # skin still simulates and renders; all fish contact goes through the bone hulls.
        n_soft = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{env0_ns}/skeleton")):
            if prim.HasAttribute("physxDeformable:simulationHexahedralResolution") or \
                    any(a.GetName().startswith("physxDeformable:") for a in prim.GetAttributes()):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                prim.CreateAttribute("physxDeformable:selfCollision", Sdf.ValueTypeNames.Bool).Set(False)
                prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
                n_soft += 1
        print(f"[ROVChase] spawned ROV + {n} bone colliders; FEM collision disabled on {n_soft} prim(s)",
              flush=True)

    def _setup_scene(self):
        super()._setup_scene()
        from isaaclab.actuators import ImplicitActuatorCfg
        self.rov = Articulation(ArticulationCfg(
            prim_path="/World/envs/env_.*/rov", spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(pos=(-1.2, 0.0, 1.0),
                                                       joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
            actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"],
                                                  stiffness=None, damping=None)}))
        self.scene.articulations["rov"] = self.rov

    # ------------------------------------------------ spaces (ROV agent replaces fish agent)
    def _configure_gym_env_spaces(self):
        # dims MUST be set here: this runs inside super().__init__, BEFORE the agent
        # builds its network from the spaces (setting them later trains a 38-wide net
        # against 34-wide observations -> RuntimeError in running_mean_std).
        nj = len(self.cfg.rov_arm_joints) + len(self.cfg.rov_finger_joints)
        self.ROV_OBS_DIM = 22 + 2 * nj
        self.ROV_ACT_DIM = 12 + (1 if self.cfg.rov_finger_joints else 0)
        super()._configure_gym_env_spaces()
        self._fish_obs_dim = self._obs_dim if hasattr(self, "_obs_dim") else None
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.ROV_ACT_DIM,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(self.ROV_OBS_DIM,))})
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"], self.num_envs)

    # ------------------------------------------------ frozen fish policy
    def _load_fish_player(self):
        from rl_games.torch_runner import Runner
        ck = Path(self.cfg.fish_ckpt)
        ac = yaml.safe_load((ck.resolve().parents[1] / "params/agent.yaml").read_text())
        ac["params"]["config"].update({"num_actors": 1, "device": str(self.device),
                                       "device_name": str(self.device)})
        ac["params"]["load_checkpoint"] = True
        ac["params"]["load_path"] = str(ck)
        ac["params"]["config"]["env_info"] = {
            "observation_space": gym.spaces.Box(-np.inf, np.inf, (self._obs_dim,)),
            "action_space": gym.spaces.Box(-1.0, 1.0, (self._num_pca,)), "agents": 1}
        runner = Runner(); runner.load(ac)
        player = runner.create_player(); player.has_batch_dimension = True
        player.restore(str(ck))
        print(f"[ROVChase] frozen fish policy restored: {ck.name}", flush=True)
        return player

    # ------------------------------------------------ stepping
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 1) fish: frozen policy on the FISH observation (deterministic mean action)
        fish_obs = SalmonSwimPCAReach10Env._get_observations(self)["policy"]
        with torch.no_grad():
            fa = self._fish_player.model({"is_train": False, "prev_actions": None,
                                          "obs": self._fish_player._preproc_obs(fish_obs),
                                          "rnn_states": self._fish_player.states})["mus"]
        super()._pre_physics_step(fa.detach())
        # 2) ROV: agent action -> body wrench (6) + arm joint targets (6) + gripper (1)
        a = torch.nan_to_num(actions.view(self.num_envs, self.ROV_ACT_DIM), nan=0.0).clamp(-1.0, 1.0)
        self._rov_cmd = a
        if not hasattr(self, "_rov_jids"):
            jn = self.rov.data.joint_names
            self._rov_arm_ids = torch.tensor([jn.index(n) for n in self.cfg.rov_arm_joints],
                                             device=self.device)
            self._rov_fin_ids = torch.tensor([jn.index(n) for n in self.cfg.rov_finger_joints],
                                             device=self.device, dtype=torch.long)
            lims = self.rov.data.joint_pos_limits[0]
            self._arm_lo = lims[self._rov_arm_ids, 0]; self._arm_hi = lims[self._rov_arm_ids, 1]
            self._rov_jids = True
        # zero-centered mapping: action 0 = URDF zero pose (arm extended, basket level).
        # positive action moves toward the upper limit, negative toward the lower.
        a_arm = a[:, 6:12]
        # NOTE: never slew-limit a policy trained WITHOUT the limit (the ep-525 basket policy
        # collapses into a breaker-cap runaway with ANY rate limit -- A/B proven 2026-08-23).
        # For FROM-SCRATCH runs cfg.arm_slew > 0 cures the constant arm blow-up churn.
        arm_t = torch.where(a_arm >= 0, a_arm * self._arm_hi, a_arm * (-self._arm_lo))
        if self.cfg.arm_slew > 0.0:
            if not hasattr(self, "_arm_tgt_prev"):
                self._arm_tgt_prev = torch.zeros_like(arm_t)
            arm_t = self._arm_tgt_prev + (arm_t - self._arm_tgt_prev).clamp(-self.cfg.arm_slew, self.cfg.arm_slew)
            self._arm_tgt_prev = arm_t.detach()
        self.rov.set_joint_position_target(arm_t, joint_ids=self._rov_arm_ids)
        if len(self._rov_fin_ids):
            fin_t = ((a[:, 12:13] + 1.0) * 0.5).repeat(1, 2)      # 0 (open) .. 1 rad (closed)
            self.rov.set_joint_position_target(fin_t, joint_ids=self._rov_fin_ids)

    def _apply_action(self) -> None:
        super()._apply_action()                                   # fish drives + hydro + guards
        _, q, vw, ww = self._hull_state()
        F_thrust_b = self._rov_cmd[:, 0:3] * self.cfg.rov_f_max
        T_thrust_b = self._rov_cmd[:, 3:6] * self.cfg.rov_t_max
        F_thrust_w = math_utils.quat_apply(q, F_thrust_b)
        F_drag = -self.cfg.rov_drag_lin * vw - self.cfg.rov_drag_quad * vw.norm(dim=1, keepdim=True) * vw
        T_drag = -self.cfg.rov_adrag * ww
        F = (F_thrust_w + F_drag).unsqueeze(1)
        T = (math_utils.quat_apply(q, T_thrust_b) + T_drag).unsqueeze(1)
        if not hasattr(self, "_rov_base_id"):
            bn = self.rov.data.body_names
            base = "hull" if "hull" in bn else "bravo_base_link"
            self._rov_base_id = [bn.index(base)]
        self.rov.set_external_force_and_torque(F, T, body_ids=self._rov_base_id, is_global=True)
        # circuit breaker (same pattern as the fish FEM clamp): the arm+thrust combination can
        # slowly pump the chain under this scene's physics settings; cap joint & root velocities
        # per env so divergence is bounded. No-op below the caps.
        jv = self.rov.data.joint_vel
        bad_j = jv.abs().amax(dim=1) > 25.0
        if bool(bad_j.any()):
            ids = bad_j.nonzero(as_tuple=False).flatten()
            self.rov.write_joint_state_to_sim(self.rov.data.joint_pos[ids],
                                              jv[ids].clamp(-25.0, 25.0), env_ids=ids)
        rv = self.rov.data.root_com_vel_w
        bad_r = (rv[:, 0:3].norm(dim=1) > 6.0) | (rv[:, 3:6].norm(dim=1) > 25.0)
        if bool(bad_r.any()):
            ids = bad_r.nonzero(as_tuple=False).flatten()
            v = rv[ids].clone()
            v[:, 0:3] *= (6.0 / v[:, 0:3].norm(dim=1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
            v[:, 3:6] *= (25.0 / v[:, 3:6].norm(dim=1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
            self.rov.write_root_com_velocity_to_sim(v, env_ids=ids)

    # ------------------------------------------------ geometry helpers
    def _basket_frame(self):
        """Basket center (world) and rotation (world). Basket is rigid on the hand link."""
        import json as _json
        if not hasattr(self, "_bk_meta"):
            un = Path(self.cfg.rov_usd_path).name          # bravo_rov_net<suffix>.usd -> basket_meta<suffix>.json
            # bravo_rov_net<suffix>.usd -> basket_meta<suffix>.json ; plain bravo_rov.usd -> basket_meta.json
            mn = "basket_meta" + un.replace("bravo_rov_net", "").replace("bravo_rov", "").replace(".usd", "") + ".json"
            m = _json.load(open(Path(self.cfg.rov_usd_path).parent / mn))
            self._bk_hand = self.rov.data.body_names.index(m["hand_link"])
            self._bk_cl = torch.tensor(m["c_local"], device=self.device, dtype=torch.float32)
            import numpy as _np
            R = _np.array(m["R_local"])
            q = _np.roll(__import__("scipy.spatial.transform", fromlist=["Rotation"]).Rotation.from_matrix(R).as_quat(), 1)
            self._bk_ql = torch.tensor(q, device=self.device, dtype=torch.float32)  # wxyz
            self._bk_int = torch.tensor(m["interior"], device=self.device) * 0.5
            self._bk_meta = True
        hp = self.rov.data.body_link_pos_w[:, self._bk_hand]
        hq = self.rov.data.body_link_quat_w[:, self._bk_hand]
        c_w = hp + math_utils.quat_apply(hq, self._bk_cl.expand(self.num_envs, 3))
        q_w = math_utils.quat_mul(hq, self._bk_ql.expand(self.num_envs, 4))
        return c_w, q_w

    def _basket_fish_dist(self):
        c_w, _ = self._basket_frame()
        return (self.robot.data.root_state_w[:, 0:3] - c_w).norm(dim=1)

    def _fish_in_basket(self):
        c_w, q_w = self._basket_frame()
        rel = math_utils.quat_apply_inverse(q_w, self.robot.data.root_state_w[:, 0:3] - c_w)
        half = self._bk_int
        inside = (rel[:, 0].abs() < half[0]) & (rel[:, 1].abs() < half[1]) & (rel[:, 2].abs() < half[2])
        up_w = math_utils.quat_apply(q_w, torch.tensor([0.0, 0, 1.0], device=self.device).expand(self.num_envs, 3))
        upright = up_w[:, 2] > 0.866                     # tilt < 30 deg
        return inside, upright, up_w

    def _hull_state(self):
        if not hasattr(self, "_hull_i"):
            self._hull_i = self.rov.data.body_names.index("hull")
        d = self.rov.data
        i = self._hull_i
        return (d.body_link_pos_w[:, i], d.body_link_quat_w[:, i],
                d.body_link_lin_vel_w[:, i], d.body_link_ang_vel_w[:, i])

    def _tip_pos_w(self) -> torch.Tensor:
        if not hasattr(self, "_fin_body_ids"):
            bn = self.rov.data.body_names
            self._fin_body_ids = [bn.index("bravo_finger_jaws_rs2_300_link"),
                                  bn.index("bravo_finger_jaws_rs2_301_link")]
        return self.rov.data.body_link_pos_w[:, self._fin_body_ids].mean(dim=1)

    def _nearest_bone_dist(self) -> torch.Tensor:
        tip = self._tip_pos_w().unsqueeze(1)                      # (E,1,3)
        bones = self.robot.data.body_link_pos_w                   # (E,B,3)
        return (bones - tip).norm(dim=-1).min(dim=1).values       # (E,)

    # ------------------------------------------------ MDP (ROV agent)
    def _get_observations(self) -> dict:
        hp, q, hv, hw = self._hull_state()
        v_b = math_utils.quat_apply_inverse(q, hv)
        w_b = math_utils.quat_apply_inverse(q, hw)
        fwd_w = math_utils.quat_apply(q, torch.tensor([1.0, 0, 0], device=self.device).repeat(self.num_envs, 1))
        up_w = math_utils.quat_apply(q, torch.tensor([0.0, 0, 1.0], device=self.device).repeat(self.num_envs, 1))
        bk_c, bk_q = self._basket_frame()
        delta_w = self.robot.data.root_state_w[:, 0:3] - bk_c
        dist = delta_w.norm(dim=1, keepdim=True)
        dir_b = math_utils.quat_apply_inverse(q, delta_w / dist.clamp(min=1e-6))
        fish_v_b = math_utils.quat_apply_inverse(q, self.robot.data.root_state_w[:, 7:10])
        qj = torch.cat([self.rov.data.joint_pos[:, self._rov_arm_ids],
                        self.rov.data.joint_pos[:, self._rov_fin_ids]], dim=1) if hasattr(self, "_rov_jids") \
            else torch.zeros(self.num_envs, len(self.cfg.rov_arm_joints) + len(self.cfg.rov_finger_joints), device=self.device)
        qv = torch.cat([self.rov.data.joint_vel[:, self._rov_arm_ids],
                        self.rov.data.joint_vel[:, self._rov_fin_ids]], dim=1) * 0.1 if hasattr(self, "_rov_jids") \
            else torch.zeros(self.num_envs, len(self.cfg.rov_arm_joints) + len(self.cfg.rov_finger_joints), device=self.device)
        bup_w = math_utils.quat_apply(bk_q, torch.tensor([0.0, 0, 1.0], device=self.device).expand(self.num_envs, 3))
        # rest-pose pocket offset in the ROOT-LINK frame (write_root_pose_to_sim writes the
        # root link -- an ARM piece, NOT the hull -- so spawn math must use this frame).
        if not hasattr(self, "_pocket_off_rb") or bool(self.rov.data.joint_pos[0].abs().max() < 0.05):
            rq0 = self.rov.data.root_quat_w[0:1]; rp0 = self.rov.data.root_pos_w[0:1]
            self._pocket_off_rb = math_utils.quat_apply_inverse(rq0, (bk_c[0:1] - rp0))[0].clone()
            self._root_rel_q = math_utils.quat_mul(math_utils.quat_inv(q[0:1]), rq0)[0].clone()
        obs = torch.cat([v_b * 0.5, w_b * 0.2, fwd_w, up_w, dir_b, dist, fish_v_b * 0.5, qj, qv, bup_w], dim=1)
        return {"policy": torch.nan_to_num(obs).clamp(-10, 10)}

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        dist = self._basket_fish_dist()
        _, _, bup = self._fish_in_basket()
        prog = self._rov_prev_dist - dist
        hp, q, _, _ = self._hull_state()
        fwd_w = math_utils.quat_apply(q, torch.tensor([1.0, 0, 0], device=self.device).repeat(self.num_envs, 1))
        delta = self.robot.data.root_state_w[:, 0:3] - hp
        face = (fwd_w * (delta / delta.norm(dim=1, keepdim=True).clamp(min=1e-6))).sum(-1)
        r_lift = 0.0
        if self.cfg.lift_height > 0.0 and hasattr(self, "_lift"):
            lift_c = self._lift.clamp(0.0, self.cfg.lift_height)
            r_lift = self.cfg.w_lift * (lift_c - self._lift_prev)     # potential-based: no farming
            self._lift_prev = lift_c.clone()
        # hold bonus PAYS ONLY BRIEFLY (streak <= 15, ~0.5 s): with the lift criterion, an
        # uncapped hold paid ~2700/episode for cradling the fish forever WITHOUT succeeding --
        # the policy farmed it and success collapsed (reward 3000+ at 10% success).
        hold = ((self._contact_streak.float() / 8.0).clamp(0.0, 1.0)
                * (self._contact_streak <= 15).float()) if hasattr(self, "_contact_streak") \
            else torch.zeros_like(prog)
        r = (2.0 * hold
             + r_lift
             + 0.5 * bup[:, 2].clamp(0.0, 1.0)
             + c.w_progress * prog
             + c.w_face * face.clamp(min=0.0) * (prog * 30.0).clamp(0.0, 0.5)
             + c.w_success * self._rov_success.float()
             - c.w_action * (self._rov_cmd ** 2).mean(dim=1))
        self._rov_prev_dist = dist.detach()
        self.extras.setdefault("log", {}).update({
            "rov/dist": dist.mean().detach(),
            "rov/success_now": self._rov_success.float().mean().detach(),
            "rov/face_cos": face.mean().detach()})
        return torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e3, 1e3)

    def _get_dones(self):
        # keep the PREY moving: the ROV override disabled the fish's Reach10 attempt machine,
        # so a fish that reached its target sat frozen there (the frozen policy holds position
        # at a reached target). Resample a fresh target the moment a fish arrives.
        froot = self.robot.data.root_state_w[:, 0:3]
        fdist = torch.linalg.norm(self.target_positions_w[:, 0:3] - froot, dim=1)
        if not hasattr(self, "_fish_att_timer"):
            self._fish_att_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._fish_att_timer += 1
        # resample on ARRIVAL, on attempt TIMEOUT (10 s -- the frozen policy deterministically
        # stalls just outside the radius on ~25% of attempts), or if the target got too far
        renew = (fdist < self._cur_radius) | (self._fish_att_timer >= 300) | (fdist > 3.0)
        if bool(renew.any()):
            ids = renew.nonzero(as_tuple=False).flatten()
            self._sample_targets(ids, froot[ids])
            self._fish_att_timer[ids] = 0
            self._prev_distance[ids] = torch.linalg.norm(
                self.target_positions_w[ids, 0:3] - froot[ids], dim=1)
        d = self.robot.data
        nonfinite = (~torch.isfinite(d.root_state_w).all(dim=1)) | \
                    (~torch.isfinite(self.rov.data.root_state_w).all(dim=1))
        # SUSTAINED contact: a single-frame graze at speed counted as success but was invisible
        # (touch -> instant reset = one blink). Require staying within reach for 8 consecutive
        # control steps (~0.27 s) -- a hold, not a flyby.
        inside, upright, _ = self._fish_in_basket()
        near = inside & upright if self.cfg.require_upright else inside
        if not hasattr(self, "_contact_streak"):
            self._contact_streak = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        prev_streak = self._contact_streak.clone()
        self._contact_streak = torch.where(near, self._contact_streak + 1,
                                           torch.zeros_like(self._contact_streak))
        held = self._contact_streak >= 5                 # fish in the net, held ~0.17 s
        if self.cfg.lift_height > 0.0:
            # LIFT success: the netted fish must be RAISED lift_height meters above the pocket
            # height at the moment of capture (entry). Reference resets when the fish escapes.
            bk_z = self._basket_frame()[0][:, 2]
            if not hasattr(self, "_lift_ref_z"):
                self._lift_ref_z = torch.zeros(self.num_envs, device=self.device)
                self._lift_prev = torch.zeros(self.num_envs, device=self.device)
            entering = near & (prev_streak == 0)
            self._lift_ref_z[entering] = bk_z[entering]
            self._lift = torch.where(near, bk_z - self._lift_ref_z, torch.zeros_like(bk_z))
            self._rov_success = held & (self._lift >= self.cfg.lift_height)
        else:
            self._rov_success = held
        # episodic success-rate logging.
        # UNBIASED metric: per-env table of each env's LAST completed episode outcome
        # (equal weight per env). The old completion-ordered rolling list overweights
        # short success episodes ~3x (they complete more often than 45 s timeouts) --
        # it displayed 0.83 when the true per-episode rate was ~0.4. Kept as *_biased.
        if not hasattr(self, "_succ_hist"):
            self._succ_hist = []
            self._env_last_outcome = torch.full((self.num_envs,), float("nan"), device=self.device)
        for e in torch.nonzero(self._rov_success).flatten().tolist():
            self._succ_hist.append(1.0)
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        for e in torch.nonzero(timeout & ~self._rov_success).flatten().tolist():
            self._succ_hist.append(0.0)
        self._succ_hist = self._succ_hist[-400:]
        self._env_last_outcome[self._rov_success] = 1.0
        self._env_last_outcome[timeout & ~self._rov_success] = 0.0
        # NaN-terminated episodes are FAILURES too: dropping them inflated the metric when a
        # failure mode (arm blow-up mid-approach) ended episodes via nonfinite termination.
        self._env_last_outcome[nonfinite & ~self._rov_success] = 0.0
        self.extras.setdefault("log", {}).update({
            "rov/success_rate": torch.nan_to_num(self._env_last_outcome.nanmean(), nan=0.0).detach(),
            "rov/success_rate_completion_biased": torch.tensor(float(np.mean(self._succ_hist)) if self._succ_hist else 0.0),
        })
        self.extras["success"] = self._rov_success.detach().clone()
        terminated = self._rov_success | nonfinite
        return terminated, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        if not torch.is_tensor(ids):
            ids = torch.tensor(list(ids), device=self.device, dtype=torch.long)
        n = len(ids)
        fish = self.robot.data.root_state_w[ids, 0:3]
        ang = torch.rand(n, device=self.device) * 2 * torch.pi
        r = (torch.rand(n, device=self.device)
             * (self.cfg.rov_spawn_dist[1] - self.cfg.rov_spawn_dist[0]) + self.cfg.rov_spawn_dist[0])
        pos = fish.clone()
        pos[:, 0] += r * torch.cos(ang)
        pos[:, 1] += r * torch.sin(ang)
        yaw = torch.atan2(fish[:, 1] - pos[:, 1], fish[:, 0] - pos[:, 0])   # face the fish
        quat = torch.stack([torch.cos(yaw / 2), torch.zeros(n, device=self.device),
                            torch.zeros(n, device=self.device), torch.sin(yaw / 2)], dim=1)
        if self.cfg.root_write_exact and hasattr(self, "_root_rel_q"):
            quat = math_utils.quat_mul(quat, self._root_rel_q.expand(n, 4))
        if self.cfg.spawn_net_under > 0.0 and hasattr(self, "_pocket_off_rb"):
            # place the hull so the net pocket sits spawn_net_under meters BELOW the fish.
            # data.root_state_w is STALE right after the fish reset write -> use the WRITTEN
            # spawn coordinates (default state + env origin) instead of the lazy buffer.
            fish = self.robot.data.default_root_state[ids, 0:3] + self.scene.env_origins[ids]
            yaw = torch.rand(n, device=self.device) * 2 * torch.pi
            quat = torch.stack([torch.cos(yaw / 2), torch.zeros(n, device=self.device),
                                torch.zeros(n, device=self.device), torch.sin(yaw / 2)], dim=1)
            if self.cfg.root_write_exact and hasattr(self, "_root_rel_q"):
                quat = math_utils.quat_mul(quat, self._root_rel_q.expand(n, 4))
            tgt = fish.clone(); tgt[:, 2] -= self.cfg.spawn_net_under
            pos = tgt - math_utils.quat_apply(quat, self._pocket_off_rb.expand(n, 3))
            if self.cfg.spawn_standoff > 0.0:
                pos[:, 0] -= self.cfg.spawn_standoff * torch.cos(yaw)
                pos[:, 1] -= self.cfg.spawn_standoff * torch.sin(yaw)
        self.rov.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=ids)
        self.rov.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=ids)
        # ALSO reset the arm joints: a NaN'd arm otherwise survives every reset (root writes
        # don't touch joint states) and turns the env into a permanent dead NaN loop.
        self.rov.write_joint_state_to_sim(torch.zeros(n, self.rov.num_joints, device=self.device),
                                          torch.zeros(n, self.rov.num_joints, device=self.device),
                                          env_ids=ids)
        if hasattr(self, "_arm_tgt_prev"):
            self._arm_tgt_prev[ids] = 0.0
        if hasattr(self, "_lift_ref_z"):
            self._lift_ref_z[ids] = 0.0; self._lift_prev[ids] = 0.0
            self._lift[ids] = 0.0
        if hasattr(self, "_rov_prev_dist"):
            self._rov_prev_dist[ids] = self._basket_fish_dist()[ids]
        if hasattr(self, "_contact_streak"):
            self._contact_streak[ids] = 0
