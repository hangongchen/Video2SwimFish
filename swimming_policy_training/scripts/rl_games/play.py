# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import sys
from pathlib import Path

# Ensure the FISH source tree is on sys.path when running via the env python directly.
_repo_root = Path(__file__).resolve().parents[2]
_fish_source = _repo_root / "source"
if _fish_source.is_dir():
    _fish_source_str = str(_fish_source)
    if _fish_source_str not in sys.path:
        sys.path.insert(0, _fish_source_str)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--record_demo", type=str, default=None,
                    help="Record this RL rollout (FEM point cloud + root pose, WORLD frame) to a .npz, "
                         "for IL demo-imitation. Requires env.with_deformable=true env.reset_deformable=true.")
parser.add_argument("--record_demo_max", type=int, default=2000,
                    help="Max control steps to record for the demo (stops earlier on env0 episode end).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--stochastic", action="store_true", default=False,
                    help="Sample actions from the policy (FULL exploration noise) instead of the "
                         "deterministic mean. NOTE: for the auto-skeleton fish at dt=1/120 this is NOT "
                         "needed -- deterministic play was measured stable AND smoother (max jvel 2.9 vs "
                         "3.4, 0 blow-ups over ~48 s). The noise makes the gait look jittery ('sudden pose "
                         "changes'). The old 'deterministic diverges the FEM' warning applied to the "
                         "salmon at dt=1/960, not this asset. Prefer --action_noise_scale for a smooth dial.")
parser.add_argument("--action_noise_scale", type=float, default=None,
                    help="Continuous exploration-noise dial for viewing: 0.0 = deterministic mean "
                         "(smoothest gait), 1.0 = full sampled action (== --stochastic), 0.3 = light "
                         "noise. Scales the effective policy sigma. Overrides --stochastic when set. Use "
                         "a small value only if a very long soft-body session ever destabilizes.")
parser.add_argument(
    "--traj_dir",
    type=str,
    default=None,
    help="Override the root directory used by env trajectory saving during play.",
)
parser.add_argument(
    "--traj_save_every",
    type=int,
    default=None,
    help="Override trajectory saving cadence in episodes (1 saves every completed env0 episode).",
)
parser.add_argument(
    "--traj_live_every",
    type=int,
    default=None,
    help="Override live trajectory refresh cadence in control steps (0 disables live overwrites).",
)
parser.add_argument("--step_log_path", type=str, default=None, help="Write per-step play telemetry to this JSONL file.")
parser.add_argument("--eval_reach_episodes", type=int, default=0,
    help="If >0: reach-rate eval. Stop after this many completed episodes (summed over all envs) and "
         "print reaches/episode + fraction of episodes with >=1 reach (batch-invariance A/B).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# CAMERA LIGHT as the default viewport lighting: a headlight from the camera so the (translucent) FEM
# fish is lit from the viewer instead of the washed-out stage lights. This is what the viewport
# "Lighting -> Camera Light" menu sets (omni.kit.viewport.menubar.lighting -> /rtx/useViewLightingMode).
import carb  # noqa: E402
carb.settings.get_settings().set_bool("/rtx/useViewLightingMode", True)

"""Rest everything follows."""


import gymnasium as gym
import math
import numpy as np
import os
import random
import time
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.common.tr_helpers import unsqueeze_obs
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import FISH.tasks  # noqa: F401


def _rescale_actions(low: torch.Tensor, high: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    scale = (high - low) / 2.0
    mean = (high + low) / 2.0
    return action * scale + mean


def _env0_data(value):
    if isinstance(value, dict):
        return {k: _env0_data(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu()
        if data.ndim == 0:
            return data.item()
        if data.shape[0] >= 1:
            data = data[0]
        return data.tolist()
    if isinstance(value, (list, tuple)):
        return [_env0_data(v) for v in value]
    return value


def _tensor_data(value):
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu()
        return data.item() if data.ndim == 0 else data.tolist()
    return value


def _policy_step(agent: BasePlayer, obs, noise_scale: float) -> tuple[torch.Tensor, dict]:
    if agent.has_batch_dimension is False:
        obs_for_model = unsqueeze_obs(obs)
    else:
        obs_for_model = obs
    obs_for_model = agent._preproc_obs(obs_for_model)
    input_dict = {
        "is_train": False,
        "prev_actions": None,
        "obs": obs_for_model,
        "rnn_states": agent.states,
    }
    with torch.no_grad():
        res_dict = agent.model(input_dict)
    mu = res_dict.get("mus")
    sigma = res_dict.get("sigmas")
    sampled_action = res_dict.get("actions")
    agent.states = res_dict.get("rnn_states", agent.states)
    # Continuous noise dial: noise_scale=0 -> mean (deterministic, smoothest); 1 -> full sampled action.
    # sampled_action = mu + sigma*eps, so mu + s*(sampled-mu) scales the effective exploration sigma by s.
    if mu is not None and sampled_action is not None and noise_scale > 0.0:
        current_action = mu + float(noise_scale) * (sampled_action - mu)
    elif mu is not None:
        current_action = mu
    else:
        current_action = sampled_action
    if current_action is None:
        raise RuntimeError("Could not extract policy action from RL-Games player output.")
    current_action = current_action.detach()
    if agent.has_batch_dimension is False:
        current_action = torch.squeeze(current_action)
    if getattr(agent, "clip_actions", False) and hasattr(agent, "actions_low") and hasattr(agent, "actions_high"):
        final_action = _rescale_actions(agent.actions_low, agent.actions_high, torch.clamp(current_action, -1.0, 1.0))
    else:
        final_action = current_action
    trace = {
        "mu": _env0_data(mu),
        "sigma": _env0_data(sigma),
        "sampled_action": _env0_data(sampled_action),
        "action": _env0_data(final_action),
    }
    return final_action, trace


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.traj_dir is not None and hasattr(env_cfg, "traj_dir"):
        env_cfg.traj_dir = args_cli.traj_dir
        if hasattr(env_cfg, "record_trajectory"):
            env_cfg.record_trajectory = True
    if args_cli.traj_save_every is not None and hasattr(env_cfg, "traj_save_every_n_episodes"):
        env_cfg.traj_save_every_n_episodes = max(1, args_cli.traj_save_every)
        if hasattr(env_cfg, "record_trajectory"):
            env_cfg.record_trajectory = True
    if args_cli.traj_live_every is not None and hasattr(env_cfg, "traj_live_every"):
        env_cfg.traj_live_every = max(0, args_cli.traj_live_every)
        if hasattr(env_cfg, "record_trajectory"):
            env_cfg.record_trajectory = True

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    base_env = env.unwrapped

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt
    max_episode_steps = getattr(env.unwrapped, "max_episode_length", None)
    if max_episode_steps is None and hasattr(env_cfg, "episode_length_s"):
        max_episode_steps = int(round(float(env_cfg.episode_length_s) / max(dt, 1e-9)))
    if max_episode_steps is not None:
        print(
            "[play] episode horizon: "
            f"{int(max_episode_steps)} control steps (~{float(max_episode_steps) * dt:.1f}s sim time). "
            "Success count and per-episode trajectory files update when an episode ends.",
            flush=True,
        )
    if args_cli.traj_dir is not None:
        print(
            "[play] trajectory output: "
            f"{args_cli.traj_dir} "
            f"(per-episode save cadence={args_cli.traj_save_every or 'cfg'}, "
            f"live cadence={args_cli.traj_live_every if args_cli.traj_live_every is not None else 'cfg'})",
            flush=True,
        )
    step_log_file = None
    if args_cli.step_log_path is not None:
        step_log_path = os.path.abspath(args_cli.step_log_path)
        os.makedirs(os.path.dirname(step_log_path), exist_ok=True)
        step_log_file = open(step_log_path, "w", encoding="utf-8", buffering=1)
        step_log_meta = {
            "type": "meta",
            "checkpoint": resume_path,
            "task": args_cli.task,
            "device": args_cli.device,
            "num_envs": int(base_env.num_envs),
            "policy_mode": (f"noise_scale={args_cli.action_noise_scale}" if args_cli.action_noise_scale is not None
                            else ("stochastic" if args_cli.stochastic else "deterministic")),
            "step_dt": float(dt),
            "max_episode_steps": int(max_episode_steps) if max_episode_steps is not None else None,
            "traj_dir": args_cli.traj_dir,
            "control_mode": getattr(base_env.cfg, "control_mode", None),
            "with_deformable": bool(getattr(base_env.cfg, "with_deformable", False)),
            "reset_deformable": bool(getattr(base_env.cfg, "reset_deformable", False)),
            "pos_action_scale": float(getattr(base_env.cfg, "pos_action_scale", 0.0)),
            "action_scale": float(getattr(base_env.cfg, "action_scale", 0.0)),
            "joint_limit_deg": float(getattr(base_env.cfg, "joint_limit_deg", 0.0)),
            "actuator_stiffness": float(getattr(base_env.cfg.robot_cfg.actuators["all_joints"], "stiffness", 0.0)),
            "actuator_damping": float(getattr(base_env.cfg.robot_cfg.actuators["all_joints"], "damping", 0.0)),
            "actuator_armature": float(getattr(base_env.cfg.robot_cfg.actuators["all_joints"], "armature", 0.0)),
        }
        step_log_file.write(json.dumps(step_log_meta) + "\n")
        print(f"[play] step telemetry log: {step_log_path}", flush=True)

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    episodes_completed = 0
    episodes_succeeded = 0
    # --- reach-rate eval (batch-invariance A/B): count reaches by watching each env's target JUMP
    #     (in multi_target a reach immediately resamples the target). A jump that is NOT a reset =
    #     a reach. Tracks per-episode reach counts so we can report reaches/episode + frac reaching. ---
    _er_prev_tgt = None
    _er_reaches_total = 0
    _er_episodes = 0
    _er_envsteps = 0
    _er_env_accum = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
    _er_per_episode = []
    # demo recording (RL rollout -> reference trajectory of FEM point cloud + root pose, WORLD frame)
    _demo_pc, _demo_root, _demo_nvel, _demo_jpos, _demo_jvel = [], [], [], [], []
    _demo_tau = []   # applied joint torque, for the mechanical energy / cost-of-transport metric
    # per-bone world kinematics -> lets us recompute the exact analytic hydro wrench OFFLINE
    # (mujoco_fluid_wrench is a pure fn of these + the constant semi/R_pl), env stays untouched.
    _demo_bvel, _demo_bang, _demo_bquat, _demo_bpos = [], [], [], []
    if args_cli.record_demo is not None and getattr(base_env, "_soft_view", None) is None:
        raise RuntimeError("--record_demo needs the deformable soft view: run with "
                           "env.with_deformable=true env.reset_deformable=true")
    if os.environ.get("FISH_PRINT_MASS"):
        import numpy as _np
        print("\n=============== FISH MASS / VOLUME ===============", flush=True)
        try:
            m = base_env.robot.data.default_mass[0].detach().cpu().numpy()
            print(f"rigid skeleton: {len(m)} bones, per-bone mass (g) = "
                  f"{[round(float(x)*1000,2) for x in m]}", flush=True)
            print(f"  TOTAL rigid mass = {m.sum()*1000:.2f} g = {m.sum():.4f} kg", flush=True)
        except Exception as e:
            print("rigid mass read failed:", e, flush=True)
        sv = getattr(base_env, "_soft_view", None)
        if sv is not None:
            try:
                pos = _np.asarray(sv.get_simulation_mesh_nodal_positions()[0].detach().cpu().numpy(), _np.float64)
                P = pos - pos.mean(0)
                _, _, vt = _np.linalg.svd(P, full_matrices=False)
                ext = (P @ vt.T).ptp(0)
                vol_tet = None
                try:                     # exact tet volume from connectivity
                    idx = _np.asarray(sv.get_simulation_mesh_indices()[0].detach().cpu().numpy()).reshape(-1, 4)
                    a, b, c, dd = pos[idx[:,0]], pos[idx[:,1]], pos[idx[:,2]], pos[idx[:,3]]
                    vol_tet = float(_np.abs(_np.einsum('ij,ij->i', _np.cross(b-a, c-a), dd-a)).sum()/6.0)
                except Exception as _e:
                    print("  (no tet connectivity:", _e, ")", flush=True)
                from scipy.spatial import ConvexHull
                vol_hull = float(ConvexHull(pos).volume)
                print(f"FEM soft body: {pos.shape[0]} nodes, rest extents (m) = "
                      f"{ext[0]:.3f} x {ext[1]:.3f} x {ext[2]:.3f} (L x H x W)", flush=True)
                if vol_tet is not None:
                    print(f"  EXACT tet volume = {vol_tet*1e6:.1f} cm^3 ; FEM mass @1000 kg/m^3 = {vol_tet*1000*1000:.1f} g", flush=True)
                print(f"  convex-hull volume = {vol_hull*1e6:.1f} cm^3 (upper bound)", flush=True)
            except Exception as e:
                print("FEM volume read failed:", e, flush=True)
        try:
            import omni.usd
            _stage = omni.usd.get_context().get_stage()
            for _pr in _stage.Traverse():
                _pth = str(_pr.GetPath())
                if "eformableBodyMaterial" in _pth or "eformableMaterial" in _pth:
                    for _an in ("physxDeformableBodyMaterial:density", "physxDeformable:density"):
                        _av = _pr.GetAttribute(_an).Get() if _pr.GetAttribute(_an) else None
                        if _av is not None:
                            print(f"FEM material density [{_pth}] {_an} = {_av} kg/m^3", flush=True)
        except Exception as e:
            print("density read failed:", e, flush=True)
        print("==================================================\n", flush=True)
        os._exit(0)

    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # OBS-STAT PROBE: accumulate the raw observation the policy actually sees, so it can be
            # compared channel-by-channel against the checkpoint's running_mean_std -- which IS the
            # observation distribution seen during TRAINING. Any channel computed differently in play
            # shows up as a mean/var mismatch. Dumps once and keeps going.
            if globals().get("_obs_probe_on", True):
                _ot = obs if torch.is_tensor(obs) else obs.get("obs")
                if torch.is_tensor(_ot) and _ot.dim() == 2:
                    _ot = _ot.detach().float()
                    globals()["_obs_n"] = globals().get("_obs_n", 0) + _ot.shape[0]
                    globals()["_obs_s"] = globals().get("_obs_s", torch.zeros(_ot.shape[1], device=_ot.device)) + _ot.sum(0)
                    globals()["_obs_q"] = globals().get("_obs_q", torch.zeros(_ot.shape[1], device=_ot.device)) + (_ot**2).sum(0)
                    globals()["_obs_steps"] = globals().get("_obs_steps", 0) + 1
                    if globals()["_obs_steps"] == 400:
                        import numpy as _np
                        _n = globals()["_obs_n"]; _m = globals()["_obs_s"]/_n
                        _v = (globals()["_obs_q"]/_n - _m**2).clamp(min=0)
                        _np.savez("/tmp/play_obs_stats.npz", mean=_m.cpu().numpy(), var=_v.cpu().numpy(), n=_n)
                        print(f"[obs-probe] saved /tmp/play_obs_stats.npz over {_n} samples", flush=True)
                        globals()["_obs_probe_on"] = False
            pre_obs_trace = _env0_data(obs)
            # agent stepping. Noise dial: explicit --action_noise_scale wins; else --stochastic == 1.0
            # (full noise), else 0.0 (deterministic mean). A non-deterministic agent still honors the dial.
            if args_cli.action_noise_scale is not None:
                _noise_scale = max(0.0, float(args_cli.action_noise_scale))
            elif args_cli.stochastic or not agent.is_deterministic:
                _noise_scale = 1.0
            else:
                _noise_scale = 0.0
            actions, policy_trace = _policy_step(agent, obs, noise_scale=_noise_scale)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
            # ENV-SIDE metrics, computed exactly as training computes them (same code path). Use this
            # to compare like-for-like against wandb rather than against play.py's own counters.
            globals()["_envlog_n"] = globals().get("_envlog_n", 0) + 1
            if globals()["_envlog_n"] % 200 == 0:
                # read the ENV directly (base_env.extras), NOT the wrapper's reshaped `infos` --
                # the wrapper drops the "log" dict, which is why the first attempt printed nothing.
                _lg = base_env.extras.get("log", {}) if isinstance(base_env.extras, dict) else {}
                _keys = ("targets_reached", "distance_to_target", "speed_bl", "speed_to_target_bl",
                         "in_radius_frac", "success_radius_cur", "target_dist_cur")
                _pick = {k: float(_lg[k]) for k in _keys if k in _lg}
                _cr = float(getattr(base_env, "_cur_radius", -1))
                _tr = float(getattr(base_env, "_targets_reached", torch.zeros(1)).float().mean())
                print(f"[envlog {globals()['_envlog_n']}] _cur_radius={_cr:.3f} _targets_reached={_tr:.3f} | "
                      + "  ".join(f"{k}={v:.3f}" for k, v in sorted(_pick.items())), flush=True)

            # --- closest-approach + trajectory diagnostic (env0): store fish AND target (env-local) ---
            try:
                _o = base_env.scene.env_origins[0]
                _rt = (base_env.robot.data.root_state_w[0, :3] - _o).detach().cpu().numpy()   # fish (local)
                _tg = (base_env.target_positions_w[0, :3] - _o).detach().cpu().numpy()         # target (local)
                _cur_d0 = float(((_rt - _tg) ** 2).sum() ** 0.5)
                globals()["_min_d0"] = min(globals().get("_min_d0", float("inf")), _cur_d0)
                globals().setdefault("_traj0", []).append([float(_rt[0]), float(_rt[1]), float(_rt[2]),
                                                           float(_tg[0]), float(_tg[1]), float(_tg[2])])
            except Exception:  # noqa: BLE001
                pass

            if args_cli.record_demo is not None:
                _done0 = bool(torch.as_tensor(dones, device="cpu").flatten()[0].item())
                # On the done step, robot.data is ALREADY post-reset (env0 snapped back to start),
                # so skip that contaminated frame -- keep only the pre-reset swim trajectory.
                if not _done0:
                    # FULL state for Reference State Initialization: nodal pos+vel, root state,
                    # joint pos+vel -- enough to set the sim to this exact frame at reset.
                    _pc = base_env._soft_view.get_simulation_mesh_nodal_positions()[0].detach().cpu().numpy().copy()
                    _nvel = base_env._soft_view.get_simulation_mesh_nodal_velocities()[0].detach().cpu().numpy().copy()
                    _root = base_env.robot.data.root_state_w[0].detach().cpu().numpy().copy()
                    _jpos = base_env.robot.data.joint_pos[0].detach().cpu().numpy().copy()
                    _jvel = base_env.robot.data.joint_vel[0].detach().cpu().numpy().copy()
                    _tau = getattr(base_env.robot.data, "applied_torque", None)
                    _tau = (_tau[0].detach().cpu().numpy().copy() if _tau is not None
                            else np.zeros_like(_jvel))
                    _bvel = base_env.robot.data.body_link_lin_vel_w[0].detach().cpu().numpy().copy()
                    _bang = base_env.robot.data.body_link_ang_vel_w[0].detach().cpu().numpy().copy()
                    _bquat = base_env.robot.data.body_link_quat_w[0].detach().cpu().numpy().copy()
                    _bpos = base_env.robot.data.body_link_pos_w[0].detach().cpu().numpy().copy()
                    _demo_pc.append(_pc); _demo_root.append(_root)
                    _demo_nvel.append(_nvel); _demo_jpos.append(_jpos); _demo_jvel.append(_jvel)
                    _demo_tau.append(_tau)
                    _demo_bvel.append(_bvel); _demo_bang.append(_bang)
                    _demo_bquat.append(_bquat); _demo_bpos.append(_bpos)
                if _done0 or len(_demo_pc) >= args_cli.record_demo_max:
                    _outp = os.path.abspath(args_cli.record_demo)
                    os.makedirs(os.path.dirname(_outp), exist_ok=True)
                    def _np(x):
                        import numpy as _n
                        return (x.detach().cpu().numpy().copy() if hasattr(x, "detach")
                                else _n.asarray(x)) if x is not None else _n.zeros(0)
                    np.savez_compressed(_outp, pc=np.stack(_demo_pc), root=np.stack(_demo_root),
                                        nodal_vel=np.stack(_demo_nvel), joint_pos=np.stack(_demo_jpos),
                                        joint_vel=np.stack(_demo_jvel), applied_torque=np.stack(_demo_tau),
                                        body_lin_vel_w=np.stack(_demo_bvel), body_ang_vel_w=np.stack(_demo_bang),
                                        body_quat_w=np.stack(_demo_bquat), body_pos_w=np.stack(_demo_bpos),
                                        semi=_np(getattr(base_env, "_semi", None)),
                                        R_pl=_np(getattr(base_env, "_R_pl", None)),
                                        hydro_cfg=np.array([base_env.cfg.rho, base_env.cfg.visc,
                                            base_env.cfg.cd_blunt, base_env.cfg.cd_slender,
                                            base_env.cfg.cd_angular, base_env.cfg.ck, base_env.cfg.cm],
                                            dtype=np.float64),
                                        env0_success=_done0, step_dt=float(dt))
                    print(f"[record_demo] saved {len(_demo_pc)} steps (N={_demo_pc[0].shape[0]} world pts) "
                          f"-> {_outp} (env0 done={_done0})", flush=True)
                    break

            done_mask = torch.as_tensor(dones, device="cpu").bool().flatten()
            done_count = int(done_mask.sum().item()) if done_mask.numel() > 0 else 0

            # --- reach-rate eval accounting (target JUMP that is NOT a reset == a reach) ---
            if args_cli.eval_reach_episodes > 0:
                _cur_tgt = base_env.target_positions_w[:, :3].detach()
                _dm = done_mask.to(base_env.device)[: base_env.num_envs]
                if _er_prev_tgt is not None:
                    _moved = (_cur_tgt - _er_prev_tgt).norm(dim=-1) > 1e-4
                    _reach = _moved & (~_dm)
                    _er_env_accum += _reach.long()
                    _er_reaches_total += int(_reach.sum().item())
                _er_prev_tgt = _cur_tgt.clone()
                _er_envsteps += base_env.num_envs
                if done_count > 0:
                    for _e in torch.nonzero(_dm, as_tuple=False).flatten().tolist():
                        _er_per_episode.append(int(_er_env_accum[_e].item()))
                    _er_env_accum[_dm] = 0
                    _er_episodes += done_count
                    _rpe = _er_reaches_total / max(1, _er_episodes)
                    _frac = sum(1 for x in _er_per_episode if x >= 1) / max(1, len(_er_per_episode))
                    print(f"[eval_reach] N={base_env.num_envs} episodes={_er_episodes} "
                          f"reaches={_er_reaches_total} reaches/ep={_rpe:.3f} "
                          f"frac_ep_reached={_frac:.3f} envsteps={_er_envsteps}", flush=True)
                if _er_episodes >= args_cli.eval_reach_episodes:
                    print("EVAL_REACH_DONE", flush=True)
                    break

            success_count = 0
            if done_count > 0:
                success_tensor = None
                if isinstance(infos, dict) and "success" in infos:
                    success_tensor = infos["success"]
                elif hasattr(env.unwrapped, "extras") and isinstance(env.unwrapped.extras, dict):
                    success_tensor = env.unwrapped.extras.get("success")
                if success_tensor is not None:
                    success_mask = torch.as_tensor(success_tensor, device="cpu").bool().flatten()
                    if success_mask.numel() >= done_mask.numel():
                        success_count = int((success_mask[: done_mask.numel()] & done_mask).sum().item())
                episodes_completed += done_count
                episodes_succeeded += success_count
                success_rate = 100.0 * episodes_succeeded / max(1, episodes_completed)
                print(
                    "[play] global_step="
                    f"{timestep + 1} episodes ended across {base_env.num_envs} envs: "
                    f"total={episodes_completed} successes={episodes_succeeded} "
                    f"(ended this step {done_count}, successes this step {success_count}, {success_rate:.1f}%)",
                    flush=True,
                )
                if bool(done_mask[0]) and "_min_d0" in globals():
                    print(f"[closest] env0 episode CLOSEST distance-to-target = "
                          f"{globals()['_min_d0']:.3f} m  (success_radius={base_env.cfg.success_radius})", flush=True)
                    globals()["_min_d0"] = float("inf")
                    _tr = globals().get("_traj0", [])
                    if _tr:
                        import numpy as _np
                        _p = f"/tmp/misty_traj_ep{episodes_completed}.npy"
                        _np.save(_p, _np.asarray(_tr, dtype=_np.float32))
                        print(f"[traj] saved env0 path ({len(_tr)} steps, fish-rel-to-target) -> {_p}", flush=True)
                        globals()["_traj0"] = []
            if step_log_file is not None:
                success_tensor = None
                if isinstance(infos, dict) and "success" in infos:
                    success_tensor = infos["success"]
                elif isinstance(base_env.extras, dict):
                    success_tensor = base_env.extras.get("success")
                debug_step = infos.get("debug_step") if isinstance(infos, dict) else None
                if isinstance(debug_step, dict):
                    root_state = debug_step["root_state_w"][0]
                    target_pos = debug_step["target_position_w"][0]
                    target_delta = debug_step["target_delta"][0]
                    joint_pos = debug_step["joint_pos"][0]
                    joint_vel = debug_step["joint_vel"][0]
                    projected_gravity = debug_step["projected_gravity_b"][0]
                    env_actions = debug_step["actions"][0]
                    pos_targets = debug_step["pos_targets"][0]
                    torques = debug_step["torques"][0]
                    distance_to_target = float(debug_step["distance_to_target"][0].item())
                    episode_step_env0 = int(debug_step["episode_step"][0].item())
                    terminated_env0 = bool(debug_step["terminated"][0].item())
                    time_out_env0 = bool(debug_step["time_out"][0].item())
                    success_env0 = bool(debug_step["success"][0].item())
                    out_of_bounds_env0 = bool(debug_step["out_of_bounds"][0].item())
                    joint_blowup_env0 = bool(debug_step["joint_blowup"][0].item())
                    nonfinite_env0 = bool(debug_step["nonfinite"][0].item())
                    joint_vel_max_all = float(debug_step["joint_vel"].abs().max().item())
                    state_source = "debug_step_pre_reset"
                    terminated_count_all = int(debug_step["terminated"].sum().item())
                    time_out_count_all = int(debug_step["time_out"].sum().item())
                    success_count_all = int(debug_step["success"].sum().item())
                    out_of_bounds_count_all = int(debug_step["out_of_bounds"].sum().item())
                    joint_blowup_count_all = int(debug_step["joint_blowup"].sum().item())
                    nonfinite_count_all = int(debug_step["nonfinite"].sum().item())
                else:
                    root_state = base_env.robot.data.root_state_w[0]
                    target_pos = base_env.target_positions_w[0]
                    target_delta = target_pos[0:3] - root_state[0:3]
                    joint_pos = base_env.robot.data.joint_pos[0]
                    joint_vel = base_env.robot.data.joint_vel[0]
                    projected_gravity = base_env.robot.data.projected_gravity_b[0]
                    env_actions = getattr(base_env, "actions", None)
                    if isinstance(env_actions, torch.Tensor):
                        env_actions = env_actions[0]
                    pos_targets = getattr(base_env, "_pos_targets", None)
                    if isinstance(pos_targets, torch.Tensor):
                        pos_targets = pos_targets[0]
                    torques = getattr(base_env, "_torques", None)
                    if isinstance(torques, torch.Tensor):
                        torques = torques[0]
                    distance_to_target = float(torch.linalg.norm(target_delta).item())
                    episode_step_env0 = int(base_env.episode_length_buf[0])
                    terminated_env0 = bool(base_env.reset_terminated[0].item())
                    time_out_env0 = bool(base_env.reset_time_outs[0].item())
                    success_env0 = bool(torch.as_tensor(_env0_data(success_tensor)).flatten()[0].item()) if success_tensor is not None else False
                    out_of_bounds_env0 = None
                    joint_blowup_env0 = None
                    nonfinite_env0 = None
                    joint_vel_max_all = float(base_env.joint_vel.abs().max().item()) if hasattr(base_env, "joint_vel") else None
                    state_source = "post_step_env_buffers"
                    terminated_count_all = None
                    time_out_count_all = None
                    success_count_all = None
                    out_of_bounds_count_all = None
                    joint_blowup_count_all = None
                    nonfinite_count_all = None
                step_log_record = {
                    "type": "step",
                    "global_step": int(timestep + 1),
                    "episode_step_env0": episode_step_env0,
                    "episode_step_env0_post_reset": int(base_env.episode_length_buf[0]),
                    "sim_time_s": float((timestep + 1) * dt),
                    "policy_mode": (f"noise_scale={args_cli.action_noise_scale}" if args_cli.action_noise_scale is not None
                            else ("stochastic" if args_cli.stochastic else "deterministic")),
                    "state_source": state_source,
                    "pre_obs_env0": pre_obs_trace,
                    "post_obs_env0": _env0_data(obs),
                    "policy": policy_trace,
                    "env_action_env0": _tensor_data(env_actions),
                    "reward_env0": float(torch.as_tensor(rewards).detach().cpu().flatten()[0].item()),
                    "done_env0": bool(done_mask[0].item()) if done_mask.numel() > 0 else False,
                    "success_env0": success_env0,
                    "terminated_env0": terminated_env0,
                    "time_out_env0": time_out_env0,
                    "out_of_bounds_env0": out_of_bounds_env0,
                    "joint_blowup_env0": joint_blowup_env0,
                    "nonfinite_env0": nonfinite_env0,
                    "done_count_all_envs_this_step": done_count,
                    "success_count_all_envs_this_step": success_count,
                    "terminated_count_all_envs_this_step": terminated_count_all,
                    "time_out_count_all_envs_this_step": time_out_count_all,
                    "out_of_bounds_count_all_envs_this_step": out_of_bounds_count_all,
                    "joint_blowup_count_all_envs_this_step": joint_blowup_count_all,
                    "nonfinite_count_all_envs_this_step": nonfinite_count_all,
                    "episodes_completed_total": episodes_completed,
                    "episodes_succeeded_total": episodes_succeeded,
                    "root_state_w_env0": _tensor_data(root_state),
                    "joint_pos_env0": _tensor_data(joint_pos),
                    "joint_vel_env0": _tensor_data(joint_vel),
                    "projected_gravity_env0": _tensor_data(projected_gravity),
                    "target_position_w_env0": _tensor_data(target_pos),
                    "target_delta_env0": _tensor_data(target_delta),
                    "distance_to_target_env0": distance_to_target,
                    "pos_targets_env0": _tensor_data(pos_targets),
                    "torques_env0": _tensor_data(torques),
                    "joint_vel_max_env0": float(torch.as_tensor(joint_vel).abs().max().item()),
                    "joint_vel_max_all": joint_vel_max_all,
                }
                step_log_file.write(json.dumps(step_log_record) + "\n")

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if step_log_file is not None:
        step_log_file.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
