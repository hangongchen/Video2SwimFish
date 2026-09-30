# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games (PPO)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path
from distutils.util import strtobool

# Ensure the FISH source tree is on sys.path when running via isaaclab.sh.
_repo_root = Path(__file__).resolve().parents[2]
_fish_source = _repo_root / "source"
if _fish_source.is_dir():
    _fish_source_str = str(_fish_source)
    if _fish_source_str not in sys.path:
        sys.path.insert(0, _fish_source_str)
# Repo root itself, so `import bco` (the reusable success-rate function etc.) works too.
_repo_root_str = str(_repo_root)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games (PPO).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--target_offset_xy",
    type=float,
    nargs=2,
    metavar=("TARGET_X", "TARGET_Y"),
    default=None,
    help="Override the per-env XY target offset (meters) for Template-Fish tasks.",
)
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--horizon_length",
    type=int,
    default=None,
    help="Override RL-Games rollout length (steps per actor between optimizer updates).",
)
parser.add_argument(
    "--decimation",
    type=int,
    default=None,
    help="Override Fish environment decimation (physics steps per action).",
)
parser.add_argument(
    "--action_scale",
    type=float,
    default=None,
    help="Override Fish environment action_scale (torque multiplier).",
)
parser.add_argument(
    "--debug_logs",
    action="store_true",
    default=False,
    help="Print per-step action/torque/joint telemetry from the Fish env.",
)
parser.add_argument(
    "--log_alignment_overlay",
    action="store_true",
    default=False,
    help="Save Salmon IL reward-alignment overlay grids for the run.",
)
parser.add_argument("--wandb-project-name", type=str, default="fish_articulation_analytic_water", help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=os.environ.get("WANDB_ENTITY"), help="the entity (team) of wandb's project (default: $WANDB_ENTITY or your default entity)")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
parser.add_argument(
    "--weave",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="enable weave logging (uses project '<entity>/fish_articulation')",
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--debug_one_episode",
    action="store_true",
    default=False,
    help="Run only a single episode horizon for fast debugging.",
)
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

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
from datetime import datetime
import torch

import omni
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import class_to_dict, print_dict
from isaaclab.utils.io import dump_yaml

try:
    from isaaclab.utils.io import dump_pickle
except ImportError:
    import pickle

    def dump_pickle(filename: str, data: dict | object):
        """Fallback for Isaac Lab versions that do not ship dump_pickle."""
        if not filename.endswith("pkl"):
            filename += ".pkl"
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        if not isinstance(data, dict):
            data = class_to_dict(data)
        with open(filename, "wb") as f:
            pickle.dump(data, f)

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import FISH.tasks  # noqa: F401

from bco.success_metrics import extract_live_success_count


class FishWandbObserver(IsaacAlgoObserver):
    """Observer that logs RL-Games stats to wandb/weave if available."""

    def __init__(self):
        super().__init__()
        self._wandb = None
        self._weave = None
        self._success_ep_total = 0
        self._completed_ep_total = 0
        self._success_ep_epoch = 0
        self._completed_ep_epoch = 0
        try:
            import wandb

            self._wandb = wandb
        except Exception:
            self._wandb = None
        try:
            import weave

            self._weave = weave
        except Exception:
            self._weave = None

    def process_infos(self, infos, done_indices):
        super().process_infos(infos, done_indices)
        if not isinstance(infos, dict):
            return
        if "success" not in infos:
            return
        success = infos["success"]
        if not torch.is_tensor(success):
            try:
                success = torch.as_tensor(success)
            except Exception:
                return
        if done_indices is None:
            return
        if hasattr(done_indices, "numel"):
            if done_indices.numel() == 0:
                return
            done_ids = done_indices
        else:
            if len(done_indices) == 0:
                return
            done_ids = torch.as_tensor(done_indices, device=success.device)
        if done_ids.device != success.device:
            done_ids = done_ids.to(success.device)
        try:
            success_done = success[done_ids].float()
        except Exception:
            return
        success_count = int(success_done.sum().item())
        completed = int(done_ids.numel())
        self._success_ep_total += success_count
        self._completed_ep_total += completed
        self._success_ep_epoch += success_count
        self._completed_ep_epoch += completed

    def after_print_stats(self, frame, epoch_num, total_time):
        payload = {}

        # Gather ALL available scalars first (episodic averages + direct env scalars), then
        # publish ONLY the curated core/* namespace to wandb -- stable names, no metric flood.
        seen = {}
        if self.ep_infos:
            for key in self.ep_infos[0]:
                vals = []
                for ep_info in self.ep_infos:
                    v = ep_info[key]
                    if hasattr(v, "detach"):
                        v = v.detach()
                    try:
                        v = v.cpu().item() if getattr(v, "ndim", 1) == 0 else float(v.mean())
                    except Exception:
                        continue
                    vals.append(v)
                if vals:
                    seen[key] = sum(vals) / len(vals)
        for k, v in self.direct_info.items():
            try:
                seen[k] = float(v.detach().cpu().item() if hasattr(v, "detach") else v)
            except Exception:
                pass

        def pick(*subs):
            for s in subs:
                for k, v in seen.items():
                    if s in k:
                        return v
            return None

        # CORE metrics (the paper-facing four + a couple of task vitals).
        # `targets_per_ep` is fetched via THE canonical, reusable, verified-correct function
        # (bco/success_metrics.py) -- NOT `pick("succ_frac_running")`. That field ("core/
        # success_rate" in every version of this file before this one) has been DELETED from
        # this payload entirely: it is a live, in-progress running ratio, not a valid success
        # rate, and was mistakenly reported as one earlier in this project (verified ~2x too
        # low vs. the correct metric on a real run). See success_metrics.py's module
        # docstring for the full accounting of which of this task's several similarly-named
        # fields is actually correct.
        if self.mean_scores.current_size > 0:
            try:
                payload["core/return"] = float(self.mean_scores.get_mean())
                payload["train/return"] = payload["core/return"]      # literal alias, some
                                                                       # experiments ask for this name
            except Exception:
                pass
        for name, v in (("core/targets_per_ep", extract_live_success_count(seen)),
                        ("core/blowups_per_epoch", pick("reach10/blowups", "blowup")),
                        # kp=240 stability experiment's blow-up-event monitor
                        # (BlowupMonitorMixin) -- absent (None) for every other task.
                        ("train/blowup_count", pick("blowup/count")),
                        ("train/blowup_rate", pick("blowup/rate")),
                        ("train/nan_count", pick("blowup/nan_count")),
                        ("core/heading_cos", pick("reach/heading_cos")),
                        ("core/v_fwd_bl", pick("v_fwd_bl")),
                        # trajectory-following task's own custom reward components / metrics
                        # (salmon_swim_traj_follow_env.py) -- absent (None) for every other task.
                        ("core/traj_r_track", pick("traj/rew_track")),
                        ("core/traj_r_heading", pick("traj/rew_heading")),
                        ("core/traj_r_speed", pick("traj/rew_speed")),
                        ("core/traj_r_energy", pick("traj/rew_energy")),
                        ("core/traj_r_total", pick("traj/rew_total")),
                        ("core/traj_dist_bl", pick("traj/dist_bl")),
                        ("core/traj_speed_error_bl", pick("traj/speed_error_bl")),
                        ("core/traj_frac_within_tol", pick("traj/frac_within_tol")),
                        ("core/traj_phase_progress", pick("traj/phase_progress")),
                        ("core/traj_frechet_dist", pick("traj/frechet_dist")),
                        ("core/traj_completion_rate", pick("traj/completion_rate"))):
            if v is not None:
                payload[name] = v
        # benchmark: pass through every eval/* metric the env logs, plus train/blowup_rate
        for k, v in seen.items():
            if k.startswith("eval/") or k == "train/blowup_rate":
                try:
                    payload[k] = float(v)
                except Exception:
                    pass
            elif k == "rov/success_rate":      # benchmark ROV catch: the scheduler's convergence signal
                try:
                    payload["eval/success_rate"] = float(v)
                except Exception:
                    pass
        if "core/blowups_per_epoch" in payload:
            self._blowups_cum = getattr(self, "_blowups_cum", 0.0) + payload["core/blowups_per_epoch"]
            payload["core/blowups_cum"] = self._blowups_cum

        # core/success_rate_ep_level: Isaac Lab's OWN generic mechanism (infos["success"] at
        # done_indices) -- a THIRD, narrower, unrelated metric (did the TERMINAL attempt of a
        # just-ended episode succeed, not "average targets reached per completed episode").
        # Left untouched: it isn't the field this project's history flagged as wrong, just a
        # different one this observer's base class already provides.
        success_rate_epoch = 0.0
        if self._completed_ep_epoch > 0:
            success_rate_epoch = self._success_ep_epoch / self._completed_ep_epoch
        success_rate_completed = 0.0
        if self._completed_ep_total > 0:
            success_rate_completed = self._success_ep_total / self._completed_ep_total
        payload["core/success_rate_ep_level"] = success_rate_completed
        if self.writer is not None:
            for k, v in payload.items():                       # mirror core/* into tensorboard
                if k.startswith("core/"):
                    self.writer.add_scalar("Core/" + k.split("/", 1)[1], v, epoch_num)
        if self.writer is not None:
            self.writer.add_scalar("Episode/success_rate_epoch", success_rate_epoch, epoch_num)
            self.writer.add_scalar("Episode/success_rate_completed", success_rate_completed, epoch_num)
        self._success_ep_epoch = 0
        self._completed_ep_epoch = 0

        # One clean, grep/regex-friendly line per epoch with the 3 metrics scripts most often
        # need (reward, success rate, episode length) -- rl_games itself only ever writes these
        # to TensorBoard (rewards/iter, episode_lengths/iter) / wandb, never to stdout, so a
        # plain training log has no parseable per-epoch numbers without this.
        reward_mean = float(self.algo.game_rewards.get_mean()[0]) \
            if getattr(self.algo, "game_rewards", None) is not None and self.algo.game_rewards.current_size > 0 \
            else float("nan")
        ep_len_mean = float(self.algo.game_lengths.get_mean()) \
            if getattr(self.algo, "game_lengths", None) is not None and self.algo.game_lengths.current_size > 0 \
            else float("nan")
        # targets_per_ep: THE correct, debiased success metric (0..n_targets_per_episode
        # targets reached per COMPLETED episode; see bco/success_metrics.py). The old,
        # WRONG "success_rate" field (a live, in-progress running ratio -- successes-so-far /
        # attempts-so-far within each env's CURRENT, possibly-incomplete episode -- verified
        # to read ~2x too low vs. this one) has been deleted, not just relabeled.
        targets_per_ep = payload.get("core/targets_per_ep", float("nan"))
        blowup_rate = payload.get("train/blowup_rate", float("nan"))
        print(f"[BCO-METRICS] epoch={epoch_num} reward={reward_mean:.6f} "
              f"targets_per_ep={targets_per_ep:.6f} episode_length={ep_len_mean:.3f} "
              f"blowup_rate={blowup_rate:.6f}", flush=True)
        ev = {k: v for k, v in payload.items() if k.startswith("eval/") and "station_" not in k}
        if ev:
            print("[EVAL] epoch=%d " % epoch_num + " ".join(f"{k.split('/',1)[1]}={v:.4f}" for k, v in sorted(ev.items()))
                  + f" reward={reward_mean:.4f}", flush=True)

        # Let base class write to tensorboard / clear buffers.
        super().after_print_stats(frame, epoch_num, total_time)

        # Push to wandb/weave.
        if payload:
            if self._wandb is not None and self._wandb.run is not None:
                try:
                    self._wandb.log(payload, step=epoch_num)
                except Exception:
                    pass
            if self._weave is not None and hasattr(self._weave, "run"):
                try:
                    self._weave.run.log(payload)
                except Exception:
                    pass


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.target_offset_xy is not None and hasattr(env_cfg, "target_offset_xy"):
        env_cfg.target_offset_xy = tuple(args_cli.target_offset_xy)
    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "render_interval"):
            env_cfg.sim.render_interval = env_cfg.decimation
    if args_cli.action_scale is not None:
        env_cfg.action_scale = args_cli.action_scale
    if args_cli.debug_logs:
        env_cfg.debug_logs = True

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.join("logs", "rl_games", config_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name
    experiment_name = log_dir if args_cli.wandb_name is None else args_cli.wandb_name
    # initialize wandb if requested
    if args_cli.track:
        try:
            import wandb

            # Keep wandb from spawning extra processes that can hang under Kit.
            os.environ.setdefault("WANDB_START_METHOD", "thread")
            wandb.init(
                project=wandb_project,
                entity=args_cli.wandb_entity,
                name=experiment_name,
                config={
                    "task": args_cli.task,
                    "num_envs": env_cfg.scene.num_envs,
                    "device": env_cfg.sim.device,
                    "agent_name": config_name,
                    "max_epochs": agent_cfg["params"]["config"]["max_epochs"],
                },
            )
        except ImportError:
            print("[WARN] wandb is not installed; skipping tracking.")
        except Exception as exc:  # catch auth/network/version issues without aborting training
            print(f"[WARN] wandb init failed ({exc}); retrying in offline mode.")
            try:
                os.environ["WANDB_MODE"] = "offline"
                wandb.init(
                    project=wandb_project,
                    entity=args_cli.wandb_entity,
                    name=experiment_name,
                    config={
                        "task": args_cli.task,
                        "num_envs": env_cfg.scene.num_envs,
                        "device": env_cfg.sim.device,
                        "agent_name": config_name,
                        "max_epochs": agent_cfg["params"]["config"]["max_epochs"],
                    },
                    settings=wandb.Settings(mode="offline"),
                )
                print("[WARN] wandb running in offline mode; use 'wandb sync' to upload later.")
            except Exception as exc2:
                print(f"[WARN] wandb offline init also failed ({exc2}); continuing without wandb. Disable --track to silence this.")
    # initialize weave logging if requested
    if args_cli.weave:
        try:
            import weave

            weave.init(f"{args_cli.wandb_entity}/fish_articulation")
        except ImportError:
            print("[WARN] weave is not installed; skipping weave logging.")

    if hasattr(env_cfg, "pointcloud_capture_dir"):
        env_cfg.pointcloud_capture_dir = str(_repo_root / "run_pointclouds" / log_dir)
    if hasattr(env_cfg, "alignment_overlay_log_enabled"):
        env_cfg.alignment_overlay_log_enabled = bool(args_cli.log_alignment_overlay)
    if hasattr(env_cfg, "alignment_overlay_log_dir") and args_cli.log_alignment_overlay:
        env_cfg.alignment_overlay_log_dir = str(_repo_root / "run_pointclouds" / log_dir / "reward_alignment")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = os.path.join(log_root_path, log_dir)
    else:
        # omni.log is not always available in headless mode; fallback to print.
        try:
            omni.log.warn(
                "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
            )
        except AttributeError:
            print(
                "[WARN] IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
            )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # derive minibatch_size from minibatch_size_per_env so it auto-scales with num_envs
    # (and so the direct minibatch_size reads below don't KeyError when only per_env is set).
    _agent_config = agent_cfg["params"]["config"]
    if "minibatch_size" not in _agent_config and "minibatch_size_per_env" in _agent_config:
        _agent_config["minibatch_size"] = int(_agent_config["minibatch_size_per_env"]) * env.unwrapped.num_envs
        print(f"[INFO] minibatch_size = minibatch_size_per_env({_agent_config['minibatch_size_per_env']})"
              f" x num_envs({env.unwrapped.num_envs}) = {_agent_config['minibatch_size']}")

    if args_cli.debug_one_episode:
        max_episode_steps = getattr(env.unwrapped, "max_episode_length", None)
        if max_episode_steps is None:
            max_episode_steps = math.ceil(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
        agent_cfg["params"]["config"]["max_epochs"] = 1
        agent_cfg["params"]["config"]["horizon_length"] = max_episode_steps
        total_batch = max_episode_steps * env.unwrapped.num_envs
        agent_cfg["params"]["config"]["minibatch_size"] = min(
            agent_cfg["params"]["config"]["minibatch_size"], total_batch
        )
        print(
            "[INFO] Debug mode enabled: running a single episode "
            f"({max_episode_steps} steps, minibatch_size={agent_cfg['params']['config']['minibatch_size']})."
        )
    if args_cli.horizon_length is not None:
        agent_cfg["params"]["config"]["horizon_length"] = args_cli.horizon_length
        print(f"[INFO] Overriding horizon_length to {args_cli.horizon_length} steps.")
    total_batch = agent_cfg["params"]["config"]["horizon_length"] * env.unwrapped.num_envs
    minibatch_size = agent_cfg["params"]["config"]["minibatch_size"]
    if total_batch % minibatch_size != 0:
        new_minibatch = math.gcd(total_batch, minibatch_size)
        if new_minibatch == 0:
            new_minibatch = total_batch
        agent_cfg["params"]["config"]["minibatch_size"] = new_minibatch
        print(
            "[INFO] Adjusting minibatch_size "
            f"from {minibatch_size} to {new_minibatch} so it divides total batch {total_batch}."
        )
    # create runner from rl-games
    runner = Runner(FishWandbObserver())
    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent

    global_rank = int(os.getenv("RANK", "0"))
    if args_cli.track and global_rank == 0:
        try:
            import wandb

            # If we already initialized earlier, reuse the run.
            if wandb.run is None:
                wandb.init(
                    project=wandb_project,
                    entity=args_cli.wandb_entity,
                    name=experiment_name,
                    sync_tensorboard=True,
                    monitor_gym=True,
                    save_code=True,
                )
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
            wandb.config.update({"agent_cfg": agent_cfg})
        except ImportError:
            print("[WARN] wandb is not installed; skipping final run logging.")
        except Exception as exc:
            print(f"[WARN] wandb final logging skipped ({exc}).")

    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
