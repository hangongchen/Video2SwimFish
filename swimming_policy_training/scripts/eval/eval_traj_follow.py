"""Deterministic evaluation of a trained trajectory-following policy (PCA or Joint controller),
following the eval-metric spec from the phase-based-tracking redesign:

  eval/return           mean per-episode cumulative reward
  eval/frechet_dist     MEDIAN discrete Frechet distance across completed episodes (BL units)
  eval/tracking_error   mean ||p_fish - p*_t|| over all steps, in BL
  eval/speed_error      mean |v_fish - v_zef| over all steps, in BL/s
  eval/completion_rate  fraction of episodes whose phase reached the end of its own path
                        (as opposed to hitting the episode_length_s time_out backstop)

Reuses the env's OWN already-computed per-step diagnostics (self.extras["log"]["traj/dist_bl"],
["traj/speed_error_bl"]) and per-episode-end bookkeeping (self._last_frechet, self._last_completed,
populated by TrajFollowMixin._reset_idx/_compute_frechet_for BEFORE the just-ended env's path gets
replanted) -- NOT re-derived here. EVALUATION ONLY: no policy/reward/env modification.

Usage: python eval_traj_follow.py --task <traj-follow task> --checkpoint <pth> --tag <label>
Output: outputs/trajectory_fidelity/traj_follow_eval/eval_<tag>.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--tag", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--min_episodes_per_env", type=int, default=2,
                     help="stop once every env has completed at least this many episodes")
parser.add_argument("--max_steps", type=int, default=7200,
                     help="safety ceiling (4x the 1800-step/60s episode_length_s backstop)")
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym          # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402
import yaml                      # noqa: E402
import FISH.tasks                # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rl_games.torch_runner import Runner        # noqa: E402

OUT = REPO / "outputs/trajectory_fidelity/traj_follow_eval"
OUT.mkdir(parents=True, exist_ok=True)

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.seed = args.seed
env = gym.make(args.task, cfg=cfg).unwrapped
dev = env.device
N = args.num_envs

agent_cfg = yaml.safe_load((Path(args.checkpoint).resolve().parents[1] / "params" / "agent.yaml").read_text())
agent_cfg["params"]["config"].update({"num_actors": 1, "device": str(args.device),
                                      "device_name": str(args.device)})
agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = str(args.checkpoint)
agent_cfg["params"]["config"]["env_info"] = {
    "observation_space": gym.spaces.Box(-np.inf, np.inf, (env._obs_dim,)),
    "action_space": gym.spaces.Box(-1.0, 1.0, env.single_action_space.shape),
    "agents": 1,
}
runner = Runner(); runner.load(agent_cfg)
player = runner.create_player(); player.restore(str(args.checkpoint))
player.has_batch_dimension = True
print(f"### restored {args.checkpoint}", flush=True)

obs, _ = env.reset()
obs = env._get_observations()

ep_return = torch.zeros(N, device=dev)
completed_per_env = torch.zeros(N, dtype=torch.long, device=dev)

returns, frechets, completions = [], [], []
dist_bl_steps, speed_err_steps = [], []

target_total_eps = args.min_episodes_per_env * N
t = 0
for t in range(args.max_steps):
    ob = player._preproc_obs(obs["policy"])
    with torch.no_grad():
        act = player.model({"is_train": False, "prev_actions": None, "obs": ob,
                            "rnn_states": player.states})["mus"]
    obs, rew, terminated, truncated, _ = env.step(act.detach())
    ep_return += rew

    log = env.extras.get("log", {})
    if "traj/dist_bl" in log:
        dist_bl_steps.append(float(log["traj/dist_bl"]))
    if "traj/speed_error_bl" in log:
        speed_err_steps.append(float(log["traj/speed_error_bl"]))

    done = (terminated | truncated)
    done_ids = done.nonzero(as_tuple=False).squeeze(-1)
    if done_ids.numel() > 0:
        for e in done_ids.tolist():
            returns.append(float(ep_return[e].item()))
            fr = float(env._last_frechet[e].item())
            if np.isfinite(fr):
                frechets.append(fr)
            completions.append(bool(env._last_completed[e].item()))
            completed_per_env[e] += 1
        ep_return[done_ids] = 0.0

    if t % 200 == 0:
        print(f"t={t} episodes_done={len(returns)}/{target_total_eps} "
              f"min_per_env={int(completed_per_env.min().item())}", flush=True)
    if int(completed_per_env.min().item()) >= args.min_episodes_per_env:
        break

n_eps = len(returns)
results = {
    "tag": args.tag,
    "checkpoint": str(args.checkpoint),
    "task": args.task,
    "num_envs": N,
    "steps_run": t + 1,
    "n_episodes": n_eps,
    "n_frechet_valid": len(frechets),
    "eval/return_mean": float(np.mean(returns)) if returns else float("nan"),
    "eval/return_median": float(np.median(returns)) if returns else float("nan"),
    "eval/frechet_dist_median": float(np.median(frechets)) if frechets else float("nan"),
    "eval/frechet_dist_mean": float(np.mean(frechets)) if frechets else float("nan"),
    "eval/tracking_error_bl": float(np.mean(dist_bl_steps)) if dist_bl_steps else float("nan"),
    "eval/speed_error_bl_s": float(np.mean(speed_err_steps)) if speed_err_steps else float("nan"),
    "eval/completion_rate": float(np.mean(completions)) if completions else float("nan"),
}
print(json.dumps(results, indent=2), flush=True)
(OUT / f"eval_{args.tag}.json").write_text(json.dumps(results, indent=2))
print(f"### wrote {OUT}/eval_{args.tag}.json", flush=True)
print("TRAJ_FOLLOW_EVAL_DONE", flush=True)
sys.stdout.flush()
import os
os._exit(0)
