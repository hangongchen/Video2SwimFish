"""Deterministic mean-return eval for the video-state task (no traj-follow specifics needed):
used to compare the pure-BC checkpoint (no PPO) against a PPO-fine-tuned checkpoint on the
SAME env/reward, to check whether PPO fine-tuning is actually doing anything.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--tag", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--min_episodes_per_env", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=700)
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

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.seed = args.seed
env = gym.make(args.task, cfg=cfg).unwrapped
N = args.num_envs

agent_cfg = yaml.safe_load((Path(args.checkpoint).resolve().parents[1] / "params" / "agent.yaml").read_text()) \
    if (Path(args.checkpoint).resolve().parents[1] / "params" / "agent.yaml").exists() else \
    yaml.safe_load(open(REPO / "source/FISH/FISH/tasks/direct/fish/agents/rl_games_ppo_pca_videostate_cfg.yaml"))
agent_cfg["params"]["config"].update({"num_actors": 1, "device": str(args.device), "device_name": str(args.device)})
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
ep_return = torch.zeros(N, device=env.device)
completed = torch.zeros(N, dtype=torch.long, device=env.device)
returns = []

for t in range(args.max_steps):
    ob = player._preproc_obs(obs["policy"])
    with torch.no_grad():
        act = player.model({"is_train": False, "prev_actions": None, "obs": ob,
                            "rnn_states": player.states})["mus"]
    obs, rew, terminated, truncated, _ = env.step(act.detach())
    ep_return += rew
    done = terminated | truncated
    done_ids = done.nonzero(as_tuple=False).squeeze(-1)
    if done_ids.numel() > 0:
        for e in done_ids.tolist():
            returns.append(float(ep_return[e].item()))
        ep_return[done_ids] = 0.0
        completed[done_ids] += 1
    if int(completed.min().item()) >= args.min_episodes_per_env:
        break

print(f"### {args.tag}: n_episodes={len(returns)} mean_return={np.mean(returns):.3f} "
      f"median_return={np.median(returns):.3f} std={np.std(returns):.3f}", flush=True)
print("EVAL_DONE", flush=True)
import os
os._exit(0)
