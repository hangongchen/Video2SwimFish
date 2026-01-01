"""Roll out a checkpoint on the FREE-SWIM video-state task (no target, no reach protocol) and
save it in the SAME rollout_<tag>.npz schema eval_traj_fidelity.py produces, so
eval_biofidelity_suite.py can compare it against bco_pure / other tagged rollouts.

Simpler than eval_traj_fidelity.py by construction: there is no target to plant, no reach/
timeout/abort outcome to track -- just record the policy's own free-swimming motion for the
task's own episode_length_s (or until a blow-up/OOB termination), deterministically.

CAVEAT (documented, not hidden): the resulting rollout_<tag>.npz has NO "samples" field, so
eval_biofidelity_suite.py's path-Chamfer falls back to zef_reference.npz's own 82-row sample
table, comparing each env's free-chosen 20s path shape against an ARBITRARILY assigned short
real-ZeF snippet. That number measures "does this policy's path shape happen to resemble a
typical short ZeF path", NOT "did it accurately follow a given target path" the way the
Reach10-derived rollouts' Chamfer numbers do -- the two are not apples-to-apples. The
speed/turn-rate/body-deformation metrics ARE genuinely comparable (raw descriptors of the
policy's own motion, no target involved either way).

Usage: python rollout_videostate_for_biofidelity.py --checkpoint <pth> --tag <label>
Output: outputs/trajectory_fidelity/rollout_<tag>.npz
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Template-Salmon-Swim-PCA-VideoState-Misty-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--tag", required=True)
parser.add_argument("--num_envs", type=int, default=256)
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
import isaaclab.utils.math as math_utils        # noqa: E402

OUT = REPO / "outputs/trajectory_fidelity"
cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.seed = args.seed
env = gym.make(args.task, cfg=cfg).unwrapped
dev = env.device
N = args.num_envs
DT = env.cfg.decimation * env.cfg.sim.dt
BL = env._body_length
T_MAX = int(env.max_episode_length) + 5

ckpt_path = Path(args.checkpoint).resolve()
agent_yaml_sibling = ckpt_path.parents[1] / "params" / "agent.yaml"
agent_cfg = yaml.safe_load(agent_yaml_sibling.read_text()) if agent_yaml_sibling.exists() else \
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
root0 = env.robot.data.root_state_w
p0 = root0[:, 0:3].clone()
fwd0 = math_utils.quat_apply(root0[:, 3:7], torch.tensor([env._fwd_sign, 0.0, 0.0], device=dev).repeat(N, 1))
psi_a0 = torch.atan2(fwd0[:, 1], fwd0[:, 0])

nb = env.robot.data.body_link_pos_w.shape[1]
pos_hist = np.full((T_MAX, N, 3), np.nan, np.float32)
psi_hist = np.full((T_MAX, N), np.nan, np.float32)
bone_pos = np.full((T_MAX, N, nb, 3), np.nan, np.float32)
bone_quat = np.full((T_MAX, N, nb, 4), np.nan, np.float32)
end_step = np.full(N, -1, np.int32)
outcome = np.zeros(N, np.int32)          # 0=completed full episode, -1=blowup/OOB

for t in range(T_MAX):
    ob = player._preproc_obs(obs["policy"])
    with torch.no_grad():
        act = player.model({"is_train": False, "prev_actions": None, "obs": ob,
                            "rnn_states": player.states})["mus"]
    active = end_step < 0
    obs, rew, term, trunc, _ = env.step(act.detach())
    d = env.robot.data
    root = d.root_state_w
    fwd = math_utils.quat_apply(root[:, 3:7], torch.tensor([env._fwd_sign, 0.0, 0.0], device=dev).repeat(N, 1))
    pos_hist[t] = root[:, 0:3].cpu().numpy()
    psi_hist[t] = torch.atan2(fwd[:, 1], fwd[:, 0]).cpu().numpy()
    bone_pos[t] = d.body_link_pos_w.cpu().numpy()
    bone_quat[t] = d.body_link_quat_w.cpu().numpy()
    done_now = (term | trunc).cpu().numpy().astype(bool)
    blew = term.cpu().numpy().astype(bool)
    for e in np.nonzero(done_now & active)[0]:
        end_step[e] = t
        outcome[e] = -1 if blew[e] else 0
    if (end_step >= 0).all():
        break
    if t % 150 == 0:
        print(f"t={t} finished {int((end_step >= 0).sum())}/{N}", flush=True)
t_final = t
end_step[end_step < 0] = t_final

np.savez_compressed(
    OUT / f"rollout_{args.tag}.npz",
    pos=pos_hist[: t_final + 1], psi=psi_hist[: t_final + 1], end_step=end_step, outcome=outcome,
    bone_pos=bone_pos[: t_final + 1], bone_quat=bone_quat[: t_final + 1],
    p0=p0.cpu().numpy(), psi0=psi_a0.cpu().numpy(), dt=DT, body_length=BL,
    checkpoint=str(args.checkpoint))
print(f"### outcomes: completed={int((outcome == 0).sum())} blowup={int((outcome == -1).sum())}", flush=True)
print(f"### wrote {OUT}/rollout_{args.tag}.npz", flush=True)
print("ROLLOUT_DONE", flush=True)
import os
os._exit(0)
