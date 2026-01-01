"""Locomotion-fidelity rollouts: give the trained policy ZeF-derived targets and record its
spatial trajectory. EVALUATION ONLY -- no policy/reward/env-training modification.

For every sample i from zef_reference.npz (start frame t_i, target frame t_j):
  - the ZeF target displacement is expressed in the ZeF fish's INITIAL BODY FRAME at t_i
    (rotate by -psi_zef(t_i), units BL),
  - the SAME body-frame displacement is planted in front of the agent's own start pose:
    target_world = p_agent0 + R(psi_agent0) @ disp_body * BL_agent  (2D, z = target height),
  - the agent then acts on its NORMAL observation (target direction + distance etc.) --
    it never sees the ZeF trajectory,
  - one env per sample; recording stops per-env at its first attempt outcome
    (reach < 0.2 m | 20 s timeout | 3 m abort -- the same protocol thresholds as training).

Usage: ... eval_traj_fidelity.py --task <reach10 task> --checkpoint <pth> --tag pca|cpg
Output: outputs/trajectory_fidelity/rollout_<tag>.npz
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--tag", type=str, required=True)
parser.add_argument("--ref", type=str, default="", help="reference npz (default: zef_reference.npz)")
parser.add_argument("--outdir", type=str, default="", help="output dir (default: trajectory_fidelity)")
parser.add_argument("--num_envs", type=int, default=0,
                     help="0 = one env per reference sample (default, N=len(samples)); "
                          ">0 TILES the reference samples cyclically (idx %% N_samples) to "
                          "fill this many envs -- e.g. 256 replays each of the 82 real ZeF "
                          "targets ~3x, each with its own independent physics/policy rollout.")
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

OUT = Path(args.outdir) if args.outdir else REPO / "outputs/trajectory_fidelity"
Z = np.load(args.ref if args.ref else OUT / "zef_reference.npz")
S_orig = Z["samples"]                             # (N_samples, 3): t_i, t_j, horizon_idx
N_samples = S_orig.shape[0]
N = args.num_envs if args.num_envs > 0 else N_samples
tidx = np.arange(N) % N_samples                   # tile cyclically when N > N_samples
S = S_orig[tidx]                                  # (N, 3) -- every downstream line already
                                                   # treats S.shape[0] as both sample count AND
                                                   # env count, so tiling here needs no other edits
p_z, psi_z = Z["p_bl"], Z["psi"]

cfg = parse_env_cfg(args.task, device=args.device, num_envs=N)
cfg.reach_frame_every_n_episodes = 10 ** 9
cfg.seed = 7                                      # same seed both methods -> same spawn noise
env = gym.make(args.task, cfg=cfg).unwrapped
dev = env.device
DT = env.cfg.decimation * env.cfg.sim.dt
BL = env._body_length
T_MAX = env._att_timeout_steps + 20

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
root = env.robot.data.root_state_w
p0 = root[:, 0:3].clone()
fwd = math_utils.quat_apply(root[:, 3:7],
                            torch.tensor([1.0, 0, 0], device=dev).repeat(N, 1))
psi_a0 = torch.atan2(fwd[:, 1], fwd[:, 0])

# ZeF body-frame target displacement per sample (BL units, 2D)
dz = p_z[S[:, 1]] - p_z[S[:, 0]]
c0, s0 = np.cos(-psi_z[S[:, 0]]), np.sin(-psi_z[S[:, 0]])
disp_body = np.stack([c0 * dz[:, 0] - s0 * dz[:, 1],
                      s0 * dz[:, 0] + c0 * dz[:, 1]], 1)       # (N, 2) fwd/lat in BL
db = torch.tensor(disp_body, device=dev, dtype=torch.float32)
ca, sa = torch.cos(psi_a0), torch.sin(psi_a0)
tgt = p0.clone()
tgt[:, 0] = p0[:, 0] + (ca * db[:, 0] - sa * db[:, 1]) * BL
tgt[:, 1] = p0[:, 1] + (sa * db[:, 0] + ca * db[:, 1]) * BL
tgt[:, 2] = env._target_height
env.target_positions_w[:] = tgt
env._prev_distance[:] = torch.linalg.norm(tgt - p0, dim=1)
print(f"### {N} samples planted; disp_body fwd p50={np.median(disp_body[:,0]):.2f} BL "
      f"lat p50={np.median(np.abs(disp_body[:,1])):.2f} BL", flush=True)

pos_hist = np.full((T_MAX, N, 3), np.nan, np.float32)
psi_hist = np.full((T_MAX, N), np.nan, np.float32)
nb = env.robot.data.body_link_pos_w.shape[1]
bone_pos = np.full((T_MAX, N, nb, 3), np.nan, np.float32)   # for midline point-cloud keyframes
bone_quat = np.full((T_MAX, N, nb, 4), np.nan, np.float32)
end_step = np.full(N, -1, np.int32)
outcome = np.zeros(N, np.int32)                   # 1=reach 0=timeout/abort  -1=blowup
obs = env._get_observations()
for t in range(T_MAX):
    ob = player._preproc_obs(obs["policy"])
    with torch.no_grad():
        act = player.model({"is_train": False, "prev_actions": None, "obs": ob,
                            "rnn_states": player.states})["mus"]
    prev_att = env._att_idx.cpu().numpy().copy()
    obs, rew, term, trunc, _ = env.step(act.detach())
    d = env.robot.data
    active = end_step < 0
    root = d.root_state_w
    fwd = math_utils.quat_apply(root[:, 3:7], torch.tensor([1.0, 0, 0], device=dev).repeat(N, 1))
    pos_hist[t] = root[:, 0:3].cpu().numpy()
    psi_hist[t] = torch.atan2(fwd[:, 1], fwd[:, 0]).cpu().numpy()
    bone_pos[t] = d.body_link_pos_w.cpu().numpy()
    bone_quat[t] = d.body_link_quat_w.cpu().numpy()
    adv = (env._att_idx.cpu().numpy() > prev_att)
    reach = env.extras["success"].cpu().numpy().astype(bool)
    blew = term.cpu().numpy().astype(bool)
    for e in np.nonzero((adv | blew) & active)[0]:
        end_step[e] = t
        outcome[e] = -1 if blew[e] else (1 if reach[e] else 0)
    if (end_step >= 0).all():
        break
    if t % 150 == 0:
        print(f"t={t} finished {int((end_step>=0).sum())}/{N}", flush=True)
end_step[end_step < 0] = t

np.savez_compressed(
    OUT / f"rollout_{args.tag}.npz",
    pos=pos_hist[: t + 1], psi=psi_hist[: t + 1], end_step=end_step, outcome=outcome,
    bone_pos=bone_pos[: t + 1], bone_quat=bone_quat[: t + 1],
    p0=p0.cpu().numpy(), psi0=psi_a0.cpu().numpy(), target_w=tgt.cpu().numpy(),
    disp_body=disp_body, samples=S, sample_tidx=tidx, dt=DT, body_length=BL,
    checkpoint=str(args.checkpoint))
print(f"### outcomes: reach={int((outcome==1).sum())} timeout/abort={int((outcome==0).sum())} "
      f"blowup={int((outcome==-1).sum())}")
print(f"### wrote {OUT}/rollout_{args.tag}.npz", flush=True)
print("TRAJ_FID_DONE", flush=True)
sys.stdout.flush()
import os
os._exit(0)
