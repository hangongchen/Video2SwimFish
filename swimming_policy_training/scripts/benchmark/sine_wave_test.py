"""Open-loop travelling-wave test: drive the LATERAL joints with q_i(t) = A sin(2*pi*f*t - 2*pi*lam*rank_i/N)
(head->tail phase lag) and measure whether the fish swims in a straight line. Writes metrics JSON + top-view mp4."""
import argparse, sys, json, re
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "source"), str(REPO / "scripts/zef_playback")): sys.path.insert(0, _p)
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", required=True); p.add_argument("--amp", type=float, default=0.5, help="normalized amplitude (1 = joint limit)")
p.add_argument("--freq", type=float, default=1.0); p.add_argument("--wavelengths", type=float, default=1.0, help="body wavelengths head->tail")
p.add_argument("--set", action="append", default=[], help="cfg override k=v"); p.add_argument("--seconds", type=float, default=15.0); p.add_argument("--num_envs", type=int, default=1); p.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True; app = AppLauncher(a).app
import gymnasium as gym, numpy as np, torch  # noqa: E402
import FISH.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
import curvature_utils as cu  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs); cfg.seed = 1
for kv in a.set:
    k, v = kv.split("=", 1); cur = getattr(cfg, k); setattr(cfg, k, (v.lower() in ("1", "true")) if isinstance(cur, bool) else type(cur)(v)); print(f"[sine] cfg.{k} = {getattr(cfg, k)}", flush=True)
env = gym.make(a.task, cfg=cfg).unwrapped; dev = env.device; DT = env.cfg.decimation * env.cfg.sim.dt; BL = float(env._body_length)
obs, _ = env.reset()
cj = env._control_joint_ids.tolist(); names = [env.robot.data.joint_names[i] for i in cj]
# head->tail order from the calibration file (joint_names stored head-first); fallback: chain index in the name
_cal = np.load(env.cfg.pca_calib_path, allow_pickle=True) if getattr(env.cfg, "pca_calib_path", None) else None
if _cal is not None and "joint_names" in _cal.files and set(map(str, _cal["joint_names"])) == set(names):
    _hf = list(map(str, _cal["joint_names"])); rank = np.array([_hf.index(n) for n in names], dtype=float); print("[sine] joint order from calibration (head-first)", flush=True)
else:
    rank = np.array([int((re.search(r"D6Joint_(\d+)", n) or re.search(r"(\d+)", "0")).group(1)) for n in names], dtype=float)
# head->tail order: use the bone x-position of each joint's child bone as the along-body coordinate
bp0 = env.robot.data.body_link_pos_w[0].cpu().numpy(); root_q = env.robot.data.root_state_w[0:1, 3:7]
fwd_w = mu.quat_apply(root_q, torch.tensor([[float(env._fwd_sign), 0.0, 0.0]], device=dev))[0].cpu().numpy()
# joint order along body: sort by rank (chain index) -- print both so the reviewer can check
order = np.argsort(rank); s_norm = np.zeros(len(cj)); s_norm[order] = np.linspace(0, 1, len(cj))
print(f"[sine] {len(cj)} lateral joints {names[:3]}... ; along-body order by chain index: {[names[i] for i in order][:4]}...", flush=True)
T = int(a.seconds / DT); act_dim = int(env.single_action_space.shape[-1]); assert act_dim == len(cj), (act_dim, len(cj))
pd_ = np.load(env.cfg.panel_hydro_path); rng = np.random.default_rng(0); sel = rng.choice(pd_["panel_bone"].shape[0], min(500, pd_["panel_bone"].shape[0]), replace=False)
pbi, prl = pd_["panel_bone"][sel].astype(int), pd_["panel_r_local"][sel].astype(np.float64)
geo = cu.rest_geometry(env.cfg.panel_hydro_path, head_sign=int(getattr(env.cfg, "head_sign", 1)))
root0 = env.robot.data.root_state_w[0, 0:3].cpu().numpy().copy(); fwd0 = fwd_w[:2] / (np.linalg.norm(fwd_w[:2]) + 1e-9); lat0 = np.array([-fwd0[1], fwd0[0]])
R, Q, K, PW, HEAD, RESET = [], [], [], [], [], []
with torch.no_grad():
    for t in range(T):
        phase = 2 * np.pi * a.freq * t * DT - 2 * np.pi * a.wavelengths * s_norm
        act = torch.tensor(a.amp * np.sin(phase), dtype=torch.float32, device=dev).unsqueeze(0).repeat(env.num_envs, 1)
        obs, rew, term, trunc, _ = env.step(act); d = env.robot.data; RESET.append(bool((term | trunc)[0]))
        R.append(d.root_state_w[0, 0:3].cpu().numpy().copy()); Q.append(d.joint_pos[0, cj].cpu().numpy().copy())
        bp, bq = d.body_link_pos_w[0].cpu().numpy(), d.body_link_quat_w[0].cpu().numpy()
        pr = cu.kappa_from_bones(bp, bq, geo, smooth_tol_m=2e-3); K.append(pr["kappa_bl"] if pr is not None else np.full(20, np.nan))
        PW.append(bp[pbi] + cu.quat_rotate_np(bq[pbi], prl))
        fw = mu.quat_apply(d.root_state_w[0:1, 3:7], torch.tensor([[float(env._fwd_sign), 0.0, 0.0]], device=dev))[0].cpu().numpy(); HEAD.append(np.degrees(np.arctan2(fw[1], fw[0])))
R, Q, K, PW, HEAD = map(np.array, (R, Q, K, PW, HEAD))
d = R - root0; along = d[:, :2] @ fwd0; lateral = d[:, :2] @ lat0
hd = np.degrees(np.angle(np.exp(1j * np.radians(HEAD - HEAD[0]))))
res = {"task": a.task, "amp": a.amp, "freq_hz": a.freq, "wavelengths": a.wavelengths, "seconds": a.seconds, "BL": BL, "n_joints": len(cj), "n_resets": int(sum(RESET)),
       "forward_displacement_BL": float(along[-1] / BL), "lateral_displacement_BL": float(lateral[-1] / BL), "lateral_rms_BL": float(np.sqrt(np.mean(lateral ** 2)) / BL),
       "path_straightness": float(abs(along[-1]) / (np.sum(np.linalg.norm(np.diff(d[:, :2], axis=0), axis=1)) + 1e-9)),
       "mean_forward_speed_BL_s": float(along[-1] / BL / a.seconds), "heading_change_deg": float(hd[-1]), "heading_abs_max_deg": float(np.abs(hd).max()),
       "vertical_drift_BL": float(d[-1, 2] / BL), "joint_amp_deg_mean": float(np.degrees(np.abs(Q).max(0)).mean()), "tail_kappa_bl_std": float(np.nanstd(K[:, 15]))}
print("[SINE] " + json.dumps(res), flush=True); json.dump(res, open(a.out + ".json", "w"), indent=1)
# top-view video
half = max(2.0 * BL, 0.1); fig, ax = plt.subplots(figsize=(8, 8), dpi=100); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
(tr,) = ax.plot([], [], "-", color="#e07b39", lw=1.5); sc = ax.scatter(PW[0, :, 0], PW[0, :, 1], s=4, c="#2b6ca3", alpha=0.75)
ax.plot([root0[0], root0[0] + fwd0[0] * 5 * BL], [root0[1], root0[1] + fwd0[1] * 5 * BL], "--", color="0.6", lw=1, label="initial heading")
tx = ax.text(0.02, 0.99, "", transform=ax.transAxes, va="top", fontsize=10); ax.set_title(f"open-loop travelling wave: A={a.amp} f={a.freq} Hz lam={a.wavelengths} BL", fontsize=11); ax.legend(loc="lower right", fontsize=8)
w = FFMpegWriter(fps=int(round(1 / DT)), bitrate=2500)
with w.saving(fig, a.out + ".mp4", dpi=100):
    for t in range(T):
        sc.set_offsets(PW[t, :, :2]); tr.set_data(R[: t + 1, 0], R[: t + 1, 1]); ax.set_xlim(R[t, 0] - half, R[t, 0] + half); ax.set_ylim(R[t, 1] - half, R[t, 1] + half)
        tx.set_text(f"t={t*DT:4.1f}s  forward={along[t]/BL:5.2f} BL  lateral={lateral[t]/BL:5.2f} BL  heading={hd[t]:6.1f} deg"); w.grab_frame()
plt.close(fig); print("SINE_DONE", flush=True)
import os; os._exit(0)
