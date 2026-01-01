"""Top-view mp4 of ONE benchmark policy rolling out in its env (env 0 recorded).
  --policy rlgames  : rl_games checkpoint (+ --run_dir with params/agent.yaml)   e.g. BLM+RL trajfollow
  --policy bc       : bco.bc_trainer model + state dict (checkpoints/benchmark/*/bc_policy.pt)
Draws the FEM panel cloud, the fish trail, and (trajfollow envs) the reference path + look-ahead point.
"""
import argparse, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "source"), str(REPO / "scripts/zef_playback"), str(REPO)):
    sys.path.insert(0, _p)
if "--from_npz" in sys.argv:            # REPLAY from a saved rollout: no Isaac needed
    import argparse as _ap
    _q = _ap.ArgumentParser(); _q.add_argument("--from_npz", required=True); _q.add_argument("--out", required=True); _q.add_argument("--title", default="")
    _q.add_argument("--show_path", type=int, default=1); _q.add_argument("--zoom_bl", type=float, default=1.5); _q.add_argument("--n_points", type=int, default=4000)
    _r, _ = _q.parse_known_args(); import numpy as np
    sys.path.insert(0, str(REPO / "scripts/zef_playback")); import curvature_utils as cu
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.animation import FFMpegWriter
    z = np.load(_r.from_npz, allow_pickle=True); bp, bq, roots, reset, prog = z["bp"], z["bq"], z["root"], z["reset"], z["prog"]; path, pstar = z["path"], z["pstar"]
    BL, DT = float(z["BL"]), float(z["DT"]); T = len(bp); pd_ = np.load(str(z["panel_hydro_path"])); rng = np.random.default_rng(0)
    _nb, _rl = (pd_["vert_bone"], pd_["vert_local"]) if "vert_local" in pd_.files else (pd_["panel_bone"], pd_["panel_r_local"])
    sel = np.arange(_nb.shape[0]) if _nb.shape[0] <= _r.n_points else rng.choice(_nb.shape[0], _r.n_points, replace=False)
    pbi, prl = _nb[sel].astype(int), _rl[sel].astype(np.float64); _bx = np.argsort(pd_["bone_rest_pos"][:, 0])
    pw = np.stack([bp[t][pbi] + cu.quat_rotate_np(bq[t][pbi], prl) for t in range(T)])
    trail = roots.copy()
    for t in range(1, T):
        if reset[t - 1]: trail[t - 1] = np.nan
    half = max(_r.zoom_bl * BL, 0.12); fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    (pl,) = ax.plot([], [], "--", color="0.55", lw=1.5, label="reference path (real fish)"); (tr,) = ax.plot([], [], "-", color="#e07b39", lw=1.5, label="fish trail")
    sc = ax.scatter(pw[0, :, 0], pw[0, :, 1], s=2, c="#2b6ca3", alpha=0.35, label="fish body (skin)"); (bc,) = ax.plot([], [], "-o", color="k", lw=2.5, ms=3, label="skeleton (bone chain)")
    (ps,) = ax.plot([], [], "o", ms=7, color="crimson", label="look-ahead target"); tx = ax.text(0.02, 0.99, "", transform=ax.transAxes, va="top", fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(_r.title, fontsize=12)
    ax.legend(handles=([pl, ps] if _r.show_path else []) + [tr, sc, bc], loc="lower right", fontsize=8, framealpha=0.8)
    w = FFMpegWriter(fps=int(round(1 / DT)), bitrate=2500)
    with w.saving(fig, _r.out, dpi=100):
        for t in range(T):
            sc.set_offsets(pw[t, :, :2]); tr.set_data(trail[: t + 1, 0], trail[: t + 1, 1]); bc.set_data(bp[t][_bx, 0], bp[t][_bx, 1])
            if _r.show_path and len(path[t]): pl.set_data(path[t][:, 0], path[t][:, 1]); ps.set_data([pstar[t][0]], [pstar[t][1]])
            ax.set_xlim(roots[t, 0] - half, roots[t, 0] + half); ax.set_ylim(roots[t, 1] - half, roots[t, 1] + half)
            pg = f"  path progress={prog[t]*100:.0f}%" if (_r.show_path and np.isfinite(prog[t])) else ""
            tx.set_text(f"t={t*DT:4.1f}s   body length={BL*100:.1f} cm{pg}"); w.grab_frame()
    plt.close(fig); print(f"RENDER_DONE {_r.out} (replay)"); sys.exit(0)
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", required=True); p.add_argument("--policy", choices=["rlgames", "bc"], required=True)
p.add_argument("--ckpt", required=True); p.add_argument("--run_dir", default=None); p.add_argument("--bc_yaml", default=None)
p.add_argument("--seconds", type=float, default=10.0); p.add_argument("--seed", type=int, default=3)
p.add_argument("--out", required=True); p.add_argument("--title", default=""); p.add_argument("--show_path", type=int, default=1)
p.add_argument("--stochastic", type=int, default=0, help="sample actions from the policy Gaussian (training regime) instead of the mean")
p.add_argument("--from_npz", default=None, help="re-draw from a saved rollout (<out>.rollout.npz) without simulating")
p.add_argument("--zoom_bl", type=float, default=2.5, help="half-width of the view in body lengths")
p.add_argument("--num_envs", type=int, default=1, help="envs to simulate (env 0 is recorded); use the eval regime, e.g. 256")
p.add_argument("--env_set", action="append", default=[], help="cfg override k=v applied to env_cfg before the env is built (e.g. fixed_traj_ti=1234)")
p.add_argument("--record_envs", type=int, default=0, help="also save a multi-env rollout (<out>.multi.rollout.npz) of the first K envs with per-episode completion flags and per-env phase progress")
p.add_argument("--hist", type=int, default=1, help="bc policy: stack this many past observations (BLM+IL v3 uses 5)"); p.add_argument("--init_from_video", type=int, default=0, help="set the initial BLM coefficients from random real video frames (Task-2 eval protocol)"); p.add_argument("--n_modes", type=int, default=0, help="bc policy: zero action dims >= n_modes (BLM+IL v3 uses 4)")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import gymnasium as gym, numpy as np, torch, yaml            # noqa: E402
import FISH.tasks                                             # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg                # noqa: E402
import curvature_utils as cu                                  # noqa: E402
import matplotlib; matplotlib.use("Agg")                      # noqa: E402
import matplotlib.pyplot as plt                               # noqa: E402
from matplotlib.animation import FFMpegWriter                 # noqa: E402

cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs); cfg.seed = a.seed
for kv in a.env_set:
    k, v = kv.split("=", 1); cur = getattr(cfg, k); setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true")); print(f"[render] cfg.{k} = {getattr(cfg, k)}", flush=True)
env = gym.make(a.task, cfg=cfg).unwrapped
dev = env.device; DT = env.cfg.decimation * env.cfg.sim.dt; BL = float(env._body_length); T = int(a.seconds / DT)
obs, _ = env.reset(); ob0 = obs["policy"] if isinstance(obs, dict) else obs
if a.init_from_video and hasattr(env, "_a"):
    _ds = Path(a.ckpt).parent / "dataset.npz"          # BC dataset: obs block = [x,y,vx,vy,psi, a/a_max (K)] (+history)
    if _ds.exists():
        _S = np.load(_ds)["s"]; _K = int(env._a.shape[1]); _rows = np.random.default_rng(3).choice(len(_S), env.num_envs)
        _a0 = torch.tensor(_S[_rows, 5:5 + _K], dtype=torch.float32, device=dev) * env._a_max
        env._a[:] = _a0.clamp(-env._a_max, env._a_max); print(f"[init] BLM coefficients set from {env.num_envs} real video frames", flush=True)
        obs = env._get_observations()
obs_dim, act_dim = int(ob0.shape[-1]), int(env.single_action_space.shape[-1])
if a.policy == "rlgames":
    from rl_games.torch_runner import Runner
    ac = yaml.safe_load((Path(a.run_dir) / "params/agent.yaml").read_text())
    ac["params"]["config"].update({"num_actors": 1, "device": str(dev), "device_name": str(dev)})
    ac["params"]["load_checkpoint"] = True; ac["params"]["load_path"] = a.ckpt
    ac["params"]["config"]["env_info"] = {"observation_space": gym.spaces.Box(-np.inf, np.inf, (obs_dim,)),
                                          "action_space": gym.spaces.Box(-1.0, 1.0, (act_dim,)), "agents": 1}
    r = Runner(); r.load(ac); player = r.create_player(); player.has_batch_dimension = True; player.restore(a.ckpt)
    def act_fn(o):
        out = player.model({"is_train": False, "prev_actions": None, "obs": player._preproc_obs(o), "rnn_states": player.states})
        return (out["mus"] + out["sigmas"] * torch.randn_like(out["mus"])).clamp(-1, 1) if a.stochastic else out["mus"]
else:
    from bco.bc_trainer import build_rl_games_model
    model = build_rl_games_model(a.bc_yaml, a.hist * obs_dim, act_dim)
    ck = torch.load(a.ckpt, map_location=dev); model.load_state_dict(ck["model"] if "model" in ck else ck); model.to(dev).eval()
    _hb = []
    def act_fn(o):
        nonlocal_hb = _hb
        if a.hist > 1:
            if not nonlocal_hb: nonlocal_hb.extend([o.clone() for _ in range(a.hist)])
            nonlocal_hb.insert(0, o.clone()); del nonlocal_hb[a.hist:]; oin = torch.cat(nonlocal_hb, dim=1)
        else: oin = o
        mus = model({"is_train": False, "prev_actions": None, "obs": oin})["mus"]
        _asf = Path(a.ckpt).parent / "action_scale.json"        # BLM+IL v4: labels were normalised by the smoothed p99 -> rescale
        if _asf.exists():
            import json as _j; mus = (mus * torch.tensor(_j.load(open(_asf))["action_scale"], device=mus.device, dtype=mus.dtype)).clamp(-1, 1)
        if a.n_modes > 0: mus[:, a.n_modes:] = 0.0
        return mus
print(f"[render] {a.task} policy={a.policy} obs={obs_dim} act={act_dim} BL={BL:.3f} T={T}", flush=True)

pd_ = np.load(env.cfg.panel_hydro_path); rng = np.random.default_rng(0)
if "vert_local" in pd_.files:      # dense skin vertices rigidly attached to their nearest bone (24k for a 6.7 cm white bass)
    _nb, _rl = pd_["vert_bone"], pd_["vert_local"]
else:                               # fallback: the coarse hydro panels
    _nb, _rl = pd_["panel_bone"], pd_["panel_r_local"]
sel = np.arange(_nb.shape[0]) if _nb.shape[0] <= 4000 else rng.choice(_nb.shape[0], 4000, replace=False)
pbi, prl = _nb[sel].astype(int), _rl[sel].astype(np.float64)
HAS_TRAJ = hasattr(env, "_traj_path")
rec = {"bp": [], "bq": [], "root": [], "path": [], "pstar": [], "prog": [], "reset": []}
K = a.record_envs; mrec = {"bp": [], "bq": [], "root": [], "pstar": [], "reset": [], "completed": [], "progress": [], "path": []} if K > 0 else None
def snap_path_e(e):
    n = int(env._traj_len[e].item()); return env._traj_path[e, :n].cpu().numpy().copy()
cur_paths = [snap_path_e(e) for e in range(K)] if (K > 0 and HAS_TRAJ) else None
_bx = np.argsort(pd_["bone_rest_pos"][:, 0])   # bone order along the body for the chain polyline
def snap_path():
    if not HAS_TRAJ: return None
    n = int(env._traj_len[0].item()); return env._traj_path[0, :n].cpu().numpy().copy()
cur_path = snap_path()
with torch.no_grad():
    for t in range(T):
        o = obs["policy"] if isinstance(obs, dict) else obs
        obs, rew, term, trunc, _ = env.step(act_fn(o).detach())
        d = env.robot.data
        was_reset = bool((term | trunc)[0]); rec["reset"].append(was_reset)
        if was_reset: cur_path = snap_path()
        rec["bp"].append(d.body_link_pos_w[0].cpu().numpy().copy()); rec["bq"].append(d.body_link_quat_w[0].cpu().numpy().copy())
        rec["root"].append(d.root_state_w[0, 0:3].cpu().numpy().copy()); rec["path"].append(cur_path)
        rec["pstar"].append(env._cur_p_star[0].cpu().numpy().copy() if HAS_TRAJ else None)
        lg = env.extras.get("log", {}); rec["prog"].append(float(lg["traj/phase_progress"]) if "traj/phase_progress" in lg else float("nan"))
        if K > 0:
            rs = (term | trunc)[:K].cpu().numpy()
            mrec["bp"].append(d.body_link_pos_w[:K].cpu().numpy().copy()); mrec["bq"].append(d.body_link_quat_w[:K].cpu().numpy().copy())
            mrec["root"].append(d.root_state_w[:K, 0:3].cpu().numpy().copy()); mrec["reset"].append(rs)
            mrec["pstar"].append(env._cur_p_star[:K].cpu().numpy().copy() if HAS_TRAJ else np.full((K, 3), np.nan))
            mrec["completed"].append(env._last_completed[:K].cpu().numpy().copy() if HAS_TRAJ else np.zeros(K, bool))   # valid on reset steps
            mrec.setdefault("traj_ti", []).append(env._traj_ti[:K].cpu().numpy().copy() if hasattr(env, "_traj_ti") else np.full(K, -1))
            if HAS_TRAJ:
                _s, _ph = env._project_phase(); mrec["progress"].append((_ph[:K].float() / (env._traj_len[:K] - 1).clamp(min=1).float()).cpu().numpy())
                mrec["path"].append([p.copy() for p in cur_paths])
                for e in np.nonzero(rs)[0]: cur_paths[e] = snap_path_e(int(e))
            else: mrec["progress"].append(np.full(K, np.nan))
np.savez_compressed(a.out.replace(".mp4", "") + ".rollout.npz", bp=np.array(rec["bp"]), bq=np.array(rec["bq"]), root=np.array(rec["root"]),
                    reset=np.array(rec["reset"]), prog=np.array(rec["prog"]), path=np.array([p if p is not None else np.zeros((0, 3)) for p in rec["path"]], dtype=object),
                    pstar=np.array([p if p is not None else np.full(3, np.nan) for p in rec["pstar"]]), BL=BL, DT=DT, panel_hydro_path=str(env.cfg.panel_hydro_path), task=a.task)
print("[render] rollout saved ->", a.out.replace(".mp4", "") + ".rollout.npz", flush=True)
if K > 0:
    Tm = len(mrec["bp"]); Pmax = max(len(p) for fr in mrec["path"] for p in fr) if mrec["path"] else 0
    parr = np.full((Tm, K, Pmax, 3), np.nan, np.float32); plen = np.zeros((Tm, K), np.int32)
    for t, fr in enumerate(mrec["path"]):
        for e, p in enumerate(fr): parr[t, e, :len(p)] = p; plen[t, e] = len(p)
    np.savez_compressed(a.out.replace(".mp4", "") + ".multi.rollout.npz", bp=np.array(mrec["bp"]), bq=np.array(mrec["bq"]), root=np.array(mrec["root"]),
                        pstar=np.array(mrec["pstar"]), reset=np.array(mrec["reset"]), completed=np.array(mrec["completed"]), progress=np.array(mrec["progress"]),
                        path=parr, path_len=plen, traj_ti=np.array(mrec.get("traj_ti", [])), BL=BL, DT=DT, panel_hydro_path=str(env.cfg.panel_hydro_path), task=a.task)
    print("[render] multi-env rollout saved ->", a.out.replace(".mp4", "") + ".multi.rollout.npz", flush=True)
roots = np.array(rec["root"])
trail = roots.copy()
for t in range(1, T):
    if rec["reset"][t - 1]: trail[t - 1] = np.nan     # do not connect the trail across an episode reset
pw = np.stack([rec["bp"][t][pbi] + cu.quat_rotate_np(rec["bq"][t][pbi], prl) for t in range(T)])
half = max(a.zoom_bl * BL, 0.12)
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
(pl,) = ax.plot([], [], "--", color="0.55", lw=1.5, label="reference path (real fish)")
(tr,) = ax.plot([], [], "-", color="#e07b39", lw=1.5, label="fish trail")
sc = ax.scatter(pw[0, :, 0], pw[0, :, 1], s=2, c="#2b6ca3", alpha=0.35, label="fish body (skin)")
(bc,) = ax.plot([], [], "-o", color="k", lw=2.5, ms=3, label="skeleton (bone chain)")
(ps,) = ax.plot([], [], "o", ms=7, color="crimson", label="look-ahead target")
tx = ax.text(0.02, 0.99, "", transform=ax.transAxes, va="top", fontsize=10)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(a.title, fontsize=12)
ax.legend(handles=[h for h in ([pl, ps] if a.show_path else []) + [tr, sc, bc]], loc="lower right", fontsize=8, framealpha=0.8)
w = FFMpegWriter(fps=int(round(1 / DT)), bitrate=2500)
with w.saving(fig, a.out, dpi=100):
    for t in range(T):
        sc.set_offsets(pw[t, :, :2]); tr.set_data(trail[: t + 1, 0], trail[: t + 1, 1]); bc.set_data(rec["bp"][t][_bx, 0], rec["bp"][t][_bx, 1])
        if a.show_path and rec["path"][t] is not None:
            pl.set_data(rec["path"][t][:, 0], rec["path"][t][:, 1])
            if rec["pstar"][t] is not None: ps.set_data([rec["pstar"][t][0]], [rec["pstar"][t][1]])
        else:
            pl.set_data([], []); ps.set_data([], [])
        ax.set_xlim(roots[t, 0] - half, roots[t, 0] + half); ax.set_ylim(roots[t, 1] - half, roots[t, 1] + half)
        prog = f"  path progress={rec['prog'][t]*100:.0f}%" if (a.show_path and np.isfinite(rec["prog"][t])) else ""
        tx.set_text(f"t={t*DT:4.1f}s   body length={BL*100:.1f} cm{prog}")
        w.grab_frame()
plt.close(fig); print(f"RENDER_DONE {a.out}", flush=True); sys.stdout.flush()
import os; os._exit(0)
