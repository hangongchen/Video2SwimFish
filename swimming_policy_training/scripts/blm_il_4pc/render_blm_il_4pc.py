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
p.add_argument("--hist", type=int, default=1, help="bc policy: stack this many past observations (BLM+IL v3 uses 5)"); p.add_argument("--init_from_video", type=int, default=0, help="set the initial BLM coefficients from random real video frames (Task-2 eval protocol)"); p.add_argument("--n_modes", type=int, default=0, help="bc policy: zero action dims >= n_modes (BLM+IL v3 uses 4)")
p.add_argument("--drop_xyp", type=int, default=0, help="LF_Test: zero obs dims 0,1,4 (x,y,psi) before the policy")
p.add_argument("--real_ref", type=int, default=0, help="LF_Test: start env 0 from a real video frame and overlay the REAL fish path of the same segment (grey dashed) + its current position (red dot)"); p.add_argument("--ref_seed", type=int, default=3)
p.add_argument("--calib", default=None, help="LF_Test: override cfg.pca_calib_path (recalibrated kappa->joint map)")
p.add_argument("--manifold_dir", default=None, help="LF_Test: manifold dir for --real_ref (curvature/reference/basis)"); p.add_argument("--basis", default=None, help="LF_Test: override cfg.pca_basis_path")
p.add_argument("--max_kbl", type=float, default=4.0)
p.add_argument("--n_pc", type=int, default=4, help="LF_Test: env uses only the first n_pc PCA modes")
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
if hasattr(cfg, "pca_num_modes"): cfg.pca_num_modes = a.n_pc   # LF_Test: first n_pc PCs only
if a.calib: cfg.pca_calib_path = a.calib; print("[LF_Test] pca_calib_path ->", a.calib, flush=True)
if a.basis: cfg.pca_basis_path = a.basis; print("[LF_Test] pca_basis_path ->", a.basis, flush=True)
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
REAL_PATH = None
if a.real_ref and hasattr(env, "_a"):
    import re as _re
    _m = _re.search(r"Bench-([a-z_]+)_fish(\d+)", a.task); _sp, _num = _m.group(1), _m.group(2)
    _MAN = Path(a.manifold_dir) if a.manifold_dir else REPO / f"data/species_manifold/{_sp}_fix/top"
    _c = np.load(_MAN / f"fish{_num}_curvature.npz"); _r = np.load(_MAN / f"fish{_num}_reference.npz", allow_pickle=True); _b = np.load(_MAN / f"fish{_num}_pca_basis.npz")
    _K = int(env._a.shape[1]); _V = _b["components"][:_K].astype(np.float64); _mk = _b["mean"].astype(np.float64)
    _kap = _c["kappa_bl"].astype(np.float64); _t = _c["t_sec"].astype(np.float64); _ok = _c["valid"].astype(bool) & (np.nan_to_num(np.abs(_kap), nan=np.inf).max(1) < float(a.max_kbl))
    _p = _r["p_bl"].astype(np.float64) * np.array([1.0, -1.0]); _psi = -_r["psi"].astype(np.float64)      # same handedness fix as the BC data (--mirror_traj 1)
    _segs = []
    for _a0, _b0 in _r["runs"]:
        _i = np.arange(int(_a0), int(_b0)); _o = _ok[_i]
        if _o.sum() < 8: continue
        _tt = _t[_i]; _g = np.arange(_tt[0], _tt[-1], DT); _it = lambda y: np.stack([np.interp(_g, _tt[_o], y[_o, j]) for j in range(y.shape[1])], 1)
        _A = _it((_kap[_i] - _mk) @ _V.T); _A = np.stack([np.convolve(np.pad(_A[:, j], (2, 2), mode="edge"), np.ones(5) / 5, mode="valid") for j in range(_K)], 1)
        _P = _it(_p[_i]); _PS = np.interp(_g, _tt[_o], np.unwrap(_psi[_i])[_o])
        for _s0 in range(0, len(_g) - T - 1, T // 2): _segs.append((_A[_s0], _P[_s0:_s0 + T + 1], _PS[_s0:_s0 + T + 1]))
    _pick = _segs[np.random.default_rng(a.ref_seed).integers(len(_segs))]
    env._a[0] = torch.tensor(_pick[0], dtype=torch.float32, device=dev).clamp(-env._a_max, env._a_max); obs = env._get_observations()
    import isaaclab.utils.math as _mu
    _root0 = env.robot.data.root_state_w[0]; _fwd = _mu.quat_apply(_root0[3:7].unsqueeze(0), torch.tensor([[float(env._fwd_sign), 0.0, 0.0]], device=dev))[0]
    _yaw0 = float(torch.atan2(_fwd[1], _fwd[0])); _ang = _yaw0 - _pick[2][0]; _R = np.array([[np.cos(_ang), -np.sin(_ang)], [np.sin(_ang), np.cos(_ang)]])
    _xy = _root0[0:2].cpu().numpy() + ((_pick[1] - _pick[1][0]) @ _R.T) * BL
    REAL_PATH = np.concatenate([_xy, np.full((len(_xy), 1), float(_root0[2]))], 1)
    print(f"[real_ref] env 0 starts from a real frame; real path of the same {a.seconds:.0f}s segment overlaid ({len(_segs)} segments available)", flush=True)
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
        if a.drop_xyp: o = o.clone(); o[:, [0, 1, 4]] = 0.0   # LF_Test: drop x,y,psi
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
HAS_TRAJ = hasattr(env, "_traj_path") or REAL_PATH is not None
rec = {"bp": [], "bq": [], "root": [], "path": [], "pstar": [], "prog": [], "reset": []}
_bx = np.argsort(pd_["bone_rest_pos"][:, 0])   # bone order along the body for the chain polyline
def snap_path():
    if REAL_PATH is not None: return REAL_PATH
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
        rec["pstar"].append(REAL_PATH[min(t + 1, len(REAL_PATH) - 1)].copy() if REAL_PATH is not None else (env._cur_p_star[0].cpu().numpy().copy() if HAS_TRAJ else None))
        lg = env.extras.get("log", {}); rec["prog"].append(float(lg["traj/phase_progress"]) if "traj/phase_progress" in lg else float("nan"))
np.savez_compressed(a.out.replace(".mp4", "") + ".rollout.npz", bp=np.array(rec["bp"]), bq=np.array(rec["bq"]), root=np.array(rec["root"]),
                    reset=np.array(rec["reset"]), prog=np.array(rec["prog"]), path=np.array([p if p is not None else np.zeros((0, 3)) for p in rec["path"]], dtype=object),
                    pstar=np.array([p if p is not None else np.full(3, np.nan) for p in rec["pstar"]]), BL=BL, DT=DT, panel_hydro_path=str(env.cfg.panel_hydro_path), task=a.task)
print("[render] rollout saved ->", a.out.replace(".mp4", "") + ".rollout.npz", flush=True)
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
(ps,) = ax.plot([], [], "o", ms=7, color="crimson", label=("real fish (now)" if REAL_PATH is not None else "look-ahead target"))
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
