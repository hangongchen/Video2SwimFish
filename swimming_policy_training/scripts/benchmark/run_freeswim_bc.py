#!/usr/bin/env python
"""Free-swimming IL baselines for one fish (no task reward, no PPO):
  --baseline blm_il   : BC directly from the fish's own video in the BLM (PCA) action space
                        (dataset built here from <fish>_curvature.npz + <fish>_reference.npz + PCA basis),
                        policy = rl_games actor for Bench-<tag>-FreeSwim-VideoState-v0.
  --baseline bco_pure : BCO -- IDM trained on random rollouts of Bench-<tag>-TrajFollow-Joint-v0, real
                        video curvature decoded to joint_pos -> IDM pseudo-actions -> BC (bco/ package).
Then the free-swim eval protocol: 20 real initial states (random video frames), the fish's PCA
coefficients / joint pose set to the real frame's, 5 s free swimming, no target; metrics vs the real
fish's own next-5 s trajectory + curvature statistics (eval/*), logged to wandb + results json.
Prints "[BC] epoch=N train_loss=.. val_loss=.." per epoch (scheduler stop rule) and stops itself when
val_loss has not improved for 10 epochs.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "source"), str(REPO / "scripts/zef_playback"), str(REPO / "source/FISH/FISH/tasks/direct/fish")):
    if _p not in sys.path: sys.path.insert(0, _p)

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--baseline", choices=["blm_il", "bco_pure"], required=True)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--wandb-name", default=None)
ap.add_argument("--bc_epochs", type=int, default=300); ap.add_argument("--bc_lr", type=float, default=1e-3); ap.add_argument("--bc_batch", type=int, default=256)
ap.add_argument("--patience", type=int, default=10); ap.add_argument("--n_init", type=int, default=20); ap.add_argument("--eval_seconds", type=float, default=5.0)
ap.add_argument("--idm_transitions", type=int, default=100000); ap.add_argument("--skip_eval", action="store_true")
ap.add_argument("--n_modes", type=int, default=0, help="BLM+IL: learn only the first n PCA modes (0 = all); the rest are held at 0")
ap.add_argument("--smooth_win", type=int, default=1, help="BLM+IL: moving-average window (frames @30 Hz) applied to the video coefficient series before differencing")
ap.add_argument("--hist", type=int, default=1, help="BLM+IL: number of past observations stacked as policy input")
ap.add_argument("--coef_noise", type=float, default=0.0, help="BLM+IL: Gaussian noise (std, units of a/a_max) added to the coefficient part of the training obs; "
                "with --coef_aug copies per frame. Breaks the history-extrapolation shortcut (diag 2026-09-24).")
ap.add_argument("--coef_aug", type=int, default=4)
ap.add_argument("--variant", default="", help="suffix for the checkpoint dir / run name (e.g. _v3)")
ap.add_argument("--stage", choices=["all", "bc", "eval"], default="all",
                help="bco_pure needs a fresh Isaac process for the eval env (a second gym.make in one process hangs): 'all' runs bc then spawns 'eval'")
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args(); args.headless = True
app = AppLauncher(args).app          # Isaac Sim is needed for bco_pure (IDM) and for the eval rollout of both

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import FISH.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import biofidelity_metrics as bm  # noqa: E402
from bco.bc_trainer import build_rl_games_model, save_rl_games_checkpoint, warm_start_running_mean_std  # noqa: E402

torch.manual_seed(args.seed); np.random.seed(args.seed)
tag = args.tag; sp, num = tag.rsplit("_fish", 1); dev = str(getattr(args, "device", None) or "cuda:0")
MAN = REPO / f"data/species_manifold/{sp}_fix/top"; DS = REPO / f"data/fish_assets/{tag}"
OUT = REPO / f"checkpoints/benchmark/{('bco' if args.baseline == 'bco_pure' else 'blm_il') + args.variant}/{tag}"; OUT.mkdir(parents=True, exist_ok=True)
AG = REPO / "source/FISH/FISH/tasks/direct/fish/agents"
name = args.wandb_name or f"freeswim_{tag}_{args.baseline}"

import wandb  # noqa: E402
wb = wandb.init(project=os.environ.get("WANDB_PROJECT", "video2swimfish_benchmark"), entity=os.environ.get("WANDB_ENTITY"), name=name,
                tags=[t for t in os.environ.get("WANDB_TAGS", "").split(",") if t], config=vars(args) | {"fish": tag, "species": sp})

curv = np.load(MAN / f"fish{num}_curvature.npz"); ref = np.load(MAN / f"fish{num}_reference.npz", allow_pickle=True); basis = np.load(MAN / f"fish{num}_pca_basis.npz", allow_pickle=True)
kap = curv["kappa_bl"].astype(np.float64); valid = curv["valid"].astype(bool) & (np.nan_to_num(np.abs(kap), nan=np.inf).max(1) < 4.0)
fps = float(curv["fps"]); t_sec = curv["t_sec"].astype(np.float64); p_bl = ref["p_bl"].astype(np.float64); psi = ref["psi"].astype(np.float64)
runs = [(int(a), int(b)) for a, b in ref["runs"]]
V = basis["components"].astype(np.float64); mean_k = basis["mean"].astype(np.float64); K = V.shape[0]
coeffs_b = basis["coeffs"]; okb = np.isfinite(coeffs_b).all(1)
a_max = np.percentile(np.abs(coeffs_b[okb]), 99.0, axis=0); da = coeffs_b[:-2] - coeffs_b[2:]; da = da[np.isfinite(da).all(1)]
da_max = np.percentile(np.abs(da), 99.0, axis=0)                      # same formulas as salmon_swim_pca_env._build_pca_mapping
CTRL_DT = 1.0 / 30.0


def runs_at_30hz():
    """Yield per-run 30 Hz resampled (t, x, y, psi_rel, a(K)) inside clean frames of each reference run."""
    for a, b in runs:
        idx = np.arange(a, b); ok = valid[idx]
        if ok.sum() < 8: continue
        tt = t_sec[idx]; grid = np.arange(tt[0], tt[-1], CTRL_DT)
        if len(grid) < 4: continue
        coef = (kap[idx] - mean_k) @ V.T                                  # (n, K) a_t
        good = ok
        def interp(y): return np.stack([np.interp(grid, tt[good], y[good, j]) for j in range(y.shape[1])], 1)
        psi_u = np.unwrap(psi[idx]); psi0 = psi_u[0]; c0, s0 = np.cos(-psi0), np.sin(-psi0)
        d = p_bl[idx] - p_bl[a]; xy = np.stack([c0 * d[:, 0] - s0 * d[:, 1], s0 * d[:, 0] + c0 * d[:, 1]], 1)
        yield grid, interp(xy), np.interp(grid, tt[good], (psi_u - psi0)[good]), interp(coef)


def _smooth(x, win):
    if win <= 1: return x
    k = np.ones(win) / win; pad = win // 2
    xp = np.pad(x, ((pad, win - 1 - pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)
def _stack_hist(obs_seq, hist):
    """obs_seq (T, D) -> (T, hist*D): [o_t, o_{t-1}, ..., o_{t-hist+1}], repeating the first frame at the start."""
    if hist <= 1: return obs_seq
    T = len(obs_seq); idx = np.clip(np.arange(T)[:, None] - np.arange(hist)[None, :], 0, T - 1)
    return obs_seq[idx].reshape(T, -1)
def build_blm_dataset():
    S, A = [], []
    # normaliser for the labels: p99 of the SMOOTHED per-step increment. Normalising smoothed increments by the
    # env's da_max (p99 of the RAW, noise-dominated increments) made every label ~0.01 and the BC policy learned
    # to output ~0 -> the fish never moved its joints (diag 2026-09-24: |a|=0.004, jvel 11 deg/s). At eval the
    # policy output is multiplied back by da_s/da_max so the env applies the intended increment.
    _all = np.concatenate([np.abs(np.diff(_smooth(c, args.smooth_win), axis=0)) for _, _, _, c in runs_at_30hz()], 0)
    da_s = np.maximum(np.percentile(_all, 99, axis=0), 1e-6)
    json.dump({"action_scale": (da_s / da_max).tolist(), "da_s": da_s.tolist(), "da_max": np.asarray(da_max).tolist()}, open(OUT / "action_scale.json", "w"), indent=1)
    print(f"[freeswim] BLM label scale da_s/da_max (first 4 modes): {(da_s / da_max)[:4].round(3).tolist()}", flush=True)
    for grid, xy, psir, coef in runs_at_30hz():
        vxy = np.gradient(xy, CTRL_DT, axis=0)
        cs = _smooth(coef, args.smooth_win)                        # low-pass the noisy video coefficients
        obs = np.stack([np.concatenate([xy[t], vxy[t], [psir[t]], coef[t] / a_max]) for t in range(len(grid))]).astype(np.float32)
        rng = np.random.default_rng(args.seed + len(S))
        copies = [obs] + [obs + np.concatenate([np.zeros_like(obs[:, :5]), rng.normal(0, args.coef_noise, obs[:, 5:].shape)], 1).astype(np.float32)
                          for _ in range(args.coef_aug)] if args.coef_noise > 0 else [obs]
        for ob in copies:
            ob = _stack_hist(ob, args.hist)
            for t in range(len(grid) - 1):
                a = np.clip((cs[t + 1] - cs[t]) / da_s, -1, 1).astype(np.float32)
                if args.n_modes > 0: a[args.n_modes:] = 0.0            # only the leading modes carry real motion
                S.append(ob[t]); A.append(a)
    S, A = np.stack(S), np.stack(A); np.savez(OUT / "dataset.npz", s=S, a_hat=A, a_max=a_max, da_max=da_max, dt=CTRL_DT, K=K)
    print(f"[freeswim] BLM dataset {S.shape} from {len(runs)} runs", flush=True); return S, A


def train_bc(model, S, A, obs_norm_from=None):
    from torch.utils.data import DataLoader, TensorDataset
    S = torch.tensor(S, device=dev); A = torch.tensor(A, device=dev); n = len(S)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)); nv = max(1, int(0.1 * n))
    vi, ti = perm[:nv], perm[nv:]
    warm_start_running_mean_std(model, S[ti] if obs_norm_from is None else obs_norm_from)
    model.to(dev); opt = torch.optim.Adam(model.parameters(), lr=args.bc_lr); lf = torch.nn.MSELoss()
    loader = DataLoader(TensorDataset(S[ti], A[ti]), batch_size=min(args.bc_batch, len(ti)), shuffle=True)
    best, bad, best_state = float("inf"), 0, None
    for ep in range(1, args.bc_epochs + 1):
        model.train(); model.running_mean_std.eval() if hasattr(model, "running_mean_std") else None; tl = 0.0; nb = 0
        for ob, ac in loader:
            out = model({"is_train": False, "prev_actions": None, "obs": ob}); loss = lf(out["mus"], ac)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += float(loss); nb += 1
        model.eval()
        with torch.no_grad(): vl = float(lf(model({"is_train": False, "prev_actions": None, "obs": S[vi]})["mus"], A[vi]))
        tl /= max(nb, 1); wb.log({"bc/train_loss": tl, "bc/val_loss": vl, "bc/epoch": ep}, step=ep)
        print(f"[BC] epoch={ep} train_loss={tl:.6f} val_loss={vl:.6f}", flush=True)
        if vl < best - 1e-6: best, bad, best_state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else: bad += 1
        if bad >= args.patience: print(f"[BC] early stop at epoch {ep} (no val improvement for {args.patience})", flush=True); break
    if best_state: model.load_state_dict(best_state)
    save_rl_games_checkpoint(model, str(OUT / "bc_policy.pt")); wb.summary["bc/best_val_loss"] = best
    return model


# ------------------------------------------------------------------------------------ BC stage
model = None
if args.stage == "eval":
    EVAL_TASK = f"Bench-{tag}-FreeSwim-VideoState-v0" if args.baseline == "blm_il" else f"Bench-{tag}-TrajFollow-Joint-v0"
elif args.baseline == "blm_il":
    S, A = build_blm_dataset()
    model = build_rl_games_model(str(AG / "rl_games_ppo_pca_videostate_cfg.yaml"), S.shape[1], K)
    model = train_bc(model, S, A)
    EVAL_TASK = f"Bench-{tag}-FreeSwim-VideoState-v0"
else:
    from bco.idm import train_idm, save_idm
    from bco.replay_buffer import TransitionReplayBuffer
    from bco.bco_config import BCOConfig
    EVAL_TASK = f"Bench-{tag}-TrajFollow-Joint-v0"
    cfg_e = parse_env_cfg(EVAL_TASK, device=dev, num_envs=128); cfg_e.seed = args.seed
    env = gym.make(EVAL_TASK, cfg=cfg_e).unwrapped
    obs_dim, act_dim = int(env.observation_space.shape[-1]), int(env.action_space.shape[-1]); nj = act_dim
    buf = TransitionReplayBuffer(args.idm_transitions, obs_dim, act_dim, dev); obs, _ = env.reset(); obs = obs["policy"]
    while not buf.is_full:
        act = torch.empty(128, act_dim, device=dev).uniform_(-1, 1); nobs, *_ = env.step(act); nobs = nobs["policy"]; buf.add(obs, act, nobs); obs = nobs
    bcfg = BCOConfig(); bcfg.seed = args.seed
    idm = train_idm(buf.as_dict(), bcfg, dev); save_idm(idm, str(OUT / "idm.pt"))
    # pseudo labels: video kappa (clean frames, 30 Hz) -> joint_pos via the env's own decoder -> IDM
    Mdec = env._Mdec if torch.is_tensor(env._Mdec) else torch.tensor(env._Mdec, device=dev)
    k_rest = torch.tensor(np.load(env.cfg.pca_calib_path)["kappa_rest"], dtype=torch.float32, device=dev)
    S, A = [], []
    lo = env._soft_joint_limits[0, env._control_joint_ids, 0]; hi = env._soft_joint_limits[0, env._control_joint_ids, 1]
    for grid, xy, psir, coef in runs_at_30hz():
        kappa30 = torch.tensor(mean_k + coef @ V, dtype=torch.float32, device=dev)          # (n,20)
        q = torch.clamp((kappa30 - k_rest) @ Mdec.T.float(), lo, hi) * float(env.cfg.obs_scales.joint_pos)
        o = torch.zeros(len(grid), obs_dim, device=dev); o[:, 6:6 + nj] = q
        with torch.no_grad(): a_hat = idm(o[:-1], o[1:]).clamp(-1, 1)
        S.append(o[:-1].cpu().numpy()); A.append(a_hat.cpu().numpy())
    S, A = np.concatenate(S), np.concatenate(A); np.savez(OUT / "pseudo_labeled.npz", obs=S, pseudo_action=A)
    print(f"[freeswim] BCO pseudo-labels {S.shape}", flush=True)
    model = build_rl_games_model(str(AG / "rl_games_ppo_traj_follow_joint_cfg.yaml"), obs_dim, act_dim)
    model = train_bc(model, S, A, obs_norm_from=buf.as_dict()["obs"])
    if not args.skip_eval and args.stage == "all":
        # fresh process for the eval env
        import subprocess
        cmd = [sys.executable, __file__, "--tag", tag, "--baseline", args.baseline, "--seed", str(args.seed), "--stage", "eval",
               "--n_init", str(args.n_init), "--eval_seconds", str(args.eval_seconds), "--device", dev, "--wandb-name", name,
               "--n_modes", str(args.n_modes), "--smooth_win", str(args.smooth_win), "--hist", str(args.hist), "--variant", args.variant]
        print("[freeswim] spawning eval process:", " ".join(cmd), flush=True); sys.stdout.flush()
        wb.finish(); rc = subprocess.run(cmd, cwd=str(REPO)).returncode
        print(f"[freeswim] eval process exit {rc}", flush=True); print("FREESWIM_DONE" if rc == 0 else "FREESWIM_EVAL_FAILED", flush=True); os._exit(rc)
    args.skip_eval = True

# ------------------------------------------------------------------------------------ free-swim eval
import traceback as _tb
def _die(exc):
    _tb.print_exc(); print("FREESWIM_EVAL_FAILED", flush=True); sys.stdout.flush(); os._exit(1)
sys.excepthook = lambda et, ev, tb: _die(ev)
if not args.skip_eval:
    n = args.n_init; cfg_e = parse_env_cfg(EVAL_TASK, device=dev, num_envs=n); cfg_e.seed = args.seed
    env = gym.make(EVAL_TASK, cfg=cfg_e).unwrapped; obs, _ = env.reset(); obs = obs["policy"] if isinstance(obs, dict) else obs
    if model is None:   # --stage eval: rebuild the actor and load the BC checkpoint
        yaml = AG / ("rl_games_ppo_pca_videostate_cfg.yaml" if args.baseline == "blm_il" else "rl_games_ppo_traj_follow_joint_cfg.yaml")
        _h = args.hist if args.baseline == "blm_il" else 1
        model = build_rl_games_model(str(yaml), _h * int(env.observation_space.shape[-1]), int(env.action_space.shape[-1]))
        ck = torch.load(OUT / "bc_policy.pt", map_location=dev); model.load_state_dict(ck["model"]); print("[freeswim] loaded", OUT / "bc_policy.pt", flush=True)
    model.to(dev).eval()
    steps = int(args.eval_seconds / CTRL_DT)
    # real initial states: random frames with >= 5 s of clean data ahead inside a run
    cands = []
    for grid, xy, psir, coef in runs_at_30hz():
        for t0 in range(0, len(grid) - steps - 1, 5): cands.append((xy[t0:t0 + steps + 1], psir[t0:t0 + steps + 1], coef[t0]))
    rng = np.random.default_rng(args.seed); pick = [cands[i] for i in rng.choice(len(cands), size=min(n, len(cands)), replace=False)]
    while len(pick) < n: pick.append(pick[rng.integers(len(pick))])
    # set the fish's initial BLM coefficients to the real frame's (PCA envs keep the state in env._a)
    if args.baseline == "blm_il" and hasattr(env, "_a"):     # BLM state = the real frame's PCA coefficients (joint env: not applicable)
        env._a[:] = torch.tensor(np.stack([p[2] for p in pick]), dtype=torch.float32, device=dev)[:, :env._a.shape[1]].clamp(-env._a_max, env._a_max)
    ref_stats = bm.RefStats(str(MAN / f"fish{num}_curvature.npz"), str(MAN / f"fish{num}_reference.npz"))
    calib = np.load(env.cfg.pca_calib_path, allow_pickle=True); geo = bm.cu.rest_geometry(env.cfg.panel_hydro_path, head_sign=int(calib["head_sign"]) if "head_sign" in calib.files else 1)
    root0 = env.robot.data.root_state_w.clone(); fwd_b = torch.tensor([env._fwd_sign, 0.0, 0.0], device=dev).repeat(n, 1)
    import isaaclab.utils.math as mu
    psi0 = torch.atan2(*mu.quat_apply(root0[:, 3:7], fwd_b)[:, [1, 0]].T)
    pos_hist = np.zeros((steps, n, 2)); kap_hist = np.full((steps, n, 20), np.nan); psi_hist = np.zeros((steps, n))
    from FISH.tasks.direct.fish.salmon_swim_traj_follow_env import discrete_frechet
    # BUG FIX 2026-09-22: the BCO pseudo-labelled dataset has ZEROS in every obs dim except the joint
    # angles (obs[:, 6:6+nj]) -- video gives no path/velocity features -- so the BC policy must be
    # evaluated on the same masked observation, not on the env's full obs (out-of-distribution).
    def _bc_obs(o):
        if args.baseline != "bco_pure": return o
        nj = int(env.action_space.shape[-1]); m = torch.zeros_like(o); m[:, 6:6 + nj] = o[:, 6:6 + nj]; return m
    _H = args.hist if args.baseline == "blm_il" else 1
    _hb = [obs.clone() for _ in range(_H)]                          # history buffer, newest first
    def _policy_in(o):
        if _H <= 1: return _bc_obs(o)
        _hb.insert(0, o.clone()); del _hb[_H:]
        return torch.cat(_hb, dim=1)
    with torch.no_grad():
        for t in range(steps):
            act = model({"is_train": False, "prev_actions": None, "obs": _policy_in(obs)})["mus"]
            if args.baseline == "blm_il" and (OUT / "action_scale.json").exists():
                act = (act * torch.tensor(json.load(open(OUT / "action_scale.json"))["action_scale"], device=dev, dtype=act.dtype)).clamp(-1, 1)
            if args.baseline == "blm_il" and args.n_modes > 0: act[:, args.n_modes:] = 0.0
            obs, *_ = env.step(act); obs = obs["policy"] if isinstance(obs, dict) else obs
            root = env.robot.data.root_state_w; pos_hist[t] = (root[:, 0:2] - root0[:, 0:2]).cpu().numpy(); psi_hist[t] = torch.atan2(*mu.quat_apply(root[:, 3:7], fwd_b)[:, [1, 0]].T).cpu().numpy()
            bp, bq = env.robot.data.body_link_pos_w, env.robot.data.body_link_quat_w
            for e in range(n):
                pr = bm.cu.kappa_from_bones(bp[e].cpu().numpy(), bq[e].cpu().numpy(), geo, smooth_tol_m=2e-3)
                if pr is not None: kap_hist[t, e] = pr["kappa_bl"]
    BL = float(env._body_length); res = {"frechet_dist_bl": [], "forward_speed_bl_s": [], "heading_stability": []}
    c0, s0 = np.cos(-psi0.cpu().numpy()), np.sin(-psi0.cpu().numpy()); em = bm.EpisodeMetrics(np.arange(n), steps, CTRL_DT, ref_stats, window=n)
    for e in range(n):
        d = pos_hist[:, e]; sim_b = np.stack([c0[e] * d[:, 0] - s0[e] * d[:, 1], s0[e] * d[:, 0] + c0[e] * d[:, 1]], 1) / BL
        real_b = pick[e][0][1:steps + 1]
        res["frechet_dist_bl"].append(float(discrete_frechet(sim_b, real_b)))
        res["forward_speed_bl_s"].append(float(np.linalg.norm(np.diff(sim_b, axis=0), axis=1).mean() / CTRL_DT))
        dpsi = np.diff(np.unwrap(psi_hist[:, e])); res["heading_stability"].append(float(1.0 / (1.0 + np.std(dpsi) / CTRL_DT)))
        for t in range(steps): em.record({e: t}, {e: kap_hist[t, e]}, {e: res["forward_speed_bl_s"][-1]}, {e: t})
        em.finish(e, steps)
    out = {f"eval/{k}": float(np.nanmean(v)) for k, v in res.items()} | em.log_dict()
    out["eval/swim_speed_bl_s"] = out["eval/forward_speed_bl_s"]
    if np.isfinite(ref_stats.speed_bl_s): out["eval/swim_speed_error"] = abs(out["eval/swim_speed_bl_s"] - ref_stats.speed_bl_s)
    out["eval/n_init_states"] = n; out["eval/stop_reason"] = "converged"
    wb.log(out); wb.summary.update(out); json.dump(out, open(OUT / "freeswim_eval.json", "w"), indent=1)
    print("[FREESWIM-EVAL] " + " ".join(f"{k.split('/')[1]}={v:.4f}" for k, v in out.items() if isinstance(v, float) and "station" not in k), flush=True)
wb.finish(); print("FREESWIM_DONE", flush=True); sys.stdout.flush(); os._exit(0)
