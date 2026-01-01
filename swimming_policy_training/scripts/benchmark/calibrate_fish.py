"""Generic per-fish in-sim LS calibration (from calibrate_catfish001.py): python calibrate_fish.py --tag <tag>
In-sim LEAST-SQUARES calibration of the 9-joint catfish's curvature->joint mapping
(playback-style: bend each lateral joint alone, measure the 20-station kappa_bl response
with the shared spline estimator, ridge-LS solve Phi). Also DETECTS which D6 rotation
family bends the body laterally on this asset (it rests flat: up=+Z, so the family differs
from the 8-bone Misty).

Output: Video2SwimFish/outputs/catfish_fish001/calibration_catfish001.npz with the recording.npz-compatible schema
(Phi (20,J), kappa_rest (20,), joint_names (J,), fam1=arange(J) head-first) so the existing
PCA/CPG envs consume it unchanged. Prints the detected control_dof_suffix -- the cfg value
must match it.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "source"), str(REPO / "scripts/zef_playback")):
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher
import argparse
p = argparse.ArgumentParser()
p.add_argument("--calib_amp", type=float, default=0.5)
p.add_argument("--tag", required=True, help="fish tag, e.g. catfish_fish002")
p.add_argument("--kp", type=float, default=None, help="override drive stiffness (kp sweep)")
p.add_argument("--kd", type=float, default=None, help="override drive damping")
p.add_argument("--out_dir", type=str, default=None, help="write calibration.npz here instead of the dataset dir (sweeps)")
p.add_argument("--no_fem", action="store_true", help="calibrate on the skeleton only (with_deformable=False): the FEM skin tilted the rest pose "
               "by 30-70%% BL on 4/8 fish and made the lateral-axis detection / Phi degenerate (bluegill020 cond 1.4e6, 2026-09-27)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym          # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402
import FISH.tasks                # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import curvature_utils as cu     # noqa: E402

TASK = f"Bench-{args.tag}-Calib-v0"
OUT = REPO / "data/fish_assets" / args.tag
cfg = parse_env_cfg(TASK, device=args.device, num_envs=4)
if args.no_fem: cfg.with_deformable = False; print("### calibrating with the FEM skin DEACTIVATED (--no_fem)", flush=True)
if args.kp is not None:
    cfg.robot_cfg.actuators["all_joints"].stiffness = args.kp
if args.kd is not None:
    cfg.robot_cfg.actuators["all_joints"].damping = args.kd
print(f"### drive gains kp={cfg.robot_cfg.actuators['all_joints'].stiffness} kd={cfg.robot_cfg.actuators['all_joints'].damping}", flush=True)
if args.out_dir is not None:
    _OUT_W = Path(args.out_dir); _OUT_W.mkdir(parents=True, exist_ok=True)
else:
    _OUT_W = OUT
env = gym.make(TASK, cfg=cfg).unwrapped
dev = env.device
env.cfg.curriculum = False
env._cur_radius = -1.0
E = env.num_envs
names = [env.robot.data.joint_names[i] for i in env._control_joint_ids]
nact = len(names)
print(f"### {nact} controllable DOFs on {env.robot.num_bodies} bodies", flush=True)
# SELF-HEALING bone order: PhysX orders links BFS-outward from the articulation root (here
# mid-body bone21 after the root fixup) -- neither numeric nor USD-traversal. Permute the
# npz into the RUNTIME order by name and rewrite it, so every consumer (env panel hydro,
# centerline geometry) indexes body_link_* correctly.
_np = dict(np.load(OUT / "panel_hydro.npz"))
_npz_names = [str(n) for n in _np["bone_names"]]
_rt_names = list(env.robot.data.body_names)
if _npz_names != _rt_names:
    perm = np.array([_npz_names.index(n) for n in _rt_names])      # runtime i <- npz perm[i]
    inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))    # npz j -> runtime inv[j]
    _np["bone_rest_pos"] = _np["bone_rest_pos"][perm]
    _np["bone_rest_quat"] = _np["bone_rest_quat"][perm]
    _np["panel_bone"] = inv[_np["panel_bone"]]
    _np["bone_names"] = np.array(_rt_names)
    np.savez(OUT / "panel_hydro.npz", **_np)
    print(f"### npz PERMUTED to runtime bone order (root-BFS, first: {_rt_names[:5]})", flush=True)
else:
    print("### bone order verified: npz == runtime", flush=True)
# Head direction from GEOMETRY, not from bone NAMES.
# BUG (fixed 2026-09-24): this used sign(x[bone_0] - x[bone_<highest number>]).  Bone numbering is
# assigned by the a2c actor in edit order and has NOTHING to do with position -- for
# white_bass_fish008 bone_0 (x=+0.0152) and bone_14 (x=+0.0196) are both mid-body, 4.4 mm apart on a
# 67 mm fish, so the sign was pure noise and came out -1 for all 8 benchmark fish even though every
# one of them has its head at +X.  The heading reward then rewarded pointing the TAIL along the path
# and every trajfollow policy learned to swim backwards (verified in the rollouts).
# Now: the head is the end of the skeleton's x-extent that is NOT the caudal fin.  The caudal fin is
# a thin vertical blade, so its girth aspect ratio (y-span / z-span) is far larger than the snout's.
_bp = dict(np.load(OUT / "panel_hydro.npz"))
_bn = [str(n) for n in _bp["bone_names"]]
_v = _bp["vert_local"]; _x = _v[:, 0]; _xlo, _xhi = float(_x.min()), float(_x.max()); _L = _xhi - _xlo


def _aspect(mask):
    gy = float(_v[mask, 1].max() - _v[mask, 1].min())
    gz = float(_v[mask, 2].max() - _v[mask, 2].min())
    return gy / max(gz, 1e-9)


_a_lo = _aspect((_x >= _xlo) & (_x < _xlo + 0.04 * _L))     # -X extreme
_a_hi = _aspect((_x > _xhi - 0.04 * _L) & (_x <= _xhi))     # +X extreme
HEAD_SIGN = 1 if _a_lo > _a_hi else -1                      # the blade-like end is the TAIL
print(f"### HEAD_SIGN={HEAD_SIGN} (girth aspect: -X {_a_lo:.1f} vs +X {_a_hi:.1f}; "
      f"blade-like end = tail)", flush=True)
# Forward axis of the ROOT BODY: quat_apply(root_quat, [FWD_SIGN,0,0]) must point at the head.
# All bone rest quats are identity in these assets (checked: |q - I| < 1e-5), so FWD_SIGN == HEAD_SIGN.
_q = _bp["bone_rest_quat"]; _qdev = float(np.abs(_q - np.array([1.0, 0.0, 0.0, 0.0])).max())
FWD_SIGN = float(HEAD_SIGN)
if _qdev > 1e-3:
    _p = _bp["bone_rest_pos"]; _ih = int(np.argmax(_p[:, 0])); _it = int(np.argmin(_p[:, 0]))
    _ax = _p[_ih] - _p[_it]; _ax = _ax / (np.linalg.norm(_ax) + 1e-9)
    _w, _xq, _yq, _zq = _q[0]; _u = np.array([_xq, _yq, _zq]); _e = np.array([1.0, 0.0, 0.0])
    _fwd = _e + 2 * _w * np.cross(_u, _e) + 2 * np.cross(_u, np.cross(_u, _e))
    FWD_SIGN = float(np.sign(np.dot(_fwd, _ax) * HEAD_SIGN)) or 1.0
print(f"### FWD_SIGN={FWD_SIGN:+.0f} (max|rest_quat - I| = {_qdev:.1e})", flush=True)
KTOL = 2e-3                                      # 2 mm: noise-robust yet keeps adjacent joints distinct
geo = cu.rest_geometry(str(OUT / "panel_hydro.npz"), head_sign=HEAD_SIGN)
zero = torch.zeros(E, nact, device=dev)


def kappa_now():
    ks = []
    for e in range(E):
        pr = cu.kappa_from_bones(env.robot.data.body_link_pos_w[e].cpu().numpy(),
                                 env.robot.data.body_link_quat_w[e].cpu().numpy(), geo,
                                 smooth_tol_m=KTOL)
        if pr is not None:
            ks.append(pr["kappa_bl"])
    return np.median(np.stack(ks), 0)


def zspread_now():
    bp = env.robot.data.body_link_pos_w[0].cpu().numpy()
    return float(bp[:, 2].max() - bp[:, 2].min())


for _ in range(45):
    env.step(zero)
k_rest = np.mean([kappa_now() for _ in range(3)], 0)
z0 = zspread_now()
print(f"### rest: max|kappa|={np.abs(k_rest).max():.3f} z-spread={z0*1000:.1f} mm", flush=True)
# SELF-CHECK (2026-09-22): a fish whose bone chain spans >5% of its body length VERTICALLY at rest has
# pitched over (the bone-collider-vs-FEM bug) -- any Phi fitted from here is garbage. Refuse.
_BL_ = float(env._body_length)
if z0 > 0.15 * _BL_ or np.abs(k_rest).max() > 1.0:   # 15%: healthy chains have 5-7% dorsoventral bone offset (catfish001 6%, pre-bug)
    print(f"CALIB_REST_CHECK_FAILED z-spread={z0*1000:.1f} mm ({z0/_BL_*100:.0f}% BL) max|kappa|={np.abs(k_rest).max():.2f}", flush=True)
    import os as _os; _os._exit(3)

# ---- detect the LATERAL family: bend a mid-chain joint in each family, compare horizontal
# curvature response vs vertical (z) deflection ----
def _chain(nm):
    b = nm.split(":")[0]
    return int(b.split("_")[-1]) if "_" in b else 0

fams = {}
for i, n in enumerate(names):
    fams.setdefault(n.split(":")[-1], []).append(i)
mid_rank = sorted({_chain(n) for n in names})[len(set(_chain(n) for n in names)) // 2]
probe = {f: next(i for i in idxs if _chain(names[i]) == mid_rank) for f, idxs in fams.items()}
scores = {}
for f, di in probe.items():
    a = zero.clone(); a[:, di] = args.calib_amp
    for _ in range(40):
        env.step(a)
    dk = float(np.abs(kappa_now() - k_rest).max())
    dz = zspread_now() - z0
    scores[f] = (dk, dz)
    print(f"### family :{f} joint {names[di]}: horiz |dkappa|={dk:.3f}  z-spread +{dz*1000:.1f} mm",
          flush=True)
    for _ in range(35):
        env.step(zero)
LAT = max(scores, key=lambda f: scores[f][0] - 5.0 * max(scores[f][1], 0))
print(f"### DETECTED lateral family: ':{LAT}'  -> control_dof_suffix must be ':{LAT}'", flush=True)

lat_ids = sorted(fams[LAT], key=lambda i: -_chain(names[i]))     # head-first (rank desc, as misty)
# verify head-first direction: bend the FIRST of lat_ids; kappa peak should be near s=0 (head)
a = zero.clone(); a[:, lat_ids[0]] = args.calib_amp
for _ in range(40):
    env.step(a)
pk = float(np.linspace(0, 1, 20)[int(np.argmax(np.abs(kappa_now() - k_rest)))])
for _ in range(35):
    env.step(zero)
if pk > 0.5:
    lat_ids = lat_ids[::-1]
    print(f"### head-first order REVERSED (probe peak at s={pk:.2f})", flush=True)
J = len(lat_ids)
print(f"### calibrating {J} lateral joints", flush=True)

Q = np.zeros((J, J))
K = np.zeros((J, 20))
for j, di in enumerate(lat_ids):
    a = zero.clone(); a[:, di] = args.calib_amp
    for _ in range(45):
        env.step(a)
    qm = env.robot.data.joint_pos[:, env._control_joint_ids].cpu().numpy().mean(0)
    Q[j] = qm[lat_ids]
    K[j] = np.mean([kappa_now() for _ in range(3)], 0) - k_rest
    if j % 8 == 0:
        print(f"### calib {j}/{J} {names[di]}: ach={np.rad2deg(Q[j, j]):.1f}deg "
              f"peak_s={np.linspace(0,1,20)[int(np.argmax(np.abs(K[j])))]:.2f}", flush=True)
    for _ in range(25):
        env.step(zero)

Phi = np.linalg.lstsq(Q, K, rcond=None)[0].T                     # (20, J)
lat_names = [names[i] for i in lat_ids]
np.savez(_OUT_W / "calibration.npz",
         Phi=Phi.astype(np.float64), kappa_rest=k_rest.astype(np.float64),
         Q_mat=Q, K_mat=K,
         joint_names=np.array(lat_names), fam1=np.arange(J),
         control_dof_suffix=f":{LAT}", calib_amp=args.calib_amp,
         head_sign=HEAD_SIGN, body_forward_sign=FWD_SIGN, kappa_smooth_tol=KTOL)  # fwd sign derived above, NOT hard-coded (was -1.0 copied from Misty, whose asset has the head at -X)
s_pk = np.linspace(0, 1, 20)[np.abs(Phi).argmax(0)]
print(f"### Phi built (20x{J}); peak-s monotonicity corr="
      f"{np.corrcoef(np.arange(J), s_pk)[0,1]:.2f}; cond={np.linalg.cond(Phi):.0f}", flush=True)
print("CALIB_DONE", flush=True)
sys.stdout.flush()
import os
os._exit(0)
