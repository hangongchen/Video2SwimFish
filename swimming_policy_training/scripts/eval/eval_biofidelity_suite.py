"""Biological-fidelity evaluation suite: Swimming Motion + Body Deformation, real ZeF vs N
trained policies. EVALUATION ONLY (reads rollout_<tag>.npz produced by eval_traj_fidelity.py --
nothing here trains or touches the env/reward).

Swimming Motion (per rollout, in the fish's own initial body frame, /BL):
  - path-Chamfer distance: arc-length-resampled trajectory shape vs the ZeF segment it was
    given as a target (reuses analyze_traj_fidelity.py's body_frame/resample_arc/chamfer).
  - swimming speed (BL/s) and turning rate (rad/s): computed from each rollout's OWN motion,
    independent of the target (i.e. not an error term -- a raw descriptor, real ZeF vs agent).

Body Deformation (per frame, via curvature_utils on the recorded bone chain -> kappa(s,t) at
20 head-to-tail stations, same convention as curvature_dataset.npz):
  - spatial: per-station 1D WASSERSTEIN distance between the FULL distribution of kappa values
    at that station (every frame, every episode pooled -- not time-averaged first) and the real
    ZeF distribution at that same station; reported as the mean across all 20 stations. This
    replaces the earlier time-averaged-profile MAE + Pearson r: averaging over time first (the
    old approach) collapses a whole distribution to one number before comparing, which is blind
    to anything but the mean (a station that's usually near 0 but occasionally swings to +/-3
    looks identical to one that sits at a constant 0 once you average over time). Wasserstein
    distance compares the two full distributions directly, so it is sensitive to spread and
    shape mismatches the old MAE could not see, without needing a separate correlation term.
  - temporal: per-station temporal std of kappa(s,t) (how much each body location's bend
    varies over a swim cycle), MAE vs the real ZeF per-station std.
  - dominant body-wave frequency: zero-crossing rate of the posterior-half (s>=0.5) mean
    curvature signal over time (same estimator as analyze_bioplausibility.py's "f" descriptor).

Usage: python eval_biofidelity_suite.py [--tags pure_rl,bco_pure,bco_with_rl,pca]
Inputs : outputs/trajectory_fidelity/{zef_reference,rollout_<tag>}.npz,
         outputs/zef_manifold/curvature_dataset.npz
Outputs: outputs/trajectory_fidelity/biofidelity/{metrics.json,REPORT.md,profile.png}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/zef_playback"))
import curvature_utils as cu  # noqa: E402

TFD = REPO / "outputs/trajectory_fidelity"
OUT = TFD / "biofidelity"
OUT.mkdir(parents=True, exist_ok=True)
PANEL = REPO / "SimFishLib/fish_asset_pipeline/generated_usd_dataset/misty_minnow_fixed/panel_hydro.npz"

LABELS = {"pure_rl": "Pure RL", "bco_pure": "BCO pure", "bco_with_rl": "BCO + RL", "pca": "PCA",
          "cpg_rl": "CPG + RL"}
COLORS = {"pure_rl": "#2ca02c", "bco_with_rl": "#1f77b4", "bco_pure": "#ff7f0e", "pca": "#9467bd",
          "cpg_rl": "#d62728"}

GRID = np.linspace(0, 1, 101)


# ------------------------------------------------------------------ Swimming Motion (shape)
def body_frame(p, psi0, p0):
    d = p - p0
    c, s = np.cos(-psi0), np.sin(-psi0)
    return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1)


def resample_arc(traj):
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    if cum[-1] < 1e-9:
        return np.repeat(traj[:1], len(GRID), 0)
    return np.stack([np.interp(GRID * cum[-1], cum, traj[:, k]) for k in range(traj.shape[1])], 1)


def chamfer(A, B):
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    return 0.5 * (D.min(1).mean() + D.min(0).mean())


def agg(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return {"median": None, "p25": None, "p75": None, "n": 0}
    return {"median": float(np.median(v)), "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)), "mean": float(v.mean()), "n": int(len(v))}


def swimming_motion_metrics(Z, R, tag):
    """Per-sample path-Chamfer (BL), swimming speed (BL/s), turning rate (rad/s).

    Uses R["samples"] (the rollout's OWN saved sample table), not Z["samples"] (the fixed
    82-row reference): when num_envs > 82, eval_traj_fidelity.py tiles the 82 real ZeF samples
    cyclically to fill every env, and R["samples"] is the already-tiled, correctly-sized table
    -- Z["samples"] would silently only cover envs 0..81 and drop the rest."""
    S = R["samples"] if "samples" in R.files else Z["samples"]
    dt, BL = float(R["dt"]), float(R["body_length"])
    cds, speeds, turn_rates = [], [], []
    for e in range(S.shape[0]):
        T = int(R["end_step"][e]) + 1
        if T < 3:
            continue
        p = R["pos"][:T, e, 0:2].astype(np.float64)
        psi = np.unwrap(R["psi"][:T, e].astype(np.float64))
        d_ag = body_frame(p, float(R["psi0"][e]), p[0]) / BL
        ti, tj, _ = S[e]
        seg = Z["p_bl"][ti:tj + 1].astype(np.float64)
        d_zef = body_frame(seg, Z["psi"][ti], seg[0])
        cds.append(chamfer(resample_arc(d_ag), resample_arc(d_zef)))
        v = np.linalg.norm(np.diff(p, axis=0), axis=1) / BL / dt          # BL/s
        speeds.append(float(v.mean()))
        omega = np.abs(np.diff(psi)) / dt                                  # rad/s
        turn_rates.append(float(omega.mean()))
    return {"cd_path_bl": agg(cds), "speed_bl_s": agg(speeds), "turn_rate_rad_s": agg(turn_rates)}


def zef_swimming_motion_reference(Z):
    """Same speed/turning-rate descriptors computed on the real ZeF segments themselves."""
    S = Z["samples"]
    fps = float(Z["fps"])
    speeds, turn_rates = [], []
    for ti, tj, _ in S:
        seg = Z["p_bl"][ti:tj + 1].astype(np.float64)
        psi = np.unwrap(Z["psi"][ti:tj + 1].astype(np.float64))
        if len(seg) < 3:
            continue
        v = np.linalg.norm(np.diff(seg, axis=0), axis=1) * fps
        speeds.append(float(v.mean()))
        omega = np.abs(np.diff(psi)) * fps
        turn_rates.append(float(omega.mean()))
    return {"speed_bl_s": agg(speeds), "turn_rate_rad_s": agg(turn_rates)}


# ------------------------------------------------------------------ Body Deformation (curvature)
def rollout_kappa_series(R, geo, max_envs=40):
    """kappa(s,t) per env (list of (T_e,20) arrays), stations at s=linspace(0,1,20), dt in R."""
    series = []
    n_env = min(R["bone_pos"].shape[1], max_envs)
    for e in range(n_env):
        T = int(R["end_step"][e]) + 1
        rows = []
        for t in range(T):
            prof = cu.kappa_from_bones(R["bone_pos"][t, e], R["bone_quat"][t, e], geo, presmooth=True)
            if prof is not None:
                rows.append(prof["kappa_bl"])
        if len(rows) >= 8:
            series.append(np.stack(rows, 0))
    return series


def dominant_frequency_hz(kappa_ts, dt, s_stations):
    """Zero-crossing rate of the posterior-half (s>=0.5) mean curvature signal."""
    post = kappa_ts[:, s_stations >= 0.5].mean(axis=1)
    post = post - post.mean()
    signs = np.sign(post)
    signs[signs == 0] = 1
    crossings = int(np.sum(np.diff(signs) != 0))
    duration = (len(post) - 1) * dt
    if duration <= 0:
        return float("nan")
    return crossings / 2.0 / duration       # 2 crossings per full cycle


def body_deformation_metrics(rollout_series, dt_agent, zef_kappa, zef_fps, s_stations):
    """rollout_series: list of (T_e,20) kappa arrays for one policy (own dt=dt_agent).

    Per-station Wasserstein distance (spatial metric): at each of the 20 stations, pool EVERY
    frame from EVERY episode into one 1D array of kappa values (no time-averaging), do the same
    for the real ZeF data (all 900 frames), and compare the two full distributions with
    scipy.stats.wasserstein_distance. kbar_agent/kbar_zef/std_agent/std_zef are still returned
    (used by the existing profile.png time-averaged-profile plot) but are no longer the
    reported spatial metric themselves."""
    if not rollout_series:
        return None
    pooled = np.concatenate(rollout_series, axis=0)                 # (sum T_e, 20)
    kbar_agent = pooled.mean(axis=0)
    kbar_zef = zef_kappa.mean(axis=0)

    n_stations = pooled.shape[1]
    w_per_station = np.array([
        wasserstein_distance(pooled[:, i], zef_kappa[:, i]) for i in range(n_stations)
    ])
    w_mean = float(w_per_station.mean())
    w_std = float(w_per_station.std())

    std_agent = np.mean([np.std(s, axis=0) for s in rollout_series], axis=0)
    std_zef = zef_kappa.std(axis=0)
    temporal_mae = float(np.mean(np.abs(std_agent - std_zef)))

    freqs = [dominant_frequency_hz(s, dt_agent, s_stations) for s in rollout_series
             if s.shape[0] >= 8]
    freqs = [f for f in freqs if np.isfinite(f)]
    f_agent = float(np.median(freqs)) if freqs else float("nan")
    f_zef = dominant_frequency_hz(zef_kappa, 1.0 / zef_fps, s_stations)

    return {
        "wasserstein_per_station": w_per_station.tolist(),
        "wasserstein_mean": w_mean, "wasserstein_std": w_std,
        "temporal_std_mae": temporal_mae,
        "dominant_freq_hz_agent": f_agent, "dominant_freq_hz_zef": float(f_zef),
        "kbar_agent": kbar_agent.tolist(), "kbar_zef": kbar_zef.tolist(),
        "std_agent": std_agent.tolist(), "std_zef": std_zef.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="pure_rl,bco_pure,bco_with_rl,pca")
    args = ap.parse_args()
    tags = [t for t in args.tags.split(",") if t]

    Z = np.load(TFD / "zef_reference.npz")
    D = np.load(REPO / "outputs/zef_manifold/curvature_dataset.npz")
    zef_kappa = D["kappa_bl"][D["valid"].astype(bool)]
    zef_fps = float(D["fps"])
    s_stations = D["s_stations"]
    geo = cu.rest_geometry(str(PANEL))

    zef_motion_ref = zef_swimming_motion_reference(Z)

    results = {"zef_reference": {"swimming_motion": zef_motion_ref}}
    for tag in tags:
        rp = TFD / f"rollout_{tag}.npz"
        if not rp.exists():
            print(f"!! missing {rp}, skipping {tag}", flush=True)
            continue
        R = np.load(rp)
        sm = swimming_motion_metrics(Z, R, tag)
        series = rollout_kappa_series(R, geo)
        bd = body_deformation_metrics(series, float(R["dt"]), zef_kappa, zef_fps, s_stations)
        results[tag] = {"swimming_motion": sm, "body_deformation": bd,
                         "n_kappa_episodes": len(series)}
        print(f"### {tag}: cd_path={sm['cd_path_bl']['median']:.4f} BL  "
              f"speed={sm['speed_bl_s']['median']:.3f} BL/s  "
              f"turn_rate={sm['turn_rate_rad_s']['median']:.3f} rad/s  "
              f"wasserstein={bd['wasserstein_mean'] if bd else float('nan'):.3f}"
              f"+/-{bd['wasserstein_std'] if bd else float('nan'):.3f}  "
              f"f_dom={bd['dominant_freq_hz_agent'] if bd else float('nan'):.2f} Hz", flush=True)

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))

    # ---------------- report table ----------------
    lines = ["# Biological fidelity: Swimming Motion + Body Deformation\n",
             f"ZeF reference: swimming speed {zef_motion_ref['speed_bl_s']['median']:.3f} BL/s, "
             f"turning rate {zef_motion_ref['turn_rate_rad_s']['median']:.3f} rad/s, "
             f"dominant wave freq {dominant_frequency_hz(zef_kappa, 1.0 / zef_fps, s_stations):.2f} Hz\n",
             "\n## Swimming Motion\n",
             "| Policy | Path Chamfer (BL) ↓ | Swim speed (BL/s) | Turning rate (rad/s) |",
             "|---|---|---|---|"]
    for tag in tags:
        if tag not in results:
            continue
        sm = results[tag]["swimming_motion"]
        lines.append(f"| {LABELS.get(tag, tag)} | {sm['cd_path_bl']['median']:.3f} "
                      f"[{sm['cd_path_bl']['p25']:.3f}, {sm['cd_path_bl']['p75']:.3f}] | "
                      f"{sm['speed_bl_s']['median']:.3f} | {sm['turn_rate_rad_s']['median']:.3f} |")
    lines += ["\n## Body Deformation\n",
              "| Policy | Wasserstein distance (κ·L) ↓ [mean±std over 20 stations] | "
              "Temporal-std MAE ↓ | Dominant freq (Hz) |",
              "|---|---|---|---|"]
    for tag in tags:
        if tag not in results or results[tag]["body_deformation"] is None:
            continue
        bd = results[tag]["body_deformation"]
        lines.append(f"| {LABELS.get(tag, tag)} | {bd['wasserstein_mean']:.4f} "
                      f"± {bd['wasserstein_std']:.4f} | {bd['temporal_std_mae']:.4f} | "
                      f"{bd['dominant_freq_hz_agent']:.2f} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")

    # ---------------- plot: spatial curvature profile overlay ----------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    any_bd = next((results[t]["body_deformation"] for t in tags
                    if t in results and results[t]["body_deformation"]), None)
    if any_bd is not None:
        axs[0].plot(s_stations, any_bd["kbar_zef"], "k-", lw=2.5, label="ZeF (real)")
        axs[1].plot(s_stations, any_bd["std_zef"], "k-", lw=2.5, label="ZeF (real)")
        for tag in tags:
            bd = results.get(tag, {}).get("body_deformation")
            if bd is None:
                continue
            col = COLORS.get(tag, None)
            axs[0].plot(s_stations, bd["kbar_agent"], color=col, lw=1.6, label=LABELS.get(tag, tag))
            axs[1].plot(s_stations, bd["std_agent"], color=col, lw=1.6, label=LABELS.get(tag, tag))
    axs[0].set_title("time-averaged curvature profile  κ̄(s)"); axs[0].set_xlabel("s (head→tail)")
    axs[1].set_title("temporal curvature variation  std_t[κ(s,t)]"); axs[1].set_xlabel("s (head→tail)")
    for a in axs:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "profile.png", dpi=130)
    plt.close(fig)

    # ---------------- plot: per-station Wasserstein distance overlay ----------------
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for tag in tags:
        bd = results.get(tag, {}).get("body_deformation")
        if bd is None:
            continue
        ax2.plot(s_stations, bd["wasserstein_per_station"], color=COLORS.get(tag, None),
                  lw=1.8, marker="o", ms=3, label=LABELS.get(tag, tag))
    ax2.set_title("per-station Wasserstein distance vs real ZeF")
    ax2.set_xlabel("s (head→tail)"); ax2.set_ylabel("W₁ distance (κ·L)")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(OUT / "wasserstein_profile.png", dpi=130)
    plt.close(fig2)

    print("\n" + "\n".join(lines))
    print(f"\nwrote {OUT}/metrics.json, {OUT}/REPORT.md, {OUT}/profile.png, "
          f"{OUT}/wasserstein_profile.png", flush=True)


if __name__ == "__main__":
    main()
