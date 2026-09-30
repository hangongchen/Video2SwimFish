"""Analytic curvature->joint calibration matrix for a bone chain.

The in-sim calibrated Phi (calibrate_fish.py) is measured with a spline curvature estimator whose response to a
single joint bend peaks at the chain END stations (0 or 19) for every joint, so Phi is ~rank-2 with condition
number 1e6-1e7 and the ridge inverse commands ZIGZAG joint patterns (3-5 sign changes along the body) to realise
a smooth C-bend target (2026-09-28, all 8 benchmark fish). Here Phi is built from geometry instead: a joint at
body coordinate s_k contributes a turning angle q_k spread over a Gaussian of width ~ the joint spacing, so
kappa*BL(s) = sum_k q_k g(s - s_k), integral(g ds) = 1. A ramp target then maps to a smooth, same-sign joint
pattern. Joint axis signs are taken from the calibrated Phi (sum over stations), so the D6 axis convention is kept.
"""
import numpy as np


def joint_arc_positions(geo, n_joints):
    """s in [0,1] (head=0) of the n_joints inter-bone joints, from the rest centerline (nose, bones head-first, tail)."""
    bp = geo["bone_rest_pos"]; bq = geo["bone_rest_quat"]
    import sys; from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "zef_playback")); import curvature_utils as cu
    pts = cu.centerline_points(bp, bq, geo)                     # (B+2, 2) nose, bones..., tail
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.r_[0, np.cumsum(seg)]; L = cum[-1]
    core = cum[1:-1] / L                                         # bone origins along s
    mids = 0.5 * (core[1:] + core[:-1])                          # joints between consecutive bones (B-1)
    if len(mids) != n_joints:                                    # some rigs drive fewer joints than B-1: resample evenly
        mids = np.interp(np.linspace(0, 1, n_joints), np.linspace(0, 1, len(mids)), mids)
    return mids


def analytic_phi(geo, n_joints, phi_calib=None, k_stations=20, width_scale=0.6):
    s = np.linspace(0, 1, k_stations); sj = joint_arc_positions(geo, n_joints)
    sig = width_scale * float(np.mean(np.diff(sj))) if n_joints > 1 else 0.05
    Phi = np.zeros((k_stations, n_joints))
    for k in range(n_joints):
        g = np.exp(-0.5 * ((s - sj[k]) / sig) ** 2); g /= (g.sum() * (s[1] - s[0]))          # integral over s = 1
        Phi[:, k] = g
    if phi_calib is not None:                                    # keep each joint's axis sign from the measured Phi
        sgn = np.sign(phi_calib.sum(0)); sgn[sgn == 0] = 1.0; Phi *= sgn[None, :]
    return Phi, sj
