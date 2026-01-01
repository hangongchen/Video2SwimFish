"""Shared curvature helpers for the ZeF-mode playback pipeline (NO RL anywhere).

Everything here mirrors the ZeF extraction convention (scripts/zef_manifold/extract_curvature.py):
  - midline parameterized head (s=0) -> tail (s=1), uniform arc length,
  - smoothing cubic spline (splprep), kappa = (x'y'' - y'x'') / (x'^2+y'^2)^{3/2},
  - stored DIMENSIONLESS as kappa * L (L = that frame's own arc length),
  - K=20 stations at s = linspace(0, 1, 20),
  - right-handed 2D frame (ZeF flips image-y; the sim's world XY seen from +Z is already
    right-handed, so no flip here).

The sim midline is built from the 8 bone link origins plus a rigid NOSE and TAIL tip point
(taken from the extreme panels of the end bones), projected on world XY (top view -- the same
view the ZeF data was filmed in). The ZeF path had hundreds of noisy pixel points and used a
5-tap pre-smooth; the sim gives ~10 CLEAN points, so no pre-smooth and only a tiny spline
smoothing term (1 mm tolerance) is used.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import splev, splprep

K_STATIONS = 20
S_STATIONS = np.linspace(0.0, 1.0, K_STATIONS)


# ----------------------------------------------------------------------------- quaternions (numpy)
def quat_rotate_np(q, v):
    """Rotate vectors v (...,3) by quaternions q (...,4) in (w,x,y,z) order."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # R @ v written out (same math as the env's panel placement)
    rx = (1 - 2 * (y * y + z * z)) * v[..., 0] + 2 * (x * y - w * z) * v[..., 1] + 2 * (x * z + w * y) * v[..., 2]
    ry = 2 * (x * y + w * z) * v[..., 0] + (1 - 2 * (x * x + z * z)) * v[..., 1] + 2 * (y * z - w * x) * v[..., 2]
    rz = 2 * (x * z - w * y) * v[..., 0] + 2 * (y * z + w * x) * v[..., 1] + (1 - 2 * (x * x + y * y)) * v[..., 2]
    return np.stack([rx, ry, rz], axis=-1)


# --------------------------------------------------------------------- rest-pose geometry from npz
def rest_geometry(panel_npz, head_sign=+1):
    """From panel_hydro.npz: head-first link order + nose/tail tip (bone_idx, r_local) offsets.

    Verified: the env indexes body_link_* with pd['panel_bone'] DIRECTLY, so the npz bone
    index IS the robot link index -- no remapping needed.
    """
    pd = np.load(panel_npz)
    bp = pd["bone_rest_pos"].astype(np.float64)          # (B,3)
    bq = pd["bone_rest_quat"].astype(np.float64)         # (B,4) wxyz
    pbi = pd["panel_bone"].astype(int)                   # (Pn,)
    prl = pd["panel_r_local"].astype(np.float64)         # (Pn,3)
    pw = bp[pbi] + quat_rotate_np(bq[pbi], prl)          # panel rest world positions
    # head_sign=+1: head at +X (8-bone Misty). head_sign=-1: head at -X (the 42-bone root
    # fish_articulated.usd, bone1=-X=head). Orientation flips the head-first ordering and
    # which extreme panel is the nose.
    link_order = np.argsort(-head_sign * bp[:, 0])       # head-first link indices
    i_nose = int(np.argmax(head_sign * pw[:, 0]))
    i_tail = int(np.argmin(head_sign * pw[:, 0]))

    def _tip_on_axis(i_panel, sign):
        """The extreme panel can sit OFF the body axis (e.g. a fin edge -- the raw tail panel is
        4.3 cm = 8.7% BL lateral), which injects fake end curvature into every profile. Keep only
        the component of its bone-local offset along the body axis (rest world +/-x mapped into
        the bone frame), so the tip extends the midline straight ahead of its bone."""
        bi = int(pbi[i_panel])
        # rotate the world axis into the bone frame: u_local = R^T u_world = R(q^-1) u_world
        q = bq[bi].copy()
        q[1:] = -q[1:]                                   # conjugate = inverse for unit quats
        u_loc = quat_rotate_np(q, np.array([sign, 0.0, 0.0]))
        r = prl[i_panel]
        return bi, float(np.dot(r, u_loc)) * u_loc

    tips = {"nose": _tip_on_axis(i_nose, float(head_sign)),
            "tail": _tip_on_axis(i_tail, -float(head_sign))}

    # PER-BONE AXIS OFFSETS: auto-skeleton rigs can place bone origins in a static zigzag up
    # to ~15 mm off the body midline; on densely-boned chains (7 mm spacing) that aliases into
    # huge fake curvature. Fit a smoothing spline through the rest chain, store each bone's
    # rest offset from it in the BONE's local frame; centerline_points subtracts it (rotated
    # with the bone), so measurement points ride the true midline at rest and track dynamically.
    axis_off = np.zeros_like(bp)
    try:
        ordered = bp[link_order]
        tck, _ = splprep(ordered.T, s=len(ordered) * (5e-3) ** 2, k=3)
        proj = np.stack(splev(np.linspace(0, 1, len(ordered)), tck), 1)
        off_w = ordered - proj                            # world-frame rest offsets
        for k, li in enumerate(link_order):
            q = bq[li].copy(); q[1:] = -q[1:]             # conjugate
            axis_off[li] = quat_rotate_np(q, off_w[k])
    except Exception:
        pass                                              # degenerate chains keep zero offsets
    body_length = float(pw[:, 0].max() - pw[:, 0].min())
    return {"link_order": link_order, "tips": tips, "body_length": body_length,
            "bone_rest_pos": bp, "bone_rest_quat": bq, "axis_off": axis_off}


def centerline_points(bone_pos_w, bone_quat_w, geo):
    """Head-first (N,2) top-view midline points for ONE env at ONE control step.

    bone_pos_w (B,3), bone_quat_w (B,4) wxyz (torch tensors or numpy). Points: rigid nose
    tip, the 8 bone link origins in the FIXED head-first rest order, rigid tail tip.
    The order is fixed at rest (never re-sorted by world x) so a turning fish keeps a
    consistent parameterization.
    """
    bp = np.asarray(bone_pos_w, dtype=np.float64)
    bq = np.asarray(bone_quat_w, dtype=np.float64)
    nb_i, nb_r = geo["tips"]["nose"]
    tb_i, tb_r = geo["tips"]["tail"]
    nose = bp[nb_i] + quat_rotate_np(bq[nb_i], nb_r)
    tail = bp[tb_i] + quat_rotate_np(bq[tb_i], tb_r)
    off = geo.get("axis_off")
    core = bp[geo["link_order"]]
    if off is not None and np.abs(off).max() > 0:
        core = core - quat_rotate_np(bq[geo["link_order"]], off[geo["link_order"]])
    pts = np.vstack([nose[None, :], core, tail[None, :]])
    return pts[:, :2]                                    # top view: world XY


# --------------------------------------------------------------------------------- kappa profile
def presmooth_chain(pts_xy, n_out=K_STATIONS, w=7, passes=2):
    """Low-pass a DENSE bone chain to the kappa-station scale before differentiating.

    Densely-boned rigs (42 bones, ~6 mm spacing) carry a real period-2 skeleton zigzag in the
    decoder's null space: adjacent lateral joints alternate +/-10 deg while the FLESH surface
    stays smooth (~0.5 mm surface deflection at that wavelength). The ZeF reference kappa was
    measured from the smooth body silhouette, so the sim measurement must not resolve below the
    station scale either. Boxcar the interior points (tips pinned), THEN arc-resample to n_out
    -- resampling alone would alias the 10 mm zigzag into fake low-frequency curvature.
    Verified on a real rollout: worst-frame RMSE vs commanded 110 -> 20 with a clean arc
    unchanged (1.250 vs 1.251).
    """
    P = np.asarray(pts_xy, dtype=np.float64).copy()
    if len(P) <= n_out:
        return P
    k = np.ones(w) / w
    for _ in range(passes):
        pad = np.vstack([P[0][None].repeat(w // 2, 0), P[1:-1], P[-1][None].repeat(w // 2, 0)])
        P[1:-1] = np.stack([np.convolve(pad[:, 0], k, "valid"),
                            np.convolve(pad[:, 1], k, "valid")], 1)
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    u = np.linspace(0, cum[-1], n_out)
    return np.stack([np.interp(u, cum, P[:, 0]), np.interp(u, cum, P[:, 1])], 1)


def kappa_profile(pts_xy, k_stations=K_STATIONS, smooth_tol_m=1e-3, presmooth=False):
    """Signed curvature*L at K stations from head-first 2D points (meters).

    Same spline + formula + sign convention as the ZeF extraction. Returns dict with
    kappa_bl (K,), arc length L (m), midline (100,2) resampled points; or None if degenerate.
    """
    P = np.asarray(pts_xy, dtype=np.float64)
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9]
    P = P[keep]
    if presmooth:
        P = presmooth_chain(P, n_out=k_stations)
    if len(P) < 5:
        return None
    try:
        tck, _ = splprep(P.T, s=len(P) * smooth_tol_m ** 2, k=3)
    except Exception:
        return None
    uu = np.linspace(0, 1, 400)
    xy = np.stack(splev(uu, tck), 1)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    L = cum[-1]
    if L < 1e-6:
        return None
    s_sta = np.linspace(0, 1, k_stations)
    u_sta = np.interp(s_sta * L, cum, uu)
    dx, dy = splev(u_sta, tck, der=1)
    ddx, ddy = splev(u_sta, tck, der=2)
    kappa = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    u_mid = np.interp(np.linspace(0, L, 100), cum, uu)
    mid = np.stack(splev(u_mid, tck), 1)
    return {"kappa_bl": (kappa * L).astype(np.float64), "L": float(L), "midline": mid}


def kappa_from_bones(bone_pos_w, bone_quat_w, geo, **kw):
    """Convenience: bones -> centerline -> kappa profile (one env, one step)."""
    return kappa_profile(centerline_points(bone_pos_w, bone_quat_w, geo), **kw)


# ------------------------------------------------------------------------------- PCA reconstruction
def reconstruct_kappa(coeffs, basis, pcs):
    """kappa_bl (T,20) = mean + coeffs[:, pcs] @ components[pcs]. pcs = list of PC indices."""
    c = np.asarray(coeffs, dtype=np.float64)
    comp = basis["components"].astype(np.float64)
    out = np.tile(basis["mean"].astype(np.float64), (c.shape[0], 1))
    for p in pcs:
        out += np.outer(c[:, p], comp[p])
    return out


# --------------------------------------------------------------------------------------- self-test
# MEASUREMENT FLOOR (measured here, quantified per-shape in analyze.py): with only ~10 chain
# points the interpolating spline rings ~8% locally on an arc and UNDERESTIMATES peak curvature
# of full-wavelength bends. This bias cancels in the control loop because the calibration Phi is
# measured with the SAME estimator; for the fidelity numbers, analyze.py reports a kinematic
# "ideal playback" baseline through the same estimator so representation error is separated
# from dynamic tracking error.
if __name__ == "__main__":
    # a circular arc of radius R: kappa*L must be ~ L/R everywhere (away from spline ends)
    R, L_arc = 0.4, 0.5
    th = np.linspace(0, L_arc / R, 10)
    pts = np.stack([R * np.sin(th), R * (1 - np.cos(th))], 1)
    prof = kappa_profile(pts)
    kb = prof["kappa_bl"]
    expect = prof["L"] / R
    err = np.abs(kb[2:-2] - expect).max()
    print(f"arc test: L={prof['L']:.4f} (true {L_arc}), kappa_bl inner={kb[2:-2].mean():.3f} "
          f"(expect {expect:.3f}), max inner err={err:.4f}")
    assert err < 0.10 * expect, "curvature estimator off by >10% on a clean arc"
    # a straight line: kappa ~ 0
    pts = np.stack([np.linspace(0, 0.5, 10), np.zeros(10)], 1)
    kb = kappa_profile(pts)["kappa_bl"]
    print(f"line test: max|kappa_bl| = {np.abs(kb).max():.2e}")
    assert np.abs(kb).max() < 1e-6
    # sign: bend toward +y (CCW tangent rotation) must give kappa > 0
    th = np.linspace(0, 1.0, 12)
    pts = np.stack([np.sin(th), 1 - np.cos(th)], 1) * 0.3
    kb = kappa_profile(pts)["kappa_bl"]
    print(f"sign test: mean kappa_bl = {kb.mean():.3f} (must be > 0)")
    assert kb.mean() > 0
    print("curvature_utils self-test OK")
