#!/usr/bin/env python
"""Step 1: extract the fish body CURVATURE profile kappa(s) for every ZebraFish-05 top-view frame.

Pipeline (each step justified by the mask verification in verify_masks.py):
  1. Load precomputed background-subtraction mask (verified: single blob, head-anchored, stable
     area, but (a) misses the translucent caudal fin and (b) MERGES the fish with its glass-wall
     REFLECTION when it swims along a wall, e.g. frames ~386, ~395).
  2. REPAIR: clip the mask to the human-annotated GT bbox (+pad) and keep the connected component
     containing/nearest the GT head point -> removes reflection branches using human truth.
  3. MIDLINE: skeletonize the blob; graph-walk the skeleton from the endpoint nearest the GT head
     to the geodesically farthest endpoint (prunes fin/spur branches); extend both tips along the
     local tangent to the mask boundary (the skeleton stops ~half-body-width short of the tips).
  4. SMOOTH + RESAMPLE: pre-smooth the pixel path (moving average), fit a cubic smoothing spline,
     re-parameterize to UNIFORM ARC LENGTH.
  5. CURVATURE: kappa = (x'y'' - y'x'') / (x'^2 + y'^2)^{3/2} from spline derivatives, sampled at
     K stations s in [0,1] (s=0 head, s=1 tail tip of the BODY -- the caudal fin is NOT included).
     Stored dimensionless as kappa*L (curvature in units of 1/body-length) -> per-frame length
     normalization also cancels the depth-dependent pixel scale (fish z spans 0.2-14.9 cm).
     Sign: y is flipped to a right-handed frame; kappa>0 = tangent rotates CCW along the body.
  6. QC: reject frames whose body length deviates >20% from the rolling median, whose head tip is
     >25 px from the GT head, or whose skeleton is degenerate. Store a validity mask + reasons.

Output: outputs/zef_manifold/curvature_dataset.npz
  kappa_bl (T,K) float32  kappa*BL at K stations (NaN for invalid frames)
  theta    (T,K) float32  tangent angle (rad, unwrapped along s, absolute image orientation)
  midline_px (T,M,2)      dense uniform-arclength midline in full-frame px (y flipped, NaN invalid)
  s_stations (K,)         station positions in [0,1]
  valid (T,) bool ; reason (T,) int8 (0 ok, 1 no-mask, 2 short-skel, 3 len-outlier, 4 head-far)
  bodylen_px (T,), bodylen_cm (T,), frame (T,), t_sec (T,), fps, px_per_cm (scalar, ref plane)
Diagnostics: midline overlay grids, body-length timeseries, kappa heatmap (all in outputs/zef_manifold).
"""
import argparse, json, os
import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize

import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = _os.environ.get("ZEF05_DIR", "3D-ZeF/data/ZebraFish-05")   # legacy 3D-ZeF sequence (only for the original ZeF-05 basis)
MASKS = _os.path.join(_REPO, "outputs/zef05_seg/ZebraFish-05/imgT/mask")
OUT = _os.path.join(_REPO, "outputs/zef_manifold")
FPS = 60.0          # 3D-ZeF (Pedersen et al., CVPR 2020): GoPro recordings at 60 fps
K_DEFAULT = 20
M_DENSE = 100       # dense midline samples stored per frame

NBR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton_path(mask, head_xy):
    """Ordered midline pixel path head->tail from the skeleton graph. Returns (P,2) float array
    in (x, y) full-frame px, or None if degenerate."""
    skel = skeletonize(mask > 0)
    ys, xs = np.nonzero(skel)
    if len(xs) < 20:
        return None
    idx = {(y, x): i for i, (y, x) in enumerate(zip(ys, xs))}
    nbrs = [[] for _ in range(len(xs))]
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in NBR:
            j = idx.get((y + dy, x + dx))
            if j is not None:
                nbrs[i].append(j)
    deg = np.array([len(n) for n in nbrs])
    ends = np.nonzero(deg == 1)[0]
    if len(ends) < 2:
        return None
    def bfs(s):
        dist = np.full(len(xs), -1, np.int32)
        par = np.full(len(xs), -1, np.int32)
        dist[s] = 0
        q = [s]
        while q:
            nq = []
            for u in q:
                for v in nbrs[u]:
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        par[v] = u
                        nq.append(v)
            q = nq
        return dist, par

    # 2026-09-26: use the skeleton DIAMETER (longest geodesic between two endpoints) = snout->tail, instead of
    # "start at the endpoint nearest the previous head": when a pectoral fin tip was the nearest endpoint
    # (lake sturgeon 001, fins spread) the midline ran fin->tail for whole runs. Orientation (which end is the
    # head) is decided afterwards by the caller (width profile), here only the end nearer head_xy comes first.
    d0, _ = bfs(ends[0]); A = ends[d0[ends].argmax()]
    dA, _ = bfs(A); reach = ends[dA[ends] >= 0]; B = reach[dA[reach].argmax()]
    if np.hypot(xs[A] - head_xy[0], ys[A] - head_xy[1]) <= np.hypot(xs[B] - head_xy[0], ys[B] - head_xy[1]):
        src, tail = A, B
    else:
        src, tail = B, A
    dist, par = bfs(src)
    if dist[tail] < 25:
        return None
    path = []
    u = tail
    while u != -1:
        path.append(u)
        u = par[u]
    path = path[::-1]  # head -> tail
    return np.stack([xs[path], ys[path]], 1).astype(np.float64)


def extend_tip(path, mask, at_end, max_ext=25.0):
    """Extend the path end along its local tangent until it exits the mask (recovers the tip the
    skeleton cannot reach). at_end=True extends the tail end, False the head end."""
    P = path if at_end else path[::-1]
    tang = P[-1] - P[-6] if len(P) >= 6 else P[-1] - P[0]
    n = np.linalg.norm(tang)
    if n < 1e-6:
        return path
    tang /= n
    H, W = mask.shape
    p = P[-1].copy()
    steps = 0
    while steps < max_ext / 0.25:
        q = p + 0.25 * tang
        xi, yi = int(round(q[0])), int(round(q[1]))
        if not (0 <= xi < W and 0 <= yi < H) or mask[yi, xi] == 0:
            break
        p = q
        steps += 1
    if steps == 0:
        return path
    ext = np.vstack([P, p[None]])
    return ext if at_end else ext[::-1]


def frame_midline(mask_full, row, pad=12):
    """Repaired-mask midline for one frame. Returns dict or None."""
    hx, hy = row[5], row[6]
    gx, gy, gw, gh = row[7], row[8], row[9], row[10]
    H, W = mask_full.shape
    x0, y0 = max(0, int(gx - pad)), max(0, int(gy - pad))
    x1, y1 = min(W, int(gx + gw + pad)), min(H, int(gy + gh + pad))
    clip = np.zeros_like(mask_full)
    clip[y0:y1, x0:x1] = mask_full[y0:y1, x0:x1]
    n, lab, st, cent = cv2.connectedComponentsWithStats((clip > 127).astype(np.uint8))
    if n < 2:
        return None
    # component containing (or nearest to) the GT head
    best, bd = -1, 1e18
    for i in range(1, n):
        if st[i, 4] < 200:
            continue
        m = lab == i
        ys, xs = np.nonzero(m)
        d = np.min(np.hypot(xs - hx, ys - hy))
        if d < bd:
            bd, best = d, i
    if best < 0 or bd > 40:
        return None
    m = (lab == best).astype(np.uint8)
    path = skeleton_path(m, (hx, hy))
    if path is None:
        return None
    path = extend_tip(path, m, at_end=True)
    path = extend_tip(path, m, at_end=False)
    return {"path": path, "mask": m, "head_dist": float(np.hypot(path[0, 0] - hx, path[0, 1] - hy))}


def smooth_resample(path, K, M, sigma=1.0):
    """Moving-average pre-smooth -> smoothing spline -> uniform-arclength resample.
    Returns kappa (K,), theta_K (K,), midline (M,2), L (px). Works in y-FLIPPED (right-handed) coords."""
    P = path.copy()
    P[:, 1] = -P[:, 1]                          # right-handed frame
    if len(P) >= 7:                             # 5-tap pre-smooth, keep the endpoints
        ker = np.ones(5) / 5
        inner = np.stack([np.convolve(P[:, i], ker, mode="valid") for i in (0, 1)], 1)
        P = np.vstack([P[:1], inner, P[-1:]])
    # drop duplicate consecutive points (splprep requires strictly increasing u)
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9]
    P = P[keep]
    if len(P) < 10:
        return None
    try:
        tck, _ = splprep(P.T, s=len(P) * sigma ** 2, k=3)
    except Exception:
        return None
    # uniform arc length re-parameterization
    uu = np.linspace(0, 1, 400)
    xy = np.stack(splev(uu, tck), 1)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    L = cum[-1]
    if L < 1e-6:
        return None
    u_of_s = np.interp(np.linspace(0, L, 400), cum, uu)
    s_sta = np.linspace(0, 1, K)
    u_sta = np.interp(s_sta * L, cum, uu)
    dx, dy = splev(u_sta, tck, der=1)
    ddx, ddy = splev(u_sta, tck, der=2)
    kappa = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    theta = np.arctan2(dy, dx)
    # unwrap along s so theta is a smooth profile
    theta = np.unwrap(theta)
    u_mid = np.interp(np.linspace(0, L, M), cum, uu)
    mid = np.stack(splev(u_mid, tck), 1)
    return {"kappa_px": kappa, "theta": theta, "midline": mid, "L": L}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=K_DEFAULT)
    ap.add_argument("--sigma", type=float, default=1.0, help="spline residual px per point")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    gt = np.loadtxt(f"{DATA}/gt/gt.txt", delimiter=",")
    rows = {int(r[0]): r for r in gt}
    frames = sorted(rows)
    T, K = len(frames), args.K

    # px->cm at the reference plane (aquarium top corners, world z=0): mean edge scale
    import re
    raw = open(f"{DATA}/camT_references.json").read()
    refs = json.loads(re.sub(r"/\*.*?\*/", "", raw, flags=re.S))   # file has C-style comments
    px_per_cm = np.nan
    try:
        cams = [(c["camera"]["x"], c["camera"]["y"]) for c in refs]
        d01 = np.hypot(cams[1][0] - cams[0][0], cams[1][1] - cams[0][1])
        d12 = np.hypot(cams[2][0] - cams[1][0], cams[2][1] - cams[1][1])
        d23 = np.hypot(cams[3][0] - cams[2][0], cams[3][1] - cams[2][1])
        d30 = np.hypot(cams[0][0] - cams[3][0], cams[0][1] - cams[3][1])
        px_per_cm = float((d01 + d12 + d23 + d30) / 4 / 29.0)
    except Exception:
        pass

    kappa_bl = np.full((T, K), np.nan, np.float32)
    theta_a = np.full((T, K), np.nan, np.float32)
    mid_a = np.full((T, M_DENSE, 2), np.nan, np.float32)
    Lpx = np.full(T, np.nan, np.float32)
    reason = np.zeros(T, np.int8)
    head_d = np.full(T, np.nan, np.float32)

    for i, fr in enumerate(frames):
        r = rows[fr]
        m = cv2.imread(f"{MASKS}/{fr:06d}.png", cv2.IMREAD_GRAYSCALE)
        if m is None:
            reason[i] = 1
            continue
        fm = frame_midline(m, r)
        if fm is None:
            reason[i] = 2
            continue
        sr = smooth_resample(fm["path"], K, M_DENSE, sigma=args.sigma)
        if sr is None:
            reason[i] = 2
            continue
        kappa_bl[i] = (sr["kappa_px"] * sr["L"]).astype(np.float32)   # dimensionless kappa*BL
        theta_a[i] = sr["theta"].astype(np.float32)
        mid = sr["midline"].copy()
        mid[:, 1] = -mid[:, 1]                                        # back to image y for overlays
        mid_a[i] = mid.astype(np.float32)
        Lpx[i] = sr["L"]
        head_d[i] = fm["head_dist"]

    # QC pass 2: body-length rolling-median gate + head-anchor gate
    ok = reason == 0
    med = np.nanmedian(Lpx[ok])
    roll = np.copy(Lpx)
    w = 61
    for i in range(T):
        lo, hi = max(0, i - w // 2), min(T, i + w // 2 + 1)
        v = Lpx[lo:hi]
        v = v[~np.isnan(v)]
        roll[i] = np.median(v) if len(v) else med
    len_bad = ok & (np.abs(Lpx - roll) > 0.20 * roll)
    head_bad = ok & ~len_bad & (head_d > 25.0)
    reason[len_bad] = 3
    reason[head_bad] = 4
    valid = reason == 0
    for arr in (kappa_bl, theta_a, Lpx):
        arr[~valid] = np.nan
    mid_a[~valid] = np.nan

    t_sec = (np.array(frames) - frames[0]) / FPS
    np.savez(
        f"{OUT}/curvature_dataset.npz",
        kappa_bl=kappa_bl, theta=theta_a, midline_px=mid_a,
        s_stations=np.linspace(0, 1, K).astype(np.float32),
        valid=valid, reason=reason, frame=np.array(frames, np.int32), t_sec=t_sec.astype(np.float32),
        bodylen_px=Lpx, bodylen_cm=(Lpx / px_per_cm).astype(np.float32),
        fps=np.float32(FPS), px_per_cm=np.float32(px_per_cm), head_dist_px=head_d,
    )
    n_inv = int((~valid).sum())
    print(f"[extract] {valid.sum()}/{T} frames valid ({n_inv} rejected: "
          f"no-mask {int((reason==1).sum())}, degenerate-skel {int((reason==2).sum())}, "
          f"len-outlier {int((reason==3).sum())}, head-far {int((reason==4).sum())})")
    print(f"  body length px: median {np.nanmedian(Lpx):.1f}  IQR [{np.nanpercentile(Lpx,25):.1f}, "
          f"{np.nanpercentile(Lpx,75):.1f}]  -> {np.nanmedian(Lpx)/px_per_cm:.2f} cm @ref-plane "
          f"(px_per_cm {px_per_cm:.1f})")
    print(f"  kappa*BL: std {np.nanstd(kappa_bl):.2f}  p99|.| {np.nanpercentile(np.abs(kappa_bl),99):.2f}")
    print(f"  wrote {OUT}/curvature_dataset.npz")


if __name__ == "__main__":
    main()
