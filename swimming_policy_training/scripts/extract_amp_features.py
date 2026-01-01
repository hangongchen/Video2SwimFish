#!/usr/bin/env python
"""Step 2c of the AMP pipeline: turn the carved reference clouds + trajectory features into the
AMP MOTION DATASET.

Per reference frame the AMP motion feature is:
    f = [ spine_bend_profile(K) , yaw_rate(1) , forward_speed(1) ]
where the spine-bend profile is the lateral (left/right) displacement of the body centerline along
arclength, extracted from the carved (canonical, head->+X) cloud -- this captures straight vs C-bent
(turning) vs S-wave (swimming) body shape. yaw_rate/speed come from the smoothed head track.

AMP observes transitions (f_t, f_{t+1}); pairs must stay WITHIN a contiguous segment, so we also
store segment_id (frames from different windows / with time gaps are different segments).

Output: amp_reference.npz  { feat (N, K+2), frame (N,), is_turn (N,), segment_id (N,), K }
"""
from __future__ import annotations

import argparse
from pathlib import Path

_R = str(Path(__file__).resolve().parents[1])

import numpy as np


def spine_bend_profile(pts, K, n_bins=24):
    """Lateral centerline offset y(s) along the body (X) axis, resampled to K, body-length-normalized.
    pts: (M,3) canonical cloud (X=body length, Y=lateral, Z=dorsoventral)."""
    x = pts[:, 0]
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    if hi - lo < 1e-6:
        return np.zeros(K, np.float32)
    edges = np.linspace(lo, hi, n_bins + 1)
    cx, cy = [], []
    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() >= 3:
            cx.append(0.5 * (edges[i] + edges[i + 1]))
            cy.append(float(pts[m, 1].mean()))
    if len(cx) < 4:
        return np.zeros(K, np.float32)
    cx = np.asarray(cx); cy = np.asarray(cy) - np.mean(cy)     # center the lateral offset
    body_len = hi - lo
    s = (cx - cx[0]) / max(cx[-1] - cx[0], 1e-6)               # normalized arclength [0,1]
    prof = np.interp(np.linspace(0, 1, K), s, cy) / max(body_len, 1e-6)
    return prof.astype(np.float32)


def head_anchored_bend_profile(pts, K, n_bins=24):
    """Lateral offset y(s) measured RELATIVE TO THE HEAD, resampled to K, body-length-normalized.

    Identical to spine_bend_profile EXCEPT the profile is anchored at the head (prof -= prof[head])
    instead of mean-centered. Mean-centering inflates the head-station amplitude (the DC swing of the
    whole body is redistributed across stations), which flattens the tail/head envelope ratio to ~2
    even on clean data; head-anchoring measures bend as displacement from the head heading, giving the
    biologically-correct rising-to-tail envelope (tail/head ~4-6, node at the head). Used IDENTICALLY
    on the real reference (top-view midline) and the sim FEM body cloud so the discriminator compares
    like with like (AMP requires the same observation map Phi on both). pts col0 = along-body with the
    HEAD at the low-x end, col1 = lateral."""
    x = pts[:, 0]
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi - lo < 1e-6:
        return np.zeros(K, np.float32)
    edges = np.linspace(lo, hi, n_bins + 1)
    cx, cy = [], []
    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() >= 3:
            cx.append(0.5 * (edges[i] + edges[i + 1]))
            cy.append(float(pts[m, 1].mean()))
    if len(cx) < 5:
        return np.zeros(K, np.float32)
    cx = np.asarray(cx); cy = np.asarray(cy)
    body_len = hi - lo
    s = (cx - cx[0]) / max(cx[-1] - cx[0], 1e-6)
    prof = np.interp(np.linspace(0, 1, K), s, cy)
    # ANCHOR to the anterior body (mean of the first ~15% of stations), not a single station:
    # subtracting one noisy head station injects its jitter into every station (autocorr collapses);
    # the anterior mean is a robust head-heading baseline.
    prof = prof - prof[: max(2, K // 7)].mean()
    return (prof / max(body_len, 1e-6)).astype(np.float32)


def anterior_aligned_profile(along, lateral, K, anterior_frac=0.25, n_bins=24):
    """Bend profile measured in a frame ALIGNED TO THE ANTERIOR BODY TANGENT (fixes Problem 1).

    head_anchored_bend_profile assumes the caller's (along, lateral) axis is the body axis, but the
    head->centroid / root-heading axes MIS-ALIGN on sharply-bent (turning) frames: the axis points into
    the bend, so the fish renders ~perpendicular to it and the tail deflection is projected onto the
    wrong axis. Here we fit a line to the ANTERIOR (nearly-rigid, first `anterior_frac` of the body by
    along-position), rotate the cloud so that anterior tangent is horizontal, THEN extract. The tail
    bend is then measured relative to where the HEAD actually points -- correct for cruise AND turns.
    `along` (head at low), `lateral` = perpendicular, both 1-D arrays in the same length unit."""
    along = np.asarray(along, float); lateral = np.asarray(lateral, float)
    lo = np.percentile(along, 1); hi = np.percentile(along, 99)
    span = hi - lo
    if span < 1e-6:
        return np.zeros(K, np.float32)
    ant = along < lo + anterior_frac * span
    if ant.sum() > 8:
        slope = np.polyfit(along[ant], lateral[ant], 1)[0]        # anterior tangent slope
        theta = -np.arctan(slope)
        c, s = np.cos(theta), np.sin(theta)
        along, lateral = c * along - s * lateral, s * along + c * lateral   # rotate to anterior tangent
    return head_anchored_bend_profile(np.stack([along, lateral, np.zeros_like(along)], 1), K, n_bins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clouds", default=_R + "/outputs/zef_amp_ref/pointclouds")
    ap.add_argument("--traj", default=_R + "/outputs/zef_amp_ref/traj_feats.npz")
    ap.add_argument("--out", default=_R + "/outputs/zef_amp_ref/amp_reference.npz")
    ap.add_argument("--profile-len", type=int, default=20, help="K: spine-bend profile length")
    ap.add_argument("--gap-frames", type=int, default=4, help="frame gap above which a new segment starts")
    args = ap.parse_args()

    clouds_dir = Path(args.clouds)
    carved = np.load(clouds_dir / "carved_frames.npy")
    files = sorted(clouds_dir.glob("zefref_*.npy"))
    assert len(files) == len(carved), f"{len(files)} clouds vs {len(carved)} frames"

    tf = np.load(args.traj)
    yaw_by_frame = {int(f): float(y) for f, y in zip(tf["frame"], tf["yaw_rate"])}
    spd_by_frame = {int(f): float(s) for f, s in zip(tf["frame"], tf["speed"])}
    turn_by_frame = {int(f): bool(t) for f, t in zip(tf["frame"], tf["is_turn"])}

    K = args.profile_len
    feats, frames, is_turn = [], [], []
    for fp, fr in zip(files, carved):
        pts = np.load(fp)
        prof = spine_bend_profile(pts, K)
        yaw = yaw_by_frame.get(int(fr), 0.0)
        spd = spd_by_frame.get(int(fr), 0.0)
        feats.append(np.concatenate([prof, [yaw, spd]]).astype(np.float32))
        frames.append(int(fr)); is_turn.append(turn_by_frame.get(int(fr), False))

    frames = np.array(frames); feats = np.stack(feats)
    seg = np.zeros(len(frames), np.int64)
    for i in range(1, len(frames)):
        seg[i] = seg[i - 1] + (1 if frames[i] - frames[i - 1] > args.gap_frames else 0)

    np.savez(args.out, feat=feats, frame=frames, is_turn=np.array(is_turn),
             segment_id=seg, K=K)
    n_pairs = int(np.sum(seg[1:] == seg[:-1]))
    print(f"[extract_amp_features] {len(frames)} frames -> feat {feats.shape} "
          f"(K={K} bend + yaw + speed), {seg.max()+1} segments, {n_pairs} valid AMP transition pairs")
    print(f"  turn frames {int(np.sum(is_turn))}/{len(frames)}  "
          f"bend std {feats[:, :K].std():.3f}  yaw range [{feats[:,K].min():.1f},{feats[:,K].max():.1f}] rad/s")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
