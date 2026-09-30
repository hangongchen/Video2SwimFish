#!/usr/bin/env python
"""GT-FREE real-video midline/curvature extraction for a new species (no 3D-ZeF gt.txt
available) -- same algorithm as extract_curvature.py (skeletonize -> BFS head->tail path ->
tip extension -> smoothing-spline arclength resample -> kappa/theta), reusing that script's
own skeleton_path/extend_tip/smooth_resample functions verbatim (they only ever needed a mask
+ a head_xy seed point, never gt.txt directly).

What replaces gt.txt (documented, since this is the one real gap flagged by investigation):
  - BLOB SELECTION: background-subtraction (median-of-N-frames) + Otsu threshold + morphology
    + connected components; the fish blob = the component closest to the PREVIOUS frame's
    centroid (temporal tracking), falling back to the single largest component on frame 0.
  - HEAD SEEDING (per skeleton_path's own `head_xy` argument): frame 0 uses a WIDTH heuristic
    (the fish's head end is bulkier than its tail end -- compare mask width near each skeleton
    endpoint); every later frame uses the PREVIOUS frame's tracked head pixel as the seed, so
    skeleton_path picks the endpoint nearest it -- a temporal-consistency tracker, not a
    per-frame absolute detector.

CORRECTION (found via a smoke test on catfish/top001.mp4, which showed 60% of frames rejected
by the length-outlier QC even though blob-detection and skeletonization never failed): these
tanks are NOT single-fish -- catfish/top001.mp4 has TWO catfish sharing the frame, crossing
paths and occasionally touching (visually confirmed at raw frame 400). A bare nearest-centroid
tracker with no gate silently (a) swapped identity onto the other fish, and (b) at true-contact
frames, connected-components MERGES both fish into one blob, whose skeleton/length is garbage
for either individual. The original length-outlier QC gate (reason=3) was, in hindsight, doing
its job correctly -- rejecting exactly these contaminated frames -- but it was the LAST line of
defense catching a problem that should be stopped at blob-pick time (two similarly-sized fish
can coincidentally satisfy the length tolerance while carrying the WRONG identity's head/tail
continuity). Fix: `pick_blob` now gates candidates by (1) AREA plausibility against a running
EMA of the tracked fish's own area (rejects merged-blob frames, which are ~1.5-2x too big, and
fragment/occlusion blobs, which are too small) and (2) a max per-frame CENTROID JUMP (rejects
an instantaneous swap onto the other fish, which is never co-located with the tracked one except
during an actual crossing). No candidate passing both gates => the frame is marked invalid
(reason=1) rather than force-picking the nearest blob regardless of plausibility. A long gap
with no valid pick resets the tracker (prev_centroid/prev_head -> None) so it re-acquires via
the same area-plausibility test instead of dead-reckoning from a stale, possibly now-wrong,
position.

SECOND correction, found while re-testing the above: a bare area+jump gate still lets a clean
IDENTITY SWAP through when two similarly-sized catfish briefly touch at low relative speed (area
and displacement both stay inside-gate through the crossing) -- the tracker then locks onto the
OTHER fish and stays self-consistent from then on (its area/length settle onto a new, stable
plateau), so nothing about the blob itself looks wrong anymore. There is no appearance cue
available to disambiguate two same-species fish with no gt.txt, so this is only ever partially
fixable. Mitigation: an immediate LOCAL body-length check against a short trailing window
(`recent_lengths`, last 10 accepted picks) -- reason=4 -- rejects a frame whose skeleton length
jumps >30% from that recent local median, and crucially does NOT commit the tracker state
(prev_centroid/prev_head/expected_area) on rejection, so the tracker keeps waiting for the
ORIGINAL fish to reappear near its last known state rather than adopting the swap. This does not
guarantee perfect single-individual identity for an entire video; it guarantees no MERGED-blob
or sudden-mismatched-length frame ever contaminates the output, and it splits a video across a
swap into separate locally-consistent valid runs (each internally a single real fish) -- which is
exactly what the downstream trajectory step already expects (`runs` = contiguous valid stretches),
and is sufficient for a species-level pooled curvature/speed manifold even though it cannot
promise one continuous individual for the full 3-minute video.

Resource tradeoff (documented, not hidden): these videos are 150-180s at 32 fps (~5000-5800
raw frames) vs ZeF-05's curated ~900-frame/15s clip. Processing every raw frame at full
2736x1824 resolution would take hours per fish. This script (a) subsamples to --stride raw
frames, (b) caps total processed frames at --max_frames, and (c) crops to a coarse
background-subtraction ROI BEFORE skeletonizing (skeletonize cost scales with pixel count,
and the fish only ever occupies a small fraction of the frame).

Usage:
  python scripts/extract_species_curvature.py --video raw_datasets/catfish/top001.mp4 \
      --species catfish --fish_id 001 --view top
Output: <SPECIES_MANIFOLD_ROOT>/<species>/<view>/fish<fish_id>_curvature.npz
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import splev, splprep
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_curvature import extend_tip, skeleton_path, smooth_resample  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2sf_paths as P  # noqa: E402

OUT_ROOT = P.SPECIES_MANIFOLD_ROOT


def median_background(cap, upto, n_sample=30):
    """Sample frames for the background median via a SINGLE sequential decode pass over
    [0, upto) -- these videos have essentially no keyframes after frame 0 (confirmed via
    ffprobe: key_frame=1 only at t=0), so cv2's cap.set(CAP_PROP_POS_FRAMES) forced a
    full re-decode from frame 0 on EVERY seek, making random-access sampling O(total^2).
    Assumes cap is currently positioned at frame 0."""
    idxs = set(np.linspace(0, upto - 1, n_sample).astype(int).tolist())
    frames = []
    for i in range(upto):
        ok, fr = cap.read()
        if not ok:
            break
        if i in idxs:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def orient_head_first(path, mask, frac=0.35):
    """Return (path, flipped): reverse the skeleton path if the mask is wider near its END than near
    its START (fish are bulkier at the head). Uses the distance transform = half-width along the path,
    averaged over the first / last `frac` of the path, so a per-frame decision is robust to the tip
    pixels. Measured 2026-09-25: the frame-0-only decision left white_bass_fish008 tracked TAIL-FIRST
    for whole runs (green s=0 marker on the thin end), which reverses kappa(s) and the heading psi."""
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    xs = np.clip(path[:, 0].astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(path[:, 1].astype(int), 0, mask.shape[0] - 1)
    w = dt[ys, xs]
    n = max(3, int(frac * len(w)))
    if w[-n:].mean() > w[:n].mean():
        return path[::-1].copy(), True
    return path, False


def segment_frame(gray, bg, thresh=25, min_area=400, open_px=3, close_px=9, ratio_thresh=None, ratio_min_diff=20.0):
    # thresh: background-difference cutoff. The default 25 was tuned on the small zebrafish
    # videos; on the catfish videos it also admits the fish's SHADOW (gray ~75-125 vs body
    # ~40-60, background ~143), so the skeleton ran off the head into the shadow and off the
    # tail along the translucent fin (measured: |kappa*BL| p99 = 64 vs ZeF's 3). open_px: the
    # opening kernel; 3 px kept the catfish's barbels (4-6 px wide) in the mask, which the
    # skeleton then followed. Both are CLI-tunable; the defaults stay the zebrafish values.
    if ratio_thresh is not None:
        # CONTRAST threshold: fish pixels are darker than the local background by a fixed RATIO
        # (white bass body gray/bg ~0.15-0.25, its shadow ~0.7-0.9) whereas the absolute difference
        # depends on how bright the background is where the fish happens to be (bg 54 at the shaded
        # end of the tank -> max diff 54 < thresh 60 -> "no blob" for the whole run).
        g32, b32 = gray.astype(np.float32), bg.astype(np.float32)
        # + an absolute floor (bg - gray >= ratio_min_diff): in near-black tank regions (bg ~ 5-20) sensor noise
        # satisfies the ratio alone and produced huge false blobs (bluegill 020 tracked a dark floor patch)
        mask = ((g32 < ratio_thresh * np.maximum(b32, 1.0)) & (b32 - g32 >= ratio_min_diff)).astype(np.uint8) * 255
    else:
        diff = cv2.absdiff(gray, bg)
        _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(mask)
    comps = [(i, st[i, 4], cent[i]) for i in range(1, n) if st[i, 4] >= min_area]
    return lab, comps


def pick_blob(lab, comps, prev_centroid, expected_area, max_jump):
    """Pick the single-fish blob, gated by area plausibility (rejects merged 2-fish blobs and
    occlusion fragments) and, when tracking, a max centroid jump (rejects an instantaneous
    identity swap onto another fish). Returns (mask, centroid, area) or None if no candidate
    survives -- callers must treat None as "no usable blob this frame", not "pick anyway"."""
    if not comps:
        return None
    if expected_area is not None:
        lo, hi = 0.55 * expected_area, 1.6 * expected_area
        gated = [c for c in comps if lo <= c[1] <= hi]
    else:
        gated = comps
    if not gated:
        return None
    if prev_centroid is None:
        i, area, cen = max(gated, key=lambda c: c[1])
    else:
        i, area, cen = min(gated, key=lambda c: np.hypot(c[2][0] - prev_centroid[0],
                                                          c[2][1] - prev_centroid[1]))
        if max_jump is not None and np.hypot(cen[0] - prev_centroid[0],
                                              cen[1] - prev_centroid[1]) > max_jump:
            return None
    return (lab == i).astype(np.uint8) * 255, cen, area


def width_at_endpoint(mask, path, at_end, span=8):
    """Mean mask width perpendicular to the path near one tip -- used ONLY to pick which
    skeleton endpoint is the head on frame 0 (heads are bulkier than tails)."""
    P = path if at_end else path[::-1]
    n = min(span, len(P) - 1)
    if n < 2:
        return 0.0
    seg = P[-n:] if at_end else P[:n]
    tang = seg[-1] - seg[0]
    norm = np.linalg.norm(tang)
    if norm < 1e-6:
        return 0.0
    perp = np.array([-tang[1], tang[0]]) / norm
    widths = []
    H, W = mask.shape
    for pt in seg:
        for d in range(0, 30):
            xi, yi = int(round(pt[0] + d * perp[0])), int(round(pt[1] + d * perp[1]))
            if not (0 <= xi < W and 0 <= yi < H) or mask[yi, xi] == 0:
                widths.append(d)
                break
        else:
            widths.append(30)
    return float(np.mean(widths)) * 2


def crop_bbox(mask, pad=20):
    ys, xs = np.nonzero(mask)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
    return x0, y0, x1, y1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--fish_id", required=True)
    ap.add_argument("--view", required=True, choices=["front", "top"])
    ap.add_argument("--stride", type=int, default=4, help="process every Nth raw frame")
    ap.add_argument("--max_frames", type=int, default=1400)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--thresh", type=int, default=25, help="background-diff threshold (see segment_frame)")
    ap.add_argument("--ratio_min_diff", type=float, default=20.0, help="with --ratio_thresh: also require bg-gray >= this (gray levels)")
    ap.add_argument("--ratio_thresh", type=float, default=None,
                    help="segment by gray < ratio*background instead of the absolute --thresh (see segment_frame)")
    ap.add_argument("--open_px", type=int, default=3, help="morphological opening kernel (px); raise to strip barbels/fin rays")
    ap.add_argument("--close_px", type=int, default=9)
    ap.add_argument("--sigma_bl", type=float, default=None,
                    help="spline smoothing as a FRACTION OF BODY LENGTH (px) instead of absolute --sigma. The absolute "
                         "kernel over-smooths small fish: sigma=7 is 1.5%% of a 473 px catfish but 3.6%% of a 195 px "
                         "white bass, which flattened kappa(s) to a straight line in 88%% of frames "
                         "(PCA_INVESTIGATION_white_bass_fish008.md). 0.015 reproduces the catfish setting for every fish.")
    ap.add_argument("--open_bl", type=float, default=None,
                    help="morphological opening kernel as a fraction of body length (0.019 = 9 px on the 473 px catfish); "
                         "uses --open_px until 3 frames have been accepted (no length estimate yet)")
    ap.add_argument("--gap_reset_frames", type=int, default=15,
                     help="processed frames with no valid pick before the tracker forgets its "
                          "last position/area and re-acquires fresh (avoids dead-reckoning onto "
                          "a stale, possibly now-wrong, location after a long occlusion/crossing)")
    ap.add_argument("--roi", default=None,
                    help="x0,y0,x1,y1 pixel box; motion OUTSIDE it is ignored (default: whole frame). "
                         "Added 2026-09-20: on the bluegill/white_bass top videos a wet patch on the ledge "
                         "above the tank is the largest 'moving' blob, so restrict to the tank interior.")
    args = ap.parse_args()
    roi = None
    if args.roi:
        roi = tuple(int(v) for v in args.roi.split(","))
        assert len(roi) == 4, "--roi must be x0,y0,x1,y1"

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"cannot open {args.video}"
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = src_fps / args.stride
    print(f"[extract_species] {args.video}: src_fps={src_fps:.1f} stride={args.stride} "
          f"-> effective fps={out_fps:.2f}", flush=True)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_raw = min(total, args.max_frames * args.stride)
    frame_idxs = list(range(0, n_raw, args.stride))
    bg = median_background(cap, n_raw)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # one seek back to the start is fine -- it's frame 0,
                                          # the only real keyframe, so it's cheap

    T, K, M = len(frame_idxs), args.K, args.M
    kappa_bl = np.full((T, K), np.nan, np.float32)
    theta_a = np.full((T, K), np.nan, np.float32)
    mid_a = np.full((T, M, 2), np.nan, np.float32)
    Lpx = np.full(T, np.nan, np.float32)
    flipped = np.zeros(T, bool)    # orient_head_first reversed the skeleton path this frame
    reason = np.zeros(T, np.int8)  # 0 ok, 1 no-blob, 2 degenerate-skel, 3 global-len-outlier,
                                    # 4 local-length-jump (likely identity swap / partial merge)

    prev_centroid, prev_head = None, None
    expected_area = None       # EMA of the tracked fish's own blob area, in pixels
    gap_count = 0               # processed frames since the last valid pick
    recent_lengths = deque(maxlen=10)   # short trailing window of ACCEPTED body lengths (px)

    def _reject(i, code):
        nonlocal gap_count, prev_centroid, prev_head, expected_area
        reason[i] = code
        gap_count += 1
        if gap_count > args.gap_reset_frames:
            prev_centroid, prev_head, expected_area = None, None, None
            recent_lengths.clear()      # 2026-09-26: the trailing-length window must reset with the tracker, otherwise a
                                        # few noise blobs accepted at start-up (white bass 020: L~70 px) pin the length gate
                                        # forever and every real-fish frame is rejected as a "local length jump" (2370/2400)

    # single sequential decode pass over the raw video -- see median_background's docstring for
    # why: cap.set() forces a full re-decode from frame 0 on this codec, so per-frame seeking
    # (the original approach) was O(n_raw^2/stride) instead of O(n_raw).
    i = 0
    for fidx in range(n_raw):
        ok, fr = cap.read()
        if not ok:
            break
        if fidx % args.stride != 0:
            continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if roi is not None:
            # outside the ROI make the frame identical to the background -> zero difference there
            x0, y0, x1, y1 = roi
            masked = bg.copy()
            masked[y0:y1, x0:x1] = gray[y0:y1, x0:x1]
            gray = masked
        open_px = args.open_px
        if args.open_bl is not None and len(recent_lengths) >= 3:
            open_px = max(1, int(round(args.open_bl * float(np.median(recent_lengths)))))
        lab, comps = segment_frame(gray, bg, thresh=args.thresh, open_px=open_px, close_px=args.close_px,
                                   ratio_thresh=args.ratio_thresh, ratio_min_diff=args.ratio_min_diff)
        # max plausible per-processed-frame displacement: a few body-lengths, scaled from area
        max_jump = 3.0 * np.sqrt(expected_area) if expected_area is not None else None
        picked = pick_blob(lab, comps, prev_centroid, expected_area, max_jump)
        if picked is None:
            _reject(i, 1)
            i += 1
            continue
        mask_full, cen, area = picked
        x0, y0, x1, y1 = crop_bbox(mask_full)
        mask = mask_full[y0:y1, x0:x1]

        if prev_head is None:
            skel = skeletonize(mask > 0)
            ys, xs = np.nonzero(skel)
            if len(xs) < 20:
                _reject(i, 2)
                i += 1
                continue
            # seed with an arbitrary endpoint, then decide head by width comparison below
            path0 = skeleton_path(mask, (xs[0], ys[0]))
            if path0 is None:
                _reject(i, 2)
                i += 1
                continue
            w_start = width_at_endpoint(mask, path0, at_end=False)
            w_end = width_at_endpoint(mask, path0, at_end=True)
            head_xy_local = tuple(path0[0]) if w_start >= w_end else tuple(path0[-1])
        else:
            head_xy_local = (prev_head[0] - x0, prev_head[1] - y0)

        path = skeleton_path(mask, head_xy_local)
        if path is None:
            _reject(i, 2)
            i += 1
            continue
        path = extend_tip(path, mask, at_end=True)
        path = extend_tip(path, mask, at_end=False)
        path, flipped[i] = orient_head_first(path, mask)   # per-frame head/tail check (was frame-0 only)
        path_full = path + np.array([x0, y0])
        candidate_head = tuple(path_full[0])

        sigma = args.sigma
        if args.sigma_bl is not None:
            # raw skeleton polyline length of THIS frame (px) -> kernel scales with the fish, not the camera
            L_raw = float(np.linalg.norm(np.diff(path_full.astype(np.float64), axis=0), axis=1).sum())
            sigma = args.sigma_bl * L_raw
        sr = smooth_resample(path_full, K, M, sigma=sigma)
        if sr is None:
            _reject(i, 2)
            i += 1
            continue

        L = sr["L"]
        if len(recent_lengths) >= 3:
            med = float(np.median(recent_lengths))
            if abs(L - med) > 0.3 * med:
                # do NOT commit prev_centroid/prev_head/expected_area here -- keep waiting for
                # the ORIGINAL fish rather than adopting whatever this frame tracked instead
                _reject(i, 4)
                i += 1
                continue

        # accepted: commit tracker state and write outputs
        gap_count = 0
        prev_centroid = cen
        expected_area = area if expected_area is None else 0.9 * expected_area + 0.1 * area
        prev_head = candidate_head
        recent_lengths.append(L)

        kappa_bl[i] = (sr["kappa_px"] * L).astype(np.float32)
        theta_a[i] = sr["theta"].astype(np.float32)
        mid = sr["midline"].copy()
        mid[:, 1] = -mid[:, 1]
        mid_a[i] = mid.astype(np.float32)
        Lpx[i] = L
        i += 1

    # if the decode ended early (shouldn't normally happen since n_raw <= real frame count),
    # mark any never-reached tail slots invalid rather than leaving them as default reason=0
    if i < T:
        reason[i:] = 1

    ok = reason == 0
    if ok.sum() > 5:
        med = np.nanmedian(Lpx[ok])
        roll = np.copy(Lpx)
        w = min(61, T)
        for i in range(T):
            lo, hi = max(0, i - w // 2), min(T, i + w // 2 + 1)
            v = Lpx[lo:hi]
            v = v[~np.isnan(v)]
            roll[i] = np.median(v) if len(v) else med
        len_bad = ok & (np.abs(Lpx - roll) > 0.25 * roll)
        reason[len_bad] = 3
    valid = reason == 0
    for arr in (kappa_bl, theta_a, Lpx):
        arr[~valid] = np.nan
    mid_a[~valid] = np.nan

    out_dir = OUT_ROOT / args.species / args.view
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fish{args.fish_id}_curvature.npz"
    np.savez(
        out_path, kappa_bl=kappa_bl, theta=theta_a, midline_px=mid_a,
        s_stations=np.linspace(0, 1, K).astype(np.float32),
        valid=valid, reason=reason, frame=np.array(frame_idxs, np.int32),
        t_sec=(np.arange(T) / out_fps).astype(np.float32),
        bodylen_px=Lpx, fps=np.float32(out_fps), src_fps=np.float32(src_fps),
        stride=np.int32(args.stride), video=str(args.video),
        flipped=flipped, ratio_thresh=np.float32(args.ratio_thresh if args.ratio_thresh is not None else np.nan),
        sigma=np.float32(args.sigma), sigma_bl=np.float32(args.sigma_bl if args.sigma_bl is not None else np.nan),
        open_px=np.int32(args.open_px), open_bl=np.float32(args.open_bl if args.open_bl is not None else np.nan),
    )
    n_inv = int((~valid).sum())
    print(f"[extract_species] head/tail flips applied on {int(flipped[valid].sum())}/{int(valid.sum())} valid frames", flush=True)
    print(f"[extract_species] {valid.sum()}/{T} frames valid ({n_inv} rejected: "
          f"no-blob={int((reason==1).sum())} degenerate-skel={int((reason==2).sum())} "
          f"global-len-outlier={int((reason==3).sum())} "
          f"local-length-jump={int((reason==4).sum())})", flush=True)
    print(f"[extract_species] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
