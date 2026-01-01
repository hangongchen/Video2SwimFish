#!/usr/bin/env python
"""Crop each already-selected canonical frame down to just the fish region, using CLASSICAL
CV (background-subtraction + connected components) instead of Qwen3-VL -- reuses the exact
same segment_frame/pick_blob/crop_bbox/median_background functions already built and
validated in scripts/extract_species_curvature.py for the full-video midline
extraction, rather than a second, independent detector.

Why this works without per-image temporal context: each canonical JPG is a single still frame
copied out of one of the ORIGINAL <view>NNN.mp4 videos (see select_canonical_frame_qwen.py) at
the SAME resolution, so a proper background image can still be built from that same video (a
median over many sampled raw frames) even without knowing which exact frame the canonical JPG
was. With no previous-frame identity to track, blob selection falls back to "largest blob" --
the same frame-0 fallback the video pipeline itself uses, and a reasonable choice here since
select_canonical_frame_qwen.py's own prompt already biased toward frames with one clear,
unoccluded, in-focus fish (usually the nearer/larger blob when more than one fish is visible).

Usage:
  python scripts/crop_canonical_to_fish_cv.py --species catfish \
      --canonical_dir raw_datasets/catfish_canonical_frames \
      --video_dir raw_datasets/catfish --view front
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_species_curvature import (  # noqa: E402
    crop_bbox, median_background, pick_blob, segment_frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--canonical_dir", required=True, help="dir with <species>_fish<id>_canonical.jpg")
    ap.add_argument("--video_dir", required=True, help="dir with the source <view>NNN.mp4 files")
    ap.add_argument("--view", default="front")
    ap.add_argument("--out_dir", default=None, help="default: <canonical_dir>_cropped")
    ap.add_argument("--margin", type=float, default=0.15,
                     help="fractional padding around the detected bbox -- kept modest (unlike the "
                          "0.35 used for the Qwen VLM boxes) because a background-subtraction mask "
                          "bbox is a direct pixel measurement, not a VLM's approximate localization, "
                          "so it is not prone to the same systematic under-inclusion")
    ap.add_argument("--bg_samples", type=int, default=40)
    args = ap.parse_args()

    canonical_dir = Path(args.canonical_dir).resolve()
    video_dir = Path(args.video_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else canonical_dir.parent / f"{canonical_dir.name}_cropped"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(canonical_dir.glob(f"{args.species}_fish*_canonical.jpg"))
    assert images, f"no canonical images found in {canonical_dir}"
    print(f"[crop_cv] {len(images)} canonical images in {canonical_dir}", flush=True)

    n_ok, n_fallback = 0, 0
    for img_path in images:
        m = re.search(r"fish(\d+)", img_path.stem)
        fish_id = m.group(1) if m else None
        video = video_dir / f"{args.view}{fish_id}.mp4"
        out_path = out_dir / img_path.name
        im_bgr = cv2.imread(str(img_path))
        if im_bgr is None:
            print(f"[crop_cv] {img_path.name}: could not read image, skipping", flush=True)
            continue

        if fish_id is None or not video.exists():
            print(f"[crop_cv] {img_path.name}: no matching video ({video}), keeping uncropped", flush=True)
            cv2.imwrite(str(out_path), im_bgr)
            n_fallback += 1
            continue

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[crop_cv] {img_path.name}: cannot open {video}, keeping uncropped", flush=True)
            cv2.imwrite(str(out_path), im_bgr)
            n_fallback += 1
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        bg = median_background(cap, total, n_sample=args.bg_samples)
        cap.release()

        gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        if gray.shape != bg.shape:
            bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]))
        lab, comps = segment_frame(gray, bg)
        picked = pick_blob(lab, comps, prev_centroid=None, expected_area=None, max_jump=None)
        if picked is None:
            print(f"[crop_cv] {img_path.name}: no fish blob detected, keeping uncropped", flush=True)
            cv2.imwrite(str(out_path), im_bgr)
            n_fallback += 1
            continue
        mask_full, cen, area = picked
        x0, y0, x1, y1 = crop_bbox(mask_full, pad=0)
        H, W = mask_full.shape
        bw, bh = x1 - x0, y1 - y0
        if (bw * bh) > 0.8 * (W * H):
            # background subtraction picked up diffuse tank-wide noise (debris, a moving
            # reflection through the glass) as one giant connected blob instead of isolating
            # the fish -- report this HONESTLY as a failed detection, don't silently count a
            # near-no-op "crop" (that covers ~the whole frame) as a success
            print(f"[crop_cv] {img_path.name}: detected blob covers {100*bw*bh/(W*H):.0f}% of "
                  f"the frame (background-subtraction false positive) -- keeping uncropped", flush=True)
            cv2.imwrite(str(out_path), im_bgr)
            n_fallback += 1
            continue
        mx, my = int(bw * args.margin), int(bh * args.margin)
        cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
        cx1, cy1 = min(W, x1 + mx), min(H, y1 + my)
        cropped = im_bgr[cy0:cy1, cx0:cx1]
        cv2.imwrite(str(out_path), cropped)
        print(f"[crop_cv] {img_path.name}: bbox=({x0},{y0},{x1},{y1}) area={area:.0f} "
              f"-> crop {cropped.shape[1]}x{cropped.shape[0]}", flush=True)
        n_ok += 1

    print(f"[crop_cv] done: {n_ok} cropped via classical CV, {n_fallback} kept uncropped "
          f"(no blob/video found)", flush=True)


if __name__ == "__main__":
    main()
