#!/usr/bin/env python
"""Species-level analog of scripts/zef_pca_rl/extract_zef_reference.py, applied per fish.

For every data/species_manifold/<species>/<view>/fish*_curvature.npz produced by
extract_species_curvature.py, this replicates extract_zef_reference.py's own logic EXACTLY
(same p_bl/psi definitions, same contiguous-run threshold >=60 frames, same 5-frame smoothing,
same STRIDE=30 / short=0.5,medium=1.0,long=2.0 BL horizons) to build one *_reference.npz per
fish, structurally IDENTICAL to zef_reference.npz -- any script that already reads that format
(e.g. eval_biofidelity_suite.py's zef_swimming_motion_reference) can read one of these unchanged.

Then pools every fish's per-sample mean speed (the exact same "per-(ti,tj)-segment mean of
norm(diff(p_bl))*fps, then aggregate" formula eval_biofidelity_suite.py's
zef_swimming_motion_reference uses to produce ZeF's own reported 3.443 BL/s) into one
species+view summary, so the number is directly comparable to that reference.

Usage: python extract_species_reference.py --species catfish --view top
Output: data/species_manifold/<species>/<view>/fish<id>_reference.npz  (per fish)
        data/species_manifold/<species>/<view>/speed_summary.npz      (pooled)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "data" / "species_manifold"

HORIZONS = {"short": 0.5, "medium": 1.0, "long": 2.0}
STRIDE = 30
MIN_RUN = 60


def smooth(x, k=5):
    ker = np.ones(k) / k
    out = x.copy()
    for c in range(x.shape[1]):
        out[k // 2:-(k // 2), c] = np.convolve(x[:, c], ker, mode="valid")
    return out


def contiguous_runs(valid, min_len=MIN_RUN):
    runs = []
    i = 0
    while i < len(valid):
        if valid[i]:
            j = i
            while j < len(valid) and valid[j]:
                j += 1
            if j - i >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def build_reference(npz_path: Path):
    """EXACT port of extract_zef_reference.py's per-file logic."""
    d = np.load(npz_path)
    mid = d["midline_px"].astype(np.float64)   # (T, 100, 2), already y-flipped by the extractor
    valid = d["valid"].astype(bool)
    fps = float(d["fps"])
    blpx = d["bodylen_px"].astype(np.float64)

    if valid.sum() < MIN_RUN:
        return None, f"only {int(valid.sum())} valid frames (<{MIN_RUN})"

    BLPX = float(np.nanmedian(blpx[valid]))
    p = np.nanmean(mid, axis=1) / BLPX
    head = mid[:, 0, :] / BLPX
    s30 = mid[:, 30, :] / BLPX
    psi = np.arctan2(head[:, 1] - s30[:, 1], head[:, 0] - s30[:, 0])

    runs = contiguous_runs(valid)
    if not runs:
        return None, "no run >=60 frames"

    samples = []
    for (a, b) in runs:
        p[a:b] = smooth(p[a:b])
        for ti in range(a, b - 30, STRIDE):
            dist = np.linalg.norm(p[ti:b] - p[ti], axis=1)
            for hi, (hname, D) in enumerate(HORIZONS.items()):
                hit = np.nonzero(dist >= D)[0]
                if len(hit) > 0:
                    samples.append((ti, ti + int(hit[0]), hi))
    samples = np.array(samples, dtype=np.int64) if samples else np.zeros((0, 3), dtype=np.int64)

    ref = dict(p_bl=p.astype(np.float32), psi=psi.astype(np.float32), valid=valid,
               runs=np.array(runs), samples=samples, fps=fps, bl_px=BLPX,
               horizons=np.array(list(HORIZONS.values())),
               horizon_names=np.array(list(HORIZONS.keys())),
               source=str(npz_path))
    return ref, None


def sample_speeds(ref: dict) -> np.ndarray:
    """Same formula as eval_biofidelity_suite.py's zef_swimming_motion_reference: per-(ti,tj)
    segment mean of norm(diff(p_bl))*fps."""
    S, p, fps = ref["samples"], ref["p_bl"].astype(np.float64), ref["fps"]
    out = []
    for ti, tj, _ in S:
        seg = p[ti:tj + 1]
        if len(seg) < 3:
            continue
        v = np.linalg.norm(np.diff(seg, axis=0), axis=1) * fps
        out.append(float(v.mean()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--view", required=True, choices=["front", "top"])
    args = ap.parse_args()

    in_dir = ROOT / args.species / args.view
    files = sorted(in_dir.glob("fish*_curvature.npz"))
    assert files, f"no curvature npz found in {in_dir}"

    all_speeds, all_turn = [], []
    per_fish = {}
    for f in files:
        fish_id = f.stem.replace("_curvature", "")
        ref, err = build_reference(f)
        if ref is None:
            print(f"[extract_species_ref] {fish_id}: SKIPPED ({err})", flush=True)
            per_fish[fish_id] = {"status": "skipped", "reason": err}
            continue
        out_path = in_dir / f"{fish_id}_reference.npz"
        np.savez(out_path, **ref)
        speeds = sample_speeds(ref)
        turn = None
        if len(speeds):
            # unwrap PER-SAMPLE SLICE, not the whole-timeline psi array: psi has NaN gaps
            # between runs, and np.unwrap's cumulative correction propagates a NaN forward
            # through every later value once it hits one -- unwrapping the full array poisons
            # every run after the first gap. Each (ti,tj) slice is guaranteed to lie entirely
            # inside one contiguous valid run, so it has no internal NaN to propagate.
            psi_raw = ref["psi"].astype(np.float64)
            fps = ref["fps"]
            turn_list = []
            for ti, tj, _ in ref["samples"]:
                seg = np.unwrap(psi_raw[ti:tj + 1])
                if len(seg) < 3:
                    continue
                turn_list.append(float(np.abs(np.diff(seg)).mean() * fps))
            turn = np.array(turn_list)
        n_valid = int(ref["valid"].sum())
        n_runs = len(ref["runs"])
        print(f"[extract_species_ref] {fish_id}: {n_valid} valid frames, {n_runs} runs, "
              f"{len(speeds)} speed samples"
              + (f", median speed {np.median(speeds):.3f} BL/s" if len(speeds) else ", NO SAMPLES")
              , flush=True)
        per_fish[fish_id] = {
            "status": "ok", "n_valid": n_valid, "n_runs": n_runs, "n_samples": len(speeds),
            "median_speed_bl_s": float(np.median(speeds)) if len(speeds) else None,
        }
        if len(speeds):
            all_speeds.append(speeds)
        if turn is not None and len(turn):
            all_turn.append(turn)

    pooled_speed = np.concatenate(all_speeds) if all_speeds else np.array([])
    pooled_turn = np.concatenate(all_turn) if all_turn else np.array([])

    summary = {
        "species": args.species, "view": args.view, "n_fish_total": len(files),
        "n_fish_with_samples": len(all_speeds),
        "n_samples_total": int(len(pooled_speed)),
        "speed_bl_s_median": float(np.median(pooled_speed)) if len(pooled_speed) else None,
        "speed_bl_s_mean": float(np.mean(pooled_speed)) if len(pooled_speed) else None,
        "speed_bl_s_p25": float(np.percentile(pooled_speed, 25)) if len(pooled_speed) else None,
        "speed_bl_s_p75": float(np.percentile(pooled_speed, 75)) if len(pooled_speed) else None,
        "turn_rate_rad_s_median": float(np.median(pooled_turn)) if len(pooled_turn) else None,
        "per_fish": per_fish,
    }
    out_path = in_dir / "speed_summary.npz"
    np.savez(out_path, pooled_speed=pooled_speed, pooled_turn=pooled_turn,
             summary_json=json.dumps(summary))
    print(f"\n[extract_species_ref] === {args.species}/{args.view} POOLED ({len(pooled_speed)} "
          f"samples from {len(all_speeds)}/{len(files)} fish) ===")
    if len(pooled_speed):
        print(f"  speed_bl_s: median={summary['speed_bl_s_median']:.3f} "
              f"mean={summary['speed_bl_s_mean']:.3f} "
              f"p25={summary['speed_bl_s_p25']:.3f} p75={summary['speed_bl_s_p75']:.3f}")
    if len(pooled_turn):
        print(f"  turn_rate_rad_s: median={summary['turn_rate_rad_s_median']:.3f}")
    print(f"[extract_species_ref] wrote {out_path}")


if __name__ == "__main__":
    main()
