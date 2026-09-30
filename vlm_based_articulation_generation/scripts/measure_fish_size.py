#!/usr/bin/env python
"""Real-world fish size from tank-wall pixel calibration.

User-provided ground truth: tank interior is 30" long x 17.5" deep x 11.75" wide.

METHOD: the tank's own walls (visible in the median background of each video, fish-free)
are a built-in ruler. Measure the walls' pixel span in a view, divide the known real span
(inches) by it -> pixels-per-inch for that view. Apply that scale to the already-measured
`bodylen_px` (median over valid frames, from extract_species_curvature.py's curvature.npz)
to get the fish's real length.

CALIBRATION, from manual pixel inspection of the tank walls in the median background
(see conversation record for the annotated verification images) -- both catfish and
lake_sturgeon share the same physical tank + fixed camera rig (confirmed: landmark edges
[e.g. the foam-support/wall transition at y=103] land at IDENTICAL pixel positions across
different fish of the same species, and the tank/apparatus look pixel-identical between
species), so ONE calibration per view covers all fish:

  TOP view: the tank's LENGTH (30") runs off the LEFT edge of the frame (confirmed: the
    visible width-based scale, from the fully-in-frame top/bottom walls, predicts a 30"
    span of ~3120px -- wider than the 2736px frame). The WIDTH (11.75", top/bottom walls,
    both fully visible) is the only usable reference here:
      TOP_WALL_Y=228, BOTTOM_WALL_Y=1450  ->  (1450-228)/11.75 = 104.0 px/inch
    This calibrates the CROSS-axis, not the fish's nose-to-tail length (which runs along
    the clipped length axis in this view) -- so top-view bodylen_px is NOT converted here.

  FRONT view: the FULL tank (both end walls) fits in frame, giving a direct nose-to-tail
    LENGTH-axis calibration:
      LEFT_WALL_X=20, RIGHT_WALL_X=2675  ->  (2675-20)/30 = 88.5 px/inch
    This is the axis used for real fish LENGTH (front view is a lateral/side view, so the
    fish's nose-to-tail extent runs along this same horizontal/length axis).

HONEST UNCERTAINTY: the length-axis and depth-axis (17.5", vertical in front view) scale
estimates disagree by 15-20% depending on which pair of walls is used (cross-checked at
both the top view and the front view, independently) -- e.g. front view's depth-based
estimate is ~75 px/inch vs 88.5 px/inch from length. This survived repeated, careful
re-measurement, so it looks like real, uncorrected lens/camera-angle distortion in this
single consumer-camera rig, not a pixel-reading mistake. Treat all sizes from this script
as a best estimate with roughly +/-15% systematic uncertainty, not a lab-grade measurement.

Usage:
  python measure_fish_size.py --species catfish
  python measure_fish_size.py --species lake_sturgeon
Output: <SPECIES_MANIFOLD_ROOT>/<species>/fish_sizes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2sf_paths as P  # noqa: E402

ROOT = P.SPECIES_MANIFOLD_ROOT      # <root>/<species>/front/fish<id>_curvature.npz -> <species>/fish_sizes.json

# ---- calibration (shared rig, both species -- see module docstring) ----
FRONT_PX_PER_INCH = 88.5      # length axis (horizontal), cross-checked against depth axis
FRONT_PX_PER_INCH_UNCERTAINTY = 0.18   # +/-18%, from the length-vs-depth cross-check spread

# Per-species override of the FRONT length-axis scale. bluegill / white_bass (added 2026-09-20)
# were filmed in the SAME tank but the camera was re-set: in their median background the tank's
# FRONT face spans x=32..2713 (white_bass) and x=83..2716 (bluegill) for the 30 in length, i.e.
# the same camera distance as catfish within 1-2% (catfish: x=20..2675). Measured from the
# strongest vertical Sobel edges of the median background (see /tmp/newsp/bg_*_front_grid.jpg).
#
# PER-SPECIES px/inch TABLE (front view, length axis; 30 in tank length / measured wall span):
#   catfish        88.5   measured  (x=20..2675)
#   lake_sturgeon  88.5   measured  (same rig as catfish)
#   white_bass     89.4   measured  (x=32..2713)
#   bluegill       87.8   measured  (x=83..2716)
#   brown_trout    89.0   measured  (documented in deploy/README_H200.md)
#   brook_trout    89.8   NOT MEASURED -- FLAGGED ESTIMATE. The automatic estimate (78.7) is wrong: the right
#                         tank wall is too low-contrast. 89.8 is the five-species median quoted in
#                         deploy/README_H200.md (five other species measured on the trout rig fell in
#                         89.0-90.2). Re-measure by hand if precision matters.
# A species that is not in the table falls back to FRONT_PX_PER_INCH (88.5) and prints a warning.
BROOK_TROUT_PX_PER_INCH_IS_ESTIMATE = True
FRONT_PX_PER_INCH_BY_SPECIES = {
    "catfish": 88.5,
    "lake_sturgeon": 88.5,
    "white_bass": (2713 - 32) / 30.0,    # 89.4
    "bluegill": (2716 - 83) / 30.0,      # 87.8
    "brown_trout": 89.0,                  # measured (README_H200)
    "brook_trout": 89.8,                  # ESTIMATE (five-species median from README_H200), flagged
}
TOP_WIDTH_PX_PER_INCH = 104.0  # cross-axis only -- NOT usable for fish nose-to-tail length

INCH_TO_M = 0.0254

# sanity bound for these juvenile catfish/lake-sturgeon fingerlings in a ~30in tank -- a
# median bodylen_px implying a fish longer than this is almost certainly multi-fish blob
# contamination or a tracking failure slipping through (same failure class documented in
# extract_species_curvature.py), not a real fish; flag rather than silently report it
PLAUSIBLE_MAX_IN = 20.0


def fish_length_from_front(species: str, fish_id: str) -> dict | None:
    npz_path = ROOT / species / "front" / f"fish{fish_id}_curvature.npz"
    if not npz_path.exists():
        return None
    d = np.load(npz_path)
    valid = d["valid"]
    if valid.sum() < 10:
        return None
    bodylen_px = float(np.nanmedian(d["bodylen_px"][valid]))
    ppi = FRONT_PX_PER_INCH_BY_SPECIES.get(species, FRONT_PX_PER_INCH)
    length_in = bodylen_px / ppi
    return {
        "fish_id": fish_id,
        "bodylen_px_median": bodylen_px,
        "n_valid_frames": int(valid.sum()),
        "length_in": length_in,
        "length_in_lo": bodylen_px / (ppi * (1 + FRONT_PX_PER_INCH_UNCERTAINTY)),
        "length_in_hi": bodylen_px / (ppi * (1 - FRONT_PX_PER_INCH_UNCERTAINTY)),
        "length_cm": length_in * 2.54,
        "length_m": length_in * INCH_TO_M,
        "suspect": length_in > PLAUSIBLE_MAX_IN,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    args = ap.parse_args()

    if args.species not in FRONT_PX_PER_INCH_BY_SPECIES:
        print(f"[measure_fish_size] WARNING: no px/in entry for '{args.species}', using the default "
              f"{FRONT_PX_PER_INCH} px/in (catfish rig). Add a measured entry to "
              f"FRONT_PX_PER_INCH_BY_SPECIES.", flush=True)
    if args.species == "brook_trout":
        print("[measure_fish_size] NOTE: brook_trout px/in (89.8) is a FLAGGED ESTIMATE, not a measurement.",
              flush=True)
    front_dir = ROOT / args.species / "front"
    if not front_dir.is_dir():
        sys.exit(f"[measure_fish_size] no curvature data at {front_dir}. Run scripts/extract_species_curvature.py "
                 f"--view front for each front video first (deploy/run_species_e2e.sh phase 2).")
    fish_ids = sorted(p.stem.replace("fish", "").replace("_curvature", "")
                       for p in front_dir.glob("fish*_curvature.npz"))

    results = {}
    lengths_in = []
    n_suspect = 0
    for fid in fish_ids:
        r = fish_length_from_front(args.species, fid)
        if r is None:
            print(f"[measure_fish_size] fish{fid}: insufficient front-view data, skipped", flush=True)
            continue
        results[fid] = r
        flag = " ** SUSPECT (likely multi-fish contamination, excluded from summary) **" if r["suspect"] else ""
        print(f"[measure_fish_size] fish{fid}: {r['length_in']:.2f} in "
              f"({r['length_in_lo']:.2f}-{r['length_in_hi']:.2f} in, "
              f"{r['length_cm']:.1f} cm) from {r['n_valid_frames']} valid frames{flag}", flush=True)
        if r["suspect"]:
            n_suspect += 1
        else:
            lengths_in.append(r["length_in"])

    # ---- robustness pass (added 2026-09-20 for bluegill / white_bass) ----
    # The front-view segmentation is unreliable on small fish: a few fish come out with very few
    # valid frames and/or a length far outside the rest of the group (multi-fish blobs, shadows),
    # and some have no usable front data at all. Rather than dropping those fish from the dataset
    # or scaling their assets to a wrong size, fall back to the species median and FLAG them.
    MIN_VALID_FRAMES = 150
    LO_FRAC, HI_FRAC = 0.4, 2.5
    reliable = [r["length_in"] for r in results.values()
                if r["n_valid_frames"] >= MIN_VALID_FRAMES and not r["suspect"]]
    if reliable:
        med = float(np.median(reliable))
        for fid, r in results.items():
            bad = (r["n_valid_frames"] < MIN_VALID_FRAMES
                   or not (LO_FRAC * med <= r["length_in"] <= HI_FRAC * med))
            if bad:
                r["measured_length_in_raw"] = r["length_in"]
                r["length_in"] = med
                r["length_cm"] = med * 2.54
                r["length_m"] = med * INCH_TO_M
                r["length_in_lo"] = med / (1 + FRONT_PX_PER_INCH_UNCERTAINTY)
                r["length_in_hi"] = med / (1 - FRONT_PX_PER_INCH_UNCERTAINTY)
                r["suspect"] = True
                r["size_source"] = "species_median_fallback"
                print(f"[measure_fish_size] fish{fid}: raw {r['measured_length_in_raw']:.2f} in from "
                      f"{r['n_valid_frames']} frames is unreliable -> species median {med:.2f} in", flush=True)
        # fish with no usable front data at all: still give them the median so the dataset keeps them
        all_ids = sorted(p.stem.replace("fish", "").replace("_curvature", "")
                         for p in front_dir.glob("fish*_curvature.npz"))
        for fid in all_ids:
            if fid not in results:
                results[fid] = {"fish_id": fid, "bodylen_px_median": None, "n_valid_frames": 0,
                                "length_in": med, "length_in_lo": med / (1 + FRONT_PX_PER_INCH_UNCERTAINTY),
                                "length_in_hi": med / (1 - FRONT_PX_PER_INCH_UNCERTAINTY),
                                "length_cm": med * 2.54, "length_m": med * INCH_TO_M,
                                "suspect": True, "size_source": "species_median_fallback"}
                print(f"[measure_fish_size] fish{fid}: no front data -> species median {med:.2f} in", flush=True)
        lengths_in = reliable
        n_suspect = sum(1 for r in results.values() if r["suspect"])
        results = dict(sorted(results.items()))

    lengths_in = np.array(lengths_in)
    summary = {
        "species": args.species,
        "calibration": {
            "front_px_per_inch": FRONT_PX_PER_INCH_BY_SPECIES.get(args.species, FRONT_PX_PER_INCH),
            "uncertainty_frac": FRONT_PX_PER_INCH_UNCERTAINTY,
            "method": "front-view tank wall span (30in length), see module docstring",
            "px_per_inch_is_estimate": bool(args.species == "brook_trout"),
        },
        "n_fish": len(results),
        "n_suspect_excluded": n_suspect,
        "median_length_in": float(np.median(lengths_in)) if len(lengths_in) else None,
        "median_length_cm": float(np.median(lengths_in) * 2.54) if len(lengths_in) else None,
        "min_length_in": float(lengths_in.min()) if len(lengths_in) else None,
        "max_length_in": float(lengths_in.max()) if len(lengths_in) else None,
        "per_fish": results,
    }
    out_path = ROOT / args.species / "fish_sizes.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[measure_fish_size] {args.species}: {len(results)} fish measured, "
          f"median length {summary['median_length_in']:.2f} in "
          f"({summary['median_length_cm']:.1f} cm)" if results else
          f"\n[measure_fish_size] {args.species}: NO fish measured", flush=True)
    print(f"[measure_fish_size] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
