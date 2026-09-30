#!/usr/bin/env python
"""Create the lowercase, space-free species dir the pipeline's tag regex requires.

Tags are built as `<species>_fish<NNN>` and must match `^(.*)_fish(\\d+)$`; a directory
named "Brook Trout" breaks both the tag and every shell path. This mirrors the
bluegill/white_bass precedent: symlinks, not copies.

    python deploy/make_species_dir.py --species brook_trout     # RAW_VIDEO_ROOT/"Brook Trout" -> brook_trout
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2sf_paths as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--species", required=True)
a = ap.parse_args()
dst = P.RAW_VIDEO_ROOT / a.species
if dst.exists():
    print(f"{dst} exists ({len(list(dst.glob('*.mp4')))} videos)")
    sys.exit(0)
pretty = a.species.replace("_", " ").title()
src = P.RAW_VIDEO_ROOT / pretty
if not src.is_dir():
    sys.exit(f"no raw dir for '{a.species}': tried {src}")
dst.symlink_to(src)
ids = sorted(p.stem[5:] for p in dst.glob("front*.mp4"))
print(f"{dst} -> {src}  ({len(ids)} front videos)")
print(f"fish ids: {','.join(ids)}")
if [int(i) for i in ids] != list(range(1, len(ids) + 1)):
    print("NOTE: ids are NOT contiguous -- never loop with `seq 1 N`, enumerate the files.")
