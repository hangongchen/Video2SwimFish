#!/usr/bin/env python
"""Pick ONE canonical frame per fish for Meshy image-to-3D input.

For each <view>NNN.mp4 in --dataset (default view: front, the lateral/side camera
-- Meshy needs a clean full-body-profile image, which is what the "front" cam
gives per this project's own carve_voxel_from_two_views.py convention: imgF ==
front == the (X,Z) height-profile / lateral view):

  1. ffmpeg-extract --n_candidates evenly-spaced frames (cheap: ~12 frames per
     video, not the full clip -- this is NOT the segment_zef_fish.py pipeline,
     no background subtraction, no gt.txt needed).
  2. Show all candidates to Qwen3-VL in ONE call, ask it to pick the index of
     the best canonical frame (full lateral body visible, unoccluded, in
     focus, not touching frame edges, body extended not curled).
  3. Save that single full-resolution frame as the canonical output image.

Usage:
  python scripts/select_canonical_frame_qwen.py --dataset raw_datasets/catfish --species catfish
  python scripts/select_canonical_frame_qwen.py --dataset raw_datasets/lake_sturgeon --species lake_sturgeon
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # package root: v2sf_paths + simfishlib/
import v2sf_paths as P  # noqa: E402

from simfishlib.inference.qwen3vl import Qwen3VLClient  # noqa: E402

DEFAULT_MODEL = str(P.QWEN_MODEL_PATH)

PROMPT = """You are shown {n} candidate video frames (numbered 1 to {n}, in that
order) of the SAME fish, each from a different moment in a swimming video.

Pick the single BEST frame to use as input to an image-to-3D mesh reconstruction
tool (Meshy). The best frame is the one where:
  - the ENTIRE fish body is visible, from nose to tail tip
  - the fish is in a clean LATERAL (side-profile) view, not foreshortened
  - the body is fairly straight/extended, not sharply curled or bent
  - the image is in focus (not motion-blurred)
  - the fish is not touching or cut off by the frame edges
  - the fish is not overlapping/occluded by another fish or object

Respond with ONLY the single number (1-{n}) of the best frame, nothing else."""


def extract_candidates(video: Path, n: int, out_dir: Path) -> list[Path]:
    # key=value output (NOT positional csv=p=0 -- ffprobe reorders csv fields to its own
    # canonical order regardless of the order requested in -show_entries, e.g. it prints
    # r_frame_rate,duration,nb_frames even though we ask for nb_frames,r_frame_rate,duration;
    # parsing that positionally silently reads nb_frames as duration).
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True, check=True).stdout.strip()
    duration = float(out) if out and out != "N/A" else None
    if duration is None:
        duration = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, check=True).stdout.strip())

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    margin = duration * 0.05          # skip the very start/end (camera settling, fish entering frame)
    span = duration - 2 * margin
    for i in range(n):
        t = margin + span * (i + 0.5) / n
        out_path = out_dir / f"cand_{i + 1:02d}.jpg"
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
             "-q:v", "2", str(out_path)],
            capture_output=True, text=True, check=True)
        if not out_path.exists():
            raise RuntimeError(f"ffmpeg exited 0 but wrote no frame at t={t:.3f}s "
                               f"(video duration={duration:.2f}s): {video}\nstderr:\n{r.stderr[-800:]}")
        paths.append(out_path)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dir with <view>NNN.mp4 files")
    ap.add_argument("--species", required=True, help="used only in output filenames")
    ap.add_argument("--view", default="front")
    ap.add_argument("--n_candidates", type=int, default=12)
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None, help="default: <dataset>/../<species>_canonical_frames")
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    out_dir = Path(args.out) if args.out else dataset.parent / f"{args.species}_canonical_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(dataset.glob(f"{args.view}*.mp4"))
    assert videos, f"no {args.view}*.mp4 found in {dataset}"
    print(f"[select_canonical_frame] {len(videos)} '{args.view}' videos found in {dataset}", flush=True)

    print(f"[select_canonical_frame] loading Qwen3-VL from {args.model_path} ...", flush=True)
    client = Qwen3VLClient(model_path=args.model_path)
    client.load()
    print("[select_canonical_frame] model loaded", flush=True)

    results = []
    for video in videos:
        m = re.search(r"(\d+)", video.stem)
        fish_id = m.group(1) if m else video.stem
        with tempfile.TemporaryDirectory() as tmp:
            cand_paths = extract_candidates(video, args.n_candidates, Path(tmp))
            prompt = PROMPT.format(n=len(cand_paths))
            reply = client.generate_from_images(cand_paths, prompt, max_new_tokens=16)
            m2 = re.search(r"\d+", reply)
            idx = int(m2.group()) if m2 else 1
            idx = max(1, min(idx, len(cand_paths)))
            chosen = cand_paths[idx - 1]
            out_path = out_dir / f"{args.species}_fish{fish_id}_canonical.jpg"
            shutil.copy(chosen, out_path)
        print(f"[select_canonical_frame] {video.name}: Qwen picked candidate {idx}/{len(cand_paths)} "
              f"(raw reply={reply.strip()!r}) -> {out_path}", flush=True)
        results.append({"fish_id": fish_id, "video": str(video), "chosen_index": idx,
                        "raw_reply": reply.strip(), "out_path": str(out_path)})

    print(f"[select_canonical_frame] wrote {len(results)} canonical frames to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
