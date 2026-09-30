#!/usr/bin/env python
"""VLM CRITIC for the Video2SwimFish pipeline: renders the generated mesh and asks Qwen3-VL
to visually inspect it for defects before the asset is accepted as done.

Why this exists: a real failure slipped through unnoticed -- catfish_fish001's Meshy
reconstruction has a visible dark HOLE in the body skin (a non-watertight-mesh artifact,
confirmed by rendering the skin mesh ALONE with every skeleton/bone object hidden: the hole
is still there, so it is not a bone poking through skin, it is Meshy's own geometry). Nothing
in the pipeline checked the finished mesh's visual quality before calling it "DONE" -- this
script is that check: a generator (the deterministic Meshy + fish_asset_pipeline steps,
the "actor") produces the mesh, this VLM step (the "critic") inspects renders of it and
either passes or rejects it, the same self-verification pattern already used for the
canonical-frame crop step (crop_canonical_to_fish_qwen.py's crop-then-verify loop).

Usage:
  python verify_mesh_vlm.py --glb mesh.glb --out_json verify.json
  python verify_mesh_vlm.py --usd fish_articulated.usda --mesh_name Mesh_0 --out_json verify.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # package root: v2sf_paths + simfishlib/
import v2sf_paths as P  # noqa: E402
from simfishlib.inference.qwen3vl import Qwen3VLClient  # noqa: E402

DEFAULT_MODEL = str(P.QWEN_MODEL_PATH)
SCRIPT_DIR = Path(__file__).resolve().parent

PROMPT = """You are a 3D-asset quality reviewer. These are three views (perspective, top,
side) of ONE reconstructed fish mesh, rendered solid-shaded (no lighting tricks).

Look carefully for FABRICATION DEFECTS -- NOT for anatomical realism or texture quality.
Specifically check for:
  1. Any HOLE or GAP in the body surface -- a dark patch, slot, or gash where you can see
     into the inside of the model or through to the background, where the surrounding body
     is otherwise a solid closed skin.
  2. Any RIGID PART (a bone, a joint, an internal structure) visibly POKING OUT THROUGH the
     outer body surface where it should be hidden inside.
  3. Any obviously broken/inverted/missing geometry -- a chunk of the body that looks
     inside-out, collapsed, or is simply absent.
  4. MORE THAN ONE FISH: the mesh must be exactly ONE fish body. If you see a second fish
     (or a large second body/blob that is not part of the same fish) anywhere in the views,
     that is a defect -> FAIL. (The source photo sometimes contained two fish.)

Do NOT flag: normal body seams, fin edges, scale texture patterns, the mouth/gill opening,
or minor asymmetry -- those are normal fish anatomy, not defects.

Step 1: describe in one sentence what you see overall.
Step 2: state whether you see any of the 4 defect types above, and exactly where (e.g.
"a dark rectangular hole on the back, just behind the head").

End with a final line in EXACTLY this format:
VERDICT: PASS or VERDICT: FAIL"""


def render_views(out_prefix: str, glb: str | None = None, usd: str | None = None,
                  mesh_name: str | None = None) -> list[str]:
    cmd = [P.blender_bin(), "-b", "--python", str(SCRIPT_DIR / "render_mesh_views.py"), "--",
           "--out_prefix", out_prefix]
    if glb:
        cmd += ["--mesh", glb]
    else:
        cmd += ["--usd", usd]
        if mesh_name:
            cmd += ["--mesh_name", mesh_name]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"render_mesh_views.py failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    views = [f"{out_prefix}_{tag}.png" for tag in ("persp", "top", "side")]
    for v in views:
        assert Path(v).exists(), f"expected render {v} was not produced"
    return views


def parse_verdict(reply: str) -> bool:
    """Returns True if PASS."""
    m = re.search(r"VERDICT:\s*(PASS|FAIL)", reply, re.IGNORECASE)
    if not m:
        return False  # unparseable reply -> fail closed, don't silently accept
    return m.group(1).upper() == "PASS"


def verify(glb: str | None = None, usd: str | None = None, mesh_name: str | None = None,
           model_path: str = DEFAULT_MODEL, client: Qwen3VLClient | None = None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = str(Path(tmp) / "view")
        views = render_views(out_prefix, glb=glb, usd=usd, mesh_name=mesh_name)
        owns_client = client is None
        if client is None:
            client = Qwen3VLClient(model_path=model_path)
            client.load()
        reply = client.generate_from_images(views, PROMPT, max_new_tokens=400, temperature=0.0)
        if owns_client:
            del client
    passed = parse_verdict(reply)
    return {"source": glb or usd, "passed": passed, "raw_reply": reply.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", default=None)
    ap.add_argument("--usd", default=None)
    ap.add_argument("--mesh_name", default=None, help="for --usd: hide every other mesh object")
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    assert args.glb or args.usd, "need --glb or --usd"

    result = verify(glb=args.glb, usd=args.usd, mesh_name=args.mesh_name, model_path=args.model_path)
    print(json.dumps(result, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2))
        print(f"[verify_mesh_vlm] wrote {args.out_json}", flush=True)
    if not result["passed"]:
        print("[verify_mesh_vlm] VERDICT: FAIL", file=sys.stderr)
        sys.exit(1)
    print("[verify_mesh_vlm] VERDICT: PASS", flush=True)


if __name__ == "__main__":
    main()
