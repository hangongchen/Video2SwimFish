#!/usr/bin/env python
"""Fail fast, before any expensive stage, if this machine cannot run the pipeline.

    python deploy/preflight.py [--species brook_trout,brown_trout] [--fish 41] [--no_meshy]

Checks every dependency the stages actually use, in the order they are needed, and reports the
Meshy credit balance (a free read-only call) so a run is not started that cannot finish.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2sf_paths as P  # noqa: E402

OK, BAD = "  OK  ", " FAIL "
rows, fatal = [], 0


def chk(name, ok, detail=""):
    global fatal
    rows.append(f"[{OK if ok else BAD}] {name:38s} {detail}")
    if not ok:
        fatal += 1
    return ok


ap = argparse.ArgumentParser()
ap.add_argument("--species", default="brook_trout,brown_trout")
ap.add_argument("--fish", type=int, default=41, help="fish count, for the Meshy credit estimate")
ap.add_argument("--no_meshy", action="store_true", help="skip the Meshy balance check")
a = ap.parse_args()

print("=" * 72, "\nVideo2SwimFish (VLM articulation half) preflight\n", "=" * 72, sep="")

chk("V2SF_ROOT", P.ROOT.is_dir(), str(P.ROOT))
chk("  fish_asset_pipeline/scripts", P.PIPE_SCRIPTS.is_dir(), str(P.PIPE_SCRIPTS))
chk("  usd_physics_defaults.json", P.USD_PHYSICS_CFG.is_file())
chk("  14 template .blend files", len(list(P.SKELETON_DATASET.glob("*/*.blend"))) == 14,
    f"{len(list(P.SKELETON_DATASET.glob('*/*.blend')))} found in {P.SKELETON_DATASET}")
chk("QWEN_MODEL_PATH (63 GB)", P.QWEN_MODEL_PATH.is_dir(), str(P.QWEN_MODEL_PATH))
chk("ACTOR_LORA adapter", (P.ACTOR_LORA / "adapter_config.json").is_file()
    and (P.ACTOR_LORA / "adapter_model.safetensors").is_file(), str(P.ACTOR_LORA))

# Blender must be EXACTLY 5.0.1
try:
    bb, bpy = P.require_blender()
    chk("BLENDER_BIN == 5.0.1", True, str(bb))
    chk("  Blender bundled python", Path(bpy).exists(), str(bpy))
    r = subprocess.run([str(bpy), "-c", "import pxr; print(pxr.__file__)"], capture_output=True, text=True)
    chk("  pxr (usd-core) in Blender py", r.returncode == 0, r.stdout.strip() or r.stderr.strip()[:60])
except RuntimeError as e:
    chk("BLENDER_BIN == 5.0.1", False, str(e)[:100])

# VLM env
r = subprocess.run([P.VLM_PYTHON, "-c",
                    "import torch,transformers,peft,scipy,PIL,cv2,skimage,trimesh;"
                    "print(torch.__version__, transformers.__version__, peft.__version__,"
                    "      torch.cuda.is_available(), torch.cuda.device_count())"],
                   capture_output=True, text=True)
chk("VLM python (torch/transformers/peft)", r.returncode == 0, r.stdout.strip() or r.stderr.strip()[-80:])
if r.returncode == 0 and "True" in r.stdout:
    g = subprocess.run([P.VLM_PYTHON, "-c",
                        "import torch;p=torch.cuda.get_device_properties(0);"
                        "print(p.name, round(p.total_memory/2**30), 'GiB')"],
                       capture_output=True, text=True).stdout.strip()
    gib = int(g.split()[-2]) if g else 0
    chk("  GPU >= 64 GiB for the 32B VLM", gib >= 64, g)

chk("ffmpeg / ffprobe", subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0
    and subprocess.run(["which", "ffprobe"], capture_output=True).returncode == 0)

# raw videos
for sp in a.species.split(","):
    d = P.RAW_VIDEO_ROOT / sp
    n = len(list(d.glob("front*.mp4"))) if d.is_dir() else 0
    chk(f"raw videos {sp} (front*.mp4)", n > 0, f"{n} videos in {d}")

# Isaac is OPTIONAL
if P.ISAAC_PYTHON:
    rows.append(f"[{OK}] ISAAC_PYTHON (FEM cook)             {P.ISAAC_PYTHON}")
else:
    rows.append("[ NOTE ] ISAAC_PYTHON unset               skeleton/USD/spine stages still run; "
                "the FEM cook is skipped")

# Meshy: read-only balance call, costs nothing. The key itself is never printed.
if not a.no_meshy:
    try:
        sys.path.insert(0, str(P.SCRIPTS_DIR))
        from meshy_client import MeshyClient  # noqa: E402
        bal = MeshyClient().balance()
        need = a.fish * 30
        chk(f"Meshy credits (need ~{need} for {a.fish} fish)", bal >= need,
            f"balance={bal}" + ("" if bal >= need else f"  -> only {bal // 30} fish"))
    except Exception as e:                                                    # noqa: BLE001
        chk("Meshy API key + balance", False, type(e).__name__ + ": " + str(e)[:90])

print("\n".join(rows))
print("=" * 72)
print(f"{fatal} blocking problem(s)." if fatal else "All checks passed.")
sys.exit(1 if fatal else 0)
