"""Single source of truth for every path and interpreter used by the VLM-based articulation
generation half of Video2SwimFish.

Everything is resolved RELATIVE TO THIS FILE by default, and can be overridden by environment
variables (see `env.sh`).  Nothing in the package hard-codes a machine path.

    V2SF_ROOT            package root (default: the directory of this file)
    QWEN_MODEL_PATH      Qwen3-VL-32B-Instruct weights (63 GB)       [ROOT/models/Qwen3-VL-32B-Instruct]
    ACTOR_LORA           fine-tuned actor LoRA adapter dir            [ROOT/checkpoints/actor_lora_v2]
    BLENDER_BIN          Blender executable, EXACTLY 5.0.1            [auto-search, see find_blender]
    BLENDER_PY           Blender's bundled python3.11 (has `pxr`)     [<blender dir>/5.0/python/bin/python3.11]
    VLM_PYTHON           python with torch/transformers/peft          [sys.executable]
    ISAAC_PYTHON         Isaac Sim 5.1 python; ONLY the FEM cook      [unset]
    MESHY_API_KEY_FILE   file that holds the Meshy key                [~/.meshy_api_key]
    ASSET_OUT            final per-fish dataset dir (K_final.usd ...) [ROOT/dataset]
    MESH_OUT             Meshy meshes: MESH_OUT/<tag>/mesh.glb        [ROOT/outputs]
    RAW_VIDEO_ROOT       raw <species>/{front,top}NNN.mp4 videos      [ROOT/raw_datasets]
    SPECIES_MANIFOLD_ROOT  <species>/fish_sizes.json + curvature npz  [ROOT/species_manifold]
    EVAL_OUT             evaluation renders / IoU / sheets           [ROOT/eval_out]
    POLICY_ROOT          sibling `swimming_policy_training` repo      [ROOT/../swimming_policy_training]
                         (only run_dataset_v2sf.py phases D and E use it)

Stdlib only, Python >= 3.8 syntax, so it is safe to import from Blender's bundled python.
Scripts that run under Blender insert ROOT into sys.path themselves (from `__file__`) and then
`import v2sf_paths as P`.
"""
from __future__ import annotations

import functools
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

BLENDER_VERSION = "5.0.1"          # pinned: 4.x and 5.1 change the bpy / USD API this code uses


def _env_path(name: str, default) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else Path(default)


ROOT = _env_path("V2SF_ROOT", Path(__file__).resolve().parent)

# ---- code layout (fixed, relative to ROOT) ---------------------------------------------------
ARTICULATION_DIR = ROOT / "video2swimfish" / "articulation"
FINETUNE_DIR = ROOT / "video2swimfish" / "finetune"
SCRIPTS_DIR = ROOT / "scripts"
PIPE_ROOT = ROOT / "fish_asset_pipeline"
PIPE_SCRIPTS = PIPE_ROOT / "scripts"
PIPE_CONFIGS = PIPE_ROOT / "configs"
USD_PHYSICS_CFG = PIPE_CONFIGS / "usd_physics_defaults.json"
TEMPLATE_INDEX = PIPE_CONFIGS / "skeleton_template_index.json"
SKELETON_DATASET = ROOT / "data" / "fish_skeleton_dataset"      # the 14 hand-built GT fish

# ---- big / external artifacts (NOT in git) ---------------------------------------------------
QWEN_MODEL_PATH = _env_path("QWEN_MODEL_PATH", ROOT / "models" / "Qwen3-VL-32B-Instruct")
ACTOR_LORA = _env_path("ACTOR_LORA", ROOT / "checkpoints" / "actor_lora_v2")

# ---- data / output dirs ----------------------------------------------------------------------
ASSET_OUT = _env_path("ASSET_OUT", ROOT / "dataset")
MESH_OUT = _env_path("MESH_OUT", ROOT / "outputs")
RAW_VIDEO_ROOT = _env_path("RAW_VIDEO_ROOT", ROOT / "raw_datasets")
SPECIES_MANIFOLD_ROOT = _env_path("SPECIES_MANIFOLD_ROOT", ROOT / "species_manifold")
EVAL_OUT = _env_path("EVAL_OUT", ROOT / "eval_out")
POLICY_ROOT = _env_path("POLICY_ROOT", ROOT.parent / "swimming_policy_training")

# ---- interpreters / credentials --------------------------------------------------------------
VLM_PYTHON = os.environ.get("VLM_PYTHON") or sys.executable
ISAAC_PYTHON = os.environ.get("ISAAC_PYTHON") or None            # optional: FEM cook only
MESHY_API_KEY_FILE = _env_path("MESHY_API_KEY_FILE", Path.home() / ".meshy_api_key")


def meshy_key() -> str:
    """Read the Meshy key from MESHY_API_KEY_FILE.  Never print or log the return value."""
    if not MESHY_API_KEY_FILE.exists():
        raise RuntimeError(
            f"Meshy API key file not found: {MESHY_API_KEY_FILE}. Put the key in that file "
            f"(chmod 600) or point MESHY_API_KEY_FILE at it. Never commit it.")
    return MESHY_API_KEY_FILE.read_text().strip()


# ---- Blender ---------------------------------------------------------------------------------
def find_blender() -> Path | None:
    """BLENDER_BIN if set, else the first `blender-5.0.1*/blender` under ~/blender, /opt,
    /usr/local, or $BLENDER_ROOT, else `blender` on PATH.  None if nothing is found."""
    b = os.environ.get("BLENDER_BIN")
    if b:
        return Path(b).expanduser()
    roots = [os.environ.get("BLENDER_ROOT"), str(Path.home() / "blender"), "/opt", "/usr/local"]
    for r in roots:
        if r:
            hits = sorted(glob.glob(os.path.join(r, f"blender-{BLENDER_VERSION}*", "blender")))
            if hits:
                return Path(hits[0])
    w = shutil.which("blender")
    return Path(w) if w else None


@functools.lru_cache(maxsize=1)
def require_blender() -> tuple[Path, Path]:
    """Return (blender_executable, blender_bundled_python), verifying the pinned version.
    Raises RuntimeError with an actionable message otherwise."""
    bb = find_blender()
    if bb is None or not bb.exists():
        raise RuntimeError(
            f"Blender {BLENDER_VERSION} not found. Set BLENDER_BIN=/path/to/blender-{BLENDER_VERSION}*/blender "
            f"(the version is pinned: other versions change the bpy/USD API used here).")
    ver = subprocess.run([str(bb), "--version"], capture_output=True, text=True).stdout
    if BLENDER_VERSION not in ver:
        raise RuntimeError(f"{bb} is not Blender {BLENDER_VERSION} (got: {ver.splitlines()[:1]}).")
    py = os.environ.get("BLENDER_PY")
    if py:
        return bb, Path(py).expanduser()
    got = sorted(glob.glob(str(bb.parent / "5.0" / "python" / "bin" / "python3*")))
    got = [g for g in got if not g.endswith("-config")]
    if not got:
        raise RuntimeError(f"Blender's bundled python not found under {bb.parent}/5.0/python/bin/ "
                           f"(set BLENDER_PY).")
    return bb, Path(got[0])


def blender_bin() -> str:
    return str(require_blender()[0])


def blender_py() -> str:
    return str(require_blender()[1])


def add_to_syspath(*paths) -> None:
    """Idempotently put directories on sys.path (front)."""
    for p in paths:
        p = str(p)
        if p not in sys.path:
            sys.path.insert(0, p)


def summary() -> dict:
    """Human-readable dump used by deploy/preflight.py (never includes the Meshy key)."""
    return {k: str(v) for k, v in dict(
        ROOT=ROOT, QWEN_MODEL_PATH=QWEN_MODEL_PATH, ACTOR_LORA=ACTOR_LORA, ASSET_OUT=ASSET_OUT,
        MESH_OUT=MESH_OUT, EVAL_OUT=EVAL_OUT, RAW_VIDEO_ROOT=RAW_VIDEO_ROOT, SPECIES_MANIFOLD_ROOT=SPECIES_MANIFOLD_ROOT,
        VLM_PYTHON=VLM_PYTHON, ISAAC_PYTHON=ISAAC_PYTHON, BLENDER_BIN=find_blender(),
        MESHY_API_KEY_FILE=MESHY_API_KEY_FILE).items()}


if __name__ == "__main__":
    for k, v in summary().items():
        print(f"{k:24s} {v}")
