#!/usr/bin/env bash
# Example environment for the VLM-based articulation generation half of Video2SwimFish.
#   source env.sh            (edit the values marked EDIT for your machine)
#
# Every variable is OPTIONAL except the two interpreters (BLENDER_BIN is auto-searched, see
# v2sf_paths.py:find_blender). Unset variables fall back to paths relative to this directory.

export V2SF_ROOT="${V2SF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# --- models (NOT in git) ------------------------------------------------------------------
# Qwen3-VL-32B-Instruct, 63 GB:  python scripts/download_qwen3_vl.py --output "$V2SF_ROOT/models/Qwen3-VL-32B-Instruct"
export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-$V2SF_ROOT/models/Qwen3-VL-32B-Instruct}"
# fine-tuned actor LoRA adapter dir (adapter_config.json + adapter_model.safetensors); see README
export ACTOR_LORA="${ACTOR_LORA:-$V2SF_ROOT/checkpoints/actor_lora_v2}"

# --- interpreters -------------------------------------------------------------------------
# Blender is PINNED to EXACTLY 5.0.1 (its bundled python3.11 is the only interpreter with `pxr`).
# EDIT:  export BLENDER_BIN=/opt/blender-5.0.1-stable+v50.a3db93c5b259-linux.x86_64-release/blender
# export BLENDER_BIN=...
# export BLENDER_PY=...        # default: <blender dir>/5.0/python/bin/python3.11
# python env with torch / transformers / peft / scipy / opencv (requirements.txt):
export VLM_PYTHON="${VLM_PYTHON:-$(command -v python)}"     # EDIT to e.g. /path/to/conda/envs/v2sf/bin/python
# OPTIONAL. Only the FEM deformable cook needs Isaac Sim 5.1's python; without it K_final.usd
# is still a valid articulation (bones + D6 joints + attachments + mass) but the skin is not cooked.
# export ISAAC_PYTHON=/path/to/env_isaaclab/bin/python

# --- credentials --------------------------------------------------------------------------
# Meshy image-to-3D key lives in a FILE (chmod 600). Never commit it; never put it on a command line.
export MESHY_API_KEY_FILE="${MESHY_API_KEY_FILE:-$HOME/.meshy_api_key}"

# --- data / outputs (defaults are under $V2SF_ROOT; override to put big data elsewhere) ----
# export RAW_VIDEO_ROOT=/data/raw_datasets            # <species>/{front,top}NNN.mp4
# export ASSET_OUT=/data/v2sf/dataset                 # <tag>/K_final.usd, stage1_report.json ...
# export MESH_OUT=/data/v2sf/outputs                  # <tag>/mesh.glb (Meshy output)
# export SPECIES_MANIFOLD_ROOT=/data/v2sf/species_manifold   # <species>/fish_sizes.json
# export POLICY_ROOT=/path/to/swimming_policy_training       # only for run_dataset_v2sf.py phases D, E

_ok(){ [ -e "$1" ] && echo OK || echo MISSING; }
echo "V2SF_ROOT=$V2SF_ROOT"
echo "QWEN_MODEL_PATH=$QWEN_MODEL_PATH   $(_ok "$QWEN_MODEL_PATH")"
echo "ACTOR_LORA=$ACTOR_LORA             $(_ok "$ACTOR_LORA")"
echo "BLENDER_BIN=${BLENDER_BIN:-<auto-search>}"
echo "VLM_PYTHON=$VLM_PYTHON"
echo "ISAAC_PYTHON=${ISAAC_PYTHON:-<unset: FEM cook will be skipped>}"
echo "MESHY_API_KEY_FILE=$MESHY_API_KEY_FILE   $(_ok "$MESHY_API_KEY_FILE")"
