#!/usr/bin/env bash
# End-to-end: raw front/top mp4  ->  spine-boned K_final.usd, for ONE species.
#
#   source env.sh
#   deploy/run_species_e2e.sh brook_trout [--phases 012ABCF]
#
# Phases (each skips fish whose output already exists, so this is resumable):
#   0  lowercase symlink species dir                                   free, cpu
#   1  canonical frame: Qwen3-VL picks 1 of 12 candidates + CV crop    GPU (VLM)
#   2  front-view curvature -> measure_fish_size.py -> fish_sizes.json cpu, slow (~10 min/fish)
#   A  Meshy image-to-3D + two-fish mesh cleaner                       *** SPENDS 30 CREDITS/FISH ***
#   B  a2c VLM skeleton (mesh critic + actor + critic) + geometric verifier + REPAIR
#      pass + physics/USD export -> K_final.usd                        GPU (VLM) + Blender
#   C  stage-1 verification report (containment/coverage/alignment)    Blender only
#   F  box bones -> fish-spine geometry                                cpu, needs pxr
#   D  panel hydro (OPTIONAL)                                          needs ISAAC_PYTHON + sibling repo
set -uo pipefail
SP="${1:?usage: run_species_e2e.sh <species> [--phases 012ABCF]}"; shift || true
PHASES="012ABCF"
while [ $# -gt 0 ]; do case "$1" in --phases) PHASES="$2"; shift 2;; *) shift;; esac; done

ROOT="${V2SF_ROOT:?source env.sh first}"
cd "$ROOT"
: "${VLM_PYTHON:?source env.sh first}"
RAW="${RAW_VIDEO_ROOT:-$ROOT/raw_datasets}"
MANIFOLD="${SPECIES_MANIFOLD_ROOT:-$ROOT/species_manifold}"
ASSET="${ASSET_OUT:-$ROOT/dataset}"
BPY="${BLENDER_PY:-}"
if [ -z "$BPY" ]; then   # ask the single source of truth (also checks the Blender version)
  BPY="$("$VLM_PYTHON" -c 'import sys; sys.path.insert(0,"'"$ROOT"'"); import v2sf_paths as P; print(P.blender_py())')" || exit 1
fi
RUN="$ROOT/dataset_run_${SP}"; mkdir -p "$RUN"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RUN/driver.log"; }

log "species=$SP phases=$PHASES"

if [[ $PHASES == *0* ]]; then
  log "phase 0: species dir"
  "$VLM_PYTHON" "$ROOT/deploy/make_species_dir.py" --species "$SP" 2>&1 | tee -a "$RUN/driver.log"
fi

if [[ $PHASES == *1* ]]; then
  log "phase 1: canonical frames (Qwen3-VL select + CV crop)"
  "$VLM_PYTHON" "$ROOT/scripts/select_canonical_frame_qwen.py" \
      --dataset "$RAW/$SP" --species "$SP" \
      --model_path "${QWEN_MODEL_PATH:-$ROOT/models/Qwen3-VL-32B-Instruct}" >>"$RUN/phase1.log" 2>&1
  "$VLM_PYTHON" "$ROOT/scripts/crop_canonical_to_fish_cv.py" \
      --species "$SP" --canonical_dir "$RAW/${SP}_canonical_frames" --video_dir "$RAW/$SP" \
      --view front >>"$RUN/phase1.log" 2>&1
  log "  -> $RAW/${SP}_canonical_frames_cropped/ ($(ls "$RAW/${SP}_canonical_frames_cropped" 2>/dev/null | wc -l) frames)"
fi

if [[ $PHASES == *2* ]]; then
  log "phase 2: front curvature + fish_sizes.json"
  for v in "$RAW/$SP"/front*.mp4; do
    id=$(basename "$v" .mp4); id=${id#front}
    out="$MANIFOLD/$SP/front/fish${id}_curvature.npz"
    [ -f "$out" ] && continue
    "$VLM_PYTHON" "$ROOT/scripts/extract_species_curvature.py" \
      --video "$v" --species "$SP" --fish_id "$id" --view front \
      --sigma 7.0 --thresh 60 --open_px 9 --stride 2 --max_frames 3000 \
      >>"$RUN/phase2.log" 2>&1 || log "  WARN front$id failed"
  done
  "$VLM_PYTHON" "$ROOT/scripts/measure_fish_size.py" --species "$SP" 2>&1 | tee -a "$RUN/driver.log"
fi

if [[ $PHASES == *A* ]]; then
  log "phase A: Meshy image-to-3D  *** SPENDS CREDITS -- one call per fish, never resubmit ***"
  "$VLM_PYTHON" "$ROOT/scripts/run_dataset_v2sf.py" --species "$SP" --phases A \
      --run_dir "$RUN" 2>&1 | tee -a "$RUN/driver.log"
fi

if [[ $PHASES == *B* ]]; then
  log "phase B: a2c VLM skeleton + geometric verifier + repair pass + USD export"
  # the dataset driver reads MESH_OUT/<tag>/mesh.glb, runs ONE shared Qwen load per wave and
  # writes ASSET_OUT/<tag>/K_final.usd (phase A's meshes are picked up automatically)
  "$VLM_PYTHON" "$ROOT/scripts/run_dataset_v2sf.py" --species "$SP" --phases B \
      --a2c_max_iterations 6 --run_dir "$RUN" 2>&1 | tee -a "$RUN/phaseB.log"
fi

if [[ $PHASES == *C* ]]; then
  log "phase C: stage-1 verification (Blender only, report not a gate)"
  "$VLM_PYTHON" "$ROOT/scripts/run_dataset_v2sf.py" --species "$SP" --phases C \
      --run_dir "$RUN" 2>&1 | tee -a "$RUN/driver.log"
fi

if [[ $PHASES == *F* ]]; then
  log "phase F: box bones -> fish spine (shape donor: ${TEMPLATE_FROM:-catfish_fish001} must exist in $ASSET)"
  "$BPY" "$ROOT/scripts/box_bones_to_spine.py" \
      --dataset "$ASSET" --out "${ASSET}_spine" --template_from "${TEMPLATE_FROM:-catfish_fish001}" \
      --link_siblings 2>&1 | tee -a "$RUN/phaseF.log"
fi

if [[ $PHASES == *D* ]]; then
  if [ -z "${ISAAC_PYTHON:-}" ]; then
    log "phase D: SKIPPED (ISAAC_PYTHON unset -- panel hydro needs Isaac Sim)"
  else
    log "phase D: panel hydro (Isaac Sim + sibling swimming_policy_training repo)"
    "$VLM_PYTHON" "$ROOT/scripts/run_dataset_v2sf.py" --species "$SP" --phases D \
        --run_dir "$RUN" 2>&1 | tee -a "$RUN/driver.log"
  fi
fi
log "done. USDs: $ASSET/${SP}_fish*/K_final.usd ; spine: ${ASSET}_spine/"
