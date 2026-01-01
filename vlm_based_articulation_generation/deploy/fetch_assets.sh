#!/usr/bin/env bash
# Download everything the pipeline needs that is not in git, from Hugging Face.
#
#   source env.sh
#   deploy/fetch_assets.sh brook_trout brown_trout [--with-qwen]
#
#   raw videos   video2swimfish/video2swimfish-dataset (public dataset)  -> $RAW_VIDEO_ROOT/<species>/   (~6.5 GB / species)
#   actor LoRA   video2swimfish/actor-lora-v2           (model; access may be restricted)  -> $ACTOR_LORA                  (547 MB)
#   spine donor  data/usd_assets/catfish_fish001/K_final.usd (dataset)   -> $ASSET_OUT/catfish_fish001/   (2.5 MB, phase F)
#   --with-qwen  Qwen/Qwen3-VL-32B-Instruct                              -> $QWEN_MODEL_PATH             (63 GB)
#
# The LoRA repo may require access: run `hf auth login` (a token with read access to the video2swimfish org) first.
# Already-present files are skipped, so the script can be re-run.
set -uo pipefail
ROOT="${V2SF_ROOT:?source env.sh first}"
RAW="${RAW_VIDEO_ROOT:-$ROOT/raw_datasets}"
ASSET="${ASSET_OUT:-$ROOT/dataset}"
LORA="${ACTOR_LORA:-$ROOT/checkpoints/actor_lora_v2}"
QWEN="${QWEN_MODEL_PATH:-$ROOT/models/Qwen3-VL-32B-Instruct}"
HF_DATASET="${HF_DATASET:-video2swimfish/video2swimfish-dataset}"
HF_LORA="${HF_LORA:-video2swimfish/actor-lora-v2}"
command -v hf >/dev/null || { echo "need the 'hf' CLI:  pip install -U huggingface_hub"; exit 1; }

SPECIES=(); WITH_QWEN=0
for a in "$@"; do case "$a" in --with-qwen) WITH_QWEN=1;; *) SPECIES+=("$a");; esac; done
[ ${#SPECIES[@]} -gt 0 ] || { echo "usage: fetch_assets.sh <species>... [--with-qwen]"; exit 1; }

for sp in "${SPECIES[@]}"; do
  if [ -d "$RAW/$sp" ] && [ "$(ls "$RAW/$sp"/*.mp4 2>/dev/null | wc -l)" -ge 2 ]; then
    echo "[fetch] videos for $sp already in $RAW/$sp ($(ls "$RAW/$sp"/*.mp4 | wc -l) files)"; continue
  fi
  echo "[fetch] raw videos: $sp"
  tmp="$ROOT/.hf_tmp"; mkdir -p "$tmp" "$RAW/$sp"
  hf download "$HF_DATASET" --repo-type dataset --include "data/raw_videos/$sp/*" --local-dir "$tmp" >/dev/null || exit 1
  cp -n "$tmp/data/raw_videos/$sp/"*.mp4 "$RAW/$sp/" && echo "  -> $(ls "$RAW/$sp"/*.mp4 | wc -l) videos in $RAW/$sp"
done
rm -rf "$ROOT/.hf_tmp"

if [ -f "$LORA/adapter_model.safetensors" ]; then echo "[fetch] actor LoRA already at $LORA"
else echo "[fetch] actor LoRA -> $LORA"; hf download "$HF_LORA" --local-dir "$LORA" >/dev/null || echo "  FAILED (access restricted? run 'hf auth login')"; fi

DONOR="$ASSET/catfish_fish001/K_final.usd"
if [ -f "$DONOR" ]; then echo "[fetch] spine donor already at $DONOR"
else
  echo "[fetch] spine donor -> $DONOR"
  hf download "$HF_DATASET" --repo-type dataset --include "data/usd_assets/catfish_fish001/K_final.usd" --local-dir "$ROOT/.hf_tmp" >/dev/null \
    && mkdir -p "$ASSET/catfish_fish001" && cp "$ROOT/.hf_tmp/data/usd_assets/catfish_fish001/K_final.usd" "$DONOR"; rm -rf "$ROOT/.hf_tmp"
fi

if [ "$WITH_QWEN" = 1 ]; then
  if [ -f "$QWEN/config.json" ]; then echo "[fetch] Qwen3-VL already at $QWEN"
  else echo "[fetch] Qwen3-VL-32B-Instruct (63 GB) -> $QWEN"; "${VLM_PYTHON:-python}" "$ROOT/scripts/download_qwen3_vl.py" --output "$QWEN"; fi
fi
echo "[fetch] done. Next: deploy/preflight.py, then deploy/run_species_e2e.sh <species> --phases 012"
