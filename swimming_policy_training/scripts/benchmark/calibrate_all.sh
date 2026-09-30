#!/bin/bash
# sequential in-sim calibration for every benchmark fish lacking calibration.npz
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p run_logs/benchmark
PY=${PY:-python}
for tag in $(python3 -c "import json;d=json.load(open('scripts/benchmark/fish_list.json'));print(' '.join(sorted(set(d['traj'])|set(d['freeswim']))))"); do
  if [ -f data/fish_assets/$tag/calibration.npz ]; then echo "[calib] $tag exists"; continue; fi
  echo "[calib] $(date +%T) $tag start"
  WANDB_MODE=disabled timeout 1200 $PY scripts/benchmark/calibrate_fish.py --tag $tag --device cuda:1 --no_fem > run_logs/benchmark/calib_$tag.log 2>&1
  echo "[calib] $(date +%T) $tag exit $? $(grep -E 'DETECTED|Phi built|CALIB_DONE|Error|error' run_logs/benchmark/calib_$tag.log | tail -3 | tr '\n' ' | ')"
done
echo "[calib] ALL DONE"
