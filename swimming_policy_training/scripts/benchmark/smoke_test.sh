#!/bin/bash
# 1-2 epoch smoke tests of every benchmark env/baseline on one calibrated fish
cd "$(dirname "${BASH_SOURCE[0]}")/../.."; mkdir -p run_logs/benchmark; PY=${PY:-python}; TAG=${1:-bluegill_fish015}; DEV=${2:-cuda:0}
export WANDB_MODE=disabled
for kind in Joint PCA CPG; do
  echo "[smoke] $(date +%T) $TAG TrajFollow-$kind start"
  timeout 900 $PY scripts/rl_games/train_ppo.py --task Bench-$TAG-TrajFollow-$kind-v0 --num_envs 64 --seed 1 --headless --max_iterations 2 --device $DEV > run_logs/benchmark/smoke_${TAG}_$kind.log 2>&1
  echo "[smoke] $(date +%T) $kind exit $? | $(grep -E '^\[EVAL\]|^\[BCO-METRICS\] epoch=2|Traceback|Error:' run_logs/benchmark/smoke_${TAG}_$kind.log | tail -2 | cut -c1-260 | tr '\n' ' ')"
done
echo "[smoke] $(date +%T) freeswim blm_il start"
timeout 1200 $PY scripts/benchmark/run_freeswim_bc.py --tag $TAG --baseline blm_il --bc_epochs 3 --n_init 4 --eval_seconds 2 --device $DEV > run_logs/benchmark/smoke_${TAG}_blm_il.log 2>&1
echo "[smoke] $(date +%T) blm_il exit $? | $(grep -E '^\[BC\]|FREESWIM-EVAL|FREESWIM_DONE|Traceback|Error:' run_logs/benchmark/smoke_${TAG}_blm_il.log | tail -3 | cut -c1-260 | tr '\n' ' ')"
echo "[smoke] $(date +%T) freeswim bco_pure start"
timeout 1500 $PY scripts/benchmark/run_freeswim_bc.py --tag $TAG --baseline bco_pure --bc_epochs 3 --n_init 4 --eval_seconds 2 --idm_transitions 5000 --device $DEV > run_logs/benchmark/smoke_${TAG}_bco_pure.log 2>&1
echo "[smoke] $(date +%T) bco_pure exit $? | $(grep -E '^\[BC\]|FREESWIM-EVAL|FREESWIM_DONE|Traceback|Error:' run_logs/benchmark/smoke_${TAG}_bco_pure.log | tail -3 | cut -c1-260 | tr '\n' ' ')"
echo "[smoke] ALL DONE"
