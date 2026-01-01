#!/bin/bash
# 4-PC variant: BLM+IL (first 4 PCs only), fixed PCA, NO-HEAD USD (white_bass_fish008_nohead), same BC settings as blm_il_v6, + mirror fix + obs re-rooting every 5 s (FINAL settings: n_pc 4, mirror, reroot 5 s, hist 5).
cd "$(dirname "${BASH_SOURCE[0]}")"; ROOT="$(cd ../.. && pwd)"
# needs WANDB_API_KEY in the environment (never commit a key)
export WANDB_MODE=${WANDB_MODE:-online} WANDB_DIR=$ROOT/wandb
export WANDB_TAGS=freeswim,white_bass,blm_il,4pc,nohead,mirror,final PYTHONDONTWRITEBYTECODE=1
exec ${PY:-python} run_blm_il_4pc.py --tag white_bass_fish008 --baseline blm_il \
  --n_pc 4 --smooth_win 5 --coef_noise 0.3 --coef_aug 4 --variant _4pc_nohead_final --mirror_traj 1 --reroot_s 5 --hist 5 --task_tag white_bass_fish008_nohead --device ${DEV:-cuda:1} \
  --wandb-name freeswim_white_bass_fish008_blm_il_4pc_nohead_final
