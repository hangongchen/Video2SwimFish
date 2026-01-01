# Swimming Policy Training (Video2SwimFish benchmark)

This half of the repository trains, evaluates and benchmarks **swimming policies** for the controllable
fish assets that the [VLM articulation pipeline](../vlm_based_articulation_generation/README.md) produces
(`K_final.usd`, one per real fish). It contains

* the Isaac Lab task family `Bench-<fish_tag>-<Kind>-v0` (one set of tasks per fish),
* **all baseline training scripts** of the paper (Task 1, Task 2 and the ROV case study),
* the video → curvature → **BLM (PCA)** extraction, the per-fish calibration, the panel-hydro precompute,
* the job scheduler with the paper's stopping rules, the metrics (Fréchet, W1(κ), frequency, speed) and evaluation scripts,
* the assets of the **9 benchmark fish** (about 60 MB) so that everything can be run right after cloning.

> Everything below was executed on: 2 × RTX PRO 6000 Blackwell (96 GB), Isaac Sim 5.1.0 (pip), Isaac Lab 0.54.x,
> Python 3.11, torch 2.7.0+cu128, rl_games 1.6.1. Multi-GPU (NCCL) is broken on that box: run **one process per GPU**.

---------------------------------------------------------------------------------------------------

## 1. Method ↔ code map (paper Section 4 / Appendix)

| Paper name | Task | Task id (`<tag>` = e.g. `catfish_fish002`) | Env class (`source/FISH/FISH/tasks/direct/fish/`) | PPO yaml (`agents/`) |
|---|---|---|---|---|
| **BLM + RL (ours)** | Task 1 | `Bench-<tag>-TrajFollow-PCA-v0` | `salmon_swim_traj_follow_env.py:SalmonSwimPCATrajFollowEnv` | `rl_games_ppo_traj_follow_pca_cfg.yaml` |
| Joint RL | Task 1 | `Bench-<tag>-TrajFollow-Joint-v0` | `SalmonSwimJointTrajFollowEnv` | `rl_games_ppo_traj_follow_joint_cfg.yaml` |
| Joint RL + AMP | Task 1 | `Bench-<tag>-TrajFollow-Joint-AMP-v0` | `SalmonSwimJointAMPTrajFollowEnv` (Joint env + AMP discriminator) | `rl_games_ppo_traj_follow_joint_cfg.yaml` |
| CPG + RL | Task 1 | `Bench-<tag>-TrajFollow-CPG-v0` | `SalmonSwimCPGTrajFollowEnv` | `rl_games_ppo_traj_follow_pca_cfg.yaml` |
| BCO + RL | Task 1 | `Bench-<tag>-TrajFollow-Joint-v0` **+ `--checkpoint <bco bc_policy.pt>`** | same as Joint RL (PPO warm-started from the BCO policy) | joint yaml |
| **BLM + IL** | Task 2 | `Bench-<tag>-FreeSwim-VideoState-v0` (eval env) | `salmon_swim_pca_videostate_env.py`, trained by `scripts/benchmark/run_freeswim_bc.py --baseline blm_il` | `rl_games_ppo_pca_videostate_cfg.yaml` |
| BCO (pure) | Task 2 | `Bench-<tag>-TrajFollow-Joint-v0` (random rollouts for the IDM) + `FreeSwim-VideoState` (eval) | `scripts/benchmark/run_freeswim_bc.py --baseline bco_pure` + `bco/` package | – |
| ROV basket capture | Case study | `Bench-<tag>-ROVCatch-v0` | `salmon_rov_catch_bench_env.py:BenchROVCatchEnv` → `salmon_rov_chase_env.py` | `rl_games_ppo_rov_chase_cfg.yaml` |
| (probe) skeleton calibration | – | `Bench-<tag>-Calib-v0` | `salmon_swim_env.py:SalmonSwimEnv` | `rl_games_ppo_cfg.yaml` |

How a method is selected: **by task id** (separate env + cfg class per method). All Task-1 envs share
`TrajFollowMixin` (`salmon_swim_traj_follow_env.py`), so the reward, reference path, phase variable, completion test
and metrics are *identical* for every baseline — only the action space differs:

| Action space | What the policy outputs | Where |
|---|---|---|
| **BLM** (`PCA`) | `â ∈ [-1,1]^d`; coefficients updated `a ← clip(a + â ⊙ Δa_max, −a_max, a_max)`; target curvature `μ + V a` → joint targets by the per-fish curvature→joint decoder | `salmon_swim_pca_env.py` (`_build_pca_mapping`, `_pre_physics_step`) |
| Joint | one PD position target per lateral joint (`a ≥ 0 → a·hi`, `a < 0 → a·(−lo)`) | `SalmonSwimJointTrajFollowEnv` |
| CPG | `[ΔA, Δf, Δb]` of a travelling-wave generator fitted to the fish's own curvature (`cpg_params.npz`), decoded by the **same** decoder as BLM | `SalmonSwimCPGTrajFollowEnv` |

`a_max` is the 99-th percentile of `|a|` and `Δa_max` the 99-th percentile of the per-control-step change
`|a_t − a_{t+1}|` of the fish's own video coefficients (the code takes a 2-frame gap of the 60 fps coefficients =
one 1/30 s control step, hence **the PCA coefficients must be stored at 60 fps**, see §4).

**Task 1 protocol** (`salmon_swim_traj_follow_cfg.py`, `TrajFollowMixin._get_rewards`): each episode plants a random 5 s
window of the fish's own real path at the simulated start pose. Phase `s_t` = arc length of the path point nearest to the
fish; look-ahead point 0.3 BL ahead (K = 5 look-ahead points are in the observation).
Reward = `1.0·exp(−2·dist_BL) + 0.3·cos(θ_fish − θ_path) + 0.1·(−|v − v_ref|) + 0.01·(−0.01·‖q_target‖²)`.
Completion = phase reached the path end **and** the fish is within 0.5 BL of the path end; a blown-up episode counts as *not* completed.

---------------------------------------------------------------------------------------------------

## 2. Repository layout

```
swimming_policy_training/
├── source/FISH/                       Isaac Lab extension (pip install -e source/FISH)
│   └── FISH/tasks/direct/fish/
│       ├── __init__.py                 registers the 72 Bench-* task ids  (GENERATED by gen_fish_cfgs.py)
│       ├── salmon_benchmark_fish_cfgs.py   per-fish cfg classes            (GENERATED)
│       ├── salmon_swim_env.py          base env: zero-g "water", PD joint drives, FEM cook/bind, panel hydro wrench,
│       │                               blow-up guards, bone-collider fix          (1.7k lines, the physics core)
│       ├── salmon_swim_pca_env.py      BLM/PCA controller (+ analytic Φ)
│       ├── salmon_swim_traj_follow_{env,cfg}.py   Task 1 (Joint / PCA / CPG / Joint+AMP)
│       ├── salmon_swim_pca_videostate_{env,cfg}.py Task 2 eval env (25-dim video-state observation)
│       ├── salmon_swim_reach10_{env,cfg}.py, salmon_rov_{chase,catch_bench}_env.py   ROV case study
│       ├── salmon_amp_tank_env.py, salmon_swim_amp_env.py, salmon_*_cfg.py           base classes / AMP machinery
│       ├── biofidelity_metrics.py      W1(κ), dominant frequency, speed, phase monotonicity
│       └── agents/                     rl_games PPO yamls, ROV asset (bravo_rov/), underwater background
├── scripts/
│   ├── rl_games/train_ppo.py           THE training entry point for every RL baseline (+ play.py)
│   ├── benchmark/
│   │   ├── gen_fish_cfgs.py            fish_list.json → cfg classes + gym registrations   (--kp --jlim --pca_modes)
│   │   ├── fish_list.json              which fish are in the benchmark
│   │   ├── calibrate_fish.py / calibrate_all.sh   in-simulation κ→joint calibration (Φ)
│   │   ├── analytic_phi.py             geometric Φ actually used by every Bench cfg (pca_phi_mode="analytic")
│   │   ├── calibrate_cpg_from_fish.py  CPG parameters fitted to the fish's own curvature
│   │   ├── build_amp_ref_from_fish.py  46-dim AMP reference features from the fish's own video
│   │   ├── run_freeswim_bc.py          Task 2: BLM+IL and BCO (pure) — dataset, BC, eval
│   │   ├── build_queue.py, scheduler.py   job queue + scheduler with the paper's stop rules
│   │   ├── export_fish_tables.py       per-fish table from the logs; render_benchmark_video.py; smoke_test.sh
│   │   └── validate_extraction.py, travelling_wave_report.py, pca_degeneracy_report.py, sine_wave_test.py   sanity probes
│   ├── zef_manifold/                   video → midline → curvature → PCA (the BLM)  [extract_species_curvature, fit_pca_species, extract_species_reference, fit_pca_species_pooled]
│   ├── zef_playback/curvature_utils.py shared 20-station curvature estimator (sim and video use the same code)
│   ├── precompute_catfish001_panels.py per-fish panel-hydro proxy (name is historical; works for any fish, see §4)
│   ├── blm_il_4pc/                     final 4-PC BLM+IL variant + its renderer + launcher
│   ├── eval/                           trajectory-fidelity / biofidelity evaluation scripts
│   └── extract_amp_features.py         AMP feature extractor
├── bco/                                BCO: inverse-dynamics model, pseudo-labeler, BC trainer, replay buffer
├── data/
│   ├── fish_assets/<tag>/              K_final.usd, panel_hydro.npz, calibration.npz, cpg_params.npz, amp_reference.npz
│   ├── species_manifold/<sp>_fix/top/  fish<NNN>_{curvature,pca_basis,reference}.npz  (the BLM of each fish)
│   ├── species_manifold/<sp>/fish_sizes.json   metric body length of each fish
│   └── zef_target_angles.npz           empirical target bearings (ROV / reach tasks)
└── checkpoints/benchmark/blm_il/<tag>/ released BLM+IL policies (the ROV prey policies) for 4 fish
```

Not shipped: the 39 GB of raw videos and the 63 GB VLM (see the VLM README), trained RL checkpoints, wandb logs.

---------------------------------------------------------------------------------------------------

## 3. Installation

```bash
# 1. Isaac Sim 5.1 (pip) + Isaac Lab, python 3.11  (follow the official Isaac Lab pip installation guide)
conda create -n v2sf_policy python=3.11 && conda activate v2sf_policy
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/isaac-sim/IsaacLab && (cd IsaacLab && ./isaaclab.sh --install none)   # or pip install -e each source/isaaclab*
pip install -r requirements.txt

# 2. this extension (editable)
pip install -e source/FISH
```

**Important pitfalls**

* Use the env's python **directly** (`$CONDA_PREFIX/bin/python scripts/...`), not `isaaclab.sh -p` (`isaaclab.sh` may be bound to a different python than your env).
* If another editable `FISH` package is already installed in the env (`pip list | grep -i fish`), it **shadows** this
  one for `import FISH`. Run `pip install -e source/FISH` again (or `pip uninstall fish`) first; check with
  `python -c "import FISH; print(FISH.__file__)"` after launching Kit, and read the line
  `Parsing configuration from: …/agents/…yaml` in the training log — it must point into this repository.
* `train_ppo.py` imports `distutils.util.strtobool` → Python ≤ 3.11.
* Every launch is meant to be **tracked in wandb**: `--track` with `WANDB_MODE=online` and `WANDB_API_KEY` in the environment
  (optionally `WANDB_ENTITY`, `WANDB_PROJECT`). Never `wandb login` inside a job. For a quick test use `WANDB_MODE=disabled`.
* `simulation_app.close()` hangs in Kit teardown on Blackwell + FEM. Scripts end with `os._exit`/are killed by the scheduler;
  if a run lingers, kill it **by PID**, never `pkill -f <pattern>`.
* FEM soft-body dynamics are **not** invariant to the number of environments: train and evaluate with the same `--num_envs` (256 in the paper).

---------------------------------------------------------------------------------------------------

## 4. Data: what each fish needs, and how it is produced

The env cfgs read (see `scripts/benchmark/gen_fish_cfgs.py`, which prints the exact paths):

| File | Produced by | Notes |
|---|---|---|
| `data/fish_assets/<tag>/K_final.usd` | VLM articulation pipeline (`../vlm_based_articulation_generation`) | copy / symlink it here |
| `data/fish_assets/<tag>/panel_hydro.npz` | `scripts/precompute_catfish001_panels.py` | needs Isaac; voxel = BL/30; see below |
| `data/fish_assets/<tag>/calibration.npz` | `scripts/benchmark/calibrate_fish.py --tag <tag> --no_fem` | joint order, axis sign, head sign (Φ itself is analytic) |
| `data/fish_assets/<tag>/cpg_params.npz` | `scripts/benchmark/calibrate_cpg_from_fish.py --tag <tag>` | CPG baseline only |
| `data/fish_assets/<tag>/amp_reference.npz` | `scripts/benchmark/build_amp_ref_from_fish.py --tag <tag>` | Joint+AMP baseline only |
| `data/species_manifold/<sp>_fix/top/fish<NNN>_curvature.npz` | `scripts/zef_manifold/extract_species_curvature.py` | from the **top-view** video |
| `…/fish<NNN>_pca_basis.npz` | `scripts/zef_manifold/fit_pca_species.py` | the BLM (mean, components, coeffs @ 60 fps) |
| `…/fish<NNN>_reference.npz` | `scripts/zef_manifold/extract_species_reference.py` | real path `p_bl`, heading `psi` |
| `data/species_manifold/<sp>/fish_sizes.json` | `../vlm_based_articulation_generation/scripts/measure_fish_size.py` | body length in metres (**read at import time** by the generated cfgs) |

Fish tag = `<species>_fish<NNN>`; a suffix (e.g. `white_bass_fish008_nohead`) is a *variant asset of the same real fish*
(re-uses its video data). The 9 tags shipped in `data/` are exactly the ones in `scripts/benchmark/fish_list.json`.

Commands to build the per-fish data for a NEW fish (all with the env's python, from the repo root):

```bash
PY=$CONDA_PREFIX/bin/python; SP=catfish; NUM=002; TAG=${SP}_fish${NUM}
# (a) BLM from the top-view video (CPU). Body-length-relative kernels + contrast threshold + per-frame head/tail (see pitfalls)
$PY scripts/zef_manifold/extract_species_curvature.py --video /path/top${NUM}.mp4 --species ${SP}_fix --fish_id ${NUM} --view top \
    --sigma_bl 0.015 --open_bl 0.019 --ratio_thresh 0.6 --open_px 9 --stride 2 --max_frames 3000
$PY scripts/zef_manifold/fit_pca_species.py --species ${SP}_fix --fish_id ${NUM} --view top --max_kbl 4 --resample_fps 60
$PY scripts/zef_manifold/extract_species_reference.py --species ${SP}_fix --view top
#     (fish with < 100 clean frames use a species-pooled basis:  scripts/zef_manifold/fit_pca_species_pooled.py --auto)
# (b) panel hydro from the asset (Isaac). voxel = body_length / 30
FISH_GEN_DIR=$PWD/data/fish_assets/$TAG FISH_MESH_TOKEN=Mesh_0 $PY scripts/precompute_catfish001_panels.py --voxel=0.0103
# (c) calibration (Isaac, skeleton only) + baseline-specific references
$PY scripts/benchmark/calibrate_fish.py --tag $TAG --no_fem
$PY scripts/benchmark/calibrate_cpg_from_fish.py --tag $TAG
$PY scripts/benchmark/build_amp_ref_from_fish.py --tag $TAG
# (d) add the tag to scripts/benchmark/fish_list.json, then regenerate cfgs + gym ids
$PY scripts/benchmark/gen_fish_cfgs.py            # defaults = paper preset: --kp 120 --jlim 45
```

The VLM pipeline driver (`run_dataset_v2sf.py`, phases D/E) runs steps (a) and (b) for whole species automatically.

---------------------------------------------------------------------------------------------------

## 5. Simulation and actuator preset (what the paper calls "the same physics for every fish")

| Item | Value | Where |
|---|---|---|
| Physics step / control | 1/120 s, decimation 4 → 30 Hz control | `SalmonSwimPCACfg.__post_init__` |
| Gravity / buoyancy | zero gravity ("neutrally buoyant water") | `salmon_swim_cfg.py` |
| Joints | 3-DoF D6 per bone pair; **PD position drives authored at load time** (`_prepare_env_assets`, before cloning); limit = `joint_limit_deg`; kp = `KP`, kd = `SWIM_DAMPING = 6` | `salmon_swim_env.py`, generated cfgs |
| **Paper preset** | **kp = 120, limit ±45°** (`gen_fish_cfgs.py` defaults) | `KP`, `JLIM` in `gen_fish_cfgs.py` |
| FEM skin | Young 1e5 Pa, ν 0.45, elasticity damping 0.05 (material prim is *created and bound* at load — the USD ships none) | `salmon_swim_env.py` |
| Hydrodynamics | quasi-steady resistive **panel model** on the skin: per-bone panels from `panel_hydro.npz`, normal drag Cd⊥ = 1.0, tangential Cd∥ = 0.01, applied as an external wrench every physics step | `SalmonSwimEnv._panel_hydro_wrench`, `salmon_swim_amp_misty_panels_cfg.py` |
| Blow-up guard | episode ends if any joint speed > 200 rad/s or FEM nodal-velocity limit exceeded; counts toward `train/blowup_rate`, never as completion | `salmon_swim_env.py` |
| Bone colliders | **disabled** (`keep_bone_colliders = False`). The startup log must contain `disabled 29 colliders` (not `0`) | generated cfgs |
| PPO | rl_games `a2c_continuous`, MLP [128, 64] ELU, log-σ init −0.5 (fixed), γ 0.99, λ 0.95, lr 3e-4 KL-adaptive (0.008), clip 0.2, entropy 0.003, horizon 256, 5 mini-epochs, minibatch 32/env, obs+value normalisation, 256 envs | `agents/*.yaml` |

---------------------------------------------------------------------------------------------------

## 6. Running the baselines

All commands: repo root, env python, `export WANDB_MODE=online WANDB_API_KEY=… [WANDB_ENTITY=…]` (or `WANDB_MODE=disabled` for a test).
`--device cuda:N` selects the GPU. `T=catfish_fish002` below.

```bash
PY=$CONDA_PREFIX/bin/python; T=catfish_fish002
COMMON="--num_envs 256 --seed 42 --headless --track --wandb-project-name video2swimfish_benchmark"

# ---- Task 1: trajectory following -------------------------------------------------------------
$PY scripts/rl_games/train_ppo.py --task Bench-$T-TrajFollow-PCA-v0       $COMMON --wandb-name trajfollow_${T}_blm_rl     # BLM + RL  (ours)
$PY scripts/rl_games/train_ppo.py --task Bench-$T-TrajFollow-Joint-v0     $COMMON --wandb-name trajfollow_${T}_joint_rl   # Joint RL
$PY scripts/rl_games/train_ppo.py --task Bench-$T-TrajFollow-Joint-AMP-v0 $COMMON --wandb-name trajfollow_${T}_amp_rl     # Joint RL + AMP
$PY scripts/rl_games/train_ppo.py --task Bench-$T-TrajFollow-CPG-v0       $COMMON --wandb-name trajfollow_${T}_cpg_rl     # CPG + RL

# BCO + RL: (1) behaviour-clone a joint-space policy from the video with an inverse-dynamics model, (2) PPO from it
$PY scripts/benchmark/run_freeswim_bc.py --tag $T --baseline bco_pure --skip_eval      # -> checkpoints/benchmark/bco/$T/bc_policy.pt
$PY scripts/rl_games/train_ppo.py --task Bench-$T-TrajFollow-Joint-v0 $COMMON --wandb-name trajfollow_${T}_bco_rl \
      --checkpoint checkpoints/benchmark/bco/$T/bc_policy.pt

# ---- Task 2: free swimming from video (no reward) ----------------------------------------------
$PY scripts/benchmark/run_freeswim_bc.py --tag $T --baseline blm_il  --n_modes 4 --smooth_win 5 --hist 1 --coef_noise 0.3 --coef_aug 4
$PY scripts/benchmark/run_freeswim_bc.py --tag $T --baseline bco_pure                   # BCO: random rollouts → IDM → pseudo-labels → BC → eval
#   outputs: checkpoints/benchmark/{blm_il,bco}<variant>/$T/{bc_policy.pt,freeswim_eval.json,…}; 20 real initial states, 5 s roll-outs
#   final 4-PC / history-5 variant used for the released white-bass policy:  bash scripts/blm_il_4pc/run_train_final.sh

# ---- Case study: ROV captures the fish (prey = the fish's own BLM+IL policy) -----------------------
#   needs checkpoints/benchmark/blm_il/$T/bc_policy.pt   (released for catfish_fish002, lake_sturgeon_fish016, bluegill_fish015, white_bass_fish008)
$PY scripts/rl_games/train_ppo.py --task Bench-$T-ROVCatch-v0 $COMMON --wandb-name rovcatch_$T agent.params.config.save_frequency=5
```

Whole benchmark with the paper's protocol (2 jobs per GPU, stop rules, retry, stall-resume):

```bash
python scripts/benchmark/build_queue.py --day 1 --out jobs.json               # day1 joint_rl+blm_rl, --day 2 cpg_rl+amp_rl, --day 3 bco_rl; add --freeswim for Task 2
python scripts/benchmark/scheduler.py --queue queue.json --add jobs.json      # (or first run: --queue queue.json --slots 2 --gpus 0,1)
python scripts/benchmark/scheduler.py --queue queue.json --slots 2 --gpus 0,1
python scripts/benchmark/export_fish_tables.py                                # best-completion epoch per fish from the logs
```

Stopping rules implemented in `scheduler.py::Tracker.stop_reason` (see §8 for the one difference from the paper text):
converged = completion ≥ 0.90 for 3 consecutive epochs (ROV: success ≥ 0.85); blow-up = `train/blowup_rate` > 0.30 for 3 epochs after a
10-epoch warm-up → run marked failed and retried once with seed + 1000; stall = no new epoch for 40 min (PhysX device hang) → resumed from the newest
checkpoint; plateau: see §8. Checkpoints go to `logs/rl_games/<name>/<timestamp>/nn/` (`save_frequency: 10`); the reported number should be the
**best-completion** epoch (`BEST=` in the scheduler log), not the last one (completion decays after its peak).

Evaluation logged every epoch on 16 tracked environments (`[EVAL] epoch=…` lines and wandb keys): `eval/completion_rate`,
`frechet_dist_bl`, `wasserstein_curvature` (+ per station), `dominant_freq_hz` (+ error), `swim_speed_bl_s` (+ error), `phase_monotonicity`, `train/blowup_rate`.
Post-hoc scripts: `scripts/eval/*` (trajectory fidelity, biofidelity suite, video-state rollouts), `scripts/benchmark/render_benchmark_video.py`.

---------------------------------------------------------------------------------------------------

## 7. Verified in this repository

* `python -m py_compile` on every file, `bash -n` on every shell script.
* Smoke test of the *relocated* code (from a clean clone layout, shadowing the machine's other `FISH` install):
  `train_ppo.py --task Bench-catfish_fish002-TrajFollow-PCA-v0 --num_envs 4 --max_iterations 2` — results in §10: **all 8 entry points passed**.

---------------------------------------------------------------------------------------------------

## 8. Differences between the paper text and the code that produced the results (read before reproducing)

The code is the ground truth for the released numbers; these are the places where the paper text and the code differ. Each is a one-flag change.

1. **Number of PCA modes.** Paper §3.3: "first 4 principal components (99.92 % variance)". Code: BLM+RL / CPG+RL control **all 20** modes
   (`pca_num_modes = 20` in `salmon_swim_traj_follow_cfg.py`); only the Task-2 BLM+IL runs use 4 (`--n_modes 4` / `scripts/blm_il_4pc`).
   For a 4-PC BLM+RL: `python scripts/benchmark/gen_fish_cfgs.py --pca_modes 4`. (The env slices `V[:K]`, `a_max`, `Δa_max` by K.)
2. **PD gain / joint limit.** Paper: kp = 120, ±45°. The runs of the paper used these. `gen_fish_cfgs.py` here defaults to them; the research
   checkout later moved its generator to kp = 240 / ±80° for an *unfinished* re-training sweep — reproduce with `--kp 240 --jlim 80`.
3. **Plateau rule.** Paper/appendix: "best completion improves by ≤ 0.02 over 20 epochs after ≥ 40 epochs". `scheduler.py`
   implements **≥ 80 epochs and ≤ 0.02 over the last 30** (relaxed 2026-09-27 because the 40/20 rule cut every run at epoch 40).
   Converged / blow-up / stall rules match the paper.
4. **Hydro safeguards.** The appendix mentions a per-bone impulse limiter (0.5) and a body-speed cap (1.5 BL/s). In code
   `hydro_impulse_beta = 0.5` is set only in the Reach10 cfg and `body_speed_cap_bl = 1.5` only in the ROV cfgs; the Task-1 / Task-2 cfgs use the env defaults (**off**).
5. **The ROV.** The code's ROV is the Bravo-arm ROV with a rigid basket (`agents/bravo_rov/`), not the BlueROV2 URDF named in the text.
   13 actions (6 wrench, 6 arm, 1 gripper); success needs the fish inside the basket and upright.
6. **Fish count.** Locally there are 80 processed fish (4 species × 20). The paper's 6 species / 120 fish include brook trout and brown trout,
   which were reconstructed in a second environment (`../vlm_based_articulation_generation/deploy/README_H200.md`). Only 9 tags (4 species) have
   trained benchmark results locally; `fish_list.json` lists them.
7. **Isaac Lab version.** Whatever `pip list` shows for `isaaclab` (0.54.4 in our setup, editable from a source checkout); the code needs Isaac Sim 5.1 semantics.

---------------------------------------------------------------------------------------------------

## 9. Pitfalls (each one cost a debugging session)

* **Bone colliders vs. FEM skin** — if `keep_bone_colliders` is `True` the bone colliders fight the attached skin and the fish pitches/spins by itself.
  All benchmark cfgs set `False`; the log line `disabled 29 colliders` is the check. Results produced before 2026-09-22 were void because of this.
* **Measured Φ is degenerate** (rank ≈ 2, condition ≈ 1e6): spline smoothing turns a joint kink into a global ramp on small fish and the ridge inverse
  commands zig-zag joints (a wave can even swim backwards). Every Bench cfg therefore uses `pca_phi_mode = "analytic"` (`scripts/benchmark/analytic_phi.py`);
  `calibration.npz` only supplies joint order, axis sign and head sign. Calibrate **skeleton-only** (`--no_fem`).
* **Curvature extraction**: an absolute difference threshold tracked runs tail-first and admitted shadows (|κ·BL| up to 250). Use the ratio threshold +
  per-frame head/tail decision (defaults in the command above), then validate with `scripts/benchmark/validate_extraction.py` and `travelling_wave_report.py`.
  `cpg_params.npz` and `amp_reference.npz` are derived from the curvature and must be regenerated whenever it changes. Store PCA coefficients at 60 fps.
* **Video vs. simulator handedness**: video curvature is in image coordinates while `p_bl`/`psi` come from a y-flipped midline;
  `scripts/blm_il_4pc/run_blm_il_4pc.py --mirror_traj 1` fixes this for BLM+IL. The reference heading is head/tail-flipped in about half of the runs —
  `TrajFollowMixin._disambiguate_reference_heading` repairs it at load.
* **FEM blow-ups**: the fish must be large enough for dt = 1/120 (a 6.8 cm sturgeon failed and was replaced by `lake_sturgeon_fish016`). Some assets self-propel through the FEM skin
  (`lake_sturgeon_fish001` drifts at 0.19 BL/s with zero action) — expect them to be the worst fish. Keep the nodal-velocity and joint-velocity guards on.
* **Do not read `body_link_*` buffers** in custom play loops (it perturbs the simulation); record `root_*` after `env.step`.
* **BLM+IL**: 20-mode increments are noise — use `--n_modes 4 --smooth_win 5`; history stacking lets the network extrapolate the history (use
  `--coef_noise 0.3 --coef_aug 4`); the expert video is mostly hovering, so BC matches statistics, not thrust; free-swim metrics with
  `spawn_glide_speed = 0.15 BL/s` also reward a passive glider, so read them together with joint-motion statistics.
* **BCO eval** needs a fresh process (a second `gym.make` in one Kit process hangs) — `--stage all` spawns it automatically.
* **ROV**: resumed policies collapse if `arm_slew` differs from training; success plateaus around 0.86–0.90.
* Do not `pkill -f` patterns; kill by PID. `queue.json` is live scheduler state — do not hand-edit it while the scheduler runs.

---------------------------------------------------------------------------------------------------

## 10. Smoke-test record

Run on the consolidated repository (2026-09-29/30), 4 environments, on a GPU shared with other jobs. The machine also had another editable `FISH`
install; the harness shadowed it with the code of this repository (the log line `Parsing configuration from: …/swimming_policy_training/…/agents/…yaml` proves which copy loaded).

| Test | Result |
|---|---|
| `train_ppo.py --task Bench-catfish_fish002-TrajFollow-PCA-v0 --max_iterations 2` | **passed** — FEM material created + bound, `disabled 29 colliders`, DriveAPI authored on 14 D6 joints, BLM loaded (`a_max`, `Δa_max`), epoch-1 `[EVAL]` line (completion 0.25, Fréchet 1.07 BL, W1(κ) 0.51, f 0.37 Hz, 0.30 BL/s), checkpoint written |
| `… Bench-catfish_fish002-TrajFollow-Joint-v0 --max_iterations 1` | **passed** (exit 0, checkpoint written; `reward=nan` because no episode finished in 1 epoch with 4 envs) |
| `… Bench-catfish_fish002-TrajFollow-CPG-v0 --max_iterations 1` | **passed** (exit 0; epoch-1 `[EVAL]`: completion 0.25, Fréchet 1.08 BL, f 1.48 Hz) |
| `… Bench-catfish_fish002-TrajFollow-Joint-AMP-v0 --max_iterations 1` | **passed** (exit 0; AMP discriminator + reference loaded) |
| `… Bench-catfish_fish002-ROVCatch-v0 --max_iterations 1` | **passed** (exit 0). The ROV USD loads without the URDF `meshes/` folder (not shipped, 80 MB) |
| `scripts/benchmark/run_freeswim_bc.py --baseline blm_il --n_modes 4 --smooth_win 5 --bc_epochs 3` | **passed** — BC trained, 20-state eval printed `FREESWIM-EVAL` (Fréchet, speed, heading stability, W1(κ), frequency) |
| `scripts/benchmark/run_freeswim_bc.py --baseline bco_pure --idm_transitions 3000 --bc_epochs 3` | **passed** — random rollouts → IDM → pseudo-labels → BC → eval in a fresh process (`FREESWIM_DONE`) |

All eight baseline entry points therefore run end to end from this repository. These are *plumbing* tests (the code path from task registration to a written checkpoint and metrics). They say nothing about final performance; the full
benchmark (256 envs, hundreds of epochs per run) was not repeated for this release.
