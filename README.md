# Video2SwimFish

Code for **Video2SwimFish: An End-to-End Pipeline for Reconstructing Controllable Fish Models and Biological
Locomotion from Real Fish Videos** (data: <https://huggingface.co/datasets/video2swimfish/video2swimfish-dataset>).

> **Anonymized snapshot for double-blind review.** Author names, e-mail addresses, personal URLs and machine paths
> have been removed from the files and from the git history.

Given synchronized top / front videos of one fish, the pipeline builds a metrically scaled, deformable and
**controllable** simulated fish and learns a swimming policy for it that is grounded in that animal's own motion.

```
                    ┌────────────────────────── vlm_based_articulation_generation/ ──────────────────────────┐
 front + top video  │ Qwen3-VL canonical frame → crop → Meshy image-to-3D → mesh cleaning / scaling            │
  (one fish, 32 fps)│ → VLM actor–critic skeleton (LoRA actor, geometric verifier, critic 1–5)                │
                    │ → structural completion R → Blender → PhysX USD (D6 joints, FEM skin, per-bone mass)    │
                    └───────────────────────────────────────┬────────────────────────────────────────────────┘
                                                            │  K_final.usd  (+ panel hydro proxy)
                    ┌────────────────────────── swimming_policy_training/ ───────────────────────────────────┐
 top-view video ───►│ midline → curvature → PCA = Biological Locomotion Manifold (BLM)                        │
                    │ Isaac Lab tasks (analytic panel hydro, FEM skin, PD joints)                              │
                    │ Task 1  trajectory following : BLM+RL · Joint RL · Joint RL+AMP · CPG+RL · BCO+RL        │
                    │ Task 2  free swimming from video : BLM+IL · BCO                                          │
                    │ Case study : BlueROV-style basket capture of the video-driven fish                       │
                    └──────────────────────────────────────────────────────────────────────────────────────────┘
```

| Folder | What it is | Start here |
|---|---|---|
| [`vlm_based_articulation_generation/`](vlm_based_articulation_generation/README.md) | video → canonical frame → mesh → VLM-articulated, physics-ready USD fish, incl. all post-processing (mesh cleaner, structural completion R, containment verifier, stage-1 checks, box→spine, mass, size calibration) and the actor LoRA fine-tuning | its README |
| [`swimming_policy_training/`](swimming_policy_training/README.md) | Isaac Lab tasks, **every baseline training script**, BLM extraction, calibration, scheduler, metrics, assets of the 9 benchmark fish | its README |

> The two folders are named `vlm_based_articulation_generation` and `swimming_policy_training` (snake case, no spaces, so that paths work in shells and CI).

## Two ways to use this repository

**A. Reproduce the policy benchmark from the shipped assets** (needs only Isaac Sim / Isaac Lab and a GPU):
`swimming_policy_training/data/` already contains the processed assets and BLMs of the 9 benchmark fish
(4 species). Follow `swimming_policy_training/README.md` §3 (install) and §6 (run baselines).

**B. Build new fish from new videos** (needs Blender 5.0.1, Qwen3-VL-32B, a Meshy key, ≥ 64 GB GPU memory for the VLM):
1. run the VLM pipeline → `K_final.usd` per fish (`vlm_based_articulation_generation/README.md`);
2. copy / symlink `<asset dir>/<tag>/K_final.usd` into `swimming_policy_training/data/fish_assets/<tag>/`;
3. build the per-fish BLM + hydro + calibration files and register the fish
   (`swimming_policy_training/README.md` §4), then train (§6).

## What is *not* in git (and where to get it)

| Item | Size | Where |
|---|---|---|
| Raw synchronized videos (120 fish, 6 species) and the released 120 assets | 39 GB+ | Hugging Face dataset above |
| Qwen3-VL-32B-Instruct weights | 63 GB | `python vlm_based_articulation_generation/scripts/download_qwen3_vl.py` (Hugging Face `Qwen/Qwen3-VL-32B-Instruct`) |
| Actor LoRA adapter used for the released assets (`actor_lora_v2`) | 547 MB | not in git — Hugging Face model repo `video2swimfish/actor-lora-v2` (access may be restricted during review; `deploy/fetch_assets.sh` downloads it), or retrain from the 14 hand-built fish (exact command in the VLM README) |
| Meshy API key (≈ 30 credits per fish) | – | your own account; never commit it |
| Trained RL checkpoints, wandb logs | – | produced by the scripts; the four BLM+IL prey policies used by the ROV case study are shipped |

## Requirements at a glance

| | VLM articulation half | Policy training half |
|---|---|---|
| OS / GPU | Linux, ≥ 64 GB GPU for the 32B VLM (2 × 96 GB used) | Linux, RTX-class GPU, 96 GB used for 256 FEM envs × 2 jobs |
| Key software | Blender **5.0.1 exactly**, Python 3.11, torch 2.11 + transformers 5.9 + peft 0.19 | Isaac Sim 5.1.0 (pip), Isaac Lab, Python 3.11, torch 2.7, rl_games 1.6.1, wandb |
| Optional | Isaac Sim (only for the FEM "cook" and panel hydro) | – |

## Paper ↔ code

| Paper | Code |
|---|---|
| §3.1 canonical frame (Qwen3-VL) + crop + Meshy + scaling | `vlm_based_articulation_generation/scripts/{select_canonical_frame_qwen,crop_canonical_to_fish_cv,meshy_client,measure_fish_size,clean_mesh_components,verify_mesh_vlm}.py` |
| §3.2 actor–critic articulation, geometric verifier, structural completion R | `…/video2swimfish/articulation/{actor,critic,run_auto_construction,geometric_verification,articulation}.py` (`_repair_skeleton` = R) |
| §3.5 / App. skeleton dataset + LoRA fine-tuning | `…/data/fish_skeleton_dataset/`, `…/video2swimfish/finetune/{build_sft_dataset,train_actor_lora}.py` |
| §3.3 Biological Locomotion Manifold | `swimming_policy_training/scripts/zef_manifold/`, `…/salmon_swim_pca_env.py` |
| §4.1 simulation setup, panel hydrodynamics | `…/salmon_swim_env.py`, `…/salmon_swim_amp_misty_panels_cfg.py`, `scripts/precompute_catfish001_panels.py` |
| §4.2 Task 1 baselines | `…/salmon_swim_traj_follow_{env,cfg}.py`, `scripts/rl_games/train_ppo.py`, `bco/` |
| §4.3 Task 2 (BLM+IL, BCO) | `scripts/benchmark/run_freeswim_bc.py`, `scripts/blm_il_4pc/` |
| §4.4 BlueROV capture | `…/salmon_rov_{chase,catch_bench}_env.py`, `agents/bravo_rov/` |
| Training protocol / stopping rules | `scripts/benchmark/{build_queue,scheduler}.py` |

Both READMEs contain a section **"Paper vs. code"** listing the (few) places where the text of the paper and the code that
produced the released results differ (e.g. number of PCA modes, PD gains, plateau rule, LoRA rank). Please read them before comparing numbers.

## Status and honesty notes

* This is the cleaned, path-independent version of the code: every hard-coded machine path was replaced by a repo-relative path or an environment variable.
* Verification performed on the consolidated code is listed at the end of each half's README (compile checks, smoke runs). A full re-training of the benchmark
  (days of GPU time) was **not** repeated for this release.
* License: the license file is added with the public release (the dataset is CC BY-NC 4.0 according to the paper).
