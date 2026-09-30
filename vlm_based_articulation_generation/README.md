# Video2SwimFish — VLM-based articulation generation

This folder is the **"video → controllable fish asset"** half of the Video2SwimFish pipeline.
It turns a fish video into a **USD articulated fish** (rigid box bones, D6 joints, FEM skin,
mass) that the other half, `../swimming_policy_training/`, loads into Isaac Lab.

It is self-contained: every path is resolved from one file, `v2sf_paths.py`, relative to this
folder and overridable with environment variables. Nothing is hard-coded to a machine.

```
raw front/top video (mp4)
   │  1  scripts/select_canonical_frame_qwen.py     Qwen3-VL picks 1 of 12 candidate frames
   │  1b scripts/crop_canonical_to_fish_cv.py       classical CV crop around the fish
   ▼
canonical crop (jpg)                       ┌──────────────────────────────────────────────┐
   │  2  scripts/extract_species_curvature.py  →  scripts/measure_fish_size.py            │
   │     (front video → body length in px)        (tank wall = ruler → real length, m)    │
   │                                              └── fish_sizes.json  ──────────┐        │
   ▼                                                                             │        │
   │  A  scripts/run_pipeline.py --stop_after_mesh --skip_critic                 │        │
   │     scripts/meshy_client.py  Meshy image-to-3D (30 credits)                 │        │
   │     scripts/clean_mesh_components.py  drop a 2nd, separate fish body        │        │
   ▼                                                                             │        │
mesh.glb                                                                         │        │
   │  B  video2swimfish/articulation/run_auto_construction.py  (ONE Qwen3-VL load)│       │
   │      ├─ mesh critic   scripts/verify_mesh_vlm.py       (adapter OFF)        │        │
   │      ├─ bootstrap     align + PCA-level + scale to real length  ◄───────────┘        │
   │      ├─ ACTOR–CRITIC LOOP (≤6 iters)                                                 │
   │      │    render 3 views → Actor (LoRA ON) → 1 action                                │
   │      │    Blender apply → geometric verifier (hard constraint) → render → Critic     │
   │      ├─ repair pass   articulation._repair_skeleton   (deterministic, must stay ON)  │
   │      └─ export        extract skeleton → USDA → physics → USD → [FEM cook] → mass    │
   ▼                                                                                      │
K_final.usd  (+ articulation_construction_log.json, K_final.mass_report.json)             │
   │  C  video2swimfish/articulation/verify_stage1.py   report (not a gate)               │
   │  F  scripts/box_bones_to_spine.py                  optional: box → fish-spine shape  │
   │  D/E (optional) panel hydro + PCA  → live in ../swimming_policy_training ────────────┘
   ▼
../swimming_policy_training/data/fish_assets/<tag>/K_final.usd   (hand-off)
```

Letters A–F are the phase names used by `scripts/run_dataset_v2sf.py` and
`deploy/run_species_e2e.sh`.

---------------------------------------------------------------------------------------------

## 1. Quick start

```bash
# 0. python env for the VLM (Python 3.11)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 1. edit env.sh (BLENDER_BIN, VLM_PYTHON, ...), then
source env.sh
python v2sf_paths.py                       # prints every resolved path
python deploy/preflight.py --species brook_trout --no_meshy   # checks the machine

# 2. get the big things (see section 6): Qwen weights, actor LoRA, raw videos

# 3. one species end to end (resumable; phase A SPENDS Meshy credits)
deploy/run_species_e2e.sh brook_trout --phases 012          # free phases first
deploy/run_species_e2e.sh brook_trout --phases ABCF         # then mesh → skeleton → USD → spine

# or drive the dataset loop yourself
python scripts/run_dataset_v2sf.py --species catfish,lake_sturgeon --phases ABC
```

Single fish, skeleton only (needs `MESH_OUT/<tag>/mesh.glb` and, for real-size scaling,
`SPECIES_MANIFOLD_ROOT/<species>/fish_sizes.json`):

```bash
python video2swimfish/articulation/run_auto_construction.py \
    --fish_id catfish_fish002 --max_iterations 6 --require_vlm \
    --actor_checkpoint "$ACTOR_LORA" --output_dir dataset/catfish_fish002
```

### Environment variables (all optional; defaults are relative to this folder)

| variable | default | meaning |
|---|---|---|
| `V2SF_ROOT` | folder of `v2sf_paths.py` | package root |
| `QWEN_MODEL_PATH` | `models/Qwen3-VL-32B-Instruct` | base VLM weights (63 GB) |
| `ACTOR_LORA` | `checkpoints/actor_lora_v2` | actor LoRA adapter dir |
| `BLENDER_BIN` / `BLENDER_PY` | auto-search `~/blender`, `/opt`, `/usr/local`, `$BLENDER_ROOT`, `PATH` / `<blender>/5.0/python/bin/python3.11` | **exactly Blender 5.0.1**, version checked at first use |
| `VLM_PYTHON` | `sys.executable` | python with torch/transformers/peft |
| `ISAAC_PYTHON` | unset | Isaac Sim 5.1 python; **only** the FEM cook (and hydro phase D) |
| `MESHY_API_KEY_FILE` | `~/.meshy_api_key` | file with the Meshy key (`chmod 600`) |
| `ASSET_OUT` | `dataset/` | final per-fish dirs (`<tag>/K_final.usd` …) |
| `MESH_OUT` | `outputs/` | Meshy meshes `<tag>/mesh.glb` |
| `RAW_VIDEO_ROOT` | `raw_datasets/` | `<species>/{front,top}NNN.mp4`, `<species>_canonical_frames[_cropped]/` |
| `SPECIES_MANIFOLD_ROOT` | `species_manifold/` | `<species>/front/fish<id>_curvature.npz`, `<species>/fish_sizes.json` |
| `EVAL_OUT` | `eval_out/` | renders / IoU / sheets for the evaluation scripts |
| `POLICY_ROOT` | `../swimming_policy_training` | sibling repo; used only by driver phases D and E |

Code that runs **inside Blender's python** (`articulation.py`, `verify_stage1.py`,
`geometric_verification.py`, `build_sft_dataset.py`, …) finds `v2sf_paths.py` from its own
`__file__`, so it needs no environment variable.

---------------------------------------------------------------------------------------------

## 2. Stage by stage

| # | stage | command (from this folder) | inputs | outputs | code |
|---|---|---|---|---|---|
| 1 | canonical frame | `python scripts/select_canonical_frame_qwen.py --dataset raw_datasets/<sp> --species <sp>` | `front*.mp4` | `raw_datasets/<sp>_canonical_frames/<sp>_fish<id>_canonical.jpg` | `select_canonical_frame_qwen.py:extract_candidates` (12 frames, evenly spaced, 5 % margins, ffmpeg), prompt `PROMPT` → one number |
| 1b | crop | `python scripts/crop_canonical_to_fish_cv.py --species <sp> --canonical_dir … --video_dir … --view front` | canonical jpg + its video | `<sp>_canonical_frames_cropped/` | `crop_canonical_to_fish_cv.py:main` (median background of the video → blob → bbox + 15 % margin; falls back to uncropped) |
| 2 | fish size | `python scripts/extract_species_curvature.py --video … --species <sp> --fish_id <id> --view front --sigma 7.0 --thresh 60 --open_px 9 --stride 2 --max_frames 3000`, then `python scripts/measure_fish_size.py --species <sp>` | front videos | `species_manifold/<sp>/fish_sizes.json` | `extract_species_curvature.py:segment_frame/skeleton`, `measure_fish_size.py:fish_length_from_front` |
| A | mesh | `python scripts/run_pipeline.py --image <crop> --tag <sp>_fish<id> --out_dir outputs/<tag> --species <sp> --fish_id <id> --stop_after_mesh --skip_critic` then `python scripts/clean_mesh_components.py --glb outputs/<tag>/mesh.glb` (or `run_dataset_v2sf.py --phases A`) | crop | `outputs/<tag>/mesh.glb`, `meshy_task.json`, `mesh_clean_report.json` | `meshy_client.py:MeshyClient.generate_and_download`, `clean_mesh_components.py:analyse` |
| B | skeleton + USD | `python scripts/run_dataset_v2sf.py --species <sp> --phases B` (waves of 4 fish, ONE Qwen load per wave) | `mesh.glb`, `fish_sizes.json`, LoRA | `dataset/<tag>/K_final.usd` … (see §7) | `run_auto_construction.py:run_for_fish`, `articulation.py:_blender_apply_action`, `_repair_skeleton`, `export_articulation_to_usd`, `apply_volume_proportional_mass` |
| C | stage-1 report | `python scripts/run_dataset_v2sf.py --species <sp> --phases C` (or `blender -b --python video2swimfish/articulation/verify_stage1.py -- --usd <K_final.usd> --fish_id <tag>`) | `K_final.usd` | `stage1_report.json` | `verify_stage1.py:check_interpenetration/check_coverage/check_alignment` |
| F | spine shapes | `"$BLENDER_PY" scripts/box_bones_to_spine.py --dataset dataset --out dataset_spine --template_from catfish_fish001 --link_siblings` | `K_final.usd` | `dataset_spine/<tag>/K_final.usd` | `box_bones_to_spine.py:convert` |
| — | reports | `python scripts/dataset_report.py`, `python scripts/make_dataset_sheet.py`, `python scripts/canonical_iou.py`, `python scripts/canonical_iou_best8.py` | dataset | `DATASET_REPORT.{md,csv}`, sheets, IoU tables | see §4.10 |

Notes on the stages:

* **Meshy call** (`meshy_client.py:submit_image_to_3d`): `ai_model="meshy-6"`, textured, 2k texture,
  `target_polycount=30000`, triangle topology, `should_remesh=True`, no PBR, GLB. The image is sent
  as a base64 data-URI (no public hosting). **Every task consumed 30 credits** in the 80
  `meshy_task.json` files of the released run. `poll_until_done` waits up to 900 s.
* **Mesh critic** (`verify_mesh_vlm.py:verify`): 3 solid-shaded renders (persp/top/side) and a prompt that
  asks about four defects: hole, part poking through the skin, broken geometry, *more than one fish*.
  It must end with `VERDICT: PASS|FAIL`; an unparsable reply counts as FAIL. The batch driver
  runs it with the actor adapter disabled and records the result in `mesh_critic.json`;
  a rejection is **recorded, not fatal**.
* **Driver phases D/E** (panel hydro, curvature → PCA) are *not* part of this package's paper path.
  They call scripts in `$POLICY_ROOT/scripts/` (`precompute_catfish001_panels.py`,
  `zef_manifold/{extract_species_curvature,fit_pca_species,extract_species_reference}.py`) and are
  skipped with a log line if those or `ISAAC_PYTHON` are missing.

---------------------------------------------------------------------------------------------

## 3. The actor–critic loop (paper Sec. 3.2)

Implemented in `video2swimfish/articulation/run_auto_construction.py:run_for_fish`.

```
K0 = empty articulation, skin bootstrapped in Blender (canonical frame, real length)
for k in 0 … max_iterations-1:                      # 6 for the dataset (driver default);
                                                    # CLI default 10; run_pipeline default 30
    views_k  = render(K_k)                          # top, front, side, 900x900 (renderer.py)
    a_k      = Actor(views_k, K_k, critic_feedback_{k-1}, template_library)
    K_{k+1}  = Apply(K_k, a_k)  in Blender          # articulation.py:_blender_apply_action
    K_{k+1}  = GeometricVerification(K_{k+1})       # hard constraint, geometric_verification.py
    (score, accepted, feedback) = Critic(render(K_{k+1}), K_{k+1}, measured facts)
    keep best; if accepted: K* = K_{k+1}; stop
if never accepted: K* = best-scoring K_{k+1} over all iterations
then: repair pass  →  export
```

**Actor** (`actor.py:Actor.propose`, prompt `ACTOR_PROMPT_TEMPLATE`)
* Inputs: 3 rendered views in the order `[top, front, side]`; the current articulation as JSON
  (positions/sizes shown **scaled into the actor's training units**, a 0.25 m fish:
  `ACTOR_TRAIN_LENGTH_M`, `units_k = fish_length / 0.25`); the previous critic feedback JSON (or
  "none, first iteration"); the template library text (14 keys with bone count and body-shape
  ratios, `skeleton_template.py:template_library_prompt_str`).
* Output: ONE JSON action `{action: add|remove|reposition|resize, bone_id, template, position[3],
  size[3], reasoning}`. `position`/`size` are multiplied back by `units_k`.
  The actor is **never** asked for joint parameters.
* Model: Qwen3-VL-32B-Instruct + LoRA adapter, greedy (`temperature=0`), one retry with a stricter
  reminder if the reply has no JSON; then a MOCK fallback (`_mock_action`).
* The adapter is toggled in place (`_set_adapter_enabled`): **ON for the actor, OFF for the
  critic**, so one 63 GB model instance serves both.

**Apply** (`_blender_apply_action`, runs under Blender)
* `add`: appends the template's bone meshes from its `.blend`, keeps the template bone whose relative
  x-position is closest to the requested position, resets rotation, sets `dimensions=size`, puts the
  bbox centre on the requested position. A requested centre that is **outside the skin** is projected onto
  the body centerline at the same x (`resolve_position`); non-finite/short vectors are coerced to 3 floats.
* `remove`, `reposition`, `resize` act on `bone_<id>`.

**Geometric verifier** (`geometric_verification.py:verify_bone`, "hard constraint")
* Containment test: nearest point on the skin BVH + normal side, per bone vertex.
* Anisotropic shrink: up to `max_iters=26` steps of ×0.9 on ONE axis (the axis that removes the most
  outside vertices; ties prefer not shrinking the along-body X axis), re-centring the bbox centre each
  step, until **0 vertices are outside** (`max_outside_fraction=0`). If no single-axis shrink helps it
  falls back to a uniform ×0.9. Returns counts before/after, per-axis shrink steps, `fully_contained`.

**Critic** (`critic.py:Critic.evaluate`, prompt `CRITIC_PROMPT_TEMPLATE`)
* Pretrained model only (adapter off), `temperature=0`. Inputs: the same 3 views, the proposed
  articulation JSON, and **measured facts** computed from geometry: containment of the bone just
  touched, and the midline offset of every bone (fraction of the local body half-extent, y and z).
* Criteria: nothing outside, continuous head-to-caudal-fin coverage, **midline alignment** (any
  |offset| > 0.35 or mean > 0.2 ⇒ score ≤ 2), plausible shapes, plausible bone count.
* Output `{score 1–5, accepted, feedback{interpenetration, coverage, alignment, bone_shape, other}}`;
  the prompt says `accepted=True only if score >= 4`.

**Loop control**
* Best selection: highest score; **ties go to the later iteration / the state with more bones**
  (`>=`), because the critic's criteria reward coverage.
* Cycle escape: at temperature 0 the actor falls into "add bone N / remove bone N" 2-cycles. After
  **4 alternating iterations on one bone** the *actor* (not the critic) switches to
  `temperature=0.7`; if it is still alternating after 6 more iterations the loop stops and the
  best-so-far is used (`log["cycle_escape"]`, `log["stopped_reason"]`).
* On the released run the loop **almost never converges by itself**: 1 of 80 fish was accepted
  inside the loop; the deterministic repair pass then brought 78 of 80 to a critic score ≥ 4
  (numbers from the 80 `articulation_construction_log.json` files).

---------------------------------------------------------------------------------------------

## 4. Post-processing and verification functions

### 4.1 Bootstrap of the skin (before iteration 0) — `articulation.py:_blender_apply_action`
1. `gask.align_target_mesh_like_supervised`: length → +X, head at +X (bulky end). After this step **Y is the thin axis
   (thickness) and Z the height** (the report's reason string says "height_y_thickness_z", but the measured
   `features_after` show `height_axis=2, thickness_axis=1`; `run_auto_construction` uses Y = thickness, Z = height).
2. `_pca_level_skin(min_deg=0.5, max_deg=80)`: rotate so the vertex-cloud principal axis is exactly +X
   (Meshy keeps the *photo's* pose, so a tilted fish comes out tilted). Tilts > 80° are left alone.
3. `gask.scale_target_mesh_to_real_length` using `find_real_length_m(fish_id)` (from `fish_sizes.json`);
   if no size is known it logs `applied: false` and keeps Meshy's arbitrary scale.
4. `_recenter_skin`: bbox centre → origin (the actor was trained on origin-centred fish).

### 4.2 Structural completion `R` = repair pass — `articulation.py:_repair_skeleton`
Run **once** after the loop ends (accepted, cycle-stopped or capped) with defaults
`gap_frac=0.02`, `head_area_frac=0.15`, `tail_area_frac=0.03`, `max_bones=22`
(overridable in the action dict). Centerline = `_robust_centerline` (201 samples, thickness-weighted
core per slice via `_slice_core`, 6 % moving average), so fins/barbels/scutes do not pull it.
Steps, in code order:
1. **Trunk extent**: cross-section "area" profile (101 slices). `x_head_target` = first slice from the head
   with area ≥ 15 % of the maximum (skull, not snout/barbels); `x_tail_target` = first slice from the tail
   with area ≥ 3 % (caudal-fin base, not fin membrane).
2. **Snap** every bone centre onto the centerline (y, z only), re-run `verify_bone`.
3. **Drop overlaps**: while two x-neighbours overlap by more than 50 % of the shorter one, delete the shorter
   (the cycling actor sometimes stacks bones).
4. **Fill spans** with new box bones: head margin, inner gaps wider than 2 % of body length, tail margin.
   Segment length = max(median bone length, 6 % of body length); box y/z = local body half-extent ×
   the median ratio of the existing bones (fallback 0.4 / 0.15), x-length ×0.95. Order **tail → inner →
   head** so the bone cap (22) never starves the tail.
5. **Contain**: every new box goes through `verify_bone`; if it still is not fully inside (typically the snout
   cavity) it is **dropped** rather than exported protruding.
Then the critic scores the repaired result once more (`critic_after_repair`). Bone ids are free; the joint
chain is ordered by bone centre x in `add_isaac_physics_to_usd.ordered_bones`.

### 4.3 USD export — `articulation.py:export_articulation_to_usd`
`extract_fish_skeleton.py` (largest mesh = skin, every other mesh = bone) → `export_blend_to_usd.py`
(USDA with textures) → `add_isaac_physics_to_usd.py` (rigid bodies, 9 D6 joints, attachments, mass) →
`convert_usda_to_usd.py` → `cook_deformable_isaac.py` (**optional**, Isaac) →
`apply_volume_proportional_mass`. Without `ISAAC_PYTHON` the FEM cook is skipped and recorded as
`export.fem_cooked=false`.

### 4.4 Mass — verified in the code
* `add_isaac_physics_to_usd.mass_distribution_from_mesh` first gives every bone the **same** mass
  (`volume × 1000 kg/m³ / n_bones`, volume = mesh volume if it is 15–80 % of the bbox volume, else
  0.35 × bbox volume). The deformable skin gets `1e-5` kg.
* `articulation.apply_volume_proportional_mass` then **overwrites** every bone's `physics:mass`:
  `total = (4/3)·π·(L/2)(H/2)(T/2) × 1000 kg/m³` (ellipsoid from the measured length/height/thickness of
  the canonicalised, scaled skin), split **in proportion to each bone's box volume**
  `size_x·size_y·size_z`. Written to `K_final.mass_report.json`. (Example: catfish_fish002, 0.167 m → 0.41 kg.)

### 4.5 FEM cook defaults — `cook_deformable_isaac.py` (Isaac, optional)
`hex-resolution 10`, `youngsModulus 1e5`, `poissonsRatio 0.45`, `elasticityDamping 0.05`,
`dampingScale 1.0`, `solverPositionIterationCount 96`, `vertexVelocityDamping 4.0`,
`sleepDamping 10.0`, `settlingThreshold 0.1`, no self-collision. It creates **and binds** a real
`PhysxDeformableBodyMaterialAPI` (the un-cooked asset has none, so material overrides would silently do
nothing). `usd_physics_defaults.json` carries the same 1e5 / 0.45 / 0.05 / 1.0 for the pre-cook attrs.

### 4.6 Joint preset: what the USD contains vs what training overrides
The paper says "3-DoF D6, ±45°, PD kp=120, kd=6". **The exported USD does not contain that.**
* `add_isaac_physics_to_usd.py:create_joint` authors **9 D6 joints** (10 bones), translation locked,
  `rotX/Y/Z` `PhysicsLimitAPI` **±15°** (`usd_physics_defaults.json` → `joint.rotation_limits`),
  `collisionEnabled=false`, break force/torque = inf. **No `DriveAPI` is authored.**
  The `joint.drive` block in the json (damping 1000, stiffness 0, …) is **never read** by any code.
* The ±45° limits and the PD position drive are applied **at RL-load time** by the simulation side:
  `swimming_policy_training/.../salmon_swim_env.py:_prepare_env_assets` re-writes the D6 `LimitAPI`
  (cfg `joint_limit_deg`) and authors a force-mode angular `DriveAPI` on rotX/Y/Z; the numeric
  stiffness/damping come from the task cfg's actuator (`stiffness`, `damping`), not from the USD.
  Treat kp/kd/limit as **training-side configuration**; the asset only guarantees the joint topology
  and the ±15° default. (`bones` in this repo no longer carry a `joint_config`; see `articulation.Bone`.)

### 4.7 Geometric verifier / clean_mesh_components / verify_stage1 thresholds
* `geometric_verification.verify_bone`: see §3 (26 iterations, ×0.9, 0 outside vertices).
* `clean_mesh_components.analyse` (deterministic; weld vertices → connected components → keep the largest):
  drop any other component that (a) is **disjoint** from the main body's bbox on any axis, (b) has
  more than **15 %** of the surface (`SECOND_BODY_FRAC`; no fin is that large), or (c) sticks out of
  the main bbox by more than **10 %** of its extent on an axis (`OUTGROW_FRAC`). All tests are done in the
  main component's **principal-axis frame** (a tilted fish hides fragments in an axis-aligned bbox). The
  file is rewritten only if ≥ 2 % of the area is dropped (`--min_drop_frac`); the original is kept as
  `mesh.meshy_original.glb`. `--dry_run` only writes `mesh_clean_report.json`.
* `verify_stage1.py` (imports the exported USD into Blender; a **report**, not a gate):
  1. containment — pass if 0 outside vertices, or ≤ 5 outside **and** max depth < 2 mm;
  2. coverage — union of bone x-spans / body length **≥ 0.90** and largest gap **≤ 0.10**;
  3. alignment — mean angle between each bone's PCA axis and the local midline tangent **< 15°** and max **< 30°**.
  `overall_pass` = all three. On the released 80 fish coverage failed for all 80 (median 73 %),
  containment passed for 57, alignment failed for 23, so **0/80 pass**: the repair pass is bounded by the
  rules above, not by the 0.90 target. Do not use it as a filter.

### 4.8 `box_bones_to_spine.py`
Swaps the 8-vertex **box** collision/visual mesh of every bone for the fish-spine mesh (prim
`lower_curved_triangular_fish_spine_07_mesh`, taken from a donor asset, default `catfish_fish001`,
normalised to [-1,1]³) scaled per axis so each bone's world bbox is **identical** (error > 1e-6 m aborts).
Bone count, bone xform, mass, D6 joints, attachments and the skin are untouched, so DOF is unchanged.
Bones that are already non-box (template-derived) are kept. The convex-hull collider shrinks. **A donor
dir containing that prim must exist in `--dataset`** (`--template_from`).

### 4.9 `measure_fish_size.py`
Tank interior 30" × 17.5" × 11.75". **Front view** is used for length: `length_in = bodylen_px (median over valid
frames) / px_per_inch`. `px_per_inch` per species: catfish 88.5, lake_sturgeon 88.5, white_bass 89.4,
bluegill 87.8, **brown_trout 89.0** (measured), **brook_trout 89.8 (estimate, flagged in the json as
`px_per_inch_is_estimate`)** — the automatic estimate for brook trout (78.7) is wrong because its right tank wall
has too little contrast; 89.8 is the five-species median quoted in `deploy/README_H200.md`. Unknown
species fall back to 88.5 with a warning. Uncertainty ±18 % (length- vs depth-axis disagreement).
Robustness pass: fish with < 150 valid frames, or a length outside 0.4–2.5 × the species median, or
> 20", get the **species median** and `suspect=true, size_source="species_median_fallback"`.

### 4.10 Evaluation helpers
* `canonical_iou.py`: silhouette IoU between a mesh render (`EVAL_OUT/dataset_sheet/mesh_views/<tag>_side.png`, made
  with `render_mesh_views.py`) and its canonical crop. Photo mask = U2-Net (rembg) alpha computed outside this
  package into `EVAL_OUT/canonical_iou/photo_masks/` (falls back to GrabCut); similarity alignment from
  image moments + a by-eye head-side label (`HEAD_SIDE`, only for the released fish), then a local search
  (scale ±12 %, angle ±8°, shift ±6 %). Controls: other fish of the species, and an ellipse of equal moments.
* `canonical_iou_best8.py`: 6-DOF affine refinement (Powell) for the 8 cleanest fish + a body-only IoU (fins
  removed by a morphological opening of 6 % body length).

---------------------------------------------------------------------------------------------

## 5. Actor LoRA provenance (read this before comparing with the paper)

| | paper appendix / trainer defaults | **adapter that produced the dataset (`actor_lora_v2`)** |
|---|---|---|
| rank / alpha | 8 / 16 | **16 / 32** |
| dropout | 0.05 | 0.05 |
| target modules | all-linear | all-linear (q,k,v,o,gate,up,down, ViT qkv/proj/fc1/fc2) |
| epochs | 4 | **8** (train_log: 8 × 74 rows + 8 val lines) |
| lr / grad-accum | 1e-4 / 8 | **not recorded** in the adapter files (trainer defaults assumed) |
| train / val | 12 fish / roach + trevally | 74 train rows / 11 val rows (same split) |
| val loss | | 0.302 → 0.202 (epoch 6), 0.230 at epoch 7 (last epoch saved, not the best) |

An earlier adapter, `actor_lora_v1`, *is* rank 8 / alpha 16 / 4 epochs (300 log lines = 4 × 74 + 4), i.e.
the paper setting; the dataset was built with **v2**. `data/actor_sft_v2/` keeps v2's `adapter_config`,
`train_log`, and the SFT labels (`rig_sft.jsonl`, `rig_sft_val.jsonl`: prompts + target actions, image paths
only — the 255 rendered PNGs, 229 MB with scene files, are not shipped).

`train_actor_lora.py` now exposes `--lora_rank --lora_alpha --lora_dropout --target_modules --num_epochs
--learning_rate --grad_accum_steps` (defaults = the paper) and writes `train_args.json` next to the adapter.

**Retrain the paper-setting adapter** (rank 8):
```bash
blender -b --python video2swimfish/finetune/build_sft_dataset.py -- --out_dir outputs_sft/sft_data
python video2swimfish/finetune/train_actor_lora.py --train outputs_sft/sft_data/rig_sft.jsonl \
    --val outputs_sft/sft_data/rig_sft_val.jsonl --output_dir checkpoints/actor_lora_v1_repro
```
**Retrain the shipped v2 setting**:
```bash
python video2swimfish/finetune/train_actor_lora.py --train outputs_sft/sft_data/rig_sft.jsonl \
    --val outputs_sft/sft_data/rig_sft_val.jsonl --output_dir checkpoints/actor_lora_v2_repro \
    --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --num_epochs 8 --learning_rate 1e-4 --grad_accum_steps 8
```
Caveat: `build_sft_dataset.py` re-renders the SFT images with the **current** `renderer.py`. The v2 training images
were rendered with an older renderer (dark-grey, opaque, no orange bones); the current one draws a translucent
skin, bright bones, and (for inference) a dilation of bone pixels. By default the rebuilt images
have no dilation (`--enlarge` turns it on); they are still not pixel-identical to the v2 images, so a retrained
adapter is *equivalent in spirit*, not bit-identical. Training needs ≥ 64 GB GPU memory (bf16, batch size 1,
gradient checkpointing, `device_map="auto"`). The trainer is a plain AdamW loop (no warm-up/scheduler), grad-clip 1.0.

---------------------------------------------------------------------------------------------

## 6. External requirements and how to get the big things

| need | detail |
|---|---|
| **Blender 5.0.1 exactly** | official tarball. 4.x and 5.1 change the `bpy`/USD API used here (checked at first use). Its bundled python3.11 provides `pxr` (USD) and is the interpreter for the export chain. Needs a working EEVEE/GL context for renders (`blender -b` on a headless box needs EGL/OpenGL). |
| **GPU ≥ 64 GB** | Qwen3-VL-32B-Instruct bf16 = 63 GB. Actor + critic share one instance. The released run used an RTX PRO 6000 (96 GB). |
| **Qwen3-VL-32B-Instruct** | `python scripts/download_qwen3_vl.py --output models/Qwen3-VL-32B-Instruct` (HF `Qwen/Qwen3-VL-32B-Instruct`) |
| **Actor LoRA** | not in git (547 MB). On Hugging Face (private model repo, needs `hf auth login`): `hf download video2swimfish/actor-lora-v2 --local-dir checkpoints/actor_lora_v2` (done by `deploy/fetch_assets.sh`), or **retrain** (§5). Without it the actor is the pretrained model (`actor_meta.finetuned=false`). |
| **Videos + released assets** | HF dataset `https://huggingface.co/datasets/video2swimfish/video2swimfish-dataset` (raw videos under `data/raw_videos/<species>/`, finished USD assets under `data/usd_assets/`). Put videos in `raw_datasets/<species>/`. Species dir must be lowercase without spaces (`deploy/make_species_dir.py`). |
| **Meshy API** | key in `MESHY_API_KEY_FILE`; **30 credits per fish** (41 trout ≈ 1230). `deploy/preflight.py` reads the balance (free). |
| Isaac Sim 5.1 | **optional**; only the FEM cook and phase D. |
| ffmpeg/ffprobe | frame extraction. |
| rembg (U2-Net) | optional, only `canonical_iou*.py`. |

Size of this folder in git: ~18 MB (mostly the 14 hand-built `.blend` templates, 17 MB).
Ignored (see `.gitignore`): `models/ checkpoints/ dataset/ raw_datasets/ outputs*/ species_manifold/ eval_out/`.

---------------------------------------------------------------------------------------------

## 7. Output layout of one fish

`MESH_OUT/<tag>/` (phase A): `mesh.glb` (cleaned; `mesh.meshy_original.glb` if a 2nd body was dropped),
`meshy_task.json` (Meshy task incl. `consumed_credits`), `mesh_clean_report.json`, `pipeline_report.json`.

`ASSET_OUT/<tag>/` (phases B, C):

| file | what |
|---|---|
| `K_final.usd` / `K_final.usda` | the asset: `/root/skeleton/bone_<id>` rigid bodies, `D6Joint[_NN]`, `PhysxPhysicsAttachment`s, skin with deformable schema (+ FEM cook if Isaac was used), rigid + deformable materials |
| `K_final.mass_report.json` | ellipsoid fish volume, total mass, per-bone volume and mass |
| `articulation_construction_log.json` | everything about the loop: per-iteration action + `actor_meta` (**check `source == "qwen3vl"`**), containment numbers, critic score/feedback, `cycle_escape`, `repair`, `critic_after_repair`, `final_articulation`, `export` |
| `stage1_report.json` | the three stage-1 checks (phase C) |
| `mesh_critic.json` | mesh VLM verdict (batch mode) |
| `_work/` | `iter_*_before/after_{top,front,side}.png` renders, `.blend`s, `iter_*_apply_report.json`, `repair.blend`, `repair_*.png`, `skeleton_json/`, `usd/base_export.usda`, step logs |
| `FAILED.txt` | traceback if the fish failed |

The hand-off to `swimming_policy_training` is `ASSET_OUT/<tag>/K_final.usd` (plus, optionally,
`panel_hydro.npz` next to it) copied or symlinked into `swimming_policy_training/data/fish_assets/<tag>/`.

---------------------------------------------------------------------------------------------

## 8. Pitfalls

1. **Silent mock fallback.** If Qwen fails to load (wrong path, OOM) the loop keeps running with a *mock*
   actor/critic and still writes a USD. Always pass `--require_vlm` (the dataset driver does), and check
   `actor_meta.source == "qwen3vl"` in the log. `mock_parse_failed`/`qwen3vl_retry` in a few iterations
   is normal (5 of 501 on the released run).
2. **One VLM process at a time.** A second 63 GB instance causes OOM / CPU-offload crawl.
   The driver waits while another `run_auto_construction` is running.
3. **Meshy is one-shot.** A fish with a `mesh.glb` is never resubmitted; a failed/rejected fish is
   flagged, not retried. Meshy share links are public: never commit keys or links.
4. **The mesh critic passes bad meshes** (reported in `deploy/README_H200.md`: 5 of 80 first-pass meshes were the tank
   or a glare sliver and still passed).
   Render a contact sheet (`render_mesh_views.py`, `make_dataset_sheet.py`) and look before spending VLM time.
5. **Stage 1 is a report, not a gate** (0/80 pass).
6. **The repair pass must stay ON** (`--no_repair` is for debugging only): the VLM loop alone converged for
   1/80 fish.
7. **Blender is pinned to 5.0.1.**
8. **Brook trout ids are not contiguous** (010 and 018 are missing). Never `seq 1 21`; enumerate files.
9. **Critic image order.** The pipeline passes `[top, front, side]` to both VLMs, but the critic prompt describes the
   images as "(1) side, (2) head-on, (3) top". Left as is (results and the actor's training depend on the order);
   keep in mind when reading critic feedback.
10. **`accepted` comes from the model.** `Critic._normalize` uses the model's own `accepted` field
    (falling back to `score >= 4` only when absent); it does not recompute it.
11. **Train/inference render mismatch** for the actor (§5 caveat).
12. Two runs of the same fish can differ: the actor is greedy, but it samples at temperature 0.7 after a
    detected add/remove cycle, and Blender/BVH results depend on the exact mesh.
13. Spine conversion needs a donor asset in `--dataset` (§4.8).
14. Raw phase-A/B dirs and `ASSET_OUT` can grow to tens of GB (`_work/` renders + `.usda` ≈ 8 MB per fish).

---------------------------------------------------------------------------------------------

## 9. Paper vs code (checked against the code and released logs)

| topic | paper / docs / older text | code / data | note |
|---|---|---|---|
| joint preset | 3-DoF D6, ±45°, PD kp 120, kd 6 | USD: 3-rot D6, **±15°, no drive** | limits + drive applied at RL load (§4.6) |
| actor LoRA | rank 8, 4 epochs, lr 1e-4 | released dataset used **r16/α32, 8 epochs** (v2) | §5; v1 is the paper setting |
| critic accept | score ≥ 4 accepts | model's own `accepted` flag is trusted | §8.10 |
| loop convergence | loop yields accepted articulation | 1/80 accepted in loop; **repair pass** gives 78/80 | §3 |
| iterations | — | dataset run 6 (driver default), `run_auto_construction` CLI default 10, `run_pipeline` default 30 | use 6 |
| cycle handling | (not described) | temperature 0.7 after 4 alternating iterations | §3 |
| bone mass | volume-proportional (docstring intent) | ellipsoid total ×1000 kg/m³, split by **box** volume; `add_isaac_physics` equal split is overwritten | §4.4 |
| Meshy cost | ~15 credits in old docstrings | **30 credits** measured | docstrings fixed |
| stage-1 | verification | all 80 fish fail coverage ≥ 90 % | report only |
| joint config in `Bone` | per-bone joint params (early design) | removed; always the fixed preset | `articulation.Bone` |
| render style | — | SFT images ≠ inference views | §5 caveat |
| template library | 14 hand-built fish | 14 `.blend` + `skeleton.json` in `data/fish_skeleton_dataset/` | identical to the originals (byte-checked) |
| FEM cook | hex 10, E 1e5, ν 0.45 | as coded (§4.5) | needs Isaac |
| tank size calibration | ±15–18 % | `measure_fish_size.py` | brook trout px/in estimated |

---------------------------------------------------------------------------------------------

## 10. Layout of this folder

```
v2sf_paths.py                       single source of truth for paths / interpreters
env.sh                              example environment
requirements.txt                    VLM python deps (exact versions used)
video2swimfish/articulation/        actor, critic, articulation (+ Blender-side apply/repair), renderer,
                                    geometric_verification, verify_stage1, run_auto_construction,
                                    skeleton_template, extract_centerline
video2swimfish/finetune/            build_sft_dataset.py, train_actor_lora.py
simfishlib/inference/qwen3vl.py     Qwen3VLClient (device_map="auto", PEFT adapter, _update_offload patch)
fish_asset_pipeline/scripts|configs Blender/USD chain, template index, physics defaults
data/fish_skeleton_dataset/         14 hand-built fish (.blend + skeleton.json + index.json)
data/actor_sft_v2/                  v2 provenance: adapter_config, train_log, SFT labels
scripts/                            drivers and stage scripts (see §2)
deploy/                             preflight.py, run_species_e2e.sh, make_species_dir.py, README_H200.md
```
