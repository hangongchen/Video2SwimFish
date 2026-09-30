# Running the pipeline on a fresh big-GPU machine (H200 notes)

The full documentation is in the top-level `README.md`; this page only keeps the
machine-setup checklist and the facts that were measured while preparing the trout species
(`brook_trout`: 21 fish, ids 001–009/011–017/019–023; `brown_trout`: 20 fish, ids 001–020).

```bash
git clone <this repo> && cd vlm_based_articulation_generation
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# raw videos (13 GB) from the HF dataset
hf download video2swimfish/video2swimfish-dataset --repo-type dataset \
    --include "data/raw_videos/brook_trout/*" "data/raw_videos/brown_trout/*" --local-dir /tmp/v2sf_raw
mkdir -p "raw_datasets/Brook Trout" "raw_datasets/Brown Trout"
cp /tmp/v2sf_raw/data/raw_videos/brook_trout/* "raw_datasets/Brook Trout/"
cp /tmp/v2sf_raw/data/raw_videos/brown_trout/* "raw_datasets/Brown Trout/"

python scripts/download_qwen3_vl.py            # -> models/Qwen3-VL-32B-Instruct (63 GB)
# copy actor_lora_v2 (547 MB, NOT on HF, NOT in git) into checkpoints/actor_lora_v2/

export BLENDER_BIN=/opt/blender-5.0.1-stable+v50.*/blender      # EXACTLY 5.0.1
export VLM_PYTHON=/opt/conda/envs/v2sf/bin/python
# export ISAAC_PYTHON=...                                       # optional; leave unset to skip the FEM cook
source env.sh
$VLM_PYTHON deploy/preflight.py --species brook_trout,brown_trout --fish 41

deploy/run_species_e2e.sh brook_trout --phases 012      # everything free
deploy/run_species_e2e.sh brook_trout --phases ABCF     # Meshy (30 credits/fish) -> skeleton -> USD -> spine
```

Box → spine needs a shape donor: an existing `K_final.usd` that contains the prim
`lower_curved_triangular_fish_spine_07_mesh` (default tag `catfish_fish001`, override with
`TEMPLATE_FROM=<tag>`). The released asset is in the HF dataset under
`data/usd_assets/catfish_fish001/K_final.usd`; put it at `dataset/catfish_fish001/K_final.usd`.

Facts measured on the source machine
* 41 trout ≈ 1230 Meshy credits (30 per fish).
* Historically 5 of 80 first-pass meshes were the tank or a glare sliver and the VLM mesh critic **passed** them;
  eyeball a render sheet before spending VLM time on skeletons.
* Brook trout ids 010 and 018 are missing; never `seq 1 21`.
* Tank calibration is per species (`scripts/measure_fish_size.py`): brown trout 89.0 px/in (measured);
  **brook trout could not be measured** (right tank wall too low-contrast, the automatic 78.7 is wrong), 89.8 (the
  five-species median) is used and flagged — re-measure by hand if precision matters.
* Blender renders the views with EEVEE: a cloud box without EGL/OpenGL will fail in `renderer.py`.
