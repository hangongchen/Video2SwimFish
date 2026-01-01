#!/usr/bin/env python
"""Video2SwimFish: one cropped canonical fish photo -> Meshy mesh -> (optionally) one articulated fish USD.

WHICH PART OF THIS FILE IS THE PAPER PATH
  * The ONLY path used for the paper's dataset is
        run_pipeline.py --image <crop.jpg> --tag <species>_fish<NNN> --out_dir <MESH_OUT>/<tag> \
                        --species <species> --fish_id <NNN> --stop_after_mesh --skip_critic
    i.e. Meshy image-to-3D and nothing else (scripts/run_dataset_v2sf.py phase A calls exactly
    this; the mesh critic and the skeleton are run afterwards by run_auto_construction.py).
  * `--skeleton a2c` (default when NOT stopping after the mesh) just chains into
    video2swimfish/articulation/run_auto_construction.py for a single fish -- convenient, but
    the dataset driver batches fish through ONE shared Qwen load instead.
  * `--skeleton auto` is a LEGACY path (template-matched deterministic auto-skeleton, steps
    0.5-8 below). It is kept for reference only and additionally needs scripts that are NOT
    shipped in this package (vlm_articulation.py, import_mesh_to_blend.py); it will exit with
    an explanatory message.

Original description (full chain):

    cropped canonical frame (Qwen select + classical-CV crop, already built)
      -> Meshy image-to-3D API (textured mesh, GLB)              [COSTS CREDITS]
      -> import GLB into a fresh Blender file
      -> Qwen3-VL body-ratio estimate from the SOURCE PHOTO       (compact params only,
                                                                    never writes USD -- see
                                                                    fish_asset_pipeline's own
                                                                    design rule)
      -> fish_asset_pipeline's existing DETERMINISTIC steps, unchanged in spirit, just called
         directly here instead of via generate_single_fish_usd_textured.sh so the VLM ratios
         can be threaded into step 1:
           [1] auto-skeleton fit (template-matched, VLM ratios override the bbox heuristic)
           [2] scale to target real-world length
           [3] extract skeleton.json
           [4] export USD WITH textures
           [5] author Isaac physics (bones, joints, attachments, mass)
           [6] convert USDA -> binary USD
           [7] FEM deformable cook (Isaac Lab python)
      -> <out_dir>/fish_articulated.usd

Usage:
  python run_pipeline.py --image /path/to/cropped_canonical.jpg --tag catfish_fish001 \
      --out_dir outputs/catfish_fish001 --target_length_m 0.5

Each call to Meshy costs 30 credits (measured: consumed_credits=30 in every meshy_task.json) -- this script generates a mesh EXACTLY ONCE per
invocation. Re-running with --skip_meshy and an already-downloaded <MESH_OUT>/<tag>/mesh.glb
re-runs only the free deterministic steps (useful for iterating on the VLM/skeleton fit
without spending more credits).

REAL-WORLD SIZING: pass --species and --fish_id (matching the ids used by
measure_fish_size.py, e.g. --species catfish --fish_id 001) instead of --target_length_m to
look up that fish's own tank-wall-calibrated measured length from
<SPECIES_MANIFOLD_ROOT>/<species>/fish_sizes.json and scale the USD to ITS real size,
rather than a generic default. Run measure_fish_size.py first if that file doesn't exist yet.
An explicit --target_length_m always overrides the lookup.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2sf_paths as P  # noqa: E402

V2SF = P.ROOT
SCRIPTS = P.SCRIPTS_DIR
PIPE_SCRIPTS = P.PIPE_SCRIPTS
CFG = P.USD_PHYSICS_CFG
TEMPLATE_INDEX = P.TEMPLATE_INDEX
QWEN_PY = Path(P.VLM_PYTHON)

sys.path.insert(0, str(SCRIPTS))
from meshy_client import MeshyClient  # noqa: E402


def run(cmd: list[str], log_path: Path, step: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if r.returncode != 0:
        tail = log_path.read_text()[-3000:]
        raise RuntimeError(f"[{step}] FAILED (exit {r.returncode}, {dt:.1f}s) -- see {log_path}\n"
                            f"--- tail ---\n{tail}")
    print(f"  [{step}] ok ({dt:.1f}s) -- log: {log_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="cropped canonical fish photo")
    ap.add_argument("--tag", required=True, help="short id, used for filenames/logs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_length_m", type=float, default=None,
                     help="explicit override; if omitted, looked up from --species/--fish_id's "
                          "measured real size (measure_fish_size.py), falling back to 0.5m")
    ap.add_argument("--species", default=None, help="for real-size lookup, e.g. catfish")
    ap.add_argument("--fish_id", default=None, help="for real-size lookup, e.g. 001")
    ap.add_argument("--hex_res", type=int, default=10)
    ap.add_argument("--skip_meshy", action="store_true",
                     help="reuse an already-downloaded <out_dir>/mesh.glb instead of calling the API")
    ap.add_argument("--skip_vlm", action="store_true", help="skip the VLM ratio step (bbox heuristic only)")
    ap.add_argument("--skeleton", choices=["auto", "a2c"], default="a2c",
                    help="a2c (default): VLM actor-critic articulation construction "
                         "(video2swimfish/articulation/run_auto_construction.py: fine-tuned actor, "
                         "pretrained critic with measured midline facts, cycle escape, deterministic "
                         "repair pass, then the same physics/USD/FEM chain). auto: the original "
                         "template-matched deterministic auto-skeleton (steps 1-8 below).")
    ap.add_argument("--a2c_max_iterations", type=int, default=30)
    ap.add_argument("--a2c_actor_checkpoint",
                    default=str(P.ACTOR_LORA))
    ap.add_argument("--stop_after_mesh", action="store_true",
                    help="run only Meshy + mesh critic (used by the dataset driver to batch the VLM stage)")
    ap.add_argument("--skip_critic", action="store_true",
                     help="skip the VLM mesh-defect critic gate (proceed even on a rejected mesh)")
    args = ap.parse_args()

    target_length_m = args.target_length_m
    size_source = "explicit --target_length_m"
    if target_length_m is None and args.species and args.fish_id:
        sizes_path = P.SPECIES_MANIFOLD_ROOT / args.species / "fish_sizes.json"
        if sizes_path.exists():
            sizes = json.loads(sizes_path.read_text())
            entry = sizes.get("per_fish", {}).get(args.fish_id)
            if entry and not entry.get("suspect"):
                target_length_m = entry["length_m"]
                size_source = (f"measured from {args.species}/front (tank-wall calibrated, "
                                f"{entry['length_in']:.2f}in, +/-{sizes['calibration']['uncertainty_frac']*100:.0f}%)")
            else:
                print(f"[run_pipeline] WARNING: no usable (non-suspect) measured size for "
                      f"{args.species}/fish{args.fish_id} in {sizes_path}", flush=True)
        else:
            print(f"[run_pipeline] WARNING: {sizes_path} not found -- run measure_fish_size.py "
                  f"first for real-size lookup", flush=True)
    if target_length_m is None:
        target_length_m = 0.5
        size_source = "DEFAULT (no measurement available)"
    args.target_length_m = target_length_m
    print(f"[run_pipeline] target_length_m = {target_length_m:.4f} m  (source: {size_source})", flush=True)

    out_dir = Path(args.out_dir).resolve()
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    glb_path = out_dir / "mesh.glb"
    report: dict = {"tag": args.tag, "image": str(args.image)}

    # ---- 0. Meshy image-to-3D (COSTS CREDITS unless --skip_meshy) ----
    if args.skip_meshy:
        assert glb_path.exists(), f"--skip_meshy given but {glb_path} does not exist"
        print(f"[0/8] SKIPPED Meshy call, reusing {glb_path}", flush=True)
    else:
        print(f"[0/8] Meshy image-to-3D for {args.image} ...", flush=True)
        client = MeshyClient()
        bal_before = client.balance()

        def on_progress(task_id, status, progress):
            print(f"    meshy task {task_id}: {status} {progress}%", flush=True)

        task = client.generate_and_download(args.image, str(out_dir), tag="mesh", on_progress=on_progress)
        (out_dir / "meshy_task.json").write_text(json.dumps(task, indent=2, default=str))
        if task.get("status") != "SUCCEEDED":
            raise RuntimeError(f"Meshy task did not succeed: status={task.get('status')} "
                                f"task_error={task.get('task_error')}")
        bal_after = client.balance()
        report["meshy_credits_spent"] = bal_before - bal_after
        report["meshy_task_id"] = task.get("id")
        print(f"  Meshy done: {glb_path}  (credits spent: {report['meshy_credits_spent']}, "
              f"balance now {bal_after})", flush=True)

    # ---- 0.6. VLM CRITIC: does the generated mesh actually look like an intact fish? ----
    # Catches Meshy reconstruction defects (e.g. a non-watertight mesh with a visible hole in
    # the skin -- found in catfish_fish001, confirmed NOT a skeleton/bone clipping issue by
    # rendering the skin mesh alone) BEFORE spending time on skeleton fitting / physics / FEM
    # cook for a broken asset. Actor = the deterministic Meshy+fish_asset_pipeline generation;
    # critic = this VLM visual QC gate -- same generate-then-verify pattern already used for
    # the canonical-frame crop step.
    if not args.skip_critic:
        print("[0.6/8] VLM critic: checking generated mesh for defects ...", flush=True)
        critic_json = out_dir / "mesh_critic.json"
        try:
            run([str(QWEN_PY), str(SCRIPTS / "verify_mesh_vlm.py"),
                 "--glb", str(glb_path), "--out_json", str(critic_json)],
                work / "step0_6_critic.log", "vlm-critic")
            critic_passed = True
        except RuntimeError:
            critic_passed = False
        critic_result = json.loads(critic_json.read_text()) if critic_json.exists() else {}
        report["mesh_critic"] = critic_result
        if not critic_result.get("passed", critic_passed):
            raise RuntimeError(
                f"VLM critic REJECTED the generated mesh -- {critic_result.get('raw_reply', '(no detail)')}\n"
                f"See {critic_json}. Re-run with --skip_critic to proceed anyway, or regenerate "
                f"the mesh (a fresh Meshy call, e.g. drop --skip_meshy) to try for a clean reconstruction.")
        print(f"  critic: PASS", flush=True)
    else:
        print("[0.6/8] SKIPPED VLM critic (--skip_critic)", flush=True)

    if args.stop_after_mesh:
        (out_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"[run_pipeline] --stop_after_mesh: mesh + critic done for {args.tag}", flush=True)
        return

    if args.skeleton == "auto":
        missing = [n for n in ("vlm_articulation.py", "import_mesh_to_blend.py") if not (SCRIPTS / n).exists()]
        if missing:
            raise SystemExit(f"--skeleton auto is a legacy path and needs {missing} which are not part of this "
                             f"package; use --skeleton a2c (default) or --stop_after_mesh.")

    if args.skeleton == "a2c":
        # ---- 1-8 (a2c): actor-critic construction + repair + export chain, one call ----
        print("[1-8/8] VLM actor-critic articulation (a2c) ...", flush=True)
        a2c_cmd = [str(QWEN_PY), str(V2SF / "video2swimfish" / "articulation" / "run_auto_construction.py"),
                   "--fish_id", args.tag, "--max_iterations", str(args.a2c_max_iterations),
                   "--output_dir", str(out_dir), "--actor_checkpoint", str(args.a2c_actor_checkpoint)]
        run(a2c_cmd, work / "step_a2c.log", "a2c")
        final_usd = out_dir / "K_final.usd"
        assert final_usd.exists(), f"a2c did not produce {final_usd} -- see {work / 'step_a2c.log'}"
        report["skeleton"] = "a2c"
        report["a2c_log"] = str(out_dir / "articulation_construction_log.json")
        report["target_length_m"] = target_length_m
        report["target_length_source"] = size_source
        report["final_usd"] = str(final_usd)
        (out_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"\nDONE: {final_usd}", flush=True)
        return

    # ---- 0.5. VLM body-ratio estimate from the SOURCE photo (compact params only) ----
    vlm_lhr = vlm_ltr = None
    if not args.skip_vlm:
        print("[0.5/8] Qwen3-VL body-ratio estimate ...", flush=True)
        vlm_json = out_dir / "vlm_ratios.json"
        run([str(QWEN_PY), str(SCRIPTS / "vlm_articulation.py"),
             "--image", str(args.image), "--out_json", str(vlm_json)],
            work / "step0_5_vlm.log", "vlm")
        vlm_result = json.loads(vlm_json.read_text())
        vlm_lhr = vlm_result.get("length_height_ratio")
        vlm_ltr = vlm_result.get("length_thickness_ratio")
        report["vlm"] = vlm_result
        print(f"  VLM ratios: length_height={vlm_lhr}, length_thickness={vlm_ltr}", flush=True)
    else:
        print("[0.5/8] SKIPPED VLM step (--skip_vlm)", flush=True)

    # ---- 1. import GLB -> fresh .blend ----
    BLENDER, BPY = P.blender_bin(), P.blender_py()
    ISAAC_PY = P.ISAAC_PYTHON
    print("[1/8] import GLB -> blend ...", flush=True)
    raw_blend = work / "raw_mesh.blend"
    run([str(BLENDER), "-b", "--python", str(SCRIPTS / "import_mesh_to_blend.py"), "--",
         "--glb", str(glb_path), "--output", str(raw_blend)],
        work / "step1_import.log", "import")

    # ---- 2. auto-skeleton fit (template-matched, VLM overrides) ----
    print("[2/8] auto-skeleton fit ...", flush=True)
    auto_dir = out_dir / "_auto"
    auto_skel_out = auto_dir / "auto_skeleton.blend"
    skel_cmd = [str(BLENDER), "-b", str(raw_blend), "--python",
                str(PIPE_SCRIPTS / "generate_auto_skeleton_blend.py"), "--",
                "--template-index", str(TEMPLATE_INDEX),
                "--target-length-m", str(args.target_length_m),
                "--output", str(auto_skel_out)]
    if vlm_lhr is not None:
        skel_cmd += ["--vlm-length-height-ratio", str(vlm_lhr)]
    if vlm_ltr is not None:
        skel_cmd += ["--vlm-length-thickness-ratio", str(vlm_ltr)]
    run(skel_cmd, work / "step2_autoskel.log", "auto-skeleton")
    skb = auto_skel_out / "auto_skeleton.blend"
    assert skb.exists(), f"expected {skb}"

    # ---- 3. scale to target length ----
    print("[3/8] scale ...", flush=True)
    scaled_blend = work / "scaled.blend"
    run([str(BLENDER), "-b", str(skb), "--python",
         str(PIPE_SCRIPTS / "scale_skeleton_blend_to_real_length.py"), "--",
         "--target-length-m", str(args.target_length_m), "--output", str(scaled_blend)],
        work / "step3_scale.log", "scale")

    # ---- 4. extract skeleton.json ----
    print("[4/8] extract skeleton.json ...", flush=True)
    skel_json_dir = work / "skeleton_json"
    run([str(BLENDER), "-b", str(scaled_blend), "--python",
         str(PIPE_SCRIPTS / "extract_fish_skeleton.py"), "--", "--output", str(skel_json_dir)],
        work / "step4_extract.log", "extract-skeleton")
    skj = skel_json_dir / "skeleton.json"
    assert skj.exists(), f"expected {skj}"

    # ---- 5. export USD with textures ----
    print("[5/8] export USD with textures ...", flush=True)
    usd_work = work / "usd"
    base_usda = usd_work / "base_export.usda"
    run([str(BLENDER), "-b", str(scaled_blend), "--python",
         str(PIPE_SCRIPTS / "export_blend_to_usd.py"), "--", "--output", str(base_usda)],
        work / "step5_export.log", "export-usd")
    assert base_usda.exists(), f"expected {base_usda}"
    n_tex = base_usda.read_text().count("UsdUVTexture")
    report["textures_in_export"] = n_tex
    if n_tex == 0:
        print("  WARNING: 0 UsdUVTexture prims in the base export -- fish will render flat gray", flush=True)
    if (usd_work / "textures").exists():
        import shutil
        shutil.copytree(usd_work / "textures", out_dir / "textures", dirs_exist_ok=True)

    # ---- 6. author Isaac physics ----
    print("[6/8] Isaac physics (bones, joints, attachments, mass) ...", flush=True)
    final_usda = out_dir / "fish_articulated.usda"
    run([str(BPY), str(PIPE_SCRIPTS / "add_isaac_physics_to_usd.py"),
         "--input-usd", str(base_usda), "--skeleton-json", str(skj),
         "--config", str(CFG), "--output-usd", str(final_usda)],
        work / "step6_physics.log", "physics")
    assert final_usda.exists(), f"expected {final_usda}"

    # ---- 7. convert to binary USD ----
    print("[7/8] convert USDA -> binary USD ...", flush=True)
    final_usd = out_dir / "fish_articulated.usd"
    run([str(BPY), str(PIPE_SCRIPTS / "convert_usda_to_usd.py"),
         "--input", str(final_usda), "--output", str(final_usd)],
        work / "step7_convert.log", "convert")
    assert final_usd.exists(), f"expected {final_usd}"

    # ---- 8. FEM deformable cook (Isaac Lab python) ----
    print("[8/8] FEM cook ...", flush=True)
    assert ISAAC_PY, "ISAAC_PYTHON must be set for the FEM cook"
    run([str(ISAAC_PY), str(PIPE_SCRIPTS / "cook_deformable_isaac.py"),
         "--in", str(final_usd), "--out", str(final_usd), "--hex-resolution", str(args.hex_res)],
        work / "step8_cook.log", "fem-cook")

    report["target_length_m"] = target_length_m
    report["target_length_source"] = size_source
    report["final_usd"] = str(final_usd)
    report["final_usda"] = str(final_usda)
    (out_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nDONE: {final_usd}", flush=True)
    print(f"report: {out_dir / 'pipeline_report.json'}", flush=True)


if __name__ == "__main__":
    main()
