#!/usr/bin/env python
"""Video2SwimFish over the WHOLE measured dataset: stereo video -> controllable fish asset,
resumable, one Meshy call per fish maximum.

Phases (each skips fish whose outputs already exist):
  A  mesh:      run_pipeline.py --stop_after_mesh --skip_critic  (Meshy image-to-3D; SPENDS CREDITS)
                then clean_mesh_components.py (drops a second, separate fish body).
                A fish whose mesh already exists is NEVER re-submitted (credit limit).
  B  a2c:       run_auto_construction.py --batch (ONE Qwen3-VL load for all fish):
                mesh critic + actor-critic skeleton + repair pass + physics/USD/FEM export
                -> ASSET_OUT/<tag>/K_final.usd
  C  stage1:    verify_stage1.py per fish (a report, not a gate)
  D  hydro:     panel hydro proxy per fish (needs Isaac Sim + the sibling swimming_policy_training
                repo)  -> ASSET_OUT/<tag>/panel_hydro.npz.   OPTIONAL, skipped if unavailable.
  E  pca:       top-view curvature extraction (this repo) -> per-fish PCA basis + reference
                trajectory (scripts live in the sibling swimming_policy_training repo).
                OPTIONAL, skipped if unavailable; not needed for the skeleton/USD half.
Species without fish_sizes.json / canonical frames are skipped with a note.

Paths come from v2sf_paths.py (env vars: ASSET_OUT, MESH_OUT, RAW_VIDEO_ROOT,
SPECIES_MANIFOLD_ROOT, ACTOR_LORA, VLM_PYTHON, BLENDER_BIN, ISAAC_PYTHON ...).

Usage:  python scripts/run_dataset_v2sf.py --species catfish,lake_sturgeon --phases ABC
Progress: <run_dir>/status.json + status.log   (default run_dir: ROOT/dataset_run)
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
OUT = P.MESH_OUT                # meshes (run_for_fish reads MESH_OUT/<tag>/mesh.glb)
ASSET = P.ASSET_OUT             # final controllable assets: ASSET_OUT/<tag>/K_final.usd
RUN = P.ROOT / "dataset_run"
STATUS = RUN / "status.json"
LOG = RUN / "status.log"

QWEN_PY = P.VLM_PYTHON
ISAAC_PY = P.ISAAC_PYTHON       # optional (phase D only)
ADAPTER = P.ACTOR_LORA
SCRIPTS = P.SCRIPTS_DIR
SWIM = P.POLICY_ROOT     # sibling repo that owns hydro + PCA scripts (phases D, E)


def log(msg):
    RUN.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_status():
    return json.loads(STATUS.read_text()) if STATUS.exists() else {}


def save_status(st):
    RUN.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(st, indent=1, default=str))


def sh(cmd, log_path, timeout=None):
    with open(log_path, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n")
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return r.returncode


def fish_list(species_list):
    fish = []
    for sp in species_list:
        sizes = P.SPECIES_MANIFOLD_ROOT / sp / "fish_sizes.json"
        frames = P.RAW_VIDEO_ROOT / f"{sp}_canonical_frames_cropped"
        if not sizes.exists() or not frames.exists():
            log(f"SKIP species {sp}: missing {'fish_sizes.json' if not sizes.exists() else 'canonical frames'}")
            continue
        per = json.loads(sizes.read_text()).get("per_fish", {})
        for num in sorted(per):
            img = frames / f"{sp}_fish{num}_canonical.jpg"
            if not img.exists():
                log(f"SKIP {sp}_fish{num}: no canonical frame")
                continue
            fish.append({"tag": f"{sp}_fish{num}", "species": sp, "num": num, "image": str(img),
                         "suspect_size": bool(per[num].get("suspect"))})
    return fish


def phase_A(fish, st):
    for f in fish:
        tag = f["tag"]
        out = OUT / tag
        rec = st.setdefault(tag, {})
        if (out / "mesh.glb").exists():
            rec.setdefault("mesh", "exists")
            continue
        if rec.get("mesh") in ("submitted_failed", "critic_rejected"):
            continue  # never a second Meshy call
        log(f"[A] {tag}: Meshy + critic")
        rc = sh([QWEN_PY, SCRIPTS / "run_pipeline.py", "--image", f["image"], "--tag", tag,
                 "--out_dir", out, "--species", f["species"], "--fish_id", f["num"], "--stop_after_mesh", "--skip_critic"],
                RUN / f"{tag}_A.log", timeout=3600)
        if (out / "mesh.glb").exists():
            rec["mesh"] = "ok" if rc == 0 else "ok_rc%d" % rc
            # drop a second, spatially separate body (two fish in the canonical frame -> Meshy
            # reconstructs both; catfish_fish005 got its skeleton laid across two fish)
            rc2 = sh([QWEN_PY, SCRIPTS / "clean_mesh_components.py", "--glb", out / "mesh.glb"],
                     RUN / f"{tag}_A.log", timeout=900)
            crep = out / "mesh_clean_report.json"
            if crep.exists():
                cr = json.loads(crep.read_text())
                rec["mesh_clean"] = {"n_components": cr.get("n_components"), "dropped_frac": round(cr.get("dropped_area_frac", 0), 3),
                                     "rewritten": cr.get("rewritten")}
                if cr.get("rewritten"):
                    log(f"[A] {tag}: mesh had a separate second body -> dropped {cr['dropped_area_frac']:.1%} of surface")
        else:
            rec["mesh"] = "submitted_failed"
        log(f"[A] {tag}: {rec['mesh']}")
        save_status(st)


def phase_B(fish, st, max_iter, limit=4):
    """One a2c wave: up to `limit` fish that have a mesh but no asset yet (small waves so the
    downstream phases run between them and finished fish are complete early)."""
    for _once in (0,):
        todo = [f["tag"] for f in fish if (OUT / f["tag"] / "mesh.glb").exists()
                and not (ASSET / f["tag"] / "K_final.usd").exists()
                and not (ASSET / f["tag"] / "FAILED.txt").exists()][:limit]
        if not todo:
            return
        log(f"[B] a2c wave over {len(todo)} fish (shared VLM): {todo}")
        rc = sh([QWEN_PY, V2SF / "video2swimfish/articulation/run_auto_construction.py", "--batch", ",".join(todo),
                 "--max_iterations", str(max_iter), "--output_dir", ASSET, "--actor_checkpoint", ADAPTER, "--require_vlm"],
                RUN / "B_a2c_batch.log")
        for tag in todo:
            st.setdefault(tag, {})["a2c"] = "ok" if (ASSET / tag / "K_final.usd").exists() else "failed"
        save_status(st)
        log(f"[B] wave exit {rc}; done: {sum(1 for t in todo if st[t]['a2c']=='ok')}/{len(todo)}")


def phase_C(fish, st):
    for f in fish:
        tag = f["tag"]
        usd = ASSET / tag / "K_final.usd"
        if not usd.exists() or (ASSET / tag / "stage1_report.json").exists():
            continue
        sh([P.blender_bin(), "-b", "--python", V2SF / "video2swimfish/articulation/verify_stage1.py", "--",
            "--usd", usd, "--fish_id", tag], RUN / f"{tag}_C.log", timeout=900)
        rep = ASSET / tag / "stage1_report.json"
        if rep.exists():
            r = json.loads(rep.read_text())
            st.setdefault(tag, {})["stage1"] = {
                "contain": r["check1_interpenetration"]["passed"],
                "coverage": r["check2_coverage"].get("coverage_frac"),
                "align_mean": r["check3_alignment"]["mean_angle_deg"],
                "align_max": r["check3_alignment"]["max_angle_deg"]}
            log(f"[C] {tag}: {st[tag]['stage1']}")
        save_status(st)


def phase_D(fish, st):
    """Optional: panel-hydro proxy (Isaac Sim). Script lives in the sibling swimming_policy_training repo."""
    script = SWIM / "scripts" / "precompute_catfish001_panels.py"
    if not ISAAC_PY or not script.exists():
        log(f"[D] SKIPPED: needs ISAAC_PYTHON and {script} (sibling swimming_policy_training repo)")
        return
    for f in fish:
        tag = f["tag"]
        usd = ASSET / tag / "K_final.usd"
        npz = ASSET / tag / "panel_hydro.npz"
        if not usd.exists() or npz.exists():
            continue
        log(f"[D] {tag}: panel hydro")
        env = {"FISH_GEN_DIR": str(ASSET / tag), "FISH_MESH_TOKEN": "Mesh_0"}
        # 2026-09-27: voxel = BL/30 (was a fixed 1 cm -> only 28-38 panels on 6-9 cm fish, spurious glide lift/torque;
        # the 31 cm catfish had 542). ~300-1000 panels for every fish.
        try:
            _bl = float(json.loads((ASSET / tag / "K_final.mass_report.json").read_text())["fish_length_m"])
            _vox = f"--voxel={_bl / 30:.4f}"
        except Exception:
            _vox = "--voxel=0.0030"
        cmd = ["env"] + [f"{k}={v}" for k, v in env.items()] + [ISAAC_PY, script, _vox]
        rc = sh(cmd, RUN / f"{tag}_D.log", timeout=1200)
        st.setdefault(tag, {})["hydro"] = "ok" if npz.exists() else f"failed rc={rc}"
        save_status(st)


def phase_E(fish, st):
    """Optional: top-view curvature -> PCA basis + reference. All three scripts live in the sibling
    swimming_policy_training repo (scripts/zef_manifold/); their output goes to ITS data/species_manifold."""
    zm = SWIM / "scripts" / "zef_manifold"
    need = [zm / "extract_species_curvature.py", zm / "fit_pca_species.py", zm / "extract_species_reference.py"]
    if not all(p.exists() for p in need):
        log(f"[E] SKIPPED: needs {zm}/(extract_species_curvature|fit_pca_species|extract_species_reference).py "
            f"(sibling swimming_policy_training repo; set POLICY_ROOT)")
        return
    for f in fish:
        tag, sp, num = f["tag"], f["species"], f["num"]
        man = SWIM / "data" / "species_manifold" / f"{sp}_fix" / "top"
        basis = man / f"fish{num}_pca_basis.npz"
        if basis.exists():
            continue
        video = P.RAW_VIDEO_ROOT / sp / f"top{num}.mp4"
        if not video.exists():
            log(f"[E] {tag}: no top video")
            continue
        log(f"[E] {tag}: curvature (stride 2) + PCA + reference")
        rc = sh([QWEN_PY, need[0], "--video", video,
                 "--species", f"{sp}_fix", "--fish_id", num, "--view", "top",
                 # 2026-09-25 (PCA_INVESTIGATION_white_bass_fish008.md): body-length-relative kernels, a CONTRAST
                 # (ratio) threshold instead of an absolute diff (absolute 60 found no fish in the shaded tank end,
                 # 25 admitted the shadow -> |kBL| up to 250), and per-frame head/tail orientation (the frame-0-only
                 # decision tracked whole runs tail-first). Old: --sigma 7.0 --thresh 60 --open_px 9.
                 "--sigma_bl", "0.015", "--open_bl", "0.019", "--ratio_thresh", "0.6", "--open_px", "9",
                 "--stride", "2", "--max_frames", "3000"], RUN / f"{tag}_E.log", timeout=3600)
        curv = man / f"fish{num}_curvature.npz"
        if not curv.exists():
            st.setdefault(tag, {})["pca"] = f"extract failed rc={rc}"
            save_status(st)
            continue
        # quality gate on the curvature scale (real fish |kappa*BL| p99 ~ 3; shadow/barbel noise -> 60+)
        chk = subprocess.run([QWEN_PY, "-c", f"""
import numpy as np; d=np.load('{curv}'); k=d['kappa_bl']; v=d['valid']; m=np.abs(k).max(1)
print(int(v.sum()), round(float(np.percentile(np.abs(k[v]),99)),2), int((v&(m<4)).sum()))"""],
                             capture_output=True, text=True)
        n_valid, p99, n_clean = chk.stdout.split()
        sh([QWEN_PY, need[1], "--species", f"{sp}_fix", "--fish_id", num,
            "--view", "top", "--max_kbl", "4", "--resample_fps", "60"], RUN / f"{tag}_E.log")
        sh([QWEN_PY, need[2], "--species", f"{sp}_fix",
            "--view", "top"], RUN / f"{tag}_E.log", timeout=1800)
        st.setdefault(tag, {})["pca"] = {"valid_frames": int(n_valid), "kbl_p99": float(p99),
                                          "clean_frames": int(n_clean), "basis": basis.exists()}
        log(f"[E] {tag}: {st[tag]['pca']}")
        save_status(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="catfish,lake_sturgeon")
    ap.add_argument("--phases", default="ABC",
                    help="subset of ABCDE (D=panel hydro, E=PCA need the sibling swimming_policy_training repo)")
    ap.add_argument("--a2c_max_iterations", type=int, default=6)
    ap.add_argument("--wave_size", type=int, default=4)
    ap.add_argument("--run_dir", default=None,
                     help="status/log dir (default ROOT/dataset_run). Use a separate dir when a "
                          "second driver runs concurrently, so the two never overwrite each other's status.json")
    args = ap.parse_args()
    if args.run_dir:
        global RUN, STATUS, LOG
        RUN = Path(args.run_dir); RUN.mkdir(parents=True, exist_ok=True)
        STATUS = RUN / "status.json"; LOG = RUN / "status.log"

    ASSET.mkdir(parents=True, exist_ok=True)
    fish = fish_list(args.species.split(","))
    log(f"dataset run: {len(fish)} fish, phases {args.phases}")
    st = load_status()
    import threading
    a_thread = None
    if "A" in args.phases:
        a_thread = threading.Thread(target=phase_A, args=(fish, st), daemon=True)
        a_thread.start()

    def wait_for_other_a2c():
        # a previous driver's batch may still hold the shared VLM -- never start a second one
        while subprocess.run(["pgrep", "-f", "run_auto_[c]onstruction"], capture_output=True).returncode == 0:
            time.sleep(60)

    # phase E (video curvature -> PCA) is CPU-only and independent of the assets: run it in its
    # own thread so it never delays the GPU-bound a2c waves
    e_thread = None
    if "E" in args.phases:
        e_thread = threading.Thread(target=phase_E, args=(fish, st), daemon=True)
        e_thread.start()

    def downstream():
        if "C" in args.phases:
            phase_C(fish, st)
        if "D" in args.phases:
            phase_D(fish, st)

    # interleave: finished fish get Stage 1 / hydro / PCA right away, then the next a2c wave
    while True:
        downstream()
        if "B" not in args.phases:
            break
        wait_for_other_a2c()
        todo = [f["tag"] for f in fish if (OUT / f["tag"] / "mesh.glb").exists()
                and not (ASSET / f["tag"] / "K_final.usd").exists()
                and not (ASSET / f["tag"] / "FAILED.txt").exists()]
        if not todo and (a_thread is None or not a_thread.is_alive()):
            break
        if todo:
            phase_B(fish, st, args.a2c_max_iterations, limit=args.wave_size)
        else:
            time.sleep(120)
    if a_thread is not None:
        a_thread.join()
    downstream()
    if e_thread is not None:
        e_thread.join()
    log("ALL PHASES DONE")


if __name__ == "__main__":
    main()
