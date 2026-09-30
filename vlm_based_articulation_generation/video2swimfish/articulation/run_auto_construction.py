"""Main orchestrator for the VLM-based Actor-Critic Articulation Auto-Construction loop
(paper Section 3.2, Eq. 6 / the loop pseudocode):

    K^(0) = empty articulation (no bones)
    for k in range(max_iterations):
        a^(k) = Actor(M, K^(k), f^(k-1))
        K^(k+1) = Apply(K^(k), a^(k))
        K^(k+1) = GeometricVerification(K^(k+1), M)   # hard constraint
        (r^(k), f^(k)) = Critic(M, K^(k+1))
        if r^(k).accepted: K* = K^(k+1); break
    if not converged: K* = best K^(k) by score across all iterations

Usage:
    python run_auto_construction.py --fish_id catfish_fish001 --max_iterations 6 \
        --output_dir dataset/catfish_fish001 --actor_checkpoint checkpoints/actor_lora_v2
    # (reads MESH_OUT/<fish_id>/mesh.glb; --batch tag1,tag2 shares ONE Qwen load across fish)
    # NOTE: if the VLM fails to load the loop silently runs a MOCK actor/critic -- check
    # actor_meta.source == "qwen3vl" in articulation_construction_log.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))            # package root: v2sf_paths + simfishlib/
import v2sf_paths as P  # noqa: E402

from actor import Actor  # noqa: E402
from articulation import Articulation, export_articulation_to_usd  # noqa: E402
from critic import Critic  # noqa: E402
from renderer import render_views  # noqa: E402
import skeleton_template as st  # noqa: E402

MESH_OUT = P.MESH_OUT                      # meshes are read from MESH_OUT/<fish_id>/mesh.glb
QWEN_MODEL_PATH = P.QWEN_MODEL_PATH


def load_qwen_client(actor_checkpoint: str | None = None):
    """One shared client/model for BOTH actor and critic (running two separate Qwen3-VL-32B
    instances concurrently on this box caused CUDA allocation failures / CPU-offload
    slowdowns earlier this session -- the fix was always to serialize onto one loaded
    instance). When a fine-tuned actor_checkpoint exists, it is loaded as a PEFT LoRA
    adapter ON TOP of the shared base model: Actor.propose() enables the adapter layers
    before generating and Critic.evaluate() disables them, so the SAME model object serves
    the fine-tuned actor and the untouched pretrained critic (paper Sec 3.2: critic is
    "pretrained Qwen3-VL, no fine-tuning") without loading two separate 32B models.

    BUG FIXED HERE (found by a byte-for-byte identical actor output between a pretrained-only
    run and a "fine-tuned" run): this function used to construct Qwen3VLClient WITHOUT ever
    passing adapter_path, so --actor_checkpoint was accepted, threaded through to Actor for
    bookkeeping, and then silently never actually loaded -- inference always ran the base
    model. Verified fixed by direct A/B generation test (adapter enabled vs disabled) on the
    exact actor prompt/images: outputs now differ and the enabled-adapter output matches the
    fine-tuned response style (SFT reasoning-string phrasing, metric-scale position/size)."""
    try:
        from simfishlib.inference.qwen3vl import Qwen3VLClient
        if not QWEN_MODEL_PATH.exists():
            return None, f"model path missing: {QWEN_MODEL_PATH}"
        adapter_path = None
        if actor_checkpoint and Path(actor_checkpoint).exists():
            adapter_path = str(actor_checkpoint)
        client = Qwen3VLClient(model_path=str(QWEN_MODEL_PATH), adapter_path=adapter_path)
        client.load()
        return client, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


ACTOR_TRAIN_LENGTH_M = 0.25  # finetune/build_sft_dataset.py TARGET_LENGTH_M
CYCLE_ESCAPE_TEMPERATURE = 0.7  # actor-only sampling temperature once an add/remove cycle is detected


def scale_action(action: dict, k: float) -> dict:
    out = dict(action)
    if isinstance(out.get("position"), (list, tuple)) and len(out["position"]) == 3:
        out["position"] = [float(v) * k for v in out["position"]]
    if isinstance(out.get("size"), (list, tuple)) and len(out["size"]) == 3:
        out["size"] = [float(v) * k for v in out["size"]]
    return out


def scale_articulation_dict(d: dict, k: float) -> dict:
    return {"bones": [dict(b, position=[round(v * k, 5) for v in b["position"]],
                             size=[round(v * k, 5) for v in b["size"]]) for b in d.get("bones", [])]}


def apply_action_via_blender(in_blend, mesh_glb, action: dict, out_blend: Path, out_report: Path,
                               fish_id: str | None = None):
    cmd = [P.blender_bin(), "-b", "--python", str(HERE / "articulation.py"), "--"]
    if in_blend is not None:
        cmd += ["--in_blend", str(in_blend)]
    if mesh_glb is not None:
        cmd += ["--mesh_glb", str(mesh_glb)]
    if fish_id is not None:
        cmd += ["--fish_id", str(fish_id)]
    cmd += ["--action_json", json.dumps(action), "--out_blend", str(out_blend), "--out_report", str(out_report)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not out_report.exists():
        raise RuntimeError(
            f"apply_action_via_blender failed (exit {r.returncode}):\n"
            f"--- stdout (tail) ---\n{r.stdout[-3000:]}\n--- stderr (tail) ---\n{r.stderr[-3000:]}"
        )
    return json.loads(out_report.read_text())


def run_for_fish(fish_id: str, max_iterations: int, output_dir: str, actor_checkpoint: str | None = None,
                 client=None, repair: bool = True, require_vlm: bool = False) -> dict:
    mesh_glb = MESH_OUT / fish_id / "mesh.glb"
    if not mesh_glb.exists():
        raise FileNotFoundError(f"fish mesh not found: {mesh_glb}")

    out_dir = Path(output_dir)
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    load_error = None
    if client is None:
        client, load_error = load_qwen_client(actor_checkpoint)
    if client is None and require_vlm:
        raise RuntimeError(f"--require_vlm: Qwen3-VL failed to load ({load_error}); refusing to run the "
                           f"mock actor/critic")
    finetuned_exists = actor_checkpoint is not None and Path(actor_checkpoint).exists()

    notes = []
    if not finetuned_exists:
        notes.append(
            "No fine-tuned actor checkpoint given/found (--actor_checkpoint). Using the "
            "pretrained Qwen3-VL-32B-Instruct for BOTH actor and critic (one shared instance). "
            "The paper's actor is fine-tuned on the 14-fish skeleton dataset "
            "(data/fish_skeleton_dataset/, see video2swimfish/finetune/)."
        )
    if client is None:
        notes.append(f"Qwen3-VL failed to load ({load_error}); actor and critic are both "
                      "running in mock fallback mode for this run.")
    for n in notes:
        print(f"[run_auto_construction][note] {n}", flush=True)

    actor = Actor(client=client, checkpoint_path=actor_checkpoint)
    critic = Critic(client=client)
    template_lib_str = st.template_library_prompt_str()

    log = {"fish_id": fish_id, "max_iterations": max_iterations, "notes": notes, "iterations": []}
    best = {"score": -1, "articulation": None, "blend": None, "iteration": None}

    articulation = Articulation()

    boot_blend = work / "iter_boot.blend"
    boot_report_path = work / "iter_boot_report.json"
    boot_report = apply_action_via_blender(None, mesh_glb, {"action": "bootstrap"},
                                             boot_blend, boot_report_path, fish_id=fish_id)
    current_blend = boot_blend
    mesh_bbox = boot_report["skin_bbox"]
    log["mesh_align_report"] = boot_report.get("align_to_canonical_frame")
    log["mesh_scale_report"] = boot_report.get("scale_to_real_length")
    print(f"[run_auto_construction] skin canonicalized: {boot_report.get('align_to_canonical_frame', {}).get('reason')}, "
          f"scaled: {boot_report.get('scale_to_real_length')}", flush=True)

    converged = False
    k_final_blend = current_blend

    # Fix B: the actor was fine-tuned on GT skeletons all rescaled to ACTOR_TRAIN_LENGTH_M, so
    # its position/size numbers are in "0.25 m fish" units. Feeding them to a 0.179 m fish
    # (catfish_fish006) made every bone ~40% too big for the body -> poked out -> shrunk to
    # dots; on a 0.311 m fish they were ~20% too small. Convert both ways: scale the actor's
    # output UP to this fish's real size, and show it the current articulation scaled DOWN into
    # the units it was trained on. The critic (pretrained, not fine-tuned) always sees real m.
    fish_length_m = mesh_bbox["max"][0] - mesh_bbox["min"][0]
    units_k = fish_length_m / ACTOR_TRAIN_LENGTH_M
    log["actor_units_scale"] = {"fish_length_m": fish_length_m, "train_length_m": ACTOR_TRAIN_LENGTH_M, "k": units_k}
    print(f"[run_auto_construction] actor units: fish {fish_length_m:.3f} m / train {ACTOR_TRAIN_LENGTH_M} m -> k={units_k:.3f}", flush=True)

    actor_temperature = 0.0
    cycle_escape_started = None
    for k in range(max_iterations):
        render_before = work / f"iter_{k}_before"
        views_before = render_views(str(current_blend), str(render_before))["views"]
        image_paths_before = [views_before["top"], views_before["front"], views_before["side"]]

        prev_feedback = log["iterations"][-1]["critic_feedback"] if log["iterations"] else None
        action = actor.propose(
            image_paths=image_paths_before,
            current_articulation_json=scale_articulation_dict(articulation.to_dict(), 1.0 / units_k),
            critic_feedback=prev_feedback,
            template_library=template_lib_str,
            mesh_bbox=mesh_bbox,
            temperature=actor_temperature,
        )
        action_meta = action.pop("_meta", None)
        action = scale_action(action, units_k)

        if action.get("action") == "add" and action.get("bone_id") is None:
            action["bone_id"] = articulation.next_bone_id()

        next_blend = work / f"iter_{k}_after.blend"
        apply_report_path = work / f"iter_{k}_apply_report.json"
        apply_report = apply_action_via_blender(current_blend, None, action, next_blend, apply_report_path)

        next_articulation = articulation.apply_action(action)
        verification = apply_report.get("verification")
        verification_list = [verification] if verification else []

        render_after = work / f"iter_{k}_after"
        views_after = render_views(str(next_blend), str(render_after))["views"]
        image_paths_after = [views_after["top"], views_after["front"], views_after["side"]]

        critic_result = critic.evaluate(
            image_paths=image_paths_after,
            proposed_articulation_json=next_articulation.to_dict(),
            geometric_verification_report=verification_list,
            midline_offsets=apply_report.get("midline_offsets"),
        )
        critic_meta = critic_result.pop("_meta", None)

        score = int(critic_result.get("score", 1))
        accepted = bool(critic_result.get("accepted", False))

        print(f"Fish: {fish_id} | Iter {k} | Actor: {action.get('action')} | "
              f"Critic score: {score} | Accepted: {accepted}", flush=True)

        log["iterations"].append({
            "iteration": k,
            "actor_action": action,
            "actor_meta": action_meta,
            "apply_warning": apply_report.get("warning"),
            "vertices_outside_before": verification["vertices_outside_before"] if verification else None,
            "vertices_outside_after": verification["vertices_outside_after"] if verification else None,
            "geometric_verification": verification,
            "midline_offsets": apply_report.get("midline_offsets"),
            "critic_score": score,
            "critic_accepted": accepted,
            "critic_feedback": critic_result.get("feedback"),
            "critic_meta": critic_meta,
            "num_bones": len(next_articulation.bones),
            "actor_temperature": actor_temperature,
        })

        # ... but when the tie is INSIDE an add/remove cycle, "later" alternates between the
        # with-bone and without-bone states; prefer the one with more bones (seen on
        # catfish_fish001 / lake_sturgeon_fish009 v2gen: the guard stopped on a "remove"
        # iteration, so the fallback picked the 11-bone state over the identical-score 12-bone one).
        if score > best["score"] or (score == best["score"] and
                                     len(next_articulation.bones) >= len(best["articulation"].bones if best["articulation"] else [])):
            # >= not >: among TIED scores, prefer the LATER iteration. The critic's own
            # criteria include coverage and "appropriate number of bones", so a tie should
            # favor the more-developed articulation (more bones, generally more body
            # coverage), not freeze on the first iteration that happened to reach this score.
            # Found via catfish_fish001: score sat at 3 for iterations 2-9, and the >
            # comparison kept iteration 2's 3-bone K^(2) even though later iterations had
            # added up to 9 bones -- discarding real coverage progress for no reason.
            best = {"score": score, "articulation": next_articulation, "blend": str(next_blend), "iteration": k}

        articulation = next_articulation
        current_blend = next_blend

        if accepted:
            converged = True
            k_final_blend = next_blend
            break

        # Deterministic-cycle handling. At temperature 0 the actor has been observed (all 4 fish
        # of the v2gen queue at some point) to fall into an exact 2-cycle -- "add bone N" then
        # "remove bone N" with byte-identical reasoning -- because removing the bone restores the
        # exact previous state and the actor has no memory beyond the last critic line.
        # Escape: the moment one full repeat is seen (4 alternating iterations on one bone),
        # switch the ACTOR (not the critic) to sampling at CYCLE_ESCAPE_TEMPERATURE so it can
        # pick a different action; if it is STILL cycling after 3 more full cycles at the
        # raised temperature, stop and let the best-K^(k) fallback below pick the result.
        def _alternating(n):
            acts = [(it["actor_action"].get("action"), it["actor_action"].get("bone_id"))
                    for it in log["iterations"][-n:]]
            return (len(acts) == n and len({a[1] for a in acts}) == 1 and
                    [a[0] for a in acts] in (["add", "remove"] * (n // 2), ["remove", "add"] * (n // 2)))

        if actor_temperature == 0.0 and _alternating(4):
            actor_temperature = CYCLE_ESCAPE_TEMPERATURE
            cycle_escape_started = k
            log["cycle_escape"] = {"started_at_iteration": k, "temperature": actor_temperature,
                                   "bone_id": log["iterations"][-1]["actor_action"].get("bone_id")}
            print(f"[run_auto_construction] add/remove cycle detected at iter {k} -> actor "
                  f"temperature {actor_temperature} from next iteration", flush=True)
        elif actor_temperature > 0.0 and k - cycle_escape_started >= 6 and _alternating(6):
            log["stopped_reason"] = (f"add/remove cycle persisted 3 more repeats at actor "
                                     f"temperature {actor_temperature}")
            print(f"[run_auto_construction] {log['stopped_reason']} -> stopping early", flush=True)
            break

    log["converged"] = converged
    log["converged_at_iteration"] = best["iteration"] if converged else None

    if not converged:
        if best["articulation"] is not None:
            articulation = best["articulation"]
            k_final_blend = Path(best["blend"])
        print(f"[run_auto_construction] fish={fish_id} did NOT converge in {max_iterations} "
              f"iterations; using best-scoring K^(k) from iteration {best['iteration']} "
              f"(score={best['score']}) as K*.", flush=True)

    log["vlm_articulation"] = articulation.to_dict()
    log["vlm_num_bones"] = len(articulation.bones)

    if repair:
        # deterministic structural post-pass (see articulation._repair_skeleton): snap to the
        # centerline, fill inner gaps, extend skull->caudal-fin base, re-run containment
        try:
            repaired_blend = work / "repair.blend"
            repair_report = apply_action_via_blender(k_final_blend, None, {"action": "repair"},
                                                     repaired_blend, work / "repair_report.json")
            template_by_id = {b.bone_id: b.template for b in articulation.bones}
            from articulation import Bone as _Bone
            articulation = Articulation(bones=[
                _Bone(bone_id=b["bone_id"], template=template_by_id.get(b["bone_id"], "repair_box"),
                      position=b["position"], size=b["size"], object_name=b["object_name"])
                for b in repair_report["bones"]])
            k_final_blend = repaired_blend
            log["repair"] = repair_report.get("repair")
            log["repair_midline_offsets"] = repair_report.get("midline_offsets")
            views_r = render_views(str(repaired_blend), str(work / "repair"))["views"]
            critic_r = critic.evaluate(image_paths=[views_r["top"], views_r["front"], views_r["side"]],
                                       proposed_articulation_json=articulation.to_dict(),
                                       geometric_verification_report=[],
                                       midline_offsets=repair_report.get("midline_offsets"))
            critic_r.pop("_meta", None)
            log["critic_after_repair"] = critic_r
            rp = repair_report.get("repair", {})
            print(f"[run_auto_construction] repair: {log['vlm_num_bones']} -> {rp.get('n_bones_after')} bones, "
                  f"coverage {rp.get('coverage_frac_after', 0) * 100:.1f}%, spans filled "
                  f"{[(f['kind'], f['n_added']) for f in rp.get('filled_spans', [])]}; critic after repair: "
                  f"score {critic_r.get('score')} accepted {critic_r.get('accepted')}", flush=True)
        except Exception as e:  # noqa: BLE001 -- fall back to the un-repaired K* rather than lose the run
            log["repair_error"] = f"{type(e).__name__}: {e}"
            print(f"[run_auto_construction] repair FAILED ({e}); exporting un-repaired K*", flush=True)

    log["final_num_bones"] = len(articulation.bones)
    log["final_articulation"] = articulation.to_dict()
    log["best_score_seen"] = best["score"]

    try:
        fish_length_m = mesh_bbox["max"][0] - mesh_bbox["min"][0]   # X = length
        fish_thickness_m = mesh_bbox["max"][1] - mesh_bbox["min"][1]  # Y = thickness
        fish_height_m = mesh_bbox["max"][2] - mesh_bbox["min"][2]   # Z = height
        export_report = export_articulation_to_usd(
            str(k_final_blend), str(out_dir), bones=articulation.bones,
            fish_length_m=fish_length_m, fish_height_m=fish_height_m,
            fish_thickness_m=fish_thickness_m)
        log["export"] = export_report
    except Exception as e:  # noqa: BLE001
        log["export_error"] = f"{type(e).__name__}: {e}"
        log["export_traceback"] = traceback.format_exc()
        print(f"[run_auto_construction] USD export FAILED: {e}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "articulation_construction_log.json").write_text(json.dumps(log, indent=2, default=str))

    print(f"[run_auto_construction] fish={fish_id} converged={converged} "
          f"iterations_run={len(log['iterations'])} final_bones={log['final_num_bones']} "
          f"best_score={best['score']}", flush=True)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fish_id", default=None)
    ap.add_argument("--max_iterations", type=int, default=10)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--actor_checkpoint", default=None)
    ap.add_argument("--no_repair", action="store_true",
                    help="disable the deterministic repair pass (do NOT use: the repair pass is what makes "
                         "~31/33 fish succeed; the bare VLM loop converges for ~1/33)")
    ap.add_argument("--require_vlm", action="store_true",
                    help="abort instead of silently falling back to the mock actor/critic when Qwen3-VL "
                         "cannot be loaded (recommended for real runs)")
    ap.add_argument("--batch", default=None,
                    help="comma-separated fish_ids; ONE shared Qwen3-VL load for all of them. "
                         "--output_dir is then the parent dir (one subdir per fish). Fish whose "
                         "subdir already has K_final.usd are skipped (resumable).")
    args = ap.parse_args()

    if args.batch:
        client, err = load_qwen_client(args.actor_checkpoint)
        if client is None:
            print(f"[run_auto_construction][batch] Qwen3-VL failed to load: {err}", flush=True)
            if args.require_vlm:
                sys.exit(f"--require_vlm: aborting (Qwen3-VL failed to load: {err})")
        for fish in [f.strip() for f in args.batch.split(",") if f.strip()]:
            out = Path(args.output_dir) / fish
            if (out / "K_final.usd").exists():
                print(f"[batch] {fish}: K_final.usd exists, skipping", flush=True)
                continue
            try:
                # mesh QC with the SAME loaded model (verify_mesh_vlm's own CLI would load a second
                # 32B instance per fish); a rejection is recorded, not fatal -- one Meshy call per fish
                try:
                    sys.path.insert(0, str(P.SCRIPTS_DIR))
                    import verify_mesh_vlm
                    from critic import _set_adapter_enabled
                    _set_adapter_enabled(client, False)
                    mc = verify_mesh_vlm.verify(glb=str(MESH_OUT / fish / "mesh.glb"), client=client)
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "mesh_critic.json").write_text(json.dumps(mc, indent=2, default=str))
                    print(f"[batch] {fish}: mesh critic passed={mc.get('passed')}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[batch] {fish}: mesh critic skipped ({type(e).__name__}: {e})", flush=True)
                run_for_fish(fish, args.max_iterations, str(out), args.actor_checkpoint,
                             client=client, repair=not args.no_repair, require_vlm=args.require_vlm)
            except Exception as e:  # noqa: BLE001 -- one fish must not kill the batch
                print(f"[batch] {fish} FAILED: {type(e).__name__}: {e}", flush=True)
                (out / "FAILED.txt").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    run_for_fish(args.fish_id, args.max_iterations, args.output_dir, args.actor_checkpoint,
                 repair=not args.no_repair, require_vlm=args.require_vlm)


if __name__ == "__main__":
    main()
