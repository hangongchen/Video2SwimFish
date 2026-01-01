#!/usr/bin/env python
"""Build benchmark job specs for the scheduler.
Usage: python build_queue.py --day 1 --out jobs_day1.json [--fish f1,f2] [--baselines joint_rl,blm_rl]
Baselines (trajectory following): joint_rl, blm_rl (PCA action space), cpg_rl, amp_rl, bco_rl
Free swimming (BC): bco_pure, blm_il
Run name: {task}_{fish}_{baseline}; wandb project video2swimfish_benchmark; tags [task, species, baseline]."""
import argparse, json, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; PY = sys.executable
PROJECT = "video2swimfish_benchmark"; ENTITY = os.environ.get("WANDB_ENTITY")
TASKS = {"joint_rl": "Bench-{tag}-TrajFollow-Joint-v0", "blm_rl": "Bench-{tag}-TrajFollow-PCA-v0",
         "cpg_rl": "Bench-{tag}-TrajFollow-CPG-v0", "amp_rl": "Bench-{tag}-TrajFollow-Joint-AMP-v0", "bco_rl": "Bench-{tag}-TrajFollow-Joint-v0"}
DAYS = {1: ["joint_rl", "blm_rl"], 2: ["cpg_rl", "amp_rl"], 3: ["bco_rl"]}

def rl_job(tag, baseline, seed=42):
    sp = tag.rsplit("_fish", 1)[0]; name = f"trajfollow_{tag}_{baseline}"
    cmd = [PY, "scripts/rl_games/train_ppo.py", "--task", TASKS[baseline].format(tag=tag), "--num_envs", "256", "--seed", "{seed}",
           "--headless", "--track", "--wandb-project-name", PROJECT, "--wandb-entity", ENTITY, "--wandb-name", "{name}"]
    if baseline == "bco_rl":
        cmd += ["--checkpoint", str(REPO / f"checkpoints/benchmark/bco/{tag}/bc_policy.pt")]
    return {"name": name, "task": "trajfollow", "baseline": baseline, "fish": tag, "species": sp, "kind": "rl", "cmd": cmd,
            "seed": seed, "attempt": 0, "status": "queued", "log": str(REPO / f"run_logs/benchmark/{name}.log"),
            "env": {"WANDB_TAGS": f"trajfollow,{sp},{baseline}"}, "wandb_project": PROJECT, "wandb_entity": ENTITY}

def bc_job(tag, baseline, seed=42):
    sp = tag.rsplit("_fish", 1)[0]; name = f"freeswim_{tag}_{baseline}"
    cmd = [PY, "scripts/benchmark/run_freeswim_bc.py", "--tag", tag, "--baseline", baseline, "--seed", "{seed}", "--wandb-name", "{name}"]
    return {"name": name, "task": "freeswim", "baseline": baseline, "fish": tag, "species": sp, "kind": "bc", "cmd": cmd,
            "seed": seed, "attempt": 0, "status": "queued", "log": str(REPO / f"run_logs/benchmark/{name}.log"),
            "env": {"WANDB_TAGS": f"freeswim,{sp},{baseline}"}, "wandb_project": PROJECT, "wandb_entity": ENTITY}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--day", type=int, default=1); ap.add_argument("--out", required=True)
    ap.add_argument("--fish", default=None); ap.add_argument("--baselines", default=None); ap.add_argument("--freeswim", action="store_true")
    a = ap.parse_args(); fl = json.load(open(REPO / "scripts/benchmark/fish_list.json"))
    fish = a.fish.split(",") if a.fish else fl["traj"]; bases = a.baselines.split(",") if a.baselines else DAYS[a.day]
    jobs = []
    for b in bases:
        for t in fish:
            j = rl_job(t, b)
            if b == "bco_rl":
                init_name = f"bcoinit_{t}"
                if t not in fl["freeswim"]:      # BC checkpoint not produced by a freeswim job -> add an init-only job first
                    jobs.append({**bc_job(t, "bco_pure"), "name": init_name, "task": "bcoinit", "log": str(REPO / f"run_logs/benchmark/{init_name}.log"),
                                 "cmd": bc_job(t, "bco_pure")["cmd"][:-2] + ["--wandb-name", init_name, "--skip_eval"]})
                    j["after"] = init_name
                else:
                    j["after"] = f"freeswim_{t}_bco_pure"
            jobs.append(j)
    if a.freeswim:
        jobs += [bc_job(t, b) for b in ("bco_pure", "blm_il") for t in fl["freeswim"]]
    json.dump(jobs, open(a.out, "w"), indent=1); print(f"{len(jobs)} jobs -> {a.out}")
