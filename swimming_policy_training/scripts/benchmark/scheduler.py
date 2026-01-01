#!/usr/bin/env python
"""Benchmark job scheduler: keeps every GPU at N concurrent training jobs, applies the strict
stopping rules, retries blowups once with a new seed, persists queue.json, logs to logs/scheduler.log.

Queue entry (queue.json): {"name", "task", "baseline", "fish", "species", "kind": "rl"|"bc",
  "cmd": [...], "gpu": null|int, "status": "queued|running|done|failed", "seed", "attempt",
  "pid", "log", "started", "ended", "stop_reason", "best": {...}}

RL stop rules (checked on every [EVAL]/[BCO-METRICS] line of the run's log):
  converged : eval/completion_rate >= 0.90 for 3 consecutive epochs
  plateau   : completion_rate improved by <= 0.02 over the last 30 epochs (after >= 80 epochs)
  blowup    : train/blowup_rate > 0.30 for 3 consecutive epochs  -> FAILED (retry once, seed+1000)
  timeout   : wall-clock > 6 h
BC stop rules: bc/val_loss no improvement for 10 epochs; wall-clock > 1 h.
On stop: SIGINT (rl_games saves the checkpoint on interrupt) then SIGKILL after 60 s; stop reason is
written to wandb (eval/stop_reason) via the API; next queued job starts.

Usage: python scheduler.py --queue queue.json --slots 3 --gpus 0,1 [--reserve "0:1,1:1"]
  --reserve: slots already taken by external jobs per GPU (e.g. the VLM batch / other trainings).
  add jobs while running: python scheduler.py --queue queue.json --add jobs.json
"""
from __future__ import annotations
import argparse, json, os, re, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]; LOGDIR = REPO / "logs"; LOGDIR.mkdir(exist_ok=True)
SLOG = LOGDIR / "scheduler.log"
STALL_S = 2400; ROV_CONVERGE = 0.85; BC_TIMEOUT_S = 3600
CAPPED_BASELINES = (); CONTROL_TIMEOUT_S = 6 * 3600   # ALL baselines train to convergence
RL_TIMEOUT_S = float("inf")   # no hard cap, stop only on convergence

def log(msg):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(SLOG, "a") as f: f.write(line + "\n")

def load_q(p): return json.loads(Path(p).read_text()) if Path(p).exists() else []
def save_q(p, q):
    tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(q, indent=1)); tmp.replace(p)

def gpu_util():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return {int(a): (int(b), int(c), int(d)) for a, b, c, d in (l.split(", ") for l in out.strip().splitlines())}
    except Exception:
        return {}

EVAL_RE = re.compile(r"^\[EVAL\] epoch=(\d+) (.*)$"); KV_RE = re.compile(r"(\S+)=([-+0-9.naeinf]+)")
BCO_RE = re.compile(r"^\[BCO-METRICS\] epoch=(\d+) .*blowup_rate=([-+0-9.naeinf]+)")
BC_RE = re.compile(r"^\[BC\] epoch=(\d+) .*val_loss=([-+0-9.naeinf]+)")

class Tracker:
    """Per-job metric history parsed from its log; decides stop reasons."""
    def __init__(self, job):
        self.job = job; self.pos = 0; self.comp = []; self.blow = []; self.val = []; self.best_val = float("inf"); self.no_improve = 0
        self.succ = []; self.last_progress = time.time()   # ROV success history; wall-clock of last new [EVAL]/[BC] line (stall detection)
    def ingest(self):
        p = Path(self.job["log"])
        if not p.exists(): return
        with open(p, "r", errors="ignore") as f:
            f.seek(self.pos); chunk = f.read(); self.pos = f.tell()
        for line in chunk.splitlines():
            m = EVAL_RE.match(line)
            if m:
                kv = dict(KV_RE.findall(m.group(2)))
                if "completion_rate" in kv:
                    self.comp.append(float(kv["completion_rate"]))
                    # TRUE best-completion epoch (the "best" dict below is the LAST eval, which the STOP line mislabelled as best)
                    bc = self.job.setdefault("best_completion", {"completion_rate": -1.0})
                    if float(kv["completion_rate"]) > bc["completion_rate"]:
                        self.job["best_completion"] = {"epoch": int(m.group(1)), **{k: float(v) for k, v in kv.items() if v not in ("nan", "inf")}}
                if "success_rate" in kv: self.succ.append(float(kv["success_rate"]))
                self.last_progress = time.time()
                self.job.setdefault("best", {}); 
                for k, v in kv.items():
                    try: self.job["best"][k] = float(v)
                    except ValueError: pass
                continue
            m = BCO_RE.match(line)
            if m:
                try: self.blow.append(float(m.group(2)))
                except ValueError: pass
                continue
            m = BC_RE.match(line)
            if m:
                v = float(m.group(2)); self.val.append(v); self.last_progress = time.time()
                if v < self.best_val - 1e-6: self.best_val = v; self.no_improve = 0
                else: self.no_improve += 1
    def stop_reason(self, elapsed):
        if self.job["kind"] == "bc":
            # run_freeswim_bc.py early-stops ITSELF (patience 10) and THEN runs the eval stage in a
            # subprocess; killing on plateau here killed bluegill015 bco_pure mid-eval (no metrics).
            # Only enforce the wall-clock cap; the exit branch labels the reason from the log.
            if elapsed > BC_TIMEOUT_S: return "timeout"
            return None
        if elapsed > RL_TIMEOUT_S: return "timeout"
        # Joint RL / CPG are controls -> keep the 6 h cap; BLM+RL trains to convergence
        if self.job.get("baseline") in CAPPED_BASELINES and elapsed > CONTROL_TIMEOUT_S: return "timeout"
        # STALL: Isaac/PhysX hangs (CUDA error 700 on the ROV task every few hundred epochs) leave the
        # process alive with no new epoch for a long time -> resume from the newest checkpoint
        if elapsed > 1800 and time.time() - self.last_progress > STALL_S: return "stalled"
        if self.job.get("task") == "rov":
            sc = self.succ
            if len(sc) >= 3 and all(x >= ROV_CONVERGE for x in sc[-3:]): return "converged"
            # the reference basket run needed ~525 epochs to reach 0.86-0.90; 60/20 fired at 0% for sturgeon016
            if len(sc) >= 300 and (max(sc[-100:]) - max(sc[:-100])) <= 0.02: return "plateau"
            return None
        b = [x for x in self.blow if x == x]                    # drop nan
        # blowup rule only after a 10-epoch warm-up: a random initial policy at joint_limit 45 deg
        # blows up 15-30% of episodes in epochs 1-2 on the small fish (smoke test), which is not
        # the "unstable run" the rule is meant to catch
        if len(b) >= 13 and all(x > 0.30 for x in b[-3:]): return "blowup"
        c = self.comp
        if len(c) >= 3 and all(x >= 0.90 for x in c[-3:]): return "converged"
        # 2026-09-27: 40/20 cut EVERY v3_pcafix run at exactly epoch 40 while wb008 BLM only rose 0.6->0.85 at epochs 40-58
        if len(c) >= 80 and (max(c[-30:]) - max(c[:-30])) <= 0.02: return "plateau"
        return None

def wandb_stop_reason(job, reason):
    try:
        import wandb
        _kf = Path(os.environ.get('WANDB_KEY_FILE', str(Path.home() / '.wandb_key')))
        _key = os.environ.get('WANDB_API_KEY') or (_kf.read_text().strip() if _kf.exists() else None)
        api = wandb.Api(timeout=60, api_key=_key)   # scheduler's own env may hold a different netrc login
        runs = api.runs(f"{job.get('wandb_entity') or os.environ.get('WANDB_ENTITY')}/{job.get('wandb_project','video2swimfish_benchmark')}", filters={"display_name": job["name"]})
        # the killed process's own final wandb sync can land AFTER our write and overwrite the
        # summary (seen: sturgeon020 retry had stop_reason=None) -> retry until it sticks
        for r in runs:
            for attempt in range(6):
                r.summary["eval/stop_reason"] = reason; r.summary.update()
                time.sleep(20)
                r2 = api.run(f"{r.entity}/{r.project}/{r.id}")
                if r2.summary.get("eval/stop_reason") == reason:
                    break
            else:
                log(f"  (wandb stop_reason did not stick for {job['name']} run {r.id})")
    except Exception as e:  # noqa: BLE001
        log(f"  (wandb stop_reason not written for {job['name']}: {e})")

def launch(job, gpu):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["WANDB_MODE"] = "online"
    key = Path(os.environ.get("WANDB_KEY_FILE", str(Path.home() / ".wandb_key")))
    if "WANDB_API_KEY" not in env and key.exists(): env["WANDB_API_KEY"] = key.read_text().strip()
    env.update({k: str(v) for k, v in (job.get("env") or {}).items()})
    cmd = [c.replace("{seed}", str(job["seed"])).replace("{name}", job["name"]) for c in job["cmd"]]
    Path(job["log"]).parent.mkdir(parents=True, exist_ok=True)
    # rotate the previous attempt's log: the Tracker re-reads the whole file, so an appended log carried the
    # old blow-up history into the retry and killed it after 0.01 h (sturgeon001 blm_rl_v2, 05:36)
    _lp = Path(job["log"])
    if _lp.exists() and _lp.stat().st_size > 0:
        _k = 0
        while (_lp.with_suffix(f".prev{_k}.log")).exists(): _k += 1
        _lp.rename(_lp.with_suffix(f".prev{_k}.log"))
    with open(job["log"], "a") as f:
        f.write(f"\n===== launch {datetime.now()} gpu={gpu} seed={job['seed']} attempt={job['attempt']}\n$ {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    job.update(status="running", gpu=gpu, pid=p.pid, started=time.time(), ended=None)
    log(f"START {job['name']} gpu={gpu} pid={p.pid} seed={job['seed']} attempt={job['attempt']}")
    return p

def newest_ckpt_for(job):
    import glob
    lg = Path(job["log"]); txt = lg.read_text(errors="ignore") if lg.exists() else ""
    exps = re.findall(r"Exact experiment name requested from command line: (\S+)", txt)
    cands = [d for e in exps for d in glob.glob(str(REPO / f"logs/rl_games/*/{e}/nn/*.pth"))]
    return max(cands, key=os.path.getmtime) if cands else None

def stop_proc(pid):
    try:
        os.killpg(pid, signal.SIGINT)
    except ProcessLookupError:
        return
    for _ in range(60):
        time.sleep(1)
        try: os.kill(pid, 0)
        except ProcessLookupError: return
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=str(REPO / "scripts/benchmark/queue.json")); ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--gpus", default="0,1"); ap.add_argument("--reserve", default="", help='"0:1,1:1" external jobs per GPU')
    ap.add_argument("--add", default=None, help="json list of jobs to append, then exit"); ap.add_argument("--poll", type=int, default=30)
    a = ap.parse_args(); qp = Path(a.queue)
    if a.add:
        q = load_q(qp); new = json.load(open(a.add)); names = {j["name"] for j in q}
        for j in new:
            if j["name"] not in names: j.setdefault("status", "queued"); j.setdefault("attempt", 0); q.append(j)
        save_q(qp, q); print(f"queue now {len(q)} jobs"); return
    gpus = [int(g) for g in a.gpus.split(",")]
    reserve = {int(k): int(v) for k, v in (x.split(":") for x in a.reserve.split(",") if x)}
    procs, trackers = {}, {}
    log(f"scheduler up: gpus={gpus} slots={a.slots} reserve={reserve} queue={qp}")
    last_status = 0; last_util = 0
    while True:
        q = load_q(qp)
        # adopt jobs marked running from a previous scheduler instance (crash recovery)
        for j in q:
            if j["status"] == "running" and j["name"] not in procs:
                try: os.kill(j["pid"], 0); trackers[j["name"]] = Tracker(j); procs[j["name"]] = None; log(f"ADOPT {j['name']} pid={j['pid']}")
                except (ProcessLookupError, TypeError):
                    # process gone while we were down: a BC job whose log shows the eval finished is DONE,
                    # not a crash to requeue (12:43 restart re-ran catfish002 blm_il after it had completed)
                    _lg = Path(j["log"]).read_text(errors="ignore") if j.get("log") and Path(j["log"]).exists() else ""
                    if j.get("kind") == "bc" and ("eval process exit 0" in _lg or "FREESWIM_DONE" in _lg or "--skip_eval" in " ".join(j.get("cmd", [])) and "[bc_trainer] saved" in _lg):
                        j["status"] = "done"; j["ended"] = time.time(); j["stop_reason"] = "plateau" if "[BC] early stop" in _lg else "finished"; log(f"ADOPT-DONE {j['name']} (finished while scheduler was down)")
                    else:
                        j["status"] = "queued"; j["pid"] = None
        # check running jobs
        for j in q:
            if j["status"] != "running": continue
            t = trackers.setdefault(j["name"], Tracker(j)); t.job = j; t.ingest()   # q is reloaded every loop -> rebind so best/stop_reason land in the dict that gets saved
            alive = True
            try: os.kill(j["pid"], 0)
            except ProcessLookupError: alive = False
            if alive:
                # a self-exited child of THIS scheduler is a zombie until reaped: os.kill(pid, 0) still
                # succeeds (catfish002 blm_il sat 'running' as State Z after FREESWIM_DONE) -> reap + check state
                try: os.waitpid(j["pid"], os.WNOHANG)
                except ChildProcessError: pass
                try:
                    if "State:\tZ" in Path(f"/proc/{j['pid']}/status").read_text(): alive = False
                except OSError: alive = False
            elapsed = time.time() - (j.get("started") or time.time())
            reason = t.stop_reason(elapsed)
            if not alive:
                _full = Path(j["log"]).read_text(errors="ignore") if Path(j["log"]).exists() else ""
                _tail = _full[-40000:]
                # the Isaac eval subprocess prints tens of thousands of chars AFTER the BC marker -> search the whole log
                reason = j.get("stop_reason") or ("crashed" if "Traceback" in _tail else ("plateau" if "[BC] early stop" in _full else "finished"))
            if reason == "stalled":
                log(f"STALL {j['name']} no epoch for {int(time.time()-t.last_progress)}s -> kill + resume from newest checkpoint")
                if alive: stop_proc(j["pid"])
                ck = newest_ckpt_for(j)
                if ck:
                    cmd = list(j["cmd"])
                    if "--checkpoint" in cmd: cmd[cmd.index("--checkpoint") + 1] = ck
                    else: cmd += ["--checkpoint", ck]
                    j["cmd"] = cmd; j["resumed_from"] = ck
                j["status"] = "queued"; j["pid"] = None; j["stalls"] = int(j.get("stalls", 0)) + 1
                trackers.pop(j["name"], None); procs.pop(j["name"], None); save_q(qp, q)
                continue
            if reason:
                if alive: stop_proc(j["pid"])
                j["ended"] = time.time(); j["stop_reason"] = reason
                if reason in ("blowup", "crashed") and j["attempt"] == 0:
                    j["status"] = "queued"; j["attempt"] = 1; j["seed"] = int(j["seed"]) + 1000; j["pid"] = None
                    log(f"RETRY {j['name']} reason={reason} -> seed {j['seed']}")
                else:
                    j["status"] = "failed" if reason in ("blowup", "crashed") else "done"
                    log(f"STOP  {j['name']} reason={reason} elapsed={elapsed/3600:.2f}h last={json.dumps(j.get('best',{}))[:120]} BEST={json.dumps(j.get('best_completion',{}))[:160]}")
                    wandb_stop_reason(j, reason)
                procs.pop(j["name"], None); trackers.pop(j["name"], None)
        # dynamic reserve override: scripts/benchmark/reserve.json {"0": 1, "1": 1} (external jobs per GPU)
        rp = REPO / "scripts/benchmark/reserve.json"
        if rp.exists():
            try: reserve = {int(k): int(v) for k, v in json.loads(rp.read_text()).items()}
            except Exception: pass
        # fill free slots
        for g in gpus:
            running_g = sum(1 for j in q if j["status"] == "running" and j.get("gpu") == g)
            free = a.slots - reserve.get(g, 0) - running_g
            done_names = {x["name"] for x in q if x["status"] == "done"}
            for j in q:
                if free <= 0: break
                if j["status"] == "queued" and (not j.get("after") or j["after"] in done_names):
                    procs[j["name"]] = launch(j, g); trackers[j["name"]] = Tracker(j); free -= 1
                    save_q(qp, q); time.sleep(20)        # stagger Isaac Sim startups
        save_q(qp, q)
        now = time.time()
        if now - last_util > 300:
            last_util = now
            for g, (u, m, mt) in gpu_util().items():
                if g in gpus and u < 60: log(f"WARN gpu{g} utilization {u}% (mem {m}/{mt} MB)")
        if now - last_status > 1800:
            last_status = now
            st = {k: sum(1 for j in q if j["status"] == k) for k in ("done", "running", "queued", "failed")}
            done_t = [j["ended"] - j["started"] for j in q if j["status"] == "done" and j.get("started") and j.get("ended")]
            per = (sum(done_t) / len(done_t)) if done_t else RL_TIMEOUT_S / 2
            slots = sum(a.slots - reserve.get(g, 0) for g in gpus)
            eta_h = (st["queued"] * per / max(slots, 1) + per * 0.5) / 3600 if st["queued"] else 0
            util = " ".join(f"gpu{g}:{v[0]}%/{v[1]}MB" for g, v in gpu_util().items())
            log(f"STATUS done={st['done']} running={st['running']} queued={st['queued']} failed={st['failed']} | {util} | ETA remaining ~{eta_h:.1f} h")
        if all(j["status"] in ("done", "failed") for j in q) and q:
            log("QUEUE EMPTY - all jobs finished"); break
        time.sleep(a.poll)

if __name__ == "__main__":
    main()
