"""Paper tables for ONE fish (Task 1 five baselines, Task 2 two baselines) from the benchmark logs.
Task 1 numbers = the epoch with the highest completion (ties -> lowest Frechet) of each run; also the last epoch.
Usage: python export_fish_tables.py --fish white_bass_fish008 --suffix _v2 [--out results/white_bass_fish008_v2]"""
import argparse, json, re, os, glob
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument("--fish", required=True); ap.add_argument("--suffix", default="_v2"); ap.add_argument("--out", default=None)
a = ap.parse_args(); out = Path(a.out or REPO / f"results/{a.fish}{a.suffix}"); out.mkdir(parents=True, exist_ok=True)
EVAL = re.compile(r"^\[EVAL\] epoch=(\d+) (.*)$"); KV = re.compile(r"(\S+?)=([-+0-9.naeinf]+)")
BASE1 = [("joint_rl", "Joint RL"), ("amp_rl", "Joint RL + AMP"), ("cpg_rl", "CPG + RL"), ("bco_rl", "BCO + RL"), ("blm_rl", "BLM + RL (Ours)")]
COLS1 = [("completion_rate", "Completion $\\uparrow$", "{:.2f}"), ("frechet_dist_bl", "Fr\\'echet (BL) $\\downarrow$", "{:.2f}"), ("wasserstein_curvature", "$W_1(\\kappa)$ $\\downarrow$", "{:.2f}"),
         ("dominant_freq_error_hz", "$\\Delta f$ (Hz) $\\downarrow$", "{:.2f}"), ("swim_speed_error", "$\\Delta v$ (BL/s) $\\downarrow$", "{:.2f}")]
q = {j["name"]: j for j in json.load(open(REPO / "scripts/benchmark/queue.json"))}
def parse(log):
    rows = []
    for line in open(log, errors="ignore"):
        m = EVAL.match(line.strip())
        if m:
            kv = {k: float(v) for k, v in KV.findall(m.group(2)) if v not in ("nan", "inf")}; kv["epoch"] = int(m.group(1)); rows.append(kv)
    return rows
def status(name):
    j = q.get(name); return (j.get("status", "?"), j.get("stop_reason")) if j else ("missing", None)
t1 = {}
for key, label in BASE1:
    name = f"trajfollow_{a.fish}_{key}{a.suffix}"; log = REPO / f"run_logs/benchmark/{name}.log"
    rows = parse(log) if log.exists() else []
    if rows:
        best = max(rows, key=lambda r: (r.get("completion_rate", -1), -r.get("frechet_dist_bl", 1e9))); last = rows[-1]
    else: best = last = {}
    t1[key] = {"label": label, "best": best, "last": last, "n_epochs": len(rows), "status": status(name)}
t2 = {}
for key, label, d in [("bco_pure", "BCO", "bco"), ("blm_il", "BLM + IL (Ours)", "blm_il"), ("blm_il_v3", "BLM + IL (4 modes, smoothed, 5-frame history; unscaled labels -- near-static)", "blm_il_v3"), ("blm_il_v5", "BLM + IL (Ours, 4 modes, smoothed, no history, coef.-noise aug.; rescaled labels)", "blm_il_v5"), ("blm_il_v4", "BLM + IL (Ours, 4 modes, smoothed, 5-frame history, rescaled labels)", "blm_il_v4")]:
    f = REPO / f"checkpoints/benchmark/{d}/{a.fish}/freeswim_eval.json"
    t2[key] = {"label": label, "m": json.load(open(f)) if f.exists() else {}, "status": status(f"freeswim_{a.fish}_{key}{a.suffix}")}
    if not t2[key]["m"] and key in ("blm_il_v3", "blm_il_v4", "blm_il_v5"): t2.pop(key)   # variant not run for this fish
def fmt(v, f): return "--" if v is None or v != v else f.format(v)
# ---- LaTeX Table 1
L = ["\\begin{table}[t]", f"\\caption{{Task~1 trajectory following, {a.fish.replace('_', ' ')}. Numbers are the epoch with the highest completion of each run; runs stop only on convergence or plateau.}}",
     "\\label{tab:task1-single}", "\\begin{center}\\small", "\\begin{tabular}{l" + "c" * (len(COLS1) + 2) + "}", "\\toprule",
     "Method & " + " & ".join(c[1] for c in COLS1) + " & epochs & stop \\\\", "\\midrule"]
for key, _ in BASE1:
    r = t1[key]; b = r["best"]
    L.append(f"{r['label']} & " + " & ".join(fmt(b.get(c[0]), c[2]) for c in COLS1) + f" & {r['n_epochs']} & {r['status'][1] or r['status'][0]} \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{center}", "\\end{table}"]
# ---- LaTeX Table 2
COLS2 = [("eval/frechet_dist_bl", "Fr\\'echet (BL) $\\downarrow$", "{:.2f}"), ("eval/swim_speed_error", "Speed err.\\ (BL/s) $\\downarrow$", "{:.2f}"), ("eval/heading_stability", "Heading stab.\\ $\\uparrow$", "{:.2f}"),
         ("eval/wasserstein_curvature", "$W_1(\\kappa)$ $\\downarrow$", "{:.2f}"), ("eval/dominant_freq_error_hz", "$\\Delta f$ (Hz) $\\downarrow$", "{:.3f}")]
L2 = ["\\begin{table}[t]", f"\\caption{{Task~2 free swimming from video, {a.fish.replace('_', ' ')}: 20 real initial states, 5\\,s roll-outs, no reward.}}", "\\label{tab:task2-single}",
      "\\begin{center}\\small", "\\begin{tabular}{l" + "c" * len(COLS2) + "}", "\\toprule", "Method & " + " & ".join(c[1] for c in COLS2) + " \\\\", "\\midrule"]
for key in [k for k in ("bco_pure", "blm_il", "blm_il_v3", "blm_il_v4", "blm_il_v5") if k in t2]:
    r = t2[key]; L2.append(f"{r['label']} & " + " & ".join(fmt(r["m"].get(c[0]), c[2]) for c in COLS2) + " \\\\")
L2 += ["\\bottomrule", "\\end{tabular}", "\\end{center}", "\\end{table}"]
(out / "table1_task1.tex").write_text("\n".join(L) + "\n"); (out / "table2_task2.tex").write_text("\n".join(L2) + "\n")
# ---- markdown summary (best AND last epoch) + json
md = [f"# {a.fish}{a.suffix}", "", "## Task 1 (best epoch | last epoch)", "", "| Method | epochs | stop | " + " | ".join(c[0] for c in COLS1) + " |", "|---|---|---|" + "---|" * len(COLS1)]
for key, _ in BASE1:
    r = t1[key]; md.append(f"| {r['label']} | {r['n_epochs']} | {r['status'][1] or r['status'][0]} | " + " | ".join(f"{fmt(r['best'].get(c[0]), c[2])} / {fmt(r['last'].get(c[0]), c[2])}" for c in COLS1) + " |")
md += ["", "## Task 2", "", "| Method | " + " | ".join(c[0].split('/')[1] for c in COLS2) + " |", "|---|" + "---|" * len(COLS2)]
for key in [k for k in ("bco_pure", "blm_il", "blm_il_v3", "blm_il_v4", "blm_il_v5") if k in t2]:
    r = t2[key]; md.append(f"| {r['label']} | " + " | ".join(fmt(r["m"].get(c[0]), c[2]) for c in COLS2) + " |")
(out / "summary.md").write_text("\n".join(md) + "\n"); json.dump({"task1": t1, "task2": t2}, open(out / "results.json", "w"), indent=1, default=str)
print("\n".join(md)); print(f"\nwrote {out}/table1_task1.tex, table2_task2.tex, summary.md, results.json")
