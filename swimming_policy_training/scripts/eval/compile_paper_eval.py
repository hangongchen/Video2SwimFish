"""Compile Table 1 (Reach10 success, substituted for the unimplemented Cruising/Path
Following/U-turn/FSTR tasks) and Table 2 (biofidelity) into results/final_eval_metrics.json
plus two plain LaTeX table-row bodies. Reads from already-computed rollout/biofidelity data
-- no simulation, no training, pure post-hoc compilation."""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results"
OUT.mkdir(parents=True, exist_ok=True)

LABEL = {"pure_rl_n256": "Joint-space RL", "cpg_rl_n256": "CPG + RL",
         "pca_n256": "BLM + RL (PCA)", "bco_with_rl_n256": "BCO + RL"}
ORDER = ["pure_rl_n256", "AMP", "cpg_rl_n256", "pca_n256", "bco_with_rl_n256"]
CKPT = {
    "pure_rl_n256": "logs/rl_games/fish_bco_reach10_joint/2026-09-06/nn/fish_bco_reach10_joint.pth",
    "cpg_rl_n256": "logs/rl_games/fish_random10_CPG/2026-08-14/nn/fish_random10_CPG.pth",
    "pca_n256": "logs/rl_games/fish_random10_allPCA/2026-08-28/nn/fish_random10_allPCA.pth",
    "bco_with_rl_n256": "logs/rl_games/fish_bco_reach10_joint/2026-09-06/nn/fish_bco_reach10_joint.pth",
}

TFD = REPO / "outputs/trajectory_fidelity"
bio = json.loads((TFD / "paper_eval" / "metrics.json").read_text())

results = {"notes": [
    "5th baseline 'Joint-space RL + AMP' has NO checkpoint -- never trained on the Reach10 "
    "protocol (only an older, differently-configured single-target task exists in this repo's "
    "history) -- left as null throughout, skipped rather than "
    "placeholder.",
    "Table 1 substitutes the Reach10 10-target benchmark (the one real, comparable task all 4 "
    "available checkpoints were actually trained on) for the requested Cruising/Path "
    "Following/U-turn/FSTR columns, which are not implemented anywhere in this codebase.",
    "'mean +/- std' in Table 1 is a SINGLE seed (seed=7), 256 independent episodes -- this is "
    "episode-to-episode spread within one seed, NOT cross-seed variance. A true multi-seed "
    "estimate would need several independent training + eval runs per policy, not done here.",
    "All rollouts deterministic (policy mean action, no exploration noise).",
]}

# ---------------- Table 1: Reach10 success rate ----------------
table1 = {}
for tag in ORDER:
    if tag == "AMP":
        table1["AMP"] = None
        continue
    R = np.load(TFD / f"rollout_{tag}.npz")
    o = R["outcome"]
    n = len(o)
    reach = int((o == 1).sum())
    p = reach / n
    std = float(np.sqrt(p * (1 - p)))          # within-seed episode-to-episode std, see notes
    table1[tag] = {"label": LABEL[tag], "checkpoint": CKPT[tag], "n_episodes": n,
                   "n_reach": reach, "success_rate_pct": 100 * p, "std_pct": 100 * std,
                   "blowups": int((o == -1).sum())}

# ---------------- Table 2: biofidelity ----------------
table2 = {}
for tag in ORDER:
    if tag == "AMP":
        table2["AMP"] = None
        continue
    sm = bio[tag]["swimming_motion"]
    bd = bio[tag]["body_deformation"]
    table2[tag] = {
        "label": LABEL[tag],
        "cd_path_bl_median": sm["cd_path_bl"]["median"],
        "cd_path_bl_p25": sm["cd_path_bl"]["p25"],
        "cd_path_bl_p75": sm["cd_path_bl"]["p75"],
        "speed_bl_s": sm["speed_bl_s"]["median"],
        "turn_rate_rad_s": sm["turn_rate_rad_s"]["median"],
        "wasserstein_mean": bd["wasserstein_mean"],
        "wasserstein_std": bd["wasserstein_std"],
        "wasserstein_per_station": bd["wasserstein_per_station"],
        "temporal_std_mae": bd["temporal_std_mae"],
        "dominant_freq_hz": bd["dominant_freq_hz_agent"],
    }
zef = {"speed_bl_s": 3.443, "turn_rate_rad_s": 3.441, "dominant_freq_hz": 5.73}

results["table1_reach10_success"] = table1
results["table2_biofidelity"] = table2
results["zef_reference"] = zef
(OUT / "final_eval_metrics.json").write_text(json.dumps(results, indent=2))

# ---------------- LaTeX table bodies ----------------
t1_lines = []
for tag in ORDER:
    row = table1[tag]
    if row is None:
        t1_lines.append(f"{LABEL.get('_amp_label', 'Joint-space RL + AMP') if tag=='AMP' else ''} & "
                         f"\\multicolumn{{1}}{{c}}{{--}} \\\\")
        t1_lines[-1] = "Joint-space RL + AMP & \\multicolumn{1}{c}{--} \\\\  % no checkpoint exists"
        continue
    t1_lines.append(f"{row['label']} & {row['success_rate_pct']:.1f} $\\pm$ {row['std_pct']:.1f} \\\\")

t2_lines = []
for tag in ORDER:
    row = table2[tag]
    if row is None:
        t2_lines.append("Joint-space RL + AMP & \\multicolumn{5}{c}{--} \\\\  % no checkpoint exists")
        continue
    t2_lines.append(
        f"{row['label']} & "
        f"{row['cd_path_bl_median']:.3f} [{row['cd_path_bl_p25']:.3f}, {row['cd_path_bl_p75']:.3f}] & "
        f"{row['speed_bl_s']:.3f} & {row['turn_rate_rad_s']:.3f} & "
        f"{row['wasserstein_mean']:.3f} & "
        f"{row['temporal_std_mae']:.3f} & {row['dominant_freq_hz']:.2f} \\\\"
    )

latex = (
    "% ==================== Table 1: Locomotion Performance (Reach10 substituted; see notes) ====================\n"
    "% Policy & Reach10 success rate (%%, mean +/- std, n=256 episodes, single seed=7)\n"
    + "\n".join(t1_lines)
    + "\n\n"
    "% ==================== Table 2: Biological Fidelity ====================\n"
    "% Policy & Path Chamfer (BL) [p25,p75] & Speed (BL/s) & Turn rate (rad/s) & "
    "Wasserstein distance (kappa*L, mean over 20 stations) & Temporal-std MAE & Dominant freq (Hz)\n"
    f"% ZeF reference: speed={zef['speed_bl_s']}, turn_rate={zef['turn_rate_rad_s']}, "
    f"dominant_freq={zef['dominant_freq_hz']}\n"
    + "\n".join(t2_lines) + "\n"
)
(OUT / "tables.tex").write_text(latex)
print(latex)
print(f"\nwrote {OUT}/final_eval_metrics.json and {OUT}/tables.tex")
