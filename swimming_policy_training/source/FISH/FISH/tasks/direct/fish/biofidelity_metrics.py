# Online biofidelity metrics for the benchmark (logged every epoch under eval/*).
"""Per-fish reference statistics from the fish's OWN video (curvature npz + reference npz) and an
episode accumulator that turns in-sim body curvature / speed / phase histories into:
  eval/wasserstein_curvature (mean over 20 stations), eval/wasserstein_station_{i},
  eval/dominant_freq_hz, eval/dominant_freq_error_hz, eval/swim_speed_bl_s, eval/swim_speed_error,
  eval/phase_monotonicity  (+ the caller adds frechet / completion / blowup).
Curvature is estimated in-sim with the same estimator used for the video reference
(scripts/zef_playback/curvature_utils.kappa_from_bones -> 20 stations, kappa*BL)."""
from __future__ import annotations
import sys
from collections import deque
from pathlib import Path
import numpy as np

_CU_DIR = Path(__file__).resolve().parents[6] / "scripts" / "zef_playback"
if str(_CU_DIR) not in sys.path:
    sys.path.insert(0, str(_CU_DIR))
import curvature_utils as cu  # noqa: E402

N_STATIONS = 20
_Q = np.linspace(0.0, 1.0, 101)


def _w1(a: np.ndarray, qb: np.ndarray) -> float:
    """1-D Wasserstein-1 between samples `a` and a reference given by its quantiles `qb` (on _Q)."""
    a = a[np.isfinite(a)]
    if a.size < 4:
        return float("nan")
    return float(np.mean(np.abs(np.quantile(a, _Q) - qb)))


def _dominant_freq(kappa_ts: np.ndarray, dt: float) -> float:
    """Median over stations of the FFT peak (excluding DC) of the detrended kappa series (T,20)."""
    T = kappa_ts.shape[0]
    if T < 16:
        return float("nan")
    x = kappa_ts - kappa_ts.mean(0, keepdims=True)
    x = x * np.hanning(T)[:, None]
    spec = np.abs(np.fft.rfft(x, axis=0)) ** 2
    f = np.fft.rfftfreq(T, d=dt)
    spec[0] = 0.0
    # weight stations by their variance so straight (silent) stations do not vote
    w = kappa_ts.var(0)
    if not np.isfinite(w).any() or w.sum() <= 0:
        return float("nan")
    peaks = f[np.argmax(spec, axis=0)]
    order = np.argsort(peaks)
    cw = np.cumsum(w[order]) / w.sum()
    return float(peaks[order][np.searchsorted(cw, 0.5)])


class RefStats:
    """Reference distributions from the fish's own video."""

    def __init__(self, curvature_npz: str, reference_npz: str | None, max_kbl: float = 4.0):
        d = np.load(curvature_npz)
        kap = d["kappa_bl"].astype(np.float64)
        valid = d["valid"].astype(bool) & (np.nan_to_num(np.abs(kap), nan=np.inf).max(1) < max_kbl)
        self.fps = float(d["fps"]) if "fps" in d.files else 30.0
        self.kappa_clean = kap[valid]                                     # (Nc, 20)
        self.station_q = np.stack([np.quantile(self.kappa_clean[:, s], _Q) for s in range(N_STATIONS)])
        # dominant frequency: over runs of >= 2 s of consecutive clean frames
        idx = np.where(valid)[0]
        breaks = np.where(np.diff(idx) > 1)[0] + 1
        freqs, weights = [], []
        for run in np.split(idx, breaks):
            if len(run) >= int(2 * self.fps):
                f0 = _dominant_freq(kap[run], 1.0 / self.fps)
                if np.isfinite(f0):
                    freqs.append(f0); weights.append(len(run))
        self.dominant_freq_hz = float(np.average(freqs, weights=weights)) if freqs else float("nan")
        # swim speed (BL/s) from the reference trajectory, inside its runs
        self.speed_bl_s = float("nan")
        if reference_npz and Path(reference_npz).exists():
            r = np.load(reference_npz, allow_pickle=True)
            p = r["p_bl"].astype(np.float64); fps = float(r["fps"]); v = []
            for a, b in r["runs"]:
                seg = p[int(a):int(b)]
                if len(seg) > 2:
                    v.append(np.linalg.norm(np.diff(seg, axis=0), axis=1) * fps)
            if v:
                self.speed_bl_s = float(np.mean(np.concatenate(v)))


class EpisodeMetrics:
    """Ring buffers for a subset of envs + running means over the last `window` finished episodes."""

    def __init__(self, env_ids: np.ndarray, T_max: int, dt: float, ref: RefStats, window: int = 64):
        self.env_ids = np.asarray(env_ids); self.pos = {int(e): i for i, e in enumerate(self.env_ids)}
        n = len(self.env_ids)
        self.kappa = np.full((n, T_max, N_STATIONS), np.nan, np.float32)
        self.speed = np.full((n, T_max), np.nan, np.float32)
        self.phase = np.full((n, T_max), -1, np.int64)
        self.T_max, self.dt, self.ref = T_max, dt, ref
        self.hist = {k: deque(maxlen=window) for k in ("wasserstein_curvature", "dominant_freq_hz", "swim_speed_bl_s", "phase_monotonicity")}
        self.hist_station = deque(maxlen=window)

    def record(self, step_idx: np.ndarray, kappa_rows: dict, speed_rows: dict, phase_rows: dict):
        """step_idx: per-env step index (episode_length_buf-1) for the tracked envs (dict env->idx)."""
        for e, t in step_idx.items():
            i = self.pos[e]
            if 0 <= t < self.T_max:
                if e in kappa_rows and kappa_rows[e] is not None:
                    self.kappa[i, t] = kappa_rows[e]
                self.speed[i, t] = speed_rows[e]; self.phase[i, t] = phase_rows[e]

    def finish(self, e: int, n_steps: int):
        i = self.pos[e]; m = max(0, min(n_steps, self.T_max))
        k = self.kappa[i, :m]; ok = np.isfinite(k).all(1)
        if ok.sum() >= 16:
            kk = k[ok].astype(np.float64)
            per_station = np.array([_w1(kk[:, s], self.ref.station_q[s]) for s in range(N_STATIONS)])
            self.hist_station.append(per_station); self.hist["wasserstein_curvature"].append(float(np.nanmean(per_station)))
            self.hist["dominant_freq_hz"].append(_dominant_freq(kk, self.dt))
        sp = self.speed[i, :m]; sp = sp[np.isfinite(sp)]
        if sp.size:
            self.hist["swim_speed_bl_s"].append(float(sp.mean()))
        ph = self.phase[i, :m]; ph = ph[ph >= 0]
        if ph.size >= 2:
            self.hist["phase_monotonicity"].append(float(np.mean(np.diff(ph) >= 0)))
        self.kappa[i] = np.nan; self.speed[i] = np.nan; self.phase[i] = -1

    def log_dict(self) -> dict:
        out = {}
        for k, dq in self.hist.items():
            if dq:
                out[f"eval/{k}"] = float(np.nanmean(dq))
        if "eval/dominant_freq_hz" in out and np.isfinite(self.ref.dominant_freq_hz):
            out["eval/dominant_freq_error_hz"] = abs(out["eval/dominant_freq_hz"] - self.ref.dominant_freq_hz)
        if "eval/swim_speed_bl_s" in out and np.isfinite(self.ref.speed_bl_s):
            out["eval/swim_speed_error"] = abs(out["eval/swim_speed_bl_s"] - self.ref.speed_bl_s)
        if self.hist_station:
            st = np.nanmean(np.stack(self.hist_station), 0)
            for s in range(N_STATIONS):
                out[f"eval/wasserstein_station_{s}"] = float(st[s])
        return out
