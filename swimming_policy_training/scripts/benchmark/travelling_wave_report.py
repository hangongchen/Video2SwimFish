#!/usr/bin/env python
"""Is the extracted curvature a TRAVELLING wave (phase lag head->tail, rotating PC1/PC2 trajectory) or a
STANDING C-bend (all stations in phase, PC trajectory a back-and-forth line)? A per-frame linear kappa(s)
does NOT decide this: a long-wavelength travelling wave (lambda >~ 1 BL) with a tail-growing envelope is a
ramp in every frame and still a quadrature PC pair. Usage: travelling_wave_report.py <curvature.npz> ..."""
import sys, numpy as np

def runs_of(valid, min_len=48):
    idx = np.where(valid)[0]; br = np.where(np.diff(idx) > 1)[0] + 1
    return [r for r in np.split(idx, br) if len(r) >= min_len]

def report(path, s_a=0.45, s_b=0.9):
    d = np.load(path); kap, valid, s, fps = d["kappa_bl"], d["valid"], d["s_stations"], float(d["fps"])
    valid = valid & (np.nan_to_num(np.abs(kap), nan=np.inf).max(1) < 4)
    ia, ib = int(np.argmin(abs(s - s_a))), int(np.argmin(abs(s - s_b)))
    X = kap[valid]; mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    lags, freqs, rots, n = [], [], [], 0
    for r in runs_of(valid):
        k = kap[r] - mu; ka, kb = k[:, ia] - k[:, ia].mean(), k[:, ib] - k[:, ib].mean()
        # dominant frequency of the tail station
        F = np.fft.rfft(kb * np.hanning(len(kb))); f = np.fft.rfftfreq(len(kb), 1 / fps); band = (f > 0.3) & (f < 6)
        if not band.any(): continue
        f0 = f[band][np.argmax(abs(F[band]))]
        # phase lag at f0 via cross-spectrum (positive = tail lags head = wave travels head->tail)
        Fa = np.fft.rfft(ka * np.hanning(len(ka))); j = np.argmin(abs(f - f0))
        lag_deg = np.degrees(np.angle(Fa[j] * np.conj(F[j])))
        # PC1/PC2 rotation: normalised signed area rate; |rot|~1 = circle (travelling), ~0 = line (standing)
        a = (k @ Vt[:2].T); da = np.diff(a, axis=0); cross = a[:-1, 0] * da[:, 1] - a[:-1, 1] * da[:, 0]
        rot = cross.sum() / (np.abs(cross).sum() + 1e-9)
        w = len(r); lags.append((lag_deg, w)); freqs.append((f0, w)); rots.append((rot, w)); n += w
    wavg = lambda L: sum(v * w for v, w in L) / max(1, sum(w for _, w in L))
    print(f"{path}\n  runs {len(lags)} frames {n}   tail dom. freq {wavg(freqs):.2f} Hz   phase lag s={s[ia]:.2f}->s={s[ib]:.2f}: "
          f"{wavg(lags):+.0f} deg (0 = standing C-bend; 40-120 = travelling wave)   PC1/PC2 rotation consistency {wavg(rots):+.2f} (0 line, +-1 circle)")
    per = [f"{l:+.0f}" for l, _ in lags[:12]]; print("  per-run lags:", " ".join(per))

if __name__ == "__main__":
    for p in sys.argv[1:]: report(p)
