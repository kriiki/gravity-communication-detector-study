import os
import sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_reproduce_decoding.py")).read().split("# ---- plots ----")[0])   # real data: th, gCor, expected, means, ...
import synth_gravity as S
from scipy import stats
from scipy.signal import welch
from scipy.stats import norm
from external.allan_variance.allan_variance.avar import compute_avar
OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")); os.makedirs(OUT, exist_ok=True)
pre = th < tStart; post = th > tLet[-1] + 0.01; xr = gCor[pre]
print("=== 1. Noise statistics: real background vs synthetic (noise_scale=1) ===")
syn_bits = np.zeros(240, int); R = S.generate(syn_bits, noise_scale=1.0, lead_s=0, tail_s=0, seed=3); xs = R["x"]
wmean = lambda x: np.array([x[i:i+int(25*S.FS)].mean() for i in range(0, len(x)-int(25*S.FS), int(25*S.FS))])
rows = [("real (pre+post bg)", np.r_[gCor[pre], gCor[post]]), ("synthetic", xs), ("synthetic+quantise", S.generate(syn_bits, noise_scale=1.0, lead_s=0, tail_s=0, seed=3, quantize=True)["x"])]
for nm, x in rows:
    w = wmean(x) if "real" not in nm else np.r_[wmean(gCor[pre]), wmean(gCor[post])]
    print("%-20s sample sd=%.2f exkurt=%+.2f | 25s-window-mean sd=%.2f (n=%d)" % (nm, x.std(ddof=1), stats.kurtosis(x), w.std(ddof=1), len(w)))
print("lag-1 autocorr real=%.3f synthetic=%.3f" % tuple(np.corrcoef(a[:-1], a[1:])[0, 1] for a in (xr, xs)))
# Allan deviation: real pre-background vs mean of 12 synthetic realisations of the same length
dt = 1 / S.FS; n = len(xr)
tau, vr = compute_avar(xr, dt=dt)
vs = np.mean([compute_avar(S.generate(np.zeros(n // 112 + 2, int), noise_scale=1.0, lead_s=0, tail_s=0, seed=100+i)["x"][:n], dt=dt)[1] for i in range(12)], 0)
vd = np.mean([compute_avar(S.generate(np.zeros(n // 112 + 2, int), noise_scale=1.0, lead_s=0, tail_s=0, seed=200+i, drift=S.DRIFT_FIT)["x"][:n], dt=dt)[1] for i in range(12)], 0)
for k in range(0, len(tau), max(1, len(tau)//6)):
    print("  Allan tau=%6.1fs real=%.2f synth=%.2f synth+drift(%g)=%.2f nm/s^2" % (tau[k], np.sqrt(vr[k]), np.sqrt(vs[k]), S.DRIFT_FIT, np.sqrt(vd[k])))

print("\n=== 2. Sanity: simulated BER vs Gaussian theory Q(SNR/2) ===")
rng = np.random.default_rng(7); snrs = [2, 3, 4, 5]; ber = []
for snr in snrs:
    nb = {2: 15000, 3: 15000, 4: 30000, 5: 80000}[snr]
    bits = rng.integers(0, 2, nb); R = S.generate(bits, snr=snr, lead_s=0, tail_s=0, seed=int(snr*10))
    m = S.window_means(R); dec = (m > R["amp"] / 2).astype(int); b = np.mean(dec != bits); ber.append(b)
    print("SNR=%d  simulated BER=%.5f (errors=%d/%d)  theory Q(SNR/2)=%.5f" % (snr, b, (dec != bits).sum(), nb, norm.sf(snr / 2)))
print("noise-free decode of authors' 63-bit message: errors =", int((( S.window_means(S.generate(expected, snr=1e6, seed=1)) > S.A0/2).astype(int) != expected).sum()))
Rm = S.generate(expected, noise_scale=1.0, seed=11); dm = (S.window_means(Rm) > Rm["amp"]/2).astype(int)
print("63-bit message at real noise level (SNR~%.0f): errors = %d" % (S.A0 / S.SIGMA_W_REAL, int((dm != expected).sum())))

# ---------------- figure ----------------
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
tt = (Rm["t"] - 300) / 3600
ax[0, 0].plot(tt, Rm["x"], lw=.5, label="synthetic"); ax[0, 0].plot((th - tStart), gCor, lw=.5, alpha=.6, label="real (corrected)")
ax[0, 0].set_xlim(-0.1, 1.1); ax[0, 0].axhline(S.A0/2, color="k", ls=":"); ax[0, 0].legend(loc="upper right"); ax[0, 0].set_title("'Gravity' message: real vs synthetic at real noise level"); ax[0, 0].set_xlabel("time since first bit (h)"); ax[0, 0].set_ylabel("nm/s²")
nper = 512
fr, Pr = welch(xr, fs=S.FS, nperseg=nper, noverlap=nper//2, detrend="linear")
fs_, Ps_ = welch(xs[:len(xr)*8], fs=S.FS, nperseg=nper, noverlap=nper//2, detrend="linear")
ax[0, 1].loglog(fr[1:], Pr[1:], label="real background", alpha=.8); ax[0, 1].loglog(fs_[1:], Ps_[1:], label="synthetic (8x longer, averaged)", alpha=.8)
ax[0, 1].set_title("PSD (by construction near-identical)"); ax[0, 1].set_xlabel("Hz"); ax[0, 1].set_ylabel("(nm/s²)²/Hz"); ax[0, 1].legend()
ax[0, 2].loglog(tau, np.sqrt(vr), "o-", label="real background"); ax[0, 2].loglog(tau, np.sqrt(vs), "s--", label="synthetic (mean of 12)"); ax[0, 2].loglog(tau, np.sqrt(vd), "^:", label="synthetic + drift %g nm/s²/√h" % S.DRIFT_FIT)
ax[0, 2].set_title("Allan deviation (independent check)"); ax[0, 2].set_xlabel("integration time (s)"); ax[0, 2].set_ylabel("nm/s²"); ax[0, 2].legend()
tg = S.C_T
ax[1, 0].plot(tg, S.C_UP, label="up template"); ax[1, 0].plot(tg, S.C_DOWN, label="down template"); ax[1, 0].axvline(35, color="k", ls=":")
ax[1, 0].set_title("Step responses measured from real transitions (17 up, 17 down)"); ax[1, 0].set_xlabel("s since bit start"); ax[1, 0].set_ylabel("fraction complete"); ax[1, 0].legend()
ax[1, 1].semilogy(snrs, ber, "o", label="synthetic sim"); xx = np.linspace(2, 6, 50); ax[1, 1].semilogy(xx, norm.sf(xx/2), "-", label="Gaussian theory Q(SNR/2)")
ax[1, 1].set_title("Sanity check: BER vs SNR (threshold detector)"); ax[1, 1].set_xlabel("SNR = A/σ_w"); ax[1, 1].set_ylabel("BER"); ax[1, 1].legend()
bins = np.linspace(-40, 40, 41); ax[1, 2].hist(xr, bins, density=True, alpha=.6, label="real background"); ax[1, 2].hist(xs, bins, density=True, alpha=.6, label="synthetic")
ax[1, 2].set_title("Sample amplitude distribution"); ax[1, 2].set_xlabel("nm/s²"); ax[1, 2].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_generator_validation.png", dpi=140); print("saved figure")
