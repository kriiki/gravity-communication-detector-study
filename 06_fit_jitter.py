import os
import sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_reproduce_decoding.py")).read().split("# ---- decode ----")[0])     # real data: th, gCor (nm/s^2), tBits
import synth_gravity as S
from scipy.linalg import toeplitz
FS = S.FS
word = "Gravity"
expected = np.array([1,0,1,0,1,0,1] + [int(b) for c in word for b in format(ord(c), "07b")] + [1,0,1,0,1,0,1])

# ---- model noise-only sd of an L-sample window mean (noise_scale=1) from the PSD ----
f = np.linspace(0, FS/2, 40001); P = np.interp(f, S._F, S._PSD); P[0] = 0; df = f[1]-f[0]
g = np.array([np.sum(P*np.cos(2*np.pi*f*k/FS))*df for k in range(120)])
sig_noise = lambda a: (lambda L: np.sqrt(np.ones(L) @ toeplitz(g[:L]) @ np.ones(L)) / L)(int(np.floor((60-a)*FS)))

def real_wm(a):
    return np.array([gCor[(th >= tBits[i]+a/3600) & (th < tBits[i]+60/3600)].mean() for i in range(63)])

# ---- 1. extra (non-sensor) scatter of the settled window means ----
print("=== 1. In-message window-mean scatter vs sensor-noise-only expectation ===")
ex = []
for a in [35, 40, 45, 50]:
    m = real_wm(a); r = np.where(expected == 1, m - m[expected == 1].mean(), m - m[expected == 0].mean())
    v = (r**2).sum() / (63 - 2); ev = v - sig_noise(a)**2; ex.append(ev)
    print("wait %d s: real within-class sd=%.2f | sensor-noise-only=%.2f | extra variance=%.2f -> extra sd=%.2f" % (a, np.sqrt(v), sig_noise(a), ev, np.sqrt(max(ev, 0))))
lvl = float(np.sqrt(np.mean(ex))); print("=> per-bit level jitter estimate (mean extra variance, a=35-50 s): %.2f nm/s^2" % lvl)
m = real_wm(35); r = np.where(expected == 1, m - m[expected == 1].mean(), m - m[expected == 0].mean())
print("lag-1 autocorrelation of bit-to-bit residuals (35 s window): %.2f (n=62, SE~0.13); lag-2: %.2f" % (np.corrcoef(r[:-1], r[1:])[0, 1], np.corrcoef(r[:-2], r[2:])[0, 1]))
print("residual sd by class: '1' %.2f  '0' %.2f | by transition type: " % (r[expected==1].std(ddof=1), r[expected==0].std(ddof=1)) +
      ", ".join("%s n=%d sd=%.2f" % (nm, (sel).sum(), r[sel].std(ddof=1)) for nm, sel in
                [("0->0", (expected==0)&(np.r_[0,expected[:-1]]==0)), ("0->1", (expected==1)&(np.r_[0,expected[:-1]]==0)),
                 ("1->0", (expected==0)&(np.r_[0,expected[:-1]]==1)), ("1->1", (expected==1)&(np.r_[0,expected[:-1]]==1))]))

# ---- 2. per-transition timing: shift-fit estimator, calibrated on the generator ----
TG = np.arange(-10, 20.01, 0.25)
def est_tau(trel, y, up, lo, hi):
    c = S.C_UP if up else S.C_DOWN; best = None
    cost = []
    for tau in TG:
        cc = np.interp(trel - tau, S.C_T, c)
        mod = lo + (hi-lo)*cc if up else hi + (lo-hi)*cc
        cost.append(((y - mod)**2).sum())
    return TG[int(np.argmin(cost))]
lo, hi = real_wm(35)[expected == 0].mean(), real_wm(35)[expected == 1].mean()
bits_ext = np.r_[0, expected, 0]; starts = tBits                         # 64 starts: 63 bits + return-to-lower
tr_real = {True: [], False: []}
for k in range(64):
    prev = bits_ext[k]; cur = bits_ext[k+1]
    if prev == cur: continue
    mk = (th >= tBits[k]) & (th < tBits[k] + 60/3600)
    tr_real[cur > prev].append(est_tau((th[mk]-tBits[k])*3600, gCor[mk], cur > prev, lo, hi))
print("\n=== 2. Per-transition timing (shift in s relative to the average template) ===")
print("real up   (n=%d): " % len(tr_real[True]), np.round(sorted(tr_real[True]), 2))
print("real down (n=%d): " % len(tr_real[False]), np.round(sorted(tr_real[False]), 2))
# calibration on the generator: no timing jitter, noise_scale=1, with the fitted level jitter
def sim_tau(up_sd, dn_sd, nrep=300, seed0=0):
    out = {True: [], False: []}; rng = np.random.default_rng(seed0)
    for r_ in range(nrep):
        R = S.generate(expected, noise_scale=1.0, seed=seed0 + r_, level_jitter=lvl, tau_up=up_sd, tau_dn=dn_sd, lead_s=300, tail_s=300)
        tt, xx = R["t"], R["x"]; be = np.r_[0, expected, 0]
        for k in range(64):
            if be[k] == be[k+1]: continue
            s0 = 300 + 60*k; mk = (tt >= s0) & (tt < s0+60)
            out[be[k+1] > be[k]].append(est_tau(tt[mk]-s0, xx[mk], be[k+1] > be[k], 0.0, S.A0))
    return {k: np.array(v) for k, v in out.items()}
sim0 = sim_tau(0, 0, nrep=120)
res = {}
for up in (True, False):
    rv = np.var(tr_real[up], ddof=1); sv = np.var(sim0[up], ddof=1); name = "up" if up else "down"
    res[up] = np.sqrt(max(rv - sv, 0))
    print("%-4s: real mean=%.2f sd=%.2f | generator with NO timing jitter: mean=%.2f sd=%.2f (estimator noise) | implied jitter sd=%.2f s" % (name, np.mean(tr_real[up]), np.sqrt(rv), sim0[up].mean(), np.sqrt(sv), res[up]))
print("FIT_RESULT level=%.3f tau_up=%.3f tau_dn=%.3f" % (lvl, res[True], res[False]))
np.save(os.path.join(os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")), "jitter_fit.npy"), np.array([lvl, res[True], res[False]]))
