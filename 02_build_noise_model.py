import os
import numpy as np, sys
from scipy.signal import welch
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_reproduce_decoding.py")).read().split("# ---- plots ----")[0])
pre = th < tStart; post = th > tLet[-1] + 0.01
tsec_c = th * 3600.0
nper = 512
segs = [gCor[pre], gCor[post]]
Ps = []; ws = []
for s in segs:
    f, P = welch(s, fs=fs, nperseg=nper, noverlap=nper//2, detrend="linear"); Ps.append(P); ws.append(len(s))
psd = np.average(Ps, axis=0, weights=ws)
# step responses (fraction complete) from the 17 up + 17 down real transitions
tb = tBits[:63]; tg = np.arange(0, 60.0, 1.0); up, dn = [], []
for k in range(1, 63):
    if expected[k] != expected[k-1]:
        m = (th >= tb[k]) & (th < tb[k] + 60/3600); y = np.interp(tg, (th[m]-tb[k])*3600, gCor[m])
        (up if expected[k] == 1 else dn).append(y)
up = np.mean(up, 0); dn = np.mean(dn, 0)
sm = lambda y: np.convolve(np.pad(y, 2, mode="edge"), np.ones(5)/5, "valid")
up, dn = sm(up), sm(dn)
A = float(means[expected == 1].mean() - means[expected == 0].mean())
c_up = np.clip((up - up[:2].mean()) / (up[35:].mean() - up[:2].mean()), 0, 1)
c_dn = np.clip((dn - dn[:5].mean()) / (dn[40:].mean() - dn[:5].mean()), 0, 1)
c_up = np.maximum.accumulate(c_up); c_dn = np.maximum.accumulate(c_dn); c_up[-1] = c_dn[-1] = 1.0
# real sigma_w from non-overlapping 25 s windows on the real background
def wmeans(t, g, T=25/3600):
    o = []; t0 = t[0]
    while t0 + T <= t[-1]:
        w = (t >= t0) & (t < t0 + T); o.append(g[w].mean()); t0 += T
    return np.array(o)
wm = np.r_[wmeans(th[pre], gCor[pre]), wmeans(th[post], gCor[post])]
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "synth_params.npz"), fs=fs, amp=A, freq=f, psd=psd, c_up=c_up, c_down=c_dn, c_t=tg, sigma_w_real=wm.std(ddof=1))
print("fs=%.4f amp=%.2f sigma_w_real=%.3f n_win=%d  df=%.4f Hz" % (fs, A, wm.std(ddof=1), len(wm), f[1]))
print("c_up@[0,5,10,15,20,25]:", np.round(c_up[[0,5,10,15,20,25]], 2))
print("c_dn@[0,5,10,15,20,25,30,35]:", np.round(c_dn[[0,5,10,15,20,25,30,35]], 2))
