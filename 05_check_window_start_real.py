import os
import numpy as np
exec(open("01_reproduce_decoding.py").read().split("# ---- decode ----")[0])      # real data: th, gCor (nm/s^2), tBits ...
word = "Gravity"
expected = np.array([1,0,1,0,1,0,1] + [int(b) for c in word for b in format(ord(c), "07b")] + [1,0,1,0,1,0,1])
print("wait(s)  L   errors  mean1  mean0  sd1   sd0   margin/sd (worst-class z)")
for a in [15, 20, 25, 30, 35, 40, 45, 50]:
    m = []
    for i in range(63):
        w = (th >= tBits[i] + a/3600) & (th < tBits[i] + 60/3600); m.append(gCor[w].mean())
    m = np.array(m); d = (m > 25.0).astype(int); e = int((d != expected).sum())
    m1, m0 = m[expected == 1], m[expected == 0]
    thr = (m1.mean() + m0.mean()) / 2
    z = min((m1.mean() - thr) / m1.std(ddof=1), (thr - m0.mean()) / m0.std(ddof=1))
    print("%5d   %2d  %4d   %6.1f %6.1f %5.2f %5.2f   %.1f" % (a, int(np.floor((60-a)*1.8649)), e, m1.mean(), m0.mean(), m1.std(ddof=1), m0.std(ddof=1), z))
# 1->0 bits only (where an unfinished falling step would bite)
prev = np.r_[0, expected[:-1]]; fall = (expected == 0) & (prev == 1)
for a in [20, 25, 35]:
    m = np.array([gCor[(th >= tBits[i] + a/3600) & (th < tBits[i] + 60/3600)].mean() for i in range(63)])
    print("wait %2ds: mean of '1->0' windows = %.1f nm/s2 (n=%d), mean of '0->0' windows = %.1f" % (a, m[fall].mean(), fall.sum(), m[(expected == 0) & (prev == 0)].mean()))
