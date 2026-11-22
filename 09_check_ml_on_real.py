import os
import sys, re, numpy as np
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "08_ml_detectors.py")).read().split("def detect")[0]          # feature code + training of the synthetic-trained models
exec(src)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_reproduce_decoding.py")).read().split("# ---- decode ----")[0])   # real data
word = "Gravity"; expected = np.array([1,0,1,0,1,0,1] + [int(b) for c in word for b in format(ord(c), "07b")] + [1,0,1,0,1,0,1])
t_u = np.arange(0, int((63 * 60 + 120) * FS)) / FS; x_u = np.interp(t_u, (th - tBits[0]) * 3600, gCor)    # real trace on the uniform grid, first bit at t=0
s0 = 60.0 * np.arange(63); Fr, au = feats(x_u, s0)
print("\nREAL 63-bit message, models trained ONLY on synthetic data:")
for nm, m in (("logistic regression", lr_r), ("gradient boosting", gb_r)):
    p = m.predict_proba(Fr)[:, 1]; d = (p > .5).astype(int); conf = np.abs(p - .5)
    print("  %-20s errors=%d/63 | min confidence=%.3f | 5 least-confident bits' P(1): %s" % (nm, (d != expected).sum(), conf.min(), np.round(p[np.argsort(conf)[:5]], 2)))
print("  threshold 30-60 s mean: errors=%d; authors' 35-60 s mean: errors=%d; closest bit to threshold: %.1f nm/s2 (30-60 s mean)" % (((Fr[:, 0] > A/2) != expected).sum(), ((au > A/2) != expected).sum(), np.min(np.abs(Fr[:, 0] - A/2))))
