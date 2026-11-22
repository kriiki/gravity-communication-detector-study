import os
import sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_reproduce_decoding.py")).read().split("# ---- decode ----")[0])
import synth_gravity as S
from scipy.stats import norm
FS = S.FS; A = S.A0
word = "Gravity"
expected = np.array([1,0,1,0,1,0,1] + [int(b) for c in word for b in format(ord(c), "07b")] + [1,0,1,0,1,0,1])
LVL, TUP, TDN = np.load(os.path.join(os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")), "jitter_fit.npy"))
WAITS = [15, 20, 25, 30, 35, 40, 45, 50]

def stats_from(m):
    m1, m0 = m[expected == 1], m[expected == 0]; thr = (m1.mean() + m0.mean()) / 2
    return m1.mean(), m0.mean(), m1.std(ddof=1), m0.std(ddof=1), min((m1.mean()-thr)/m1.std(ddof=1), (thr-m0.mean())/m0.std(ddof=1))
real = {a: stats_from(np.array([gCor[(th >= tBits[i]+a/3600) & (th < tBits[i]+60/3600)].mean() for i in range(63)])) for a in WAITS}
def sim_stats(**kw):
    out = {a: [] for a in WAITS}
    for r in range(300):
        R = S.generate(expected, noise_scale=1.0, seed=7000 + r, lead_s=300, tail_s=60, **kw); s0 = R["bit_starts"]; x = R["x"]
        for a in WAITS:
            L = int(np.floor((60-a)*FS)); i0 = np.ceil((s0 + a)*FS).astype(int); m = x[i0[:, None] + np.arange(L)].mean(1)
            out[a].append(stats_from(m))
    return {a: np.array(v) for a, v in out.items()}
V = {"no jitter": sim_stats(), "level jitter": sim_stats(level_jitter=LVL), "level+timing": sim_stats(level_jitter=LVL, tau_up=TUP, tau_dn=TDN)}
print("jitter used: level=%.2f nm/s2, tau_up=%.2f s, tau_dn=%.2f s\n" % (LVL, TUP, TDN))
print("wait | REAL: mean1 mean0  sd1  sd0 | z   || sim 'no jitter' sd1 sd0 z || sim 'level' sd1 sd0 z || sim 'level+timing' sd1 sd0 z  [z 5-95%]")
for a in WAITS:
    r = real[a]; line = "%4d | %6.1f %5.1f %5.2f %5.2f | %4.1f " % (a, r[0], r[1], r[2], r[3], r[4])
    for nm in V:
        v = V[nm][a]; z = v[:, 4]
        line += "|| %5.2f %5.2f %4.1f " % (v[:, 2].mean(), v[:, 3].mean(), z.mean())
    z = V["level+timing"][a][:, 4]; line += "[%.1f-%.1f] mean1/mean0=%.1f/%.1f" % (np.percentile(z, 5), np.percentile(z, 95), V["level+timing"][a][:, 0].mean(), V["level+timing"][a][:, 1].mean())
    print(line)

# ---- Re-run the SNR-4 window-start sweep (boxcar) with the realistic jitter ----
print("\n=== window-start sweep at SNR 4 (sensor noise fixed; jitter absolute) ===")
rng = np.random.default_rng(4242); nb = 6000; ch = 5; err = {k: {a: 0 for a in WAITS} for k in ("no jitter", "realistic jitter")}
for c in range(ch):
    bits = rng.integers(0, 2, nb)
    for k, kw in (("no jitter", {}), ("realistic jitter", dict(level_jitter=LVL, tau_up=TUP, tau_dn=TDN))):
        R = S.generate(bits, snr=4, lead_s=0, tail_s=60, seed=300 + c, **kw); s0 = R["bit_starts"]; x = R["x"]
        for a in WAITS:
            L = int(np.floor((60-a)*FS)); i0 = np.ceil((s0 + a)*FS).astype(int)
            err[k][a] += int(((x[i0[:, None] + np.arange(L)].mean(1) > A/2) != bits).sum())
n = nb * ch
for a in WAITS: print("wait %2ds: BER no-jitter=%.4f (%d)   realistic-jitter=%.4f (%d)" % (a, err["no jitter"][a]/n, err["no jitter"][a], err["realistic jitter"][a]/n, err["realistic jitter"][a], ))

# ---- figure: real message vs generator (window scatter and margin) ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6)); wa = np.array(WAITS)
for i, (nm, key) in enumerate([("'1' windows", 2), ("'0' windows", 3)]):
    ax[i].plot(wa, [real[a][key] for a in WAITS], "ko-", label="real message")
    ax[i].plot(wa, [V["no jitter"][a][:, key].mean() for a in WAITS], "s--", label="generator, no jitter")
    ax[i].plot(wa, [V["level jitter"][a][:, key].mean() for a in WAITS], "^-", label="generator + level jitter (%.1f nm/s\u00b2)" % LVL)
    ax[i].set_title("Scatter of window means: " + nm); ax[i].set_xlabel("window start (s after bit start)"); ax[i].set_ylabel("sd (nm/s\u00b2)"); ax[i].legend(); ax[i].grid(alpha=.3)
zs = np.array([V["level+timing"][a][:, 4] for a in WAITS])
ax[2].plot(wa, [real[a][4] for a in WAITS], "ko-", label="real message (63 bits)")
ax[2].plot(wa, [V["no jitter"][a][:, 4].mean() for a in WAITS], "s--", label="no jitter")
ax[2].plot(wa, zs.mean(1), "^-", label="level + timing jitter")
ax[2].fill_between(wa, np.percentile(zs, 5, axis=1), np.percentile(zs, 95, axis=1), alpha=.2, color="C2", label="5-95% of 300 sims")
ax[2].set_title("Worst-class margin / sd (higher = safer)"); ax[2].set_xlabel("window start (s)"); ax[2].set_ylabel("z"); ax[2].legend(); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_jitter_check.png", dpi=140); print("saved fig4_jitter_check.png")
