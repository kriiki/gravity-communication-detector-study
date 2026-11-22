import os
import sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_gravity as S
from scipy.stats import norm
from scipy.linalg import toeplitz, solve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")); os.makedirs(OUT, exist_ok=True); FS = S.FS; A = S.A0; BIT = 60.0
t00 = time.time(); res = {}

# ---------- noise autocovariance from the real-background PSD, GLS (whitened matched-filter) weights ----------
f = np.linspace(0, FS / 2, 40001); P = np.interp(f, S._F, S._PSD); P[0] = 0.0; df = f[1] - f[0]
acov = lambda m: np.array([np.sum(P * np.cos(2 * np.pi * f * k / FS)) * df for k in range(m)])
g = acov(200); print("acov(0)=%.1f (real background var ~ %.1f)" % (g[0], 9.70**2))
def weights(a):
    L = int(np.floor((BIT - a) * FS)); Sg = toeplitz(g[:L]); one = np.ones(L); z = solve(Sg, one)
    w = z / z.sum(); var_gls = 1.0 / one.dot(z); var_box = one.dot(Sg).dot(one) / L**2
    return L, w, var_box, var_gls
WA = [20, 25, 30, 35, 40, 45, 50]; W = {a: weights(a) for a in WA}
for a in WA: print("wait %2ds: L=%2d samples; boxcar sd=%.3f  GLS sd=%.3f  (SNR gain %.2f dB)" % (a, W[a][0], np.sqrt(W[a][2]), np.sqrt(W[a][3]), 10*np.log10(W[a][2]/W[a][3])))
res["gls_gain_db_wait35"] = float(10 * np.log10(W[35][2] / W[35][3]))

def errors_chunk(R, bits, waits):
    x = R["x"]; s = R["bit_starts"]; out = {}
    for a in waits:
        L, w, _, _ = W[a]; i0 = np.ceil((s + a) * FS).astype(int)
        X = x[i0[:, None] + np.arange(L)]
        out[a] = (int(((X.mean(1) > A / 2) != bits).sum()), int(((X @ w > A / 2) != bits).sum()))
    return out

# ---------- Exp A: BER vs SNR (a=35 s, the authors' window) + window-start sweep at SNR 4 ----------
plan = {1: 6000, 2: 6000, 3: 12000, 4: 24000, 5: 48000, 6: 90000}; CH = 6000; berA = {}; sweep = {}
for snr, nb in plan.items():
    e = {a: [0, 0] for a in WA}; tot = 0; rng = np.random.default_rng(snr)
    for c in range(nb // CH):
        bits = rng.integers(0, 2, CH); R = S.generate(bits, snr=snr, lead_s=0, tail_s=60, seed=1000 * snr + c)
        r = errors_chunk(R, bits, WA if snr == 4 else [35]); tot += CH
        for a, (b1, b2) in r.items(): e[a][0] += b1; e[a][1] += b2
    berA[snr] = dict(n=tot, box=e[35][0], gls=e[35][1]); print("A SNR=%d n=%d  box errors=%d (BER %.5f)  GLS errors=%d (BER %.5f)  theory %.5f  [%.0fs]" % (snr, tot, e[35][0], e[35][0]/tot, e[35][1], e[35][1]/tot, norm.sf(snr/2), time.time()-t00), flush=True)
    if snr == 4: sweep = {a: dict(box=e[a][0] / tot, gls=e[a][1] / tot, n=tot, eb=e[a][0], eg=e[a][1]) for a in WA}
res["A"] = berA; res["sweep_snr4"] = {str(k): v for k, v in sweep.items()}
for a in WA: print("sweep SNR=4 wait=%2ds: boxcar BER=%.5f (%d)  GLS BER=%.5f (%d)" % (a, sweep[a]["box"], sweep[a]["eb"], sweep[a]["gls"], sweep[a]["eg"]))

# ---------- Exp B: drift stress. fixed threshold vs drift-tracking threshold (500-bit messages ~8.3 h) ----------
PRE = np.array([1, 0, 1, 0, 1, 0, 1])
def adaptive(y, alpha=0.2):
    m1 = y[:7][PRE == 1].mean(); m0 = y[:7][PRE == 0].mean(); Ah = m1 - m0; b = m0; out = []
    for v in y[7:]:
        d = 1 if v > b + Ah / 2 else 0; out.append(d); b += alpha * (v - d * Ah - b)
    return np.array(out)
rngB = np.random.default_rng(77); drifts = [0, 10, 30, 100]; berB = {}; SNRB = 6; trials = 120
for d in drifts:
    ef = ea = n = 0
    for tr in range(trials):
        pay = rngB.integers(0, 2, 500); bits = np.r_[PRE, pay]
        R = S.generate(bits, snr=SNRB, lead_s=0, tail_s=0, drift=d, seed=50000 + 1000 * d + tr); y = S.window_means(R)
        ef += int(((y[7:] > A / 2) != pay).sum()); ea += int((adaptive(y) != pay).sum()); n += 500
    berB[d] = dict(fixed=ef / n, adaptive=ea / n, n=n, ef=ef, ea=ea)
    print("B drift=%3d nm/s2/sqrt(h): fixed-threshold BER=%.5f (%d)  adaptive BER=%.5f (%d)  n=%d" % (d, ef / n, ef, ea / n, ea, n), flush=True)
res["B"] = {str(k): v for k, v in berB.items()}
json.dump(res, open(f"{OUT}/classical_detectors.json", "w"), indent=1)

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
sn = sorted(berA); xx = np.linspace(1, 6, 100)
ax[0].semilogy(xx, norm.sf(xx / 2), "k-", lw=1, label="Gaussian theory Q(SNR/2)")
ax[0].semilogy(sn, [berA[s]["box"] / berA[s]["n"] for s in sn], "o-", label="threshold on 25 s mean (authors)")
ax[0].semilogy(sn, [berA[s]["gls"] / berA[s]["n"] for s in sn], "s--", label="whitened matched filter (GLS)")
ax[0].set_xlabel("SNR = A/σ_w (authors' window)"); ax[0].set_ylabel("BER"); ax[0].set_title("BER vs SNR (synthetic, coloured noise)"); ax[0].legend(); ax[0].grid(alpha=.3, which="both")
ax[1].semilogy(WA, [sweep[a]["box"] for a in WA], "o-", label="boxcar mean"); ax[1].semilogy(WA, [sweep[a]["gls"] for a in WA], "s--", label="GLS weights")
ax[1].axvline(35, color="k", ls=":"); ax[1].set_xlabel("window start (s after bit start; window runs to 60 s)"); ax[1].set_ylabel("BER at SNR 4 (noise fixed)"); ax[1].set_title("Effect of where the integration window starts"); ax[1].legend(); ax[1].grid(alpha=.3, which="both")
dd = [str(d) for d in drifts]; xi = np.arange(len(drifts))
ax[2].bar(xi - .18, [max(berB[d]["fixed"], 1e-5) for d in drifts], .36, label="fixed threshold A/2"); ax[2].bar(xi + .18, [max(berB[d]["adaptive"], 1e-5) for d in drifts], .36, label="drift-tracking threshold")
ax[2].set_xticks(xi); ax[2].set_xticklabels(dd); ax[2].set_yscale("log"); ax[2].set_xlabel("random-walk drift (nm/s²/√h), 500-bit messages, SNR 6"); ax[2].set_ylabel("BER"); ax[2].set_title("Drift stress test"); ax[2].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_classical_detectors.png", dpi=140); print("done %.0fs" % (time.time() - t00))
