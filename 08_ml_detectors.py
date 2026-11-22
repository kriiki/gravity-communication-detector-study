import os
import sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_gravity as S
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")); os.makedirs(OUT, exist_ok=True); FS = S.FS; A = S.A0; JIT = S.REAL_JITTER; T0 = time.time()
L30 = int(np.floor(30 * FS)); L35 = int(np.floor(25 * FS)); FEATS = ["mean", "median", "slope", "std", "mean_first3rd", "mean_last3rd", "band_0.05-0.13Hz", "band_0.2-0.7Hz", "range"]

def feats(x, s0):
    """Features of the 30-60 s part of each bit (the part the generator is validated for). Also the authors' 35-60 s mean."""
    i0 = np.ceil((s0 + 30) * FS).astype(int); X = x[i0[:, None] + np.arange(L30)]
    t = (np.arange(L30) - (L30 - 1) / 2) / FS; mean = X.mean(1)
    slope = (X * t).sum(1) / (t**2).sum(); Xd = X - mean[:, None]; Xt = Xd - slope[:, None] * t; th = L30 // 3
    F = np.abs(np.fft.rfft(Xt, axis=1))**2 / L30; fr = np.fft.rfftfreq(L30, 1 / FS)
    f = np.c_[mean, np.median(X, 1), slope, Xd.std(1), X[:, :th].mean(1), X[:, -th:].mean(1),
              F[:, (fr >= .05) & (fr < .13)].sum(1), F[:, (fr >= .2) & (fr < .7)].sum(1), X.max(1) - X.min(1)]
    j0 = np.ceil((s0 + 35) * FS).astype(int); auth = x[j0[:, None] + np.arange(L35 + 1)[: 46]].mean(1)
    return f, auth

def make(snr, nb, kind="real", seed0=0, msg=6000, **kw):
    Fs, As, ys = [], [], []; rng = np.random.default_rng(seed0); k = 0
    while sum(len(y) for y in ys) < nb:
        n = min(msg, nb - sum(len(y) for y in ys)); bits = rng.integers(0, 2, n)
        R = S.generate(bits, snr=snr, lead_s=0, tail_s=60, seed=seed0 * 1000 + k, noise_kind=kind, **JIT, **kw); k += 1
        f, a = feats(R["x"], R["bit_starts"]); Fs.append(f); As.append(a); ys.append(bits)
    return np.vstack(Fs), np.concatenate(As), np.concatenate(ys)

def train(kind, seed0):
    parts = [make(s, 12000, kind, seed0 + i) for i, s in enumerate((3, 4, 5))]
    X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[2] for p in parts])
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(X, y)
    return lr, gb
lr_r, gb_r = train("real", 10); lr_w, gb_w = train("white", 20)
print("trained (%.0fs). Logistic-regression standardised coefficients (real-noise model):" % (time.time() - T0))
for nm, c in zip(FEATS, lr_r[-1].coef_[0]): print("   %-16s %+.2f" % (nm, c))

def detect(F, auth, models):
    d = {"threshold, authors' 35-60 s mean": auth > A / 2, "threshold, 30-60 s mean": F[:, 0] > A / 2, "threshold, 30-60 s median": F[:, 1] > A / 2}
    for nm, m in models.items(): d[nm] = m.predict(F).astype(bool)
    return d
RM = {"logistic regression": lr_r, "gradient boosting": gb_r}
res = {"E1": {}, "E2": {}}

# ---- E1: BER vs SNR (real-PSD noise + level jitter), models trained on real-noise mix SNR 3-5 ----
print("\n=== E1. BER vs SNR (in-distribution) ===")
for snr, nb in {2: 20000, 3: 20000, 4: 30000, 5: 40000, 6: 60000}.items():
    F, au, y = make(snr, nb, "real", 100 + snr); d = detect(F, au, RM); res["E1"][snr] = {k: [int((v != y.astype(bool)).sum()), len(y)] for k, v in d.items()}
    print("SNR %d (n=%d, theory %.4f): " % (snr, len(y), norm.sf(snr / 2)) + " | ".join("%s %.4f (%d)" % (k.replace("threshold, ", "thr ")[:24], v[0] / v[1], v[0]) for k, v in res["E1"][snr].items()), flush=True)

# ---- E2: generalisation at SNR 4 ----
print("\n=== E2. Generalisation at SNR 4 (models trained on real-noise mix) ===")
conds = {"real noise (in-distribution)": dict(kind="real"), "white noise": dict(kind="white"), "pink (1/f) noise": dict(kind="pink"),
         "real noise + spikes (20/h, 60 nm/s2)": dict(kind="real", spike_rate_h=20, spike_amp=60, spike_tau_s=5),
         "real noise + drift 10 nm/s2/sqrt(h), 200-bit msgs": dict(kind="real", drift=10.0, msg=200)}
for i, (nm, kw) in enumerate(conds.items()):
    F, au, y = make(4, 30000, seed0=200 + i, **kw); d = detect(F, au, RM); res["E2"][nm] = {k: [int((v != y.astype(bool)).sum()), len(y)] for k, v in d.items()}
    print("%-50s " % nm + " | ".join("%s %.4f" % (k.replace("threshold, ", "thr ")[:22], v[0] / v[1]) for k, v in res["E2"][nm].items()), flush=True)
print("\n--- reverse test: models trained on WHITE noise only ---")
WM = {"logistic regression (white-trained)": lr_w, "gradient boosting (white-trained)": gb_w}
res["E2w"] = {}
for i, nm in enumerate(["real noise (in-distribution)", "pink (1/f) noise"]):
    kw = conds[nm]; F, au, y = make(4, 30000, seed0=300 + i, **kw); d = detect(F, au, WM); res["E2w"][nm] = {k: [int((v != y.astype(bool)).sum()), len(y)] for k, v in d.items()}
    print("%-34s tested on %-30s " % ("white-trained models", nm) + " | ".join("%s %.4f" % (k[:34], v[0] / v[1]) for k, v in res["E2w"][nm].items() if "white-trained" in k))
F, au, y = make(4, 30000, "white", 310); d = detect(F, au, WM); print("white-trained tested on white (in-distribution): " + " | ".join("%s %.4f" % (k[:34], (v != y.astype(bool)).mean()) for k, v in d.items() if "white-trained" in k))

# ---- E3: confidence from logistic regression ----
print("\n=== E3. Per-bit confidence (logistic regression, SNR 4, real noise) ===")
F, au, y = make(4, 40000, "real", 400); p = lr_r.predict_proba(F)[:, 1]; err = (p > .5) != y.astype(bool); conf = np.abs(p - .5)
q = np.quantile(conf, [0, .2, .4, .6, .8, 1.0]); e3 = []
for i in range(5):
    m = (conf >= q[i]) & (conf <= q[i+1] if i == 4 else conf < q[i+1]); e3.append(err[m].mean()); print("confidence quintile %d (lowest=1): BER %.4f (n=%d, errors %d)" % (i+1, err[m].mean(), m.sum(), err[m].sum()))
low = conf <= np.quantile(conf, 0.2); print("lowest 20%% of confidence holds %.0f%% of all errors (overall BER %.4f)" % (100 * err[low].sum() / err.sum(), err.mean()))
bins = np.linspace(0, 1, 6); print("calibration (predicted P(bit=1) vs observed):")
for i in range(5):
    m = (p >= bins[i]) & (p < bins[i+1] + (1e-9 if i == 4 else 0)); print("   p in [%.1f,%.1f): mean pred %.3f observed %.3f (n=%d)" % (bins[i], bins[i+1], p[m].mean(), y[m].mean(), m.sum()))
res["E3"] = dict(quintile_ber=[float(v) for v in e3], low20_error_share=float(err[low].sum() / err.sum()))
json.dump(res, open(f"{OUT}/ml_detectors.json", "w"), indent=1)

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(18, 5)); names = list(res["E1"][2].keys()); snrs = sorted(res["E1"])
ax[0].semilogy(np.linspace(2, 6, 50), norm.sf(np.linspace(2, 6, 50) / 2), "k-", lw=1, label="Gaussian theory (no jitter)")
for nm, mk in zip(names, ["o-", "s-", "d-", "^--", "v--"]): ax[0].semilogy(snrs, [res["E1"][s][nm][0] / res["E1"][s][nm][1] for s in snrs], mk, label=nm.replace("threshold, ", "thr: "), alpha=.85)
ax[0].set_xlabel("SNR (sensor noise)"); ax[0].set_ylabel("BER"); ax[0].set_title("BER vs SNR, realistic noise"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, which="both")
cn = list(res["E2"].keys()); x = np.arange(len(cn)); w = 0.16
for j, nm in enumerate(names): ax[1].bar(x + (j - 2) * w, [max(res["E2"][c][nm][0] / res["E2"][c][nm][1], 1e-4) for c in cn], w, label=nm.replace("threshold, ", "thr: "))
ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels([c.replace(" (in-distribution)", "\n(in-dist.)").replace(" + ", "\n+ ").replace(" nm/s2/sqrt(h), 200-bit msgs", "\n200-bit msgs") for c in cn], fontsize=7)
ax[1].set_ylabel("BER at SNR 4"); ax[1].set_title("Generalisation to other noise (trained on real-noise mix)"); ax[1].legend(fontsize=6)
ax[2].bar(np.arange(1, 6), e3, color="C1"); ax[2].set_xlabel("confidence quintile (1 = least confident)"); ax[2].set_ylabel("BER"); ax[2].set_title("Logistic-regression confidence vs error rate")
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_ml_detectors.png", dpi=140); print("done %.0fs" % (time.time() - T0))
