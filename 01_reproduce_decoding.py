"""Reproduce decoding of 'Gravity' from UQ data; plot raw trace and PSD.
Uses the authors' own pipeline (tide + quartic drift correction) and bundled tidegravity code."""
# Data loading, tide correction and drift correction follow the authors' generateExperimentalDataPlots.py (UQ data release, DOI 10.48610/cd26dd2).
# The tide and Allan-variance libraries are third-party code shipped with their release and are imported from there.
import sys, os, datetime
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

ROOT = os.environ.get("UQ_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fig_4_S4_S5"))
sys.path.insert(0, ROOT)
from external.tidegravity_0_5_0.tidegravity.tidegravity import solve_longman_tide

OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")); os.makedirs(OUT, exist_ok=True)
# ---- load (same steps as authors) ----
tab = pd.read_table(os.path.join(ROOT, "20241010_gz_rawdata.txt"), delimiter=",", header=None,
                    skiprows=1100, dtype={0: str, 1: str, 2: str})
tab[0] = tab[0] + "00000"
tab[1] = tab[1].str[4:].astype(np.double)
tdt = np.array([datetime.datetime.fromisoformat(s) - datetime.timedelta(hours=10) for s in tab[0]])
tsec = np.array([(t - tdt[0]).total_seconds() for t in tdt])
th = tsec / 3600.0
gRaw = tab[1].to_numpy() * 0.01 / 1e3          # mGal -> m/s^2
gRaw = gRaw - np.median(gRaw[:100])
fs = 1.0 / np.mean(np.diff(tsec))
print(f"samples={len(th)}, duration={th[-1]:.3f} h, mean sample rate={fs:.3f} Hz")

# ---- tide + drift correction (authors' method) ----
_, _, dgTot = solve_longman_tide(lat=-27.49836*np.ones_like(th), lon=153.01476*np.ones_like(th),
                                 alt=25*np.ones_like(th), time=tdt)
dgTot = -dgTot * 0.01 / 1e3
gTide = gRaw - dgTot

bitLen, dtWait, tStart = 60/3600, 35/3600, 0.46
tBits = tStart + bitLen * np.arange(64)
tLet = tBits[[0, 7, 14, 21, 28, 35, 42, 49, 56, 63]]
fit = (th < tLet[0]) | (th > tLet[-1] + 0.01) | ((th > 0.857) & (th < 0.91)) | ((th > 0.605) & (th < 0.64))
p4 = np.polyfit(th[fit], gTide[fit], 4)
gCor = (gTide - np.polyval(p4, th)) / 1e-9      # nm/s^2

# ---- decode ----
thr = 25.0
means, nsamp = [], []
for i in range(63):
    w = (th >= tBits[i] + dtWait) & (th < tBits[i] + bitLen)
    means.append(gCor[w].mean()); nsamp.append(w.sum())
means = np.array(means); decoded = (means > thr).astype(int)
word = "Gravity"
expected = [1,0,1,0,1,0,1] + [int(b) for c in word for b in format(ord(c), "07b")] + [1,0,1,0,1,0,1]
expected = np.array(expected)
errs = int((decoded != expected).sum())
chars = ["".join(map(str, decoded[7*k:7*k+7])) for k in range(9)]
txt = "".join(chr(int(c, 2)) for c in chars[1:8])
print("decoded chars (bits):", chars)
print("decoded text:", repr(txt), "| bit errors vs transmitted:", errs, "of 63")
print(f"samples per 25s window: min={min(nsamp)}, max={max(nsamp)}")
m1, m0 = means[expected == 1], means[expected == 0]
print(f"window means: '1' mean={m1.mean():.1f} sd={m1.std(ddof=1):.1f} | '0' mean={m0.mean():.1f} sd={m0.std(ddof=1):.1f} nm/s^2")
print(f"closest bit to threshold: {np.min(np.abs(means-thr)):.1f} nm/s^2 away")

# ---- background noise stats (pre-message, corrected) ----
bg = th < tStart
print(f"background corrected sample sd={gCor[bg].std(ddof=1):.2f} nm/s^2 (n={bg.sum()})")
print(f"raw 10 nm/s^2 quantisation check: unique steps ~ {np.round(np.median(np.diff(np.unique(np.round(gRaw/1e-9,3)))),2)}")

# ---- plots ----
fig, ax = plt.subplots(3, 1, figsize=(11, 10))
ax[0].plot(th - tStart, gRaw/1e-9, lw=.8, label="raw"); ax[0].plot(th - tStart, dgTot/1e-9 + (gRaw[0]-dgTot[0])/1e-9*0, "k:", lw=1, label="tide (offset arbitrary)")
ax[0].set_ylabel("raw gz (nm/s²)"); ax[0].set_xlabel("time since first bit (h)"); ax[0].legend(); ax[0].set_title("Raw trace")
ax[1].plot(th - tStart, gCor, lw=.7)
for i in range(63):
    ax[1].hlines(means[i], tBits[i]+dtWait-tStart, tBits[i]+bitLen-tStart, color="C3", lw=1.5)
ax[1].axhline(thr, color="k", ls=":"); ax[1].set_xlim(-0.1, 1.1)
ax[1].set_ylabel("corrected gz (nm/s²)"); ax[1].set_xlabel("time since first bit (h)")
ax[1].set_title(f"Corrected; decoded {txt!r}, {errs}/63 bit errors")
f_raw, P_raw = welch(gRaw[bg]/1e-9, fs=fs, nperseg=min(512, bg.sum()//2))
f_cor, P_cor = welch(gCor[bg], fs=fs, nperseg=min(512, bg.sum()//2))
ax[2].loglog(f_raw[1:], P_raw[1:], label="raw background"); ax[2].loglog(f_cor[1:], P_cor[1:], label="corrected background")
ax[2].set_xlabel("frequency (Hz)"); ax[2].set_ylabel("PSD ((nm/s²)²/Hz)"); ax[2].legend(); ax[2].set_title("Background PSD (before message)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_reproduction.png", dpi=150)
np.savetxt(f"{OUT}/bit_windows.csv", np.c_[np.arange(63), expected, decoded, means], delimiter=",",
           header="bit_index,transmitted,decoded,window_mean_nm_s2", fmt=["%d","%d","%d","%.2f"], comments="")
print("saved plot + csv")
