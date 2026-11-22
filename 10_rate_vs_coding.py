import os
"""Information rate vs error rate. Integration-window length (tau) and Hamming(7,4) coding, hard vs soft decisions.
Bit period P = 35 s (move + settle, as in the experiment) + tau. Window = [35 s, 35+tau) of each bit. Uses the validated generator
(real-PSD noise + fitted level jitter); sensor noise scaled by ns (ns=1 = the real noise level)."""
import sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_gravity as S
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")); os.makedirs(OUT, exist_ok=True); FS = S.FS; A = S.A0; JIT = S.REAL_JITTER; T0 = time.time()
P = np.array([[1,1,0],[1,0,1],[0,1,1],[1,1,1]]); G = np.c_[np.eye(4, dtype=int), P]; H = np.c_[P.T, np.eye(3, dtype=int)]
D16 = np.array([[(i >> b) & 1 for b in range(4)] for i in range(16)]); CW = D16 @ G % 2           # 16 codewords
syn2pos = {}
for j in range(7): syn2pos[int("".join(map(str, H[:, j])), 2)] = j                                  # syndrome (as int) -> error position
def hard_decode(r):                                                                                 # r: (n,7) hard bits -> info bits (n,4)
    s = r @ H.T % 2; code = s[:, 0] * 4 + s[:, 1] * 2 + s[:, 2]; out = r.copy()
    for sv, pos in syn2pos.items(): out[code == sv, pos] ^= 1
    return out[:, :4]
def soft_decode(y):                                                                                 # y: (n,7) window means -> info bits (n,4) (ML, equal-variance Gaussian)
    return D16[np.argmax((y - A / 2) @ (2 * CW - 1).T, axis=1)]

def wmeans(x, s0, tau):
    cs = np.r_[0.0, np.cumsum(x)]; i0 = np.ceil((s0 + 35) * FS).astype(int); L = int(np.floor(tau * FS)); return (cs[i0 + L] - cs[i0]) / L

TAUS = [5, 10, 15, 25]; NS = [1, 4, 6, 8]; NCH, CWPC = 25, 1000; res = {}
for ns in NS:
    err_u = {t: 0 for t in TAUS}; err_h = {t: 0 for t in TAUS}; err_s = {t: 0 for t in TAUS}; nbits = ninfo = 0
    rng = np.random.default_rng(ns)
    for c in range(NCH):
        info = rng.integers(0, 2, (CWPC, 4)); cw = info @ G % 2; bits = cw.reshape(-1)
        R = S.generate(bits, noise_scale=ns, lead_s=0, tail_s=60, seed=5000 * ns + c, **JIT)
        for tau in TAUS:
            y = wmeans(R["x"], R["bit_starts"], tau); hb = (y > A / 2).astype(int)
            err_u[tau] += int((hb != bits).sum())
            err_h[tau] += int((hard_decode(hb.reshape(-1, 7)) != info).sum()); err_s[tau] += int((soft_decode(y.reshape(-1, 7)) != info).sum())
        nbits += len(bits); ninfo += info.size
    res[ns] = {t: dict(P=35 + t, uncoded=[err_u[t], nbits], hamming_hard=[err_h[t], ninfo], hamming_soft=[err_s[t], ninfo]) for t in TAUS}
    print("noise x%d (%.0fs)" % (ns, time.time() - T0), flush=True)
    for t in TAUS:
        r = res[ns][t]; rate = 60 / (35 + t)
        print("  tau=%2ds P=%2ds | uncoded %.2f b/min BER=%.2e (%d) | Hamming hard %.2f b/min BER=%.2e (%d) | Hamming soft %.2f b/min BER=%.2e (%d)" % (
            t, 35 + t, rate, r["uncoded"][0] / nbits, r["uncoded"][0], rate * 4 / 7, r["hamming_hard"][0] / ninfo, r["hamming_hard"][0], rate * 4 / 7, r["hamming_soft"][0] / ninfo, r["hamming_soft"][0]))

TARGET = 1e-3
def ok(e, n): return (e / n if e > 0 else 3.0 / n) <= TARGET                                         # zero errors -> 95% upper bound 3/N
print("\nBest information rate (bits/min) with BER <= %.0e (zero-error cells use the 95%% bound 3/N):" % TARGET)
best = {}
for ns in NS:
    cands = []
    for t in TAUS:
        r = res[ns][t]; rate = 60 / (35 + t)
        if ok(*r["uncoded"]): cands.append((rate, "uncoded, tau=%ds" % t))
        if ok(*r["hamming_hard"]): cands.append((rate * 4 / 7, "Hamming hard, tau=%ds" % t))
        if ok(*r["hamming_soft"]): cands.append((rate * 4 / 7, "Hamming soft, tau=%ds" % t))
    best[ns] = max(cands) if cands else None
    u25 = res[ns][25]["uncoded"]
    print("  noise x%d: best = %s | reference uncoded 25 s window (1.00 b/min): BER=%.2e" % (ns, ("%.2f b/min via %s" % best[ns]) if best[ns] else "none meets target", u25[0] / u25[1]))
res["best"] = {str(k): v for k, v in best.items()}; json.dump({str(k): {str(t): v for t, v in d.items()} if isinstance(d, dict) and k != "best" else d for k, d in res.items()}, open(f"{OUT}/rate_vs_coding.json", "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8)); floor = 2e-6
for a, ns in zip(ax, [4, 6, 8]):
    for key, lab, mk, mult in (("uncoded", "uncoded", "o-", 1.0), ("hamming_hard", "Hamming(7,4), hard", "s--", 4 / 7), ("hamming_soft", "Hamming(7,4), soft", "^-", 4 / 7)):
        rates = [mult * 60 / (35 + t) for t in TAUS]; b = [max(res[ns][t][key][0] / res[ns][t][key][1], floor) for t in TAUS]
        a.semilogy(rates, b, mk, label=lab)
        for t, rr, bb in zip(TAUS, rates, b): a.annotate("%ds" % t, (rr, bb), fontsize=6, xytext=(3, 3), textcoords="offset points")
    a.axhline(TARGET, color="k", ls=":", lw=1); a.set_xlabel("information rate (bits/min)"); a.set_ylabel("information BER (floor = none seen)"); a.set_title("sensor noise x%d (labels: integration window)" % ns); a.legend(fontsize=7); a.grid(alpha=.3, which="both")
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_rate_vs_coding.png", dpi=140); print("done %.0fs" % (time.time() - T0))
