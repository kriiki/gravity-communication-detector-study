"""Synthetic gravimeter generator for the UQ gravitational-communication data.

Model (all in nm/s^2):  x(t) = signal(t) + noise_scale * coloured_gaussian(t) + drift(t) + spikes(t)
  * signal   : bit levels {0, A} with empirical up/down step responses measured from the real message
  * noise    : stationary Gaussian noise whose PSD is estimated from the REAL tide/drift-corrected background
  * drift    : optional random walk (stress parameter, not calibrated to real data)
  * spikes   : optional Poisson transients (stress parameter, not calibrated to real data)
  * quantise : optional 10 nm/s^2 rounding (the real PSD already contains the real quantisation noise)
SNR is defined as  A / sigma_w,  sigma_w = sd of the mean over the final 25 s of a bit window.
"""
import os
import numpy as np

_P = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "synth_params.npz"))
FS = float(_P["fs"]); A0 = float(_P["amp"]); BIT_S = 60.0; INT_S = 25.0; QUANT = 10.0
_F, _PSD = _P["freq"], _P["psd"]
C_UP, C_DOWN, C_T = _P["c_up"], _P["c_down"], _P["c_t"]
SIGMA_W_REAL = float(_P["sigma_w_real"])
REAL_JITTER = dict(level_jitter=2.73, tau_up=3.38, tau_dn=0.22)  # fitted to the real 63-bit message (06_fit_jitter.py): use generate(..., **REAL_JITTER)
# level_jitter: per-bit level scatter (nm/s^2) that does not average away; tau_*: per-transition timing sd (s). Validated for windows starting >= 30 s only.
DRIFT_FIT = 10.0   # nm/s^2 per sqrt(hour): rough random-walk fit to the real Allan deviation at 90-170 s (only ~10 clusters; see notes)

_VAR = float(np.sum(0.5 * (_PSD[1:] + _PSD[:-1]) * np.diff(_F)))      # total variance of the real background noise
def _psd_kind(f, kind):
    if kind == "real": return np.interp(f, _F, _PSD)
    if kind == "white": return np.full_like(f, _VAR / (FS / 2))
    if kind == "pink":                                              # 1/f, same total variance, strong slow wander
        g = 1.0 / np.maximum(f, _F[1]); c = _VAR / np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(f)); return c * g
    raise ValueError(kind)

def _coloured(n, rng, kind="real"):
    """Unit-scale Gaussian noise (zero mean). kind: 'real' = PSD of the real background; 'white'/'pink' = same variance, other spectrum."""
    w = rng.standard_normal(n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, d=1.0 / FS)
    H = np.sqrt(_psd_kind(f, kind) / (2.0 / FS))
    H[0] = 0.0
    return np.fft.irfft(W * H, n)

_SW0 = {}
def sigma_w_unit(kind="real", n=2_000_000, seed=12345):
    """sd of the decision statistic (mean of the final 25 s of a 60 s bit window) for noise_scale = 1, per noise kind.
    Calibrated with the SAME window indexing the detector uses (the real noise has a ~0.08 Hz peak, 2 cycles per 25 s)."""
    if kind not in _SW0:
        v = []
        for sd in range(10):                                   # average several long realisations: the noise has low-frequency structure
            x = _coloured(n, np.random.default_rng(seed + sd), kind); nbits = int(n / FS // BIT_S) - 1
            v.append(window_means(dict(x=x, bit_starts=BIT_S * np.arange(nbits))).var(ddof=1))
        _SW0[kind] = float(np.sqrt(np.mean(v)))
    return _SW0[kind]

def generate(bits, snr=None, noise_scale=None, amp=A0, lead_s=300, tail_s=300, drift=0.0,
             spike_rate_h=0.0, spike_amp=30.0, spike_tau_s=5.0, quantize=False, seed=0,
             level_jitter=0.0, tau_up=0.0, tau_dn=0.0, noise_kind="real"):
    """Return dict(t, x, bit_starts, bits, ...). bits: array of 0/1 (the message after lead-in).
    snr: A/sigma_w (overrides noise_scale). noise_scale=1 reproduces the real noise level (SNR ~ A/2.08).
    drift: random-walk rate in nm/s^2 per sqrt(hour)."""
    rng = np.random.default_rng(seed)
    bits = np.asarray(bits, int); nb = len(bits)
    if snr is not None:
        noise_scale = (amp / snr) / sigma_w_unit(noise_kind)
    elif noise_scale is None:
        noise_scale = 1.0
    T = lead_s + nb * BIT_S + tail_s; n = int(T * FS); t = np.arange(n) / FS
    # vectorised signal: sample i belongs to bit k = floor((t-lead)/60); level L[k], previous level L[k-1]
    # level_jitter: per-bit random level offset (nm/s^2, sd); tau_up/tau_dn: per-transition timing shift (s, sd)
    L = np.r_[0.0, amp * bits, 0.0]                         # L[0]=pre-message 0, L[nb+1]=returned to lower position
    k = np.clip(np.floor((t - lead_s) / BIT_S).astype(int) + 1, 0, nb + 1)   # 0 lead-in, 1..nb bits, nb+1 tail
    km = np.maximum(k - 1, 0)
    if level_jitter > 0 or tau_up > 0 or tau_dn > 0:
        rj = np.random.default_rng([seed, 99])              # separate stream: noise realisation unchanged by jitter settings
        dl = np.r_[0.0, rj.normal(0, level_jitter, nb), 0.0]; tu = rj.normal(0, tau_up, nb + 2); td = rj.normal(0, tau_dn, nb + 2)
    else:
        dl = np.zeros(nb + 2); tu = td = np.zeros(nb + 2)
    Lj = L + dl; cur = Lj[k]; prv = Lj[km]
    up = L[k] > L[km]; dn = L[k] < L[km]
    ph = np.where(k >= 1, (t - lead_s) - (k - 1) * BIT_S, 0.0) - np.where(up, tu[k], td[k])
    cu = np.interp(ph, C_T, C_UP); cd = np.interp(ph, C_T, C_DOWN)
    sig = np.where(up, prv + (cur - prv) * cu, np.where(dn, prv + (cur - prv) * cd, cur))
    x = sig + noise_scale * _coloured(n, rng, noise_kind)
    if drift > 0:
        x = x + np.cumsum(rng.standard_normal(n)) * drift / np.sqrt(FS * 3600.0)
    if spike_rate_h > 0:
        for t0 in rng.uniform(0, T, rng.poisson(spike_rate_h * T / 3600.0)):
            d = t - t0; m = d >= 0
            x[m] += rng.normal(0, spike_amp) * np.exp(-d[m] / spike_tau_s)
    if quantize:
        x = np.round((x + rng.uniform(0, QUANT)) / QUANT) * QUANT
    starts = lead_s + BIT_S * np.arange(nb)
    return dict(t=t, x=x, bit_starts=starts, bits=bits, noise_scale=noise_scale, signal=sig, amp=amp)

def window_means(res, wait_s=35.0):
    """Mean over the final 25 s of every bit window (the authors' decision statistic), vectorised."""
    x = res["x"]; cs = np.r_[0.0, np.cumsum(x)]
    i0 = np.ceil((res["bit_starts"] + wait_s) * FS).astype(int); i1 = np.minimum(np.ceil((res["bit_starts"] + BIT_S) * FS).astype(int), len(x))
    return (cs[i1] - cs[i0]) / (i1 - i0)
