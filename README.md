# gravity-communication-detector-study
Student data analysis of the UQ gravitational-communication experiment's open data: noise, detectors and data rate.
 Reading a gravitational message: a detector and noise study
 
This is a data-analysis project done for curiosity and practice by a second-year undergraduate student. It is not part of a course and it makes no research claims. It uses the authors' open data to study how reliably their gravitational message can be read, how noise and the choice of detector affect that, and what limits the data rate.

By line count, about 95% of the code here was written for this project. The remaining 5% (data loading, tide correction and drift correction in 01_reproduce_decoding.py) follows the authors' released script. The authors' data and the tide and Allan-variance libraries are not included.

An independent student analysis of the open data from Groszek, Woffinden, Harvey, White and Davis, *"Unblockable communication with gravity"* (Science Advances, 2026; arXiv:2511.16710), in which the word "Gravity" was sent through a wall at 1 bit/min by moving an elevator counterweight and read with a spring gravimeter.

This repository reproduces their decoding (63 of 63 bits), characterises the sensor noise, and builds a synthetic generator from the real noise. It uses the generator to compare detectors and to look at the trade-off between data rate and reliability. The sensor noise turns out to be strongly non-white, and the authors' 25 s window happens to cancel its main component. There is also about 2.7 nm/s² of extra scatter per bit beyond sensor noise. Logistic regression and gradient boosting did no better than a simple threshold, while soft-decision decoding of a small error-correcting code reduced errors a lot at higher noise. Everything rests on one 63-bit message from one run, and no literature review has been done, so no novelty is claimed.

## Results in more detail

- The published decoding reproduces: 63/63 bits correct; the closest bit is 17.6 nm/s² from the 25 nm/s² threshold.
- The sensor noise is not white: a strong peak near 0.08 Hz (about 12 s). The authors' 25 s window holds exactly two cycles and cancels it, so a whitened matched filter gains only 0.08 dB at that window.
- Inside the message the window means scatter more than sensor noise explains: about 2.7 nm/s² of extra per-bit scatter. Up transitions of the counterweight vary in timing (about 3.4 s sd); down transitions are repeatable (about 0.2 s).
- Logistic regression and gradient boosting never beat a plain threshold on the window mean (on-off signalling, real-spectrum noise, white, 1/f, spikes, drift), and models trained on white noise did as well on real noise as ones trained on it.
- Soft confidence is useful: the least-confident 20% of bits hold 96% of errors. Hamming(7,4) with soft decoding cut errors substantially at higher noise levels (e.g. 1.7% uncoded to 0.04% at 6x the real noise, at 4/7 of the rate).
- At the real noise level the rate is limited by the ~35 s needed to move and settle, not by sensor noise.

See `figures/` for the plots and `results/` for saved numbers.

## Limitations

- The generator is checked against one message. Fitted jitter (18 up and 18 down transitions) and drift (about ten Allan clusters) values could easily be off by 20% or more.
- It matches the real message for integration windows starting at 30 s or later. It does not reproduce the extra scatter in windows starting at 20-25 s, so results about early windows are not trusted.
- Windows shorter than 10 s are an extrapolation.
- Spike and drift settings are stress tests, not measured disturbances.
- The ML conclusion applies to on-off signalling with a window-mean decision. It does not say ML cannot help with richer modulation, joint decoding or the early part of the window.
- Only one error-correcting code (Hamming(7,4)) was tried.

## Setup

Python 3 with the packages in `requirements.txt` (tested with Python 3.12.3, numpy 2.5.4, scipy 1.18.1, pandas 3.0.6, matplotlib 3.11.2, scikit-learn 1.9.1).

```
pip install -r requirements.txt
```

The UQ data is not included. Download the authors' code and data release from UQ eSpace (DOI 10.48610/cd26dd2; https://espace.library.uq.edu.au/view/UQ:cd26dd2). The record states the data is CC-BY and the code MIT; please credit the authors. Unzip it so that this file exists:

```
data/fig_4_S4_S5/20241010_gz_rawdata.txt        (and data/fig_4_S4_S5/external/ with the authors' tide and Allan-variance code)
```

or point to that folder with `export UQ_DATA_DIR=/path/to/.../data/fig_4_S4_S5`.

## Run

```
bash run_all.sh
```

Outputs go to `outputs/` (override with `OUT_DIR`), logs to `logs/`. The full pipeline took about 3 minutes on my machine. A clean re-run reproduced the saved JSON and CSV results in `results/` exactly, with the versions above.

| Script | What it does |
|---|---|
| `01_reproduce_decoding.py` | Runs the authors' tide and drift correction, decodes all 63 bits, plots the raw trace and background spectrum |
| `02_build_noise_model.py` | Estimates the noise spectrum, step shapes and window noise from the real data; writes `synth_params.npz` |
| `synth_gravity.py` | The synthetic generator (module; also holds the fitted `REAL_JITTER` values) |
| `03_validate_generator.py` | Checks the generator against the real background (spectrum, Allan deviation, window statistics, BER vs Gaussian theory) |
| `04_classical_detectors.py` | Threshold vs whitened matched filter, window-start sweep, drift test |
| `05_check_window_start_real.py` | The window-start sweep on the real message |
| `06_fit_jitter.py` | Fits per-bit level scatter and transition timing scatter to the real message |
| `07_validate_jitter.py` | Checks the jittered generator against the real window statistics and draws the jitter-check figure |
| `08_ml_detectors.py` | Logistic regression and gradient boosting vs threshold detectors; generalisation to other noise; confidence |
| `09_check_ml_on_real.py` | Applies the synthetic-trained models to the real message |
| `10_rate_vs_coding.py` | Information rate vs reliability: window length and Hamming(7,4), hard and soft decoding |

## Using the generator

```python
import numpy as np, synth_gravity as S
bits = np.random.default_rng(0).integers(0, 2, 1000)
R = S.generate(bits, snr=4, **S.REAL_JITTER)     # snr = step amplitude / sd of the 25 s window mean
y = S.window_means(R)                             # decision statistic per bit
errors = ((y > R["amp"] / 2) != bits).sum()
```

`noise_scale=1` reproduces the real noise level (SNR about 25 on the background). Options include `noise_kind` (`real`, `white`, `pink`), `drift`, `spike_rate_h` and `quantize`. Treat drift and spikes as stress parameters.

## Credit

The data and the original experiment are the work of Groszek, Woffinden, Harvey, White and Davis at the University of Queensland (arXiv:2511.16710; data release DOI 10.48610/cd26dd2, CC-BY).

The loading, tide-correction and drift-correction steps in `01_reproduce_decoding.py` follow the authors' released script `generateExperimentalDataPlots.py`. The tide library (`tidegravity`) and the Allan-variance code are third-party code distributed with their release. They are not included in this repository and are loaded from the folder you download (see Setup).

The noise model, generator, detector comparison and coding study are an analysis of their data done for this project. Any errors in them are not the authors' responsibility.
