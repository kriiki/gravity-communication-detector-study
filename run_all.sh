#!/usr/bin/env bash
# Runs the whole pipeline in order. Needs the UQ data: set UQ_DATA_DIR to the folder containing 20241010_gz_rawdata.txt
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p logs outputs
export UQ_DATA_DIR="${UQ_DATA_DIR:-data/fig_4_S4_S5}"
: > logs/status.txt
for s in 01_reproduce_decoding 02_build_noise_model 03_validate_generator 04_classical_detectors 05_check_window_start_real 06_fit_jitter 07_validate_jitter 08_ml_detectors 09_check_ml_on_real 10_rate_vs_coding; do
  echo "== $s"; t0=$(date +%s)
  if python3 "$s.py" > "logs/$s.log" 2>&1; then echo "$s OK $(( $(date +%s) - t0 ))s" >> logs/status.txt; else echo "$s FAILED" >> logs/status.txt; tail -5 "logs/$s.log"; exit 1; fi
done
echo "all done"; cat logs/status.txt
