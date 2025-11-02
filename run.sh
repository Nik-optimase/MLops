#!/usr/bin/env bash
# Entrypoint script for the ML scoring service.
#
# This script runs the preprocessing followed by prediction. It assumes
# that the input CSV is mounted at `/app/input/test.csv` and that the
# output directory exists at `/app/output`.

set -euo pipefail

# Run preprocessing (writes to ./work/processed.csv)
python3 preprocess.py

# Run prediction (reads from ./work/processed.csv and writes outputs)
python3 predict.py

echo "Inference pipeline completed successfully."