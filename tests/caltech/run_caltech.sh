#!/usr/bin/env bash
set -euo pipefail

# 1) Locate this script and derive project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 2) cd into script dir so relative paths resolve
cd "$SCRIPT_DIR"

# 3) Tell Python to look in the parent folder for modules/
export PYTHONPATH="$PROJECT_ROOT"

# 4) Read positional args (or defaults)
WEIGHTS="${1:-/home/smora/model_weights/ACCLIP_100epocas_alfa04_beta07_8pred_MSE_0024.pth}"
CSV="${2:-/home/smora/caltech_pedestrian_test.csv}"
ROOT_DIR="${3:-caltech_pedestrian/frames}"
NUM_PRED="${4:-8}"
INITIAL_SEQ="${5:-10}"
HEIGHT="${6:-128}"
WIDTH="${7:-128}"
MODE="${8:-val}"

# 5) Launch the evaluation script
python evaluate_caltech.py \
  --weights      "$WEIGHTS" \
  --csv          "$CSV" \
  --root-dir     "$ROOT_DIR" \
  --num-pred     "$NUM_PRED" \
  --initial-seq  "$INITIAL_SEQ" \
  --height       "$HEIGHT" \
  --width        "$WIDTH" \
  --mode         "$MODE"


