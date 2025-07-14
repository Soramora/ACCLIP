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
WEIGHTS="${1:-/home/smora/BUDA/video_prediction_project_syntax_fixed/model_weights/ACCLIP_alfa04_beta07.pth }"
CSV="${2:-/home/smora/val_KTH_actions.csv}"
ROOT_DIR="${3:-/home/smora/KTH_actions/frames}"
INITIAL_SEQ="${4:-10}"
MODE="${5:-val}"

# 5) Launch the KTH evaluation script
python evaluate_kth.py \
  --weights      "$WEIGHTS" \ #Path to the pretrained model weights (.pth file)
  --csv          "$CSV" \ # Path to the test split CSV file (video_path, video_length)
  --root-dir     "$ROOT_DIR" \ # Path to the Caltech Pedestrian dataset root directory
  --initial-seq  "$INITIAL_SEQ" \  
  --mode         "$MODE"  
