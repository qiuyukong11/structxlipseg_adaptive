#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/data/zruan/kqy/pami/segmentation/structxlip_seg_v2_adaptive"
CONFIG_FILE="configs/sketchy_structxlipseg_5percent_debug.yaml"
LOG_DIR="${ROOT_DIR}/output/autolambda_5pct_sequence"
LOG_FILE="${LOG_DIR}/sequence.log"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

run_strategy() {
  local name="$1"
  local weight_method="$2"
  local grad_method="$3"

  echo "===== $(date '+%Y-%m-%d %H:%M:%S') START ${name}: weight=${weight_method}, grad=${grad_method} ====="
  python train_autolambda.py \
    --config-file "${CONFIG_FILE}" \
    --weight-method "${weight_method}" \
    --grad-method "${grad_method}"
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') END ${name}: weight=${weight_method}, grad=${grad_method} ====="
}

{
  run_strategy "Uncertainty" "uncert" "none"
  run_strategy "Dynamic Weight Average" "dwa" "none"
  run_strategy "Auto-Lambda" "autol" "none"
  run_strategy "GradDrop" "equal" "graddrop"
  run_strategy "PCGrad" "equal" "pcgrad"
  run_strategy "CAGrad" "equal" "cagrad"
} 2>&1 | tee -a "${LOG_FILE}"
