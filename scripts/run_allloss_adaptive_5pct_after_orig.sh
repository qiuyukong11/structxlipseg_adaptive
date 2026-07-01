#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="/mnt/data/zruan/kqy/pami/segmentation/structxlip_seg_v2_adaptive"
CONFIG_FILE="configs/sketchy_structxlipseg_5percent_debug.yaml"
WAIT_SESSION="${WAIT_SESSION:-autolambda_orig_5pct}"
LOG_DIR="${ROOT_DIR}/output/allloss_adaptive_5pct_sequence"
LOG_FILE="${LOG_DIR}/sequence.log"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') WAIT for tmux session ${WAIT_SESSION} ====="
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') still waiting for ${WAIT_SESSION} ..."
    sleep 60
  done
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') WAIT finished; start all-loss adaptive sequence ====="
} 2>&1 | tee -a "${LOG_FILE}"

run_strategy() {
  local name="$1"
  local weight_method="$2"
  local grad_method="$3"

  {
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') START ${name}: weight=${weight_method}, grad=${grad_method} ====="
    python train_allloss_autolambda.py       --config-file "${CONFIG_FILE}"       --weight-method "${weight_method}"       --grad-method "${grad_method}"       --autol-val-batches 4
    local status=$?
    if [[ ${status} -eq 0 ]]; then
      echo "===== $(date '+%Y-%m-%d %H:%M:%S') END ${name}: SUCCESS ====="
    else
      echo "===== $(date '+%Y-%m-%d %H:%M:%S') END ${name}: FAILED status=${status}; continue ====="
    fi
  } 2>&1 | tee -a "${LOG_FILE}"
}

run_strategy "Uncertainty" "uncert" "none"
run_strategy "Dynamic Weight Average" "dwa" "none"
run_strategy "GradDrop" "equal" "graddrop"
run_strategy "Auto-Lambda" "autol" "none"
run_strategy "PCGrad" "equal" "pcgrad"
run_strategy "CAGrad" "equal" "cagrad"

{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') all-loss adaptive sequence finished ====="
} 2>&1 | tee -a "${LOG_FILE}"
