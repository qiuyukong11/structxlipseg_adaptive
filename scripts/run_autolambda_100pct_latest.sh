#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/zruan/kqy/pami/segmentation/structxlip_seg_v2_adaptive"
CONFIG="configs/sketchy_structxlipseg_100percent.yaml"
WEIGHT="autol"
GRAD="none"
DATASET="sketchy_structxlipseg_100percent_st_0.25_rs_0.02_chunk_0.07_autolambda_${WEIGHT}_${GRAD}"
RUN_NAME="Seg_structxlip_ViT-B-16_latest"
LOG_DIR="${ROOT}/output/autolambda_100pct_autol_latest"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

{
  echo "===== $(date +%F_%T) TRAIN Auto-Lambda 100% Sketchy START ====="
  python train_autolambda.py --config-file "${CONFIG}" --resume --weight-method "${WEIGHT}" --grad-method "${GRAD}" TRAIN.BATCH_SIZE 8
  echo "===== $(date +%F_%T) TRAIN Auto-Lambda 100% Sketchy END ====="

  echo "===== $(date +%F_%T) TEST latest START ====="
  python test.py --config-file "${CONFIG}" --weight-method "${WEIGHT}" --grad-method "${GRAD}" TRAIN.BATCH_SIZE 8 TEST.USE_LATEST true
  echo "===== $(date +%F_%T) TEST latest END ====="

  echo "===== $(date +%F_%T) EVAL latest START ====="
  python utils/eval.py \
    --config-file "${CONFIG}" \
    --pred-dir "output/${DATASET}/seg_results/seed42/${RUN_NAME}" \
    --gt-dir "output/${DATASET}/gt_results/seed42/${RUN_NAME}"
  echo "===== $(date +%F_%T) EVAL latest END ====="
} 2>&1 | tee -a "${LOG_DIR}/run.log"
