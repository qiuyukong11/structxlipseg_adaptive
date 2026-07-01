#!/usr/bin/env bash
set -euo pipefail

SOURCE_OUTPUT="output"
SNAPSHOT_OUTPUT="output_best_cpu_snapshot"
DATASET_DIR="sketchy_structxlipseg_100percent_st_0.25_rs_0.02_chunk_0.07_gradbudget_align"
SEED="42"
RUN_NAME="Seg_structxlip_ViT-B-16_best_dice"
CKPT_NAME="${DATASET_DIR}_Seg_structxlip_ViT-B-16_best_dice.pth"
LOG_PATH="run_logs/test_best_cpu_snapshot.log"

mkdir -p "run_logs"
exec > "${LOG_PATH}" 2>&1

echo "[start] $(date)"
echo "[mode] CPU-only best_dice checkpoint test"

SRC_CKPT="${SOURCE_OUTPUT}/${DATASET_DIR}/trained_models/seed${SEED}/${CKPT_NAME}"
DST_DIR="${SNAPSHOT_OUTPUT}/${DATASET_DIR}/trained_models/seed${SEED}"
DST_CKPT="${DST_DIR}/${CKPT_NAME}"

mkdir -p "${DST_DIR}"
cp "${SRC_CKPT}" "${DST_CKPT}"
echo "[snapshot] ${SRC_CKPT} -> ${DST_CKPT}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

COMMON_OPTS=(
  STRUCTXLIP.ADAPTIVE_GRADBUDGET_ALIGN.ENABLED true
  STRUCTXLIP.ADAPTIVE_V2.ENABLED false
  STRUCTXLIP.ADAPTIVE_V3.ENABLED false
  STRUCTXLIP.ADAPTIVE_V4.ENABLED false
  STRUCTXLIP.ADAPTIVE_V6.ENABLED false
  STRUCTXLIP.ADAPTIVE_V7.ENABLED false
  STRUCTXLIP.LEARNABLE_TAU_LOSS.ENABLED false
  MODEL.DEVICE cpu
  TEST.USE_LATEST false
)

python test.py \
  --config-file configs/sketchy_structxlipseg_100percent_cpu_eval.yaml \
  --output-dir "${SNAPSHOT_OUTPUT}" \
  --seed "${SEED}" \
  "${COMMON_OPTS[@]}"

PRED_DIR="${SNAPSHOT_OUTPUT}/${DATASET_DIR}/seg_results/seed${SEED}/${RUN_NAME}"
GT_DIR="${SNAPSHOT_OUTPUT}/${DATASET_DIR}/gt_results/seed${SEED}/${RUN_NAME}"

python utils/eval.py \
  --config-file configs/sketchy_structxlipseg_100percent_cpu_eval.yaml \
  --output-dir "${SNAPSHOT_OUTPUT}" \
  --seed "${SEED}" \
  --pred-dir "${PRED_DIR}" \
  --gt-dir "${GT_DIR}" \
  "${COMMON_OPTS[@]}"

echo "[done] $(date)"
