#!/bin/bash
set -e


OUTPUT=output
SEED=42 # 666

clip_model=structxlip
CONFIG="configs/sketchy_structxlip.yaml"
DATASETS="sketchy"

echo "===================================="
echo "CLIP MODEL: ${clip_model}"
echo "SOURCE DATASET: ${DATASETS}"
echo "TARGET DATASETS: ${DATASETS}"
echo "===================================="

# -------------------------------
# Conditional backbone override
# -------------------------------
OPTS="MODEL.CLIP_MODEL ${clip_model}"

# -------------------------------
# Train once on DATASETS
# -------------------------------
python train.py \
    --config-file ${CONFIG} \
    --resume \
    --output-dir ${OUTPUT} \
    --seed ${SEED} \
    ${OPTS}

# -------------------------------
# Evaluate 
# -------------------------------
echo "Evaluating SOURCE=${DATASETS} → TARGET=${DATASETS}"

python test.py \
    --config-file ${CONFIG} \
    --output-dir ${OUTPUT} \
    --source_dataset ${DATASETS} \
    --seed ${SEED} \
    ${OPTS}

python utils/eval.py \
    --config-file ${CONFIG} \
    --output-dir ${OUTPUT} \
    --seed ${SEED} \
    ${OPTS}
