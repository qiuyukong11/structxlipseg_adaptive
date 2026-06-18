# A2: StructXLIPSeg only st
# A3: StructXLIPSeg only rs
# A4: StructXLIPSeg only chunk
# A5: StructXLIPSeg st + chunk
# A6: StructXLIPSeg st + rs
# A7: StructXLIPSeg chunk + rs

#!/usr/bin/env bash
set -e

cd /mnt/data/zruan/kqy/pami/segmentation/structxlip_seg_v2_adaptive

mkdir -p run_logs_abalation

CONFIG="configs/sketchy_structxlipseg_5percent_debug.yaml"
OUTPUT=abalation
SEED=42 # 666
DATASETS="sketchy"

# nohup python train.py \
#   --config-file ${CONFIG} --output-dir ${OUTPUT} \
#   > run_logs_abalation/sketchy_structxlipseg_5percent_debug.log 2>&1 &

# echo $! > run_logs_abalation/sketchy_structxlipseg_5percent_debug.pid
# echo "Started training. PID: $(cat run_logs_abalation/sketchy_structxlipseg_5percent_debug.pid)"
# echo "Log: run_logs_abalation/sketchy_structxlipseg_5percent_debug.log"

python train.py \
  --config-file ${CONFIG} \
  --output-dir ${OUTPUT} \
  --seed ${SEED} 

python test.py \
    --config-file ${CONFIG} \
    --output-dir ${OUTPUT} \
    --source_dataset ${DATASETS} \
    --seed ${SEED} 

python utils/eval.py \
    --config-file ${CONFIG} \
    --output-dir ${OUTPUT} \
    --seed ${SEED}