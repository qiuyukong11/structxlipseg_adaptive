#!/usr/bin/env bash
set -e

cd /mnt/data/zruan/kqy/pami/segmentation/structxlip_seg_v2

mkdir -p run_logs

# nohup python train.py \
#   --config-file configs/sketchy_clipseg_100percent.yaml \
#   > run_logs/sketchy_clipseg_100percent_train.log 2>&1 &

# echo $! > run_logs/sketchy_clipseg_100percent_train.pid
# echo "Started training. PID: $(cat run_logs/sketchy_clipseg_100percent_train.pid)"
# echo "Log: run_logs/sketchy_clipseg_100percent_train.log"


nohup python train.py \
  --config-file configs/sketchy_structxlipseg_100percent.yaml \
  > run_logs/sketchy_structxlipseg_100percent_train.log 2>&1 &

echo $! > run_logs/sketchy_structxlipseg_100percent_train.pid
echo "Started training. PID: $(cat run_logs/sketchy_structxlipseg_100percent_train.pid)"
echo "Log: run_logs/sketchy_structxlipseg_100percent_train.log"