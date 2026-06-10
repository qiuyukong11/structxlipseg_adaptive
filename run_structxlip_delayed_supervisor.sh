#!/usr/bin/env bash
set -u

WORKDIR="/mnt/data/zruan/kqy/pami/segmentation/structxlip_seg_v2"
RUN_SCRIPT="$WORKDIR/structxlip_models.sh"
LOG_DIR="$WORKDIR/run_logs"
DELAY_SECONDS=14400
CHECK_SECONDS=900
STAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="$LOG_DIR/structxlip_run_${STAMP}.log"
MONITOR_LOG="$LOG_DIR/structxlip_monitor_${STAMP}.log"
PID_FILE="$LOG_DIR/structxlip_run_${STAMP}.pid"
LATEST_MONITOR="$LOG_DIR/structxlip_monitor_latest.log"
LATEST_TRAIN="$LOG_DIR/structxlip_run_latest.log"

mkdir -p "$LOG_DIR"
ln -sfn "$MONITOR_LOG" "$LATEST_MONITOR"
ln -sfn "$TRAIN_LOG" "$LATEST_TRAIN"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "$MONITOR_LOG"
}

check_errors() {
  if grep -Eiq "Traceback|RuntimeError|CUDA error|out of memory|FileNotFoundError|No such file|Error loading|Exception|Killed" "$TRAIN_LOG" 2>/dev/null; then
    log "Potential issue detected in train log. Recent matching lines:"
    grep -Ein "Traceback|RuntimeError|CUDA error|out of memory|FileNotFoundError|No such file|Error loading|Exception|Killed" "$TRAIN_LOG" | tail -20 | tee -a "$MONITOR_LOG"
  fi
}

log "Delayed StructXLIP supervisor started."
log "Will sleep ${DELAY_SECONDS}s before launching: $RUN_SCRIPT"
log "Train log: $TRAIN_LOG"
log "Monitor log: $MONITOR_LOG"

sleep "$DELAY_SECONDS"

cd "$WORKDIR" || { log "Failed to cd into $WORKDIR"; exit 2; }
if [[ ! -x "$RUN_SCRIPT" ]]; then
  log "Run script is not executable, attempting chmod +x."
  chmod +x "$RUN_SCRIPT" || { log "chmod failed for $RUN_SCRIPT"; exit 3; }
fi

log "Starting StructXLIP run now."
bash "$RUN_SCRIPT" > "$TRAIN_LOG" 2>&1 &
child=$!
echo "$child" > "$PID_FILE"
log "Started child PID=$child"

while kill -0 "$child" 2>/dev/null; do
  log "Process PID=$child is still running. Last 20 train-log lines follow."
  tail -20 "$TRAIN_LOG" | tee -a "$MONITOR_LOG" || true
  check_errors
  sleep "$CHECK_SECONDS"
done

wait "$child"
code=$?
log "Process PID=$child finished with exit code $code. Final 80 train-log lines follow."
tail -80 "$TRAIN_LOG" | tee -a "$MONITOR_LOG" || true
check_errors

if [[ "$code" -eq 0 ]]; then
  log "StructXLIP run completed successfully."
else
  log "StructXLIP run failed. See train log: $TRAIN_LOG"
fi
exit "$code"
