#!/usr/bin/env bash
# Full pipeline: render synthetic frames, then train the U-Net.
# Logs go to logs/. Runs as a single durable background job.
set -x
cd /a/kt_spb_unet
mkdir -p logs checkpoints
LOG=logs/pipeline.log
exec > >(tee "$LOG") 2>&1

NFRAMES=${NFRAMES:-2000}
EPOCHS=${EPOCHS:-12}
BASE=${BASE:-64}
BATCH=${BATCH:-16}

echo "=== RENDER ($(date -u +%H:%M:%S)) frames=$NFRAMES ==="
time python render_all.py --frames "$NFRAMES" --workers 4

echo "=== TRAINING ($(date -u +%H:%M:%S)) base=$BASE epochs=$EPOCHS batch=$BATCH ==="
time python train.py --frames "$NFRAMES" --epochs "$EPOCHS" --base "$BASE" --batch "$BATCH"

echo "=== DONE ($(date -u +%H:%M:%S)) ==="
ls -lh checkpoints/
du -sh data/ data
