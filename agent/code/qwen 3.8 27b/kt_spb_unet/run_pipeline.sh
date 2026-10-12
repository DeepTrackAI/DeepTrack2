#!/usr/bin/env bash
# End-to-end pipeline: render 2000 frames -> train U-Net -> save checkpoint.
# Runs as one long job; progress printed to the same file as its stdout.
set -e
cd /a/kt_spb_unet

NFRAMES=${NFRAMES:-2000}
EPOCHS=${EPOCHS:-12}
BASE=${BASE:-64}
BATCH=${BATCH:-16}

stage() { echo; echo "########## $* ##########"; }

stage "STEP 1/2  Rendering ${NFRAMES} synthetic frames"
python render_all.py --frames "$NFRAMES"

stage "STEP 2/2  Training UNet (base=${BASE}, epochs=${EPOCHS}, batch=${BATCH})"
python train.py --frames "$NFRAMES" --epochs "$EPOCHS" --base "$BASE" --batch "$BATCH"

stage "Done"
ls -lh checkpoints/
