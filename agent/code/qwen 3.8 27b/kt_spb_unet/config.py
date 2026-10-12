"""Shared constants and paths for the KT/SPB U-Net project."""
from __future__ import annotations

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))          # /a/kt_spb_unet
DATA_DIR = os.environ.get(
    "KT_SPB_DATA", os.path.join(PROJECT_DIR, "data")
)                                                                # dataset dir
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

# ---- canvas / object counts ---------------------------------------------
# (H, W) must be divisible by 4 to work with 3 MaxPool2d stages.  The
# reference image is 412 px tall, so we round up to 416 for a clean fit.
WIDTH = 640                # simulation + model canvas width
HEIGHT = 416               # simulation + model canvas height
N_KT = 6                   # six kinetochores (Ndc80-GFP, green)
N_SPB = 2                  # two spindle pole bodies (Cdc11-CFP, red)

# Emitter radii (in pixel units).
KT_RADIUS_PX = 14.0
SPB_RADIUS_PX = 18.0
PIXEL_NANOMETER = 100e-9   # 100 nm per pixel  (matches Fluorescence 10x / 1 um / px)

# Camera / noise model
GAIN = 80.0                # photon-count gain for shot noise
READ_NOISE = 0.004         # uniform read-noise floor

# ---- training -------------------------------------------------------------
SEED = 20261007
BATCH = 16
EPOCHS_DEFAULT = 12
LR_DEFAULT = 3e-4
UNET_BASE_DEFAULT = 64
UNET_DEPTHS = 4
VAL_FRAC = 0.1

# ---- inference -------------------------------------------------------------
# PLACEHOLDER — replace with your real time-lapse (tiff/ome.tiff/avi/mp4/mov).
VIDEO_PATH = "/data/video/your_nde80_cdc11_timelapse.tiff"
MIN_COMPONENT_AREA = 4
THR_KT = 0.5
THR_SPB = 0.5
