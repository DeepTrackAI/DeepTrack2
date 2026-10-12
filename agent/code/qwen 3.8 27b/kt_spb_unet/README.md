# KT/SPB U-Net on DeepTrack2-synthesised Ndc80-GFP / Cdc11-CFP frames

A small end-to-end pipeline for segmenting and locating

* **6 kinetochores** (Ndc80‑GFP, green channel) and
* **2 spindle pole bodies** (Cdc11‑CFP, red channel)

in wide‑field fluorescence microscopy of a single yeast cell.

## What is in this folder

| File | Role |
|------|------|
| `config.py`           | Shared constants: canvas, object counts, paths, defaults |
| `unet.py`             | PyTorch U‑Net (2‑channel in → 2‑channel out) + `dice_bce_loss` |
| `generator.py`        | DeepTrack2 frame renderer (per‑frame determinism) |
| `render_all.py`       | Batch render script (writes `data/img_*.npy`, `data/msk_*.npy`) |
| `dataset.py`          | PyTorch `Dataset` reading the pre‑rendered frames from disk |
| `train.py`            | AdamW + cosine LR training loop, saves best‑Dice checkpoint |
| `detect.py`           | Model loading + segment → connected‑components → `(x, y)` positions |
| `detect_video.py`     | CLI: run inference on a real video → detections JSON + overlay MP4 |
| `evaluate.py`         | Dice + per‑object centroid errors on a hold‑out of the synthetic set |
| `_e2e_smoke.py`       | 40‑frame smoke test: 3‑epoch mini‑train + inference (sanity) |
| `run_full.sh`         | Render → train orchestrator (safe to run as a long background job) |
| `checkpoints/`        | Best checkpoints (`unet_base<width>_best.pt`) |
| `data/`               | Pre‑rendered frames (uint16 image + uint8 mask per frame) |
| `logs/`               | Pipeline stdout |

## How it works

1. **Synthesise**. `generator.render_frame(idx, H, W)` builds a scatterer
   field (6 `dt.Ellipsoid` KT inclusions + 2 `dt.Ellipsoid` SPB inclusions + a
   few `dt.PointParticle` satellites in each channel), images it through
   `dt.Fluorescence` (NA=1.2, 100 nm/px, 10× magnification, 525 nm for the
   green channel and 620 nm for the red), multiplies by a mild diagonal
   illumination gradient, and adds Poisson shot noise + a small read‑noise
   floor.  The *ground‑truth mask* is the analytic disc of the nominal object
   radius centred on each emitting object — a clean label whose centroid is
   exactly the object position, so the U‑Net learns the optics, not the
   analytic prior.

   Frame index is the RNG seed (`1_000_000 + idx`), so any frame is
   reproducible on demand.

2. **Render**. `render_all.py --frames 2000 --workers 4` writes
   `data/img_<idx:06d>.npy` (uint16, H×W×2) and
   `data/msk_<idx:06d>.npy` (uint8, H×W×2) to disk.  DeepTrack is not
   process‑safe, so each worker constructs its own `Fluorescence` set.
   Throughput on the current machine is ~0.4 frame/s — 2000 frames run
   **~70–90 min** as a single job.

3. **Train**. `train.py --frames 2000 --epochs 12 --base 64 --batch 16`
   trains a base‑64 U‑Net on the A100.  Loss = per‑class soft Dice + BCE,
   with the SPB channel (2 objects) weighted higher than the KT channel
   (6 objects).  Best (mean Dice) checkpoint is written to
   `checkpoints/unet_base64_best.pt`.

4. **Detect**. `detect.predict_positions(net, img, device)` runs the
   network, thresholds each channel at 0.5, finds connected components
   (scipy.ndimage), sorts them by area (largest = most confident), and
   returns the top‑6 KT centroids and top‑2 SPB centroids as `(x, y)` pairs.

5. **Run on real data**.

   ```
   python detect_video.py \
       --video /path/to/ndc80_cdc11_timelapse.tiff \
       --ckpt  checkpoints/unet_base64_best.pt \
       --max-frames 500 \
       --out-json detections.json
   ```

   produces `detections.json` (per‑frame object list) and
   `detections.mp4` (overlay of predicted centroids on the video, with the
   frame index and detected counts).

## Running the full pipeline

```bash
# 1) Render (skip if already present — the script is idempotent)
python render_all.py --frames 2000 --workers 4

# 2) Train
python train.py --frames 2000 --epochs 12 --base 64 --batch 16

# 3) Evaluate (Dice + centroid errors on 200 hold‑out frames)
python evaluate.py --ckpt checkpoints/unet_base64_best.pt --n 200

# 4) Run on the real time‑lapse — fill in your file path first
python detect_video.py --video /data/videos/my_sample.tiff
```

`run_full.sh` runs steps 1 and 2 back‑to‑back as a single job (useful as a
long background task).

## Tuning knobs

| Knob | Where |
|------|-------|
| Canvas size (currently 416×640) | `config.HEIGHT/HEIGHT`  (must be divisible by 4) |
| KT radius (px) | `config.KT_RADIUS_PX` (currently 14) |
| SPB radius (px) | `config.SPB_RADIUS_PX` (currently 18) |
| Number of KTs / SPBs | `config.N_KT` (6), `config.N_SPB` (2) |
| Noise model | `config.GAIN`, `config.READ_NOISE` |
| Training depth / width | `--base 64`, default `depths=4` in `train.py` |
| Detection threshold | `config.THR_KT` / `THR_SPB` (default 0.5) |
| Smallest component to keep | `config.MIN_COMPONENT_AREA` (default 4 px) |

## Honest limitations

* **Domain gap** is the main risk — the real video has the actual microscope's
  illumination, camera noise, cell drift, and *other* cellular GFP/CFP
  signal (ribosomes, ER, etc.) that the simulator does not.  A model trained
  only on this synthetic set may under‑transfer.  Mitigations:
  * augment the simulator (drift, out‑of‑focus variation, background
    ribosomal haze, illumination flicker);
  * fine‑tune on a small set of user‑annotated frames.
* **Position accuracy vs. segmentation**: the current model is a U‑Net
  (segmentation).  The pipeline recovers positions by connected components,
  so accuracy in *px* depends on the segmentation being tight.  On the
  smoke test (40 frames, 3 epochs) it already finds roughly the right
  centroids, but the full 12‑epoch model should be 1–2 px more accurate.
* **No temporal model**.  Per‑frame detection ignores the fact that the
  same 6 KTs and 2 SPBs are continuous across frames.  Adding a Kalman
  filter or tracking‑by‑detection layer would make it much more robust to
  a frame where one object is weak.
* **Determinism is only across seeds** — the real video has its own random
  background.

## Disk / GPU estimates

| Stage | Wall time (typical) | Disk used |
|-------|----------------------|-----------|
| Render 2000 frames | ~70–90 min on this box (≈0.4 f/s) | ~4–5 GB (uint16) |
| Train 12 epochs, base=64, batch=16 | ~10–20 min on A100‑40 | — |
| Detect 500 real frames | ~30–60 s on A100 | — |
| Overlay video | ~20 s | ~10–50 MB mp4 |
