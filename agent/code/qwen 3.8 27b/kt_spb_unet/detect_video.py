#!/usr/bin/env python3
"""Run the trained U-Net on a real time-lapse video and extract object positions.

Usage:
    python detect_video.py \
        --video /path/to/video.tiff \
        --ckpt  checkpoints/unet_base64_best.pt \
        --max-frames 500 \
        --out-json detections.json \
        --overlay-overlay overlay.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

import config
from unet import UNet
from detect import (
    load_model, predict_positions, load_video_frames, save_detections,
)


def overlay_frames(video_frames, detections, out_path):
    """Create a short GIF/MP4 overlay of detections on the video frames."""
    import cv2
    N = min(len(video_frames), len(detections))
    W, H = video_frames[0].shape[1], video_frames[0].shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 10, (W, H, 3))
    for i in range(N):
        # Normalise video frame to [0, 255]
        img = video_frames[i].copy()             # (H, W, 2)
        img_max = img.max()
        if img_max > 0:
            img = img / img_max * 255.0
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[..., 1] = img[..., 0].astype(np.uint8)  # green
        frame[..., 0] = img[..., 1].astype(np.uint8)  # red
        # Draw detections
        for (x, y) in detections[i]["kt"]:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), 1, 8)
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), 2)
        for (x, y) in detections[i]["spb"]:
            cv2.circle(frame, (int(x), int(y)), 7, (0, 0, 255), 2, 8)
            cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 2)
        cv2.putText(frame, f"f={i}  KT={len(detections[i]['kt'])} SPB={len(detections[i]['spb'])}",
                    (5, H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        out.write(frame)
    out.release()
    print(f"wrote overlay video {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default=config.VIDEO_PATH,
                    help="Real time-lapse video path")
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join(config.CKPT_DIR, "unet_base64_best.pt"))
    ap.add_argument("--max-frames", type=int, default=50)
    ap.add_argument("--thr-kt", type=float, default=config.THR_KT)
    ap.add_argument("--thr-spb", type=float, default=config.THR_SPB)
    ap.add_argument("--out-json", type=str, default="detections.json")
    ap.add_argument("--overlay", type=str, default=None,
                    help="Optional: write overlay MP4 (default = out-json with .mp4)")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"loading model from {args.ckpt}")
    net, device = load_model(args.ckpt, device=args.device)
    print(f"model loaded on {device}")

    print(f"loading video {args.video} (max {args.max_frames} frames)")
    frames = load_video_frames(args.video, max_frames=args.max_frames)
    print(f"loaded {len(frames)} frames of shape {frames[0].shape}")

    t0 = time.time()
    detections = []
    for i, img in enumerate(frames):
        kt, spb, prob = predict_positions(net, img[None] if img.ndim == 2 else img,
                                           device, thr_kt=args.thr_kt, thr_spb=args.thr_spb)
        det = {"frame": int(i), "kt": kt, "spb": spb,
               "n_kt": len(kt), "n_spb": len(spb)}
        detections.append(det)
        if (i + 1) % 25 == 0:
            print(f"[infer] {i+1}/{len(frames)}  {len(frames)/(time.time()-t0)*1e3:.0f} fps",
                  flush=True)

    out_json = args.out_json
    out_overlay = args.overlay or (os.path.splitext(out_json)[0] + ".mp4")
    save_detections(detections, out_json)
    print(f"wrote {out_json}")

    overlay_frames(frames, detections, out_overlay)

    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
