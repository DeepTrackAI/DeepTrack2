#!/usr/bin/env python3
"""Segmentation -> object-position detection, and video processing.

Given a segmentation model, extract object centroids (x, y) for the two
classes (channel 0 = KT/green, channel 1 = SPB/red) from each frame.  On a
real time-lapse this is per-frame: the network gives a segmentation, we
threshold, find connected components, and take their centroids.
"""
from __future__ import annotations

import json
import os
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

import config
from unet import UNet


def load_model(ckpt_path: str, device: str = "cuda") -> tuple[UNet, torch.device]:
    dev = torch.device(device if torch.cuda.is_available() or "cpu" in device else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = UNet(in_channels=ckpt.get("in_ch", 2),
               out_channels=ckpt.get("out_ch", 2),
               base=ckpt.get("base", 64),
               depths=ckpt.get("depths", 4))
    net.load_state_dict(ckpt["net"])
    net.to(dev).eval()
    return net, dev


@torch.no_grad()
def _segment(net: UNet, img: np.ndarray, device: torch.device) -> np.ndarray:
    """img: (H, W, 2) or (2, H, W) float[0,1] -> (2, H, W) probability map.

    The network is always in CHW, so we normalise the input to CHW internally.
    """
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[-1] == 2:                  # HWC
        a = np.ascontiguousarray(np.moveaxis(a, -1, 0))  # -> CHW
    x = torch.from_numpy(a).unsqueeze(0).to(device)      # (1,2,H,W)
    out = torch.sigmoid(net(x)[0])                        # (2,H,W)
    return out.float().cpu().numpy()


# ---- connected components + centroid --------------------------------------

def _components_and_centroids(prob: np.ndarray, thr: float, min_area: int):
    """scipy.ndimage label -> list of (x, y) centroids, large blobs first."""
    from scipy import ndimage
    mask = (prob >= thr).astype(np.uint8)
    lab, nobj = ndimage.label(mask)
    out = []
    if nobj == 0:
        return out
    sizes = ndimage.sum(input=np.ones_like(lab), labels=lab,
                        index=np.arange(1, nobj + 1))
    order = np.argsort(sizes)[::-1]
    for k in order:
        comp = (lab == (k + 1))
        if comp.sum() < min_area:
            continue
        cy, cx = ndimage.center_of_mass(comp)
        out.append((float(cx), float(cy)))
    return out


def predict_positions(net, img_hw2: np.ndarray, device,
                      thr_kt: float = 0.5, thr_spb: float = 0.5,
                      n_kt: int = config.N_KT, n_spb: int = config.N_SPB,
                      min_area: int = config.MIN_COMPONENT_AREA):
    """Return ``(kt_positions, spb_positions)`` each a list of (x, y)."""
    prob = _segment(net, img_hw2, device)         # (2, H, W)
    kt = _components_and_centroids(prob[0], thr_kt, min_area)[:n_kt]
    spb = _components_and_centroids(prob[1], thr_spb, min_area)[:n_spb]
    return kt, spb, prob


# ---- real video loading ---------------------------------------------------

def load_video_frames(path: str, max_frames: int | None = None):
    """Load a time-lapse (tiff/ome.tiff/avi/mp4) -> list of (H, W, 2) float[0,1].

    Assumes the file stores the object-of-interest channels.  For a
    3-channel RGB/fluorescence video we map green->channel0 (KT), red->channel1
    (SPB).  For a 2-channel stack we pass through.
    """
    import imageio.v3 as iio
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tiff", ".tif", ".ome.tiff"):
        arr = iio.imread(path, plugin="tifffile")
    else:
        arr = np.stack(iio.imread(path))
    if arr.ndim == 3:                        # (T, H, W) single channel -> replicate
        arr = np.repeat(arr[..., None], 2, axis=-1)
    elif arr.shape[-1] == 3:                 # RGB
        green = arr[..., 1].astype(np.float32)
        red = arr[..., 0].astype(np.float32)
        arr = np.stack([green, red], axis=-1)
    elif arr.shape[-1] == 2:
        pass
    else:
        raise ValueError(f"unsupported video channels {arr.shape}")
    mx = arr.max()
    if mx > 0:
        arr = arr.astype(np.float32) / float(mx)
    if max_frames is not None:
        arr = arr[:max_frames]
    return arr


def save_detections(detections: list[dict], out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(detections, f, indent=2)


def process_video(net, device, video_path: str, max_frames: int | None = None,
                  thr_kt: float = 0.5, thr_spb: float = 0.5,
                  out_json: str | None = None):
    """Run inference over a whole time-lapse; returns per-frame detections."""
    frames = load_video_frames(video_path, max_frames)
    detections = []
    for i, img in enumerate(frames):
        kt, spb, prob = predict_positions(net, img, device, thr_kt, thr_spb)
        detections.append({
            "frame": int(i),
            "kt": kt,
            "spb": spb,
            "n_kt": len(kt),
            "n_spb": len(spb),
        })
        if (i + 1) % 25 == 0:
            print(f"[infer] {i+1}/{len(frames)}", flush=True)
    if out_json:
        save_detections(detections, out_json)
        print(f"wrote {out_json}")
    return detections
