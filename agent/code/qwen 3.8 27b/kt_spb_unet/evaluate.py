#!/usr/bin/env python3
"""Evaluate a trained checkpoint: per-object centroid accuracy + Dice.

Given pre-rendered synthetic frames + analytic-disc ground-truth, we compute
the predicted segmentation on a holdout set and report, per class (KT / SPB):

    * Dice coefficient of the (hard-thresholded) mask
    * Mean / median per-object centroid error (px), via greedy nearest-
      neighbour match between predicted components and GT disc centres
    * Fraction of GT objects missed (outside ``--max-dist``)

Usage:
    python evaluate.py --ckpt checkpoints/unet_base64_best.pt --n 200
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
import config
from dataset import PreRenderedDataset
from detect import _segment, _components_and_centroids, load_model


def _gt_centres(mask_2d: np.ndarray) -> np.ndarray:
    """Label connected components in a binary disc mask; return (x, y) centres."""
    lab, nobj = ndimage.label(mask_2d > 0.5)
    if nobj == 0:
        return np.zeros((0, 2), dtype=np.float32)
    sizes = ndimage.sum(input=np.ones_like(lab), labels=lab,
                        index=np.arange(1, nobj + 1))
    if float(sizes.max()) < 4:
        return np.zeros((0, 2), dtype=np.float32)
    cs = ndimage.center_of_mass(input=np.ones_like(lab), labels=lab,
                                index=np.arange(1, nobj + 1))
    # cs: list of (y, x); convert to (x, y)
    return np.array([(c[1], c[0]) for c in cs], dtype=np.float32)


def _greedy_centroid_errors(pred_xy: list[tuple[float, float]],
                            gt_xy: np.ndarray,
                            max_dist: float):
    """Greedy match (largest-pred first, nearest-GT).  Returns (list of errors,
    n_missed)."""
    if len(gt_xy) == 0:
        return [], 0
    gt = gt_xy.astype(np.float32)
    # Predictions sorted by size would be ideal; we only have positions, so
    # use the natural order (already sorted by component size in detect()).
    used = [False] * len(gt)
    errors = []
    for p in pred_xy:
        p = np.asarray(p, dtype=np.float32)
        d = np.linalg.norm(gt - p, axis=1)
        idx = int(np.argmin(d))
        if used[idx] or d[idx] > max_dist:
            continue
        used[idx] = True
        errors.append(float(d[idx]))
    missed = int(len(gt) - int(np.sum(used)))
    return errors, missed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join(config.CKPT_DIR, "unet_base64_best.pt"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-dist", type=float, default=25.0,
                    help="max centroid distance to count as a match (px)")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frames-total", type=int, default=2000,
                    help="total pre-rendered frames (for holdout selection)")
    args = ap.parse_args()

    net, dev = load_model(args.ckpt, device=args.device)

    rng = np.random.default_rng(config.SEED)
    idx = rng.permutation(args.frames_total)[:args.n]
    ds = PreRenderedDataset(config.DATA_DIR, idx)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    dice_kt = dice_spb = 0.0
    err_kt: list[float] = []
    err_spb: list[float] = []
    miss_kt = miss_spb = n_kt_gt = n_spb_gt = 0

    for img, msk in dl:
        img1 = img[0].cpu().numpy()                      # (2, H, W)
        msk1 = msk[0].cpu().numpy()                      # (2, H, W)
        prob = _segment(net, img1, dev)                  # (2, H, W)

        # Dice per class (binary, threshold 0.5)
        for c in (0, 1):
            p = (prob[c] >= 0.5).astype(bool)
            t = (msk1[c] > 0.5).astype(bool)
            inter = int((p & t).sum())
            union = int((p | t).sum())
            d = inter / max(union, 1)
            if c == 0:
                dice_kt += d
            else:
                dice_spb += d

        # Ground-truth object centres
        gt_kt = _gt_centres(msk1[0])
        gt_spb = _gt_centres(msk1[1])
        n_kt_gt += len(gt_kt)
        n_spb_gt += len(gt_spb)

        # Predicted object positions (top-6 / top-2 by component size)
        pred_kt = _components_and_centroids(prob[0], 0.5,
                                           config.MIN_COMPONENT_AREA)[:config.N_KT]
        pred_spb = _components_and_centroids(prob[1], 0.5,
                                             config.MIN_COMPONENT_AREA)[:config.N_SPB]

        e_kt, m_kt = _greedy_centroid_errors(pred_kt, gt_kt, args.max_dist)
        e_spb, m_spb = _greedy_centroid_errors(pred_spb, gt_spb, args.max_dist)
        err_kt += e_kt
        err_spb += e_spb
        miss_kt += m_kt
        miss_spb += m_spb

    n = args.n
    def _fmt(errs):
        if not errs:
            return "nan"
        return f"mean={np.mean(errs):.2f}  median={np.median(errs):.2f}  p90={np.percentile(errs,90):.2f}"
    print(f"=== Evaluated {n} frames ===")
    print(f"Dice:   KT = {dice_kt/n:.4f}   SPB = {dice_spb/n:.4f}")
    print(f"KT GT  : {n_kt_gt} objects  matched={len(err_kt)}  missed_in_window={miss_kt}")
    if err_kt:
        print(f"KT err : {_fmt(err_kt)} px")
    print(f"SPB GT : {n_spb_gt} objects  matched={len(err_spb)}  missed_in_window={miss_spb}")
    if err_spb:
        print(f"SPB err: {_fmt(err_spb)} px")


if __name__ == "__main__":
    main()
