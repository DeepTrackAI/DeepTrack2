#!/usr/bin/env python3
"""Train the U-Net on the pre-rendered synthetic dataset.

Usage::

    python train.py --epochs 12 --batch 16 --base 64 --frames 2000

Saves the best (mean val Dice) checkpoint to ``checkpoints/unet_base{base}_best.pt``.
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

import config
from dataset import PreRenderedDataset
from unet import UNet, dice_bce_loss

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)


def soft_dice(pred: torch.Tensor, target: torch.Tensor, ch: int) -> float:
    p = torch.sigmoid(pred[..., ch])
    t = target[..., ch]
    eps = 1e-7
    return float((2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps))


def make_dataloaders(n_total: int):
    rng = np.random.default_rng(config.SEED + 1)
    perm = rng.permutation(n_total)
    n_val = max(int(n_total * config.VAL_FRAC), 10)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    tr = PreRenderedDataset(config.DATA_DIR, tr_idx)
    va = PreRenderedDataset(config.DATA_DIR, val_idx)
    tr_loader = DataLoader(tr, batch_size=config.BATCH, shuffle=True,
                           num_workers=0, pin_memory=False, drop_last=True)
    va_loader = DataLoader(va, batch_size=config.BATCH, num_workers=0,
                           pin_memory=False, shuffle=False)
    return tr_loader, va_loader, tr_idx, val_idx


@torch.no_grad()
def evaluate(net, loader, device) -> dict:
    net.eval()
    n_frames = 0
    d_kt = d_spb = 0.0
    t0 = time.time()
    for img, msk in loader:
        img = img.to(device, non_blocking=True)
        msk = msk.to(device, non_blocking=True)
        out = net(img)
        d_kt += soft_dice(out, msk, 0) * img.size(0)
        d_spb += soft_dice(out, msk, 1) * img.size(0)
        n_frames += img.size(0)
    return {"dice_kt": d_kt / max(n_frames, 1),
            "dice_spb": d_spb / max(n_frames, 1),
            "val_time": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.EPOCHS_DEFAULT)
    ap.add_argument("--batch", type=int, default=config.BATCH)
    ap.add_argument("--base", type=int, default=64, help="U-Net base width")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device={device} base={args.base} batch={args.batch}", flush=True)

    tr_loader, va_loader, tr_idx, val_idx = make_dataloaders(args.frames)
    print(f"train={len(tr_idx)}  val={len(val_idx)}", flush=True)

    net = UNet(in_channels=2, out_channels=2, base=args.base).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"UNet base={args.base}: {n_params/1e6:.2f}M params", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    best_path = os.path.join(config.CKPT_DIR, f"unet_base{args.base}_best.pt")

    for epoch in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        running = 0.0
        nb = 0
        for img, msk in tr_loader:
            img = img.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True)
            out = net(img)
            loss = dice_bce_loss(out, msk, weights=(1.0, 1.3))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        sched.step()
        val = evaluate(net, va_loader, device)
        cur = (val["dice_kt"] + val["dice_spb"]) / 2
        status = ""
        if cur > best:
            best = cur
            status = "  *best*"
            torch.save({"net": net.state_dict(), "base": args.base,
                        "in_ch": 2, "out_ch": 2, "depths": 4, "epoch": epoch,
                        "h": config.HEIGHT, "w": config.WIDTH}, best_path)
        print(f"epoch {epoch:2d}/{args.epochs}  loss={running/nb:.4f}  "
              f"dice KT={val['dice_kt']:.4f} SPB={val['dice_spb']:.4f}  "
              f"[{time.time()-t0:5.1f}s] val {val['val_time']:5.1f}s{status}",
              flush=True)
    print(f"done. best avg dice={best:.4f}  saved {best_path}", flush=True)


if __name__ == "__main__":
    main()
