#!/usr/bin/env python3
"""End-to-end smoke test: generate 8 frames, train a few epochs, run inference.

Proves the pipeline is wired correctly:
renderer -> dataset -> UNet -> dice/bce -> detect -> positions.

Usage::
    python _e2e_smoke.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

import config
import generator
from dataset import PreRenderedDataset
from unet import UNet, dice_bce_loss
from detect import load_model, predict_positions, _segment


def main() -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.CKPT_DIR, exist_ok=True)

    # 1. Render a tiny dataset deterministically.
    n = 40
    g_opt = generator._make_green_optics(config.HEIGHT, config.WIDTH)
    r_opt = generator._make_red_optics(config.HEIGHT, config.WIDTH)
    print(f"rendering {n} frames...")
    t0 = time.time()
    for i in range(n):
        f = generator.render_frame(i, config.HEIGHT, config.WIDTH, (g_opt, r_opt))
        np.save(os.path.join(config.DATA_DIR, f"img_{i:06d}.npy"), f["image"])
        np.save(os.path.join(config.DATA_DIR, f"msk_{i:06d}.npy"), f["masks"])
    print(f"   {n} frames in {time.time()-t0:.1f}s")

    # 2. Build small dataset for smoke test.
    from torch.utils.data import DataLoader
    ds = PreRenderedDataset(config.DATA_DIR, np.arange(n))
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    img0, msk0 = next(iter(dl))
    print(f"   sample img {tuple(img0.shape)} max={img0.max():.3f}")
    print(f"   sample mask[0] green_px={int(msk0[0,0].sum())} red_px={int(msk0[0,1].sum())}")

    # 3. Train a few epochs with a compact U-Net.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(in_channels=2, out_channels=2, base=32, depths=3).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for ep in range(3):
        net.train()
        tot = 0
        for img, msk in dl:
            img = img.to(device); msk = msk.to(device)
            loss = dice_bce_loss(net(img), msk, weights=(1.0, 1.3))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"   epoch {ep+1}/3 loss={tot/(len(dl)):.4f}")

    # 4. Save a checkpoint (small one) and load it back.
    ckpt_path = os.path.join(config.CKPT_DIR, "smoke.pt")
    torch.save({"net": net.state_dict(), "base": 32, "in_ch": 2, "out_ch": 2,
                "depths": 3, "epoch": 3, "h": config.HEIGHT, "w": config.WIDTH},
               ckpt_path)
    loaded, dev = load_model(ckpt_path, device=str(device))
    print(f"   reload model OK (base=32)")

    # 5. Inference: segment one frame and find positions.
    # image tensor from dataset is (H, W, 2); convert to (H, W, 2) float [0,1].
    img1 = img0[0].cpu().numpy()               # (H, W, 2)
    pred = predict_positions(loaded, img1, dev, thr_kt=0.4, thr_spb=0.4)
    print(f"   predicted KT count {pred[0]} (gt 6)")
    print(f"   predicted SPB count {pred[1]} (gt 2)")
    print(f"   KT  positions: {pred[0]}")
    print(f"   SPB positions: {pred[1]}")
    probs = pred[2]
    print(f"   prob map maxes green={probs[0].max():.3f} red={probs[1].max():.3f}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
