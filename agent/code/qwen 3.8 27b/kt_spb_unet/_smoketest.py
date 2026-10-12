"""Quick smoke test: render N frames and report stats."""
from __future__ import annotations
import sys, time, warnings
import numpy as np
sys.path.insert(0, "/a/kt_spb_unet")
warnings.filterwarnings("ignore")

from generator import (render_frame, _make_green_optics, _make_red_optics)

H, W = 412, 640
g_opt = _make_green_optics(H, W)
r_opt = _make_red_optics(H, W)

t0 = time.time()
for i in range(3):
    f = render_frame(i, H, W, (g_opt, r_opt))
    img = f["image"]; msk = f["masks"]
    print(f"frame {i:2d}: img {img.shape} {img.dtype} max={img.max()}; "
          f"mask {msk.shape} green_px={int(msk[...,0].sum())} red_px={int(msk[...,1].sum())}")
    print(f"           KTs={[(round(x),round(y),round(r,1)) for x,y,r in f['kt']]}")
    print(f"           SPBs={[(round(x),round(y),round(r,1)) for x,y,r in f['spb']]}")
    # determinism check
    f2 = render_frame(i, H, W, (g_opt, r_opt))
    same = np.array_equal(img, f2["image"]) and np.array_equal(msk, f2["masks"])
    print(f"           deterministic: {same}")
print(f"3 frames in {time.time()-t0:.1f}s  (~{(time.time()-t0)/3*1000:.0f}ms/frame)")

# Save one for a quick look
from PIL import Image
f = render_frame(5, H, W, (g_opt, r_opt))
rgb = np.zeros((H, W, 3), dtype=np.uint8)
rgb[..., 1] = (f["image"][..., 0].astype(np.float32) / 65535 * 255).astype(np.uint8)
rgb[..., 0] = (f["image"][..., 1].astype(np.float32) / 65535 * 255).astype(np.uint8)
Image.fromarray(rgb).save("/a/kt_spb_unet/_smoke_frame.png")
Image.fromarray(f["masks"][..., 0] * 255).save("/a/kt_spb_unet/_smoke_mask_kt.png")
Image.fromarray(f["masks"][..., 1] * 255).save("/a/kt_spb_unet/_smoke_mask_spb.png")
print("wrote smoke outputs")
