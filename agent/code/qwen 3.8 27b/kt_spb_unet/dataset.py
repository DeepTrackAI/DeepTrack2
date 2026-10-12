"""Dataset wrapper: loads pre-rendered frames from disk (memmap uint16)."""
from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PreRenderedDataset(Dataset):
    """Read from a directory of uint16 image/mask arrays written by
    ``scripts/render_all.py``.

    Expected layout::

        data/
            img_000000.npy       # (H, W, 2) uint16
            img_000001.npy
            ...
            msk_000000.npy       # (H, W, 2) uint8
            ...

    The dataset divides by 65535 to get float32 [0, 1].
    """

    def __init__(self, data_dir: str, indices: np.ndarray):
        self.data_dir = data_dir
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        img = np.load(os.path.join(self.data_dir, f"img_{idx:06d}.npy"),
                      mmap_mode="r")
        msk = np.load(os.path.join(self.data_dir, f"msk_{idx:06d}.npy"),
                      mmap_mode="r")
        # HWC -> CHW
        img = torch.from_numpy(np.ascontiguousarray(
                                  np.moveaxis(np.asarray(img, dtype=np.float32), -1, 0) / 65535.0))
        msk = torch.from_numpy(np.ascontiguousarray(
                                  np.moveaxis(np.asarray(msk, dtype=np.float32), -1, 0)))
        return img, msk


def save_frames(data_dir: str, h: int, w: int,
                renderer, n_frames: int, start: int = 0,
                log_every: int = 50) -> None:
    """Render frames to disk, skipping already-existing ones."""
    os.makedirs(data_dir, exist_ok=True)
    for i in range(start, start + n_frames):
        img_path = os.path.join(data_dir, f"img_{i:06d}.npy")
        msk_path = os.path.join(data_dir, f"msk_{i:06d}.npy")
        if os.path.exists(img_path) and os.path.exists(msk_path):
            continue
        frame = renderer(i, h, w)
        np.save(img_path, frame["image"])
        np.save(msk_path, frame["masks"])
        if (i - start) % log_every == 0:
            print(f"[render] {i - start}/{n_frames}", flush=True)
