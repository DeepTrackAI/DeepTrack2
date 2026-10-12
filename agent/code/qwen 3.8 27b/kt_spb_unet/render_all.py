#!/usr/bin/env python3
"""Parallel DeepTrack frame renderer (process pool).

Each worker process builds its own Fluorescence optics (DeepTrack is not
process-safe to share), then renders its assigned frames.  Deterministic in
frame index, so results are identical to the sequential renderer.

Usage::
    python render_all.py --frames 40 --workers 16
    python render_all.py --frames 2000 --workers 24
"""
from __future__ import annotations

import argparse
import glob
import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import config
import generator


def _worker_init(h, w):
    """One optics set per process (created in its own module globals)."""
    _worker_init.g_opt = generator._make_green_optics(h, w)
    _worker_init.r_opt = generator._make_red_optics(h, w)


def _render_one(idx):
    h, w = config.HEIGHT, config.WIDTH
    f = generator.render_frame(int(idx), h, w,
                               (_worker_init.g_opt, _worker_init.r_opt))
    os.makedirs(config.DATA_DIR, exist_ok=True)
    np.save(os.path.join(config.DATA_DIR, f"img_{idx:06d}.npy"), f["image"])
    np.save(os.path.join(config.DATA_DIR, f"msk_{idx:06d}.npy"), f["masks"])
    return idx


def _already_done(i):
    p = os.path.join(config.DATA_DIR, f"img_{i:06d}.npy")
    m = os.path.join(config.DATA_DIR, f"msk_{i:06d}.npy")
    return os.path.exists(p) and os.path.exists(m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    indices = [i for i in range(args.start, args.start + args.frames)
               if not _already_done(i)]
    if not indices:
        print("all frames already rendered; nothing to do")
        return

    print(f"rendering {len(indices)} frames with {args.workers} workers "
          f"-> {config.DATA_DIR}")
    os.makedirs(config.DATA_DIR, exist_ok=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                             initargs=(config.HEIGHT, config.WIDTH)) as ex:
        futs = [ex.submit(_render_one, i) for i in indices]
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 50 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(indices) - done) / max(rate, 1e-6)
                print(f"[render] {done}/{len(indices)}  {rate:.1f} f/s  "
                      f"eta {eta/60:.1f} min", flush=True)
    print(f"done {len(indices)} frames in {time.time()-t0:.1f}s  "
          f"({len(indices)/(time.time()-t0):.2f} f/s)")


if __name__ == "__main__":
    main()
