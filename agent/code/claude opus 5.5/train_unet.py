"""Train a U-Net to localise kinetochores (KTs) and spindle pole bodies (SPBs).

Training data is generated on the fly by the DeepTrack2 simulator in
``simulate_spindle.py`` (every batch is new, so the network never sees the
same image twice).

Model
    input  : (B, 2, H, W)  channel 0 = Ndc80-GFP (KT), 1 = Cdc11-CFP (SPB),
             normalised with ``simulate_spindle.normalize_frame``
    output : (B, 2, H, W)  logits of a Gaussian "centre heatmap" per class
Loss
    CenterNet-style penalty-reduced focal loss on the heatmaps.
Positions
    ``detect_peaks`` = local maxima above a threshold + sub-pixel centroid.

Usage::

    python train_unet.py                 # full training (~20-25 min, 32 CPU workers + 1 GPU)
    python train_unet.py --steps 200     # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
CLASS_NAMES = ("KT", "SPB")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class DoubleConv(nn.Sequential):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Plain U-Net; input H and W must be multiples of 2**depth (16)."""

    def __init__(self, in_channels: int = 2, out_channels: int = 2,
                 base: int = 32, depth: int = 4) -> None:
        super().__init__()
        chans = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        c = in_channels
        for ch in chans[:-1]:
            self.down.append(DoubleConv(c, ch))
            c = ch
        self.bottleneck = DoubleConv(chans[-2], chans[-1])
        self.up = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        for ch in reversed(chans[:-1]):
            self.up.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.up_conv.append(DoubleConv(ch * 2, ch))
        self.head = nn.Conv2d(base, out_channels, 1)
        # Start with low heatmap probabilities (focal-loss convention).
        nn.init.constant_(self.head.bias, -2.19)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.up, self.up_conv, reversed(skips)):
            x = conv(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def heatmap_focal_loss(logits: torch.Tensor, target: torch.Tensor,
                       alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Penalty-reduced pixel-wise focal loss (CenterNet, Zhou et al. 2019)."""
    pred = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.99).float()
    neg = 1.0 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** alpha * pos
    neg_loss = torch.log(1 - pred) * pred**alpha * (1 - target) ** beta * neg
    n_pos = pos.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / n_pos


# --------------------------------------------------------------------------- #
# Peak detection and evaluation
# --------------------------------------------------------------------------- #
def detect_peaks(heatmap: np.ndarray, threshold: float = 0.2,
                 min_distance: int = 1) -> np.ndarray:
    """Sub-pixel (row, col, score) of local maxima in a probability map."""
    t = torch.as_tensor(heatmap)[None, None]
    k = 2 * min_distance + 1
    is_max = t == F.max_pool2d(t, k, stride=1, padding=min_distance)
    rows, cols = np.nonzero((is_max & (t > threshold))[0, 0].numpy())
    peaks = []
    h, w = heatmap.shape
    for r, c in zip(rows, cols):
        r0, r1, c0, c1 = max(r - 1, 0), min(r + 2, h), max(c - 1, 0), min(c + 2, w)
        patch = heatmap[r0:r1, c0:c1]
        rr, cc = np.mgrid[r0:r1, c0:c1]
        weight = patch.sum()
        peaks.append(((rr * patch).sum() / weight, (cc * patch).sum() / weight,
                      heatmap[r, c]))
    return np.array(peaks, dtype=np.float32).reshape(-1, 3)


def match(pred: np.ndarray, true: np.ndarray, max_dist: float = 3.0):
    """Hungarian matching; returns (n_true_positive, list of match distances)."""
    if len(pred) == 0 or len(true) == 0:
        return 0, []
    d = np.linalg.norm(pred[:, None, :2] - true[None, :, :2], axis=-1)
    ri, ci = linear_sum_assignment(d)
    good = d[ri, ci] <= max_dist
    return int(good.sum()), list(d[ri, ci][good])


@torch.no_grad()
def predict_heatmaps(model: nn.Module, images: np.ndarray,
                     device: torch.device, batch_size: int = 64) -> np.ndarray:
    """images (N, 2, H, W) normalised -> probability maps (N, 2, H, W)."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.as_tensor(images[i:i + batch_size], device=device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(out)


def evaluate(model: nn.Module, val: dict, device: torch.device,
             threshold: float = 0.2) -> dict:
    probs = predict_heatmaps(model, val["image"], device)
    metrics = {}
    for c, name in enumerate(CLASS_NAMES):
        tp = n_pred = n_true = 0
        dists: list[float] = []
        for i in range(len(probs)):
            pred = detect_peaks(probs[i, c], threshold)
            true = val["kt_pos" if c == 0 else "spb_pos"][i]
            k, d = match(pred, true)
            tp, n_pred, n_true = tp + k, n_pred + len(pred), n_true + len(true)
            dists += d
        precision = tp / max(n_pred, 1)
        recall = tp / max(n_true, 1)
        metrics[name] = dict(
            precision=round(precision, 4), recall=round(recall, 4),
            f1=round(2 * precision * recall / max(precision + recall, 1e-9), 4),
            rmse_px=round(float(np.sqrt(np.mean(np.square(dists)))) if dists else float("nan"), 3),
        )
    return metrics


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
_SIMULATOR = None


def _init_worker(base_seed: int) -> None:
    """Build one DeepTrack2 pipeline per worker process."""
    global _SIMULATOR
    from simulate_spindle import SpindleSimulator

    np.random.seed((base_seed + os.getpid()) % 2**32)
    _SIMULATOR = SpindleSimulator()


def _simulate(seed: int | None) -> dict:
    if seed is not None:
        np.random.seed(seed)
    return _SIMULATOR.sample()


def simulated_batches(pool: Pool, batch_size: int):
    """Infinite stream of freshly simulated (image, target) batches.

    Samples travel through pipes (not /dev/shm), so this also works in
    containers with a tiny shared-memory mount.
    """
    stream = pool.imap_unordered(_simulate, iter(lambda: None, 0), chunksize=4)
    while True:
        samples = [next(stream) for _ in range(batch_size)]
        yield (torch.from_numpy(np.stack([s["image"] for s in samples])),
               torch.from_numpy(np.stack([s["target"] for s in samples])))


def make_validation_set(pool: Pool, n: int) -> dict:
    samples = pool.map(_simulate, range(10_000, 10_000 + n))
    return dict(
        image=np.stack([s["image"] for s in samples]),
        kt_pos=[s["kt_pos"] for s in samples],
        spb_pos=[s["spb_pos"] for s in samples],
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=HERE / "unet_kt_spb.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"device: {device}")

    t0 = time.time()
    # "spawn" (not fork): forked workers inherit the parent's torch/OpenMP
    # state and serialise on it (~10 samples/s total instead of ~5/s/worker).
    pool = get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=(int(t0),))
    val = make_validation_set(pool, args.n_val)
    print(f"validation set: {len(val['image'])} images ({time.time() - t0:.0f}s)")
    loader = simulated_batches(pool, args.batch_size)

    model = UNet().to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history = []
    running = 0.0
    t0 = time.time()
    for step, (x, y) in enumerate(loader, start=1):
        model.train()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x)
        loss = heatmap_focal_loss(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running += loss.item()

        if step % 100 == 0:
            print(f"step {step:5d}/{args.steps}  loss {running / 100:.4f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
            running = 0.0
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val, device)
            history.append(dict(step=step, **metrics))
            print(f"  val: {json.dumps(metrics)}", flush=True)
            torch.save({"model": model.state_dict(), "step": step,
                        "metrics": metrics}, args.out)
        if step >= args.steps:
            break
    pool.terminate()

    (args.out.with_suffix(".history.json")).write_text(json.dumps(history, indent=2))
    print(f"saved model to {args.out}")
    save_examples(model, val, device, HERE / "unet_validation_examples.png")


def save_examples(model: nn.Module, val: dict, device: torch.device,
                  path: Path, n: int = 6) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from simulate_spindle import to_rgb

    probs = predict_heatmaps(model, val["image"][:n], device)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.4))
    for i in range(n):
        axes[0, i].imshow(to_rgb(val["image"][i]))
        axes[0, i].set_title("input")
        axes[1, i].imshow(to_rgb(val["image"][i]))
        kt, spb = val["kt_pos"][i], val["spb_pos"][i]
        axes[1, i].scatter(kt[:, 1], kt[:, 0], marker="+", c="white", s=30, lw=0.8)
        axes[1, i].scatter(spb[:, 1], spb[:, 0], marker="x", c="white", s=30, lw=0.8)
        for c, color in ((0, "cyan"), (1, "magenta")):
            p = detect_peaks(probs[i, c])
            axes[1, i].scatter(p[:, 1], p[:, 0], s=70, facecolors="none",
                               edgecolors=color, lw=1)
        axes[1, i].set_title("truth (+/x) vs U-Net (o)")
        for ax in axes[:, i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
