"""Detect kinetochores (KTs) and spindle pole bodies (SPBs) in a time-lapse.

Runs the U-Net trained by ``train_unet.py`` on every frame and writes a CSV
with one row per detection::

    frame, class, row, col, score

(``row``/``col`` in pixels of the *original* video; plus ``y_um``/``x_um`` if
``--pixel-size`` is given). Optionally writes an overlay video.

Supported inputs
    * multi-channel TIFF stacks (e.g. TCYX / TYXC from Fiji / MicroManager)
    * RGB movies or images (.mp4, .avi, .gif, .png, ...): by default
      red = SPB (Cdc11-CFP, shown red) and green = KT (Ndc80-GFP)

Pixel size matters: the U-Net was trained at ``simulate_spindle.PIXEL_SIZE_UM``
(0.1 um / pixel). Pass ``--pixel-size`` (um / pixel of your camera) so frames
are rescaled to the training scale, or ``--scale`` to set the zoom directly.

Examples::

    python detect_video.py movie.tif --pixel-size 0.0645 --overlay movie_overlay.mp4
    python detect_video.py movie.tif --axes TCYX --kt-channel 0 --spb-channel 1
    python detect_video.py image.png --scale 0.15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from simulate_spindle import PIXEL_SIZE_UM, normalize_frame
from train_unet import CLASS_NAMES, UNet, detect_peaks

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_stack(path: Path) -> np.ndarray:
    """Read a video / stack / image as a numpy array (raw axis order)."""
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        import tifffile

        return tifffile.imread(path)
    if suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image.ndim == 3:
            image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
        return image[None]
    import imageio.v3 as iio

    return np.asarray(iio.imread(path))  # (T, H, W, C) for colour videos


def to_tcyx(stack: np.ndarray, axes: str | None) -> np.ndarray:
    """Reorder ``stack`` to (T, C, Y, X)."""
    if axes is None:  # guess
        if stack.ndim == 2:
            axes = "YX"
        elif stack.ndim == 3:
            axes = "YXC" if stack.shape[-1] in (2, 3, 4) else "CYX" \
                if stack.shape[0] in (2, 3, 4) else "TYX"
        elif stack.ndim == 4:
            axes = "TYXC" if stack.shape[-1] in (2, 3, 4) else "TCYX"
        else:
            raise ValueError(f"cannot guess axes of shape {stack.shape}; use --axes")
        print(f"guessed axes {axes} for shape {stack.shape}")
    axes = axes.upper()
    if len(axes) != stack.ndim:
        raise ValueError(f"--axes {axes} does not match shape {stack.shape}")
    for missing in "TC":
        if missing not in axes:
            stack, axes = stack[None], missing + axes
    return np.transpose(stack, [axes.index(a) for a in "TCYX"])


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def load_model(path: Path, device: torch.device) -> UNet:
    checkpoint = torch.load(path, map_location=device)
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"loaded {path} (step {checkpoint.get('step')}, "
          f"sim metrics {checkpoint.get('metrics')})")
    return model


@torch.no_grad()
def predict_frame(model: UNet, frame: np.ndarray, scale: float,
                  device: torch.device) -> np.ndarray:
    """frame (2, H, W) raw -> probability maps (2, H*scale, W*scale)."""
    x = torch.as_tensor(np.asarray(frame, dtype=np.float32), device=device)[None]
    if scale != 1.0:
        x = F.interpolate(x, scale_factor=scale, mode="bilinear",
                          align_corners=False)
    x = torch.as_tensor(normalize_frame(x[0].cpu().numpy()), device=device)[None]
    h, w = x.shape[-2:]
    pad_h, pad_w = (-h) % 16, (-w) % 16
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        prob = torch.sigmoid(model(x)).float()
    return prob[0, :, :h, :w].cpu().numpy()


def detect(prob: np.ndarray, scale: float, threshold: float,
           max_counts: tuple[int | None, int | None]) -> list[tuple]:
    rows = []
    for c, name in enumerate(CLASS_NAMES):
        peaks = detect_peaks(prob[c], threshold)
        peaks = peaks[np.argsort(-peaks[:, 2])]
        if max_counts[c]:
            peaks = peaks[:max_counts[c]]
        for r, col, score in peaks:
            # Map back to original pixel coordinates (pixel-centre convention).
            rows.append((name, (r + 0.5) / scale - 0.5,
                         (col + 0.5) / scale - 0.5, float(score)))
    return rows


def draw_overlay(frame: np.ndarray, detections: list[tuple]) -> np.ndarray:
    """RGB uint8 overlay: SPB red / KT green image, cyan KT and magenta SPB rings."""
    rgb = np.zeros(frame.shape[1:] + (3,), np.float32)
    for c, out in ((0, 1), (1, 0)):  # KT -> green, SPB -> red
        lo, hi = np.percentile(frame[c], (1, 99.9))
        rgb[..., out] = np.clip((frame[c] - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.ascontiguousarray((rgb * 255).astype(np.uint8))
    radius = max(3, int(round(0.02 * max(frame.shape[1:]))))
    for name, r, c, _ in detections:
        color = (0, 255, 255) if name == "KT" else (255, 0, 255)
        cv2.circle(rgb, (int(round(c)), int(round(r))),
                   radius if name == "KT" else radius + 2, color, 1, cv2.LINE_AA)
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=HERE / "unet_kt_spb.pt")
    parser.add_argument("--axes", default=None,
                        help="axis order of the input, e.g. TCYX, TYXC, TZCYX is not supported")
    parser.add_argument("--kt-channel", type=int, default=1,
                        help="channel with Ndc80-GFP (default 1 = green of RGB)")
    parser.add_argument("--spb-channel", type=int, default=0,
                        help="channel with Cdc11-CFP (default 0 = red of RGB)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pixel-size", type=float, default=None,
                       help="camera pixel size in um (sets --scale automatically)")
    group.add_argument("--scale", type=float, default=1.0,
                       help="zoom applied before the U-Net (training px / video px)")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="heatmap peak threshold (0.2 = best F1 on simulations)")
    parser.add_argument("--max-kt", type=int, default=6)
    parser.add_argument("--max-spb", type=int, default=2)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--overlay", type=Path, default=None,
                        help="write an overlay video (.mp4) or image (.png)")
    args = parser.parse_args()

    scale = args.pixel_size / PIXEL_SIZE_UM if args.pixel_size else args.scale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    stack = to_tcyx(read_stack(args.video), args.axes)
    stack = stack[:, [args.kt_channel, args.spb_channel]].astype(np.float32)
    print(f"{len(stack)} frame(s) of {stack.shape[2]}x{stack.shape[3]} px, "
          f"scale {scale:.3f}")

    records, overlays = [], []
    for t, frame in enumerate(stack):
        prob = predict_frame(model, frame, scale, device)
        detections = detect(prob, scale, args.threshold, (args.max_kt, args.max_spb))
        records += [(t, *d) for d in detections]
        if args.overlay:
            overlays.append(draw_overlay(frame, detections))

    table = pd.DataFrame(records, columns=["frame", "class", "row", "col", "score"])
    if args.pixel_size:
        table["y_um"] = table["row"] * args.pixel_size
        table["x_um"] = table["col"] * args.pixel_size
    csv = args.csv or args.video.with_name(args.video.stem + "_detections.csv")
    table.to_csv(csv, index=False)
    counts = table.groupby("class").size().to_dict() if len(table) else {}
    print(f"wrote {csv}: {counts} detections in {len(stack)} frame(s)")

    if args.overlay:
        if len(overlays) == 1 or args.overlay.suffix.lower() in (".png", ".jpg"):
            cv2.imwrite(str(args.overlay), cv2.cvtColor(overlays[0], cv2.COLOR_RGB2BGR))
        else:
            h, w = overlays[0].shape[:2]
            writer = cv2.VideoWriter(str(args.overlay),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
            for o in overlays:
                writer.write(cv2.cvtColor(o, cv2.COLOR_RGB2BGR))
            writer.release()
        print(f"wrote {args.overlay}")


if __name__ == "__main__":
    main()
