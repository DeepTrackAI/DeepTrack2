"""DeepTrack2 frame renderer.

Produces a synthetic fluorescence frame with 6 kinetochore-like inclusions
(Ndc80-GFP, green channel) and 2 spindle pole body inclusions (Cdc11-CFP,
red channel), plus a small number of satellite specks in each channel.

Output layout
-------------
``frame["image"]``  -- (H, W, 2) uint16, normalised [0, 65535]
    channel 0 = green (KT signal)
    channel 1 = red   (SPB signal)
``frame["masks"]`` -- (H, W, 2) uint8, analytic disk ground truth
``frame["kt"]``    -- list of (x, y, r) tuples  (green objects)
``frame["spb"]``   -- list of (x, y, r) tuples  (red objects)

Each frame is deterministic in ``frame_idx`` (seeded numpy RNG), so it can be
pre-rendered once to disk and used by any number of Dataloader workers.
"""
from __future__ import annotations

import warnings
import numpy as np

import deeptrack as dt


# ---------------------------------------------------------------------------
# Scattering primitives
# ---------------------------------------------------------------------------

def _make_green_optics(h: int, w: int) -> dt.Fluorescence:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # "NA/wavelength * res/mag" warning
        return dt.Fluorescence(
            NA=1.2, resolution=1e-6, magnification=10,
            wavelength=525e-9, padding=(32, 32, 32, 32),
            output_region=(0, 0, h, w),
        )


def _make_red_optics(h: int, w: int) -> dt.Fluorescence:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dt.Fluorescence(
            NA=1.2, resolution=1e-6, magnification=10,
            wavelength=620e-9, padding=(32, 32, 32, 32),
            output_region=(0, 0, h, w),
        )


def _disk(x: float, y: float, radius_px: float) -> dt.Ellipsoid:
    """Solid circular fluorescent inclusion (radius in px, 100 nm/px)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dt.Ellipsoid(
            position=(y, x, 0), position_unit="pixel",
            radius=radius_px * 1e-7,            # 100 nm per pixel  (1e-7 m)
            intensity=1.0,
        )


def _satellite(x: float, y: float, intensity: float) -> dt.PointParticle:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dt.PointParticle(
            position=(y, x, 0), position_unit="pixel", intensity=float(intensity),
        )


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _green_channel(rng: np.random.Generator,
                   g_opt: dt.Fluorescence, h: int, w: int):
    """Render the green channel and return (image_2d, kt_object_list)."""
    # 6 kinetochore positions along a diagonal band (like the reference).
    t = np.sort(rng.uniform(0.2, 0.8, 6))
    cx = t * w
    cy = (h * 0.55) + (h * 0.2) * (t - 0.5) * 2 + rng.normal(0, 8, 6)
    cr = 14.0 + rng.normal(0, 1.0, 6)

    blobs: dt.Ellipsoid | None = None
    for x, y, r in zip(cx, cy, cr):
        e = _disk(float(x), float(y), float(r))
        blobs = e if blobs is None else (blobs & e)

    # A few dim satellite specks near the KTs (typical of wide-field).
    sats: dt.PointParticle | None = None
    for _ in range(int(rng.integers(3, 9))):
        sx = float(rng.uniform(0.1 * w, 0.9 * w))
        sy = float(rng.uniform(0.15 * h, 0.85 * h))
        a = float(rng.uniform(0.4, 1.0))
        p = _satellite(sx, sy, a)
        sats = p if sats is None else (sats & p)

    sample: dt.Feature = blobs if sats is None else (blobs & sats)  # type: ignore[misc]
    img = np.asarray(g_opt(sample).resolve(), dtype=np.float64).reshape((h, w))
    return img, [(float(x), float(y), float(r)) for x, y, r in zip(cx, cy, cr)]


def _red_channel(rng: np.random.Generator,
                 r_opt: dt.Fluorescence, h: int, w: int):
    """Render the red channel and return (image_2d, spb_object_list)."""
    # 2 spindle pole bodies (Cdc11-CFP) -- larger than KT inclusions.
    sx = rng.uniform(0.15, 0.85, 2) * w
    sy = rng.uniform(0.15, 0.85, 2) * h
    # Enforce a minimum separation between the two SPBs.
    min_sep = 60.0
    while np.hypot(sx[1] - sx[0], sy[1] - sy[0]) < min_sep:
        sx[1] = rng.uniform(0.15, 0.85) * w
        sy[1] = rng.uniform(0.15, 0.85) * h
    sr = 18.0 + rng.normal(0, 1.2, 2)

    blobs: dt.Ellipsoid | None = None
    for x, y, r in zip(sx, sy, sr):
        e = _disk(float(x), float(y), float(r))
        blobs = e if blobs is None else (blobs & e)

    sats: dt.PointParticle | None = None
    for _ in range(int(rng.integers(2, 6))):
        px = float(rng.uniform(0.1 * w, 0.9 * w))
        py = float(rng.uniform(0.15 * h, 0.85 * h))
        a = float(rng.uniform(0.4, 1.0))
        p = _satellite(px, py, a)
        sats = p if sats is None else (sats & p)

    sample: dt.Feature = blobs if sats is None else (blobs & sats)  # type: ignore[misc]
    img = np.asarray(r_opt(sample).resolve(), dtype=np.float64).reshape((h, w))
    return img, [(float(x), float(y), float(r)) for x, y, r in zip(sx, sy, sr)]


def _illumination(h: int, w: int) -> np.ndarray:
    """Mild diagonal illumination gradient, values in [1 - a, 1]."""
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    return 1.0 - 0.15 * (0.3 * xx + 0.7 * yy)


def _shot_noise(rgb: np.ndarray, gain: float = 80.0, read: float = 0.004,
                seed: int = 0) -> np.ndarray:
    """Poisson shot noise + uniform read-noise floor on uint8-like 0..1 image."""
    rng = np.random.default_rng(seed)
    lam = np.clip(rgb, 0.0, None) * gain
    noisy = rng.poisson(lam).astype(np.float64) / gain
    noisy += rng.uniform(0.0, read, size=rgb.shape)
    return np.clip(noisy, 0.0, 1.0)


def _disk_mask(h: int, w: int,
               objs: list[tuple[float, float, float]]) -> np.ndarray:
    """Binary analytic disk inside the field of view."""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    m = np.zeros((h, w), dtype=np.uint8)
    for cx, cy, r in objs:
        m |= ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * r)
    return m


def render_frame(frame_idx: int, h: int = 412, w: int = 640,
                 optics: tuple[dt.Fluorescence, dt.Fluorescence] | None = None):
    """Render one synthetic frame deterministically from ``frame_idx``."""
    rng = np.random.default_rng(1_000_000 + int(frame_idx))
    if optics is None:
        g_opt, r_opt = _make_green_optics(h, w), _make_red_optics(h, w)
    else:
        g_opt, r_opt = optics

    g, kts = _green_channel(rng, g_opt, h, w)
    r, spbs = _red_channel(rng, r_opt, h, w)

    # Peak-normalise each channel (keeps the downstream noise model meaningful).
    g = np.clip(g / (g.max() + 1e-9), 0.0, 1.0)
    r = np.clip(r / (r.max() + 1e-9), 0.0, 1.0)

    illumin = _illumination(h, w)
    g *= illumin
    r *= illumin

    rgb = np.stack([g, r], axis=-1)
    rgb = _shot_noise(rgb, gain=80.0, read=0.004, seed=frame_idx * 31 + 7)
    image = (rgb * 65535).astype(np.uint16)

    mask_green = _disk_mask(h, w, kts)
    mask_red = _disk_mask(h, w, spbs)
    mask = np.stack([mask_green, mask_red], axis=-1).astype(np.uint8)

    return {"image": image, "masks": mask, "kt": kts, "spb": spbs}
