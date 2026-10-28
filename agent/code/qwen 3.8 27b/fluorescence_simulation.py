"""DeepTrack2 dual-channel (red / green) fluorescence simulation.

Reproduces the visual style of ``image.png`` held in this folder:
a near-black field of view with two distinct red spots and a diagonal
band of green spots, each carrying a handful of small bright satellite
specks, plus faint scattered background grains.

The simulation is built entirely from documented DeepTrack2 primitives:

  * :class:`dt.PointParticle`   – one emissive voxel (fluorophore);
  * ``&``                       – combine several scatterers into one sample;
  * ``tuple`` sample            – one sample per wavelength;
  * :class:`dt.Fluorescence`    – incoherent wide-field fluorescence optics;
  * :class:`dt.NormalizeMinMax` – per-channel range normalisation;
  * :class:`dt.GaussianBlur`    – out-of-focus / defocused PSF broadening;
  * :class:`dt.Lambda` / :class:`dt.Merge` – custom numpy ops (add / colour);
  * ``>>`` / ``&``              – pipeline chaining.

Run directly::

    python fluorescence_simulation.py

Produces ``simulated_image.png`` (RGB) and ``simulated_channels.png``
(the two single-channel views side-by-side) in the current folder.
"""
from __future__ import annotations

import os
import warnings
from typing import Tuple

import numpy as np
from PIL import Image

import deeptrack as dt


# ---------------------------------------------------------------------------
# Reference geometry (read from image.png in this folder)
# ---------------------------------------------------------------------------

def _reference_size() -> Tuple[int, int]:
    """Return (H, W) of the reference image, falling back to a 1:3 banner."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        with Image.open(os.path.join(here, "image.png")) as ref:
            w, h = ref.size
    except Exception:
        h, w = 411, 1278
    return h, w


H, W = _reference_size()  # ~(411, 1278) - the reference is a wide banner


# ---------------------------------------------------------------------------
# Custom numpy helpers exposed to the DeepTrack pipeline via dt.Lambda
# ---------------------------------------------------------------------------

def make_rgb_volume(**kwargs):
    """Factory for dt.Lambda: re-arrange (2, H, W, 1) stack into (H, W, 3) RGB."""
    def _rgb_volume(img_stack: np.ndarray) -> np.ndarray:
        g = np.asarray(img_stack[0], dtype=np.float64).squeeze(-1)
        r = np.asarray(img_stack[1], dtype=np.float64).squeeze(-1)
        return np.stack(
            [
                np.clip(r, 0.0, 1.0),
                np.clip(g, 0.0, 1.0),
                0.25 * np.clip(r + g, 0.0, 1.0),  # faint warm halo where channels cross
            ],
            axis=-1,
        )

    return _rgb_volume


def make_shot_noise(**kwargs):
    """Factory for dt.Lambda: gentle camera shot noise + sensor read-noise floor.

    Shot noise is modelled as Poisson sampling of the photon count, scaled by a
    per-pixel gain.  The mean of the Poisson distribution equals the signal, so
    the image is preserved in expectation.
    """
    def _shot_noise(rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        rng = np.random.default_rng(1234)
        # scale by a photon-count gain of ~80 so shot noise is visible but not dominant
        gain = 80.0
        photons = rgb * gain
        noisy = rng.poisson(lam=photons).astype(np.float64) / gain
        # small read-noise floor
        noisy += rng.uniform(0.0, 0.004, size=rgb.shape)
        return np.clip(noisy, 0.0, 1.0)

    return _shot_noise


def make_finalize(**kwargs):
    """Factory for dt.Lambda: quantize to uint8 0-255 RGB."""
    def _finalize(rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        rgb = np.clip(rgb, 0.0, 1.0) * 255.0
        return rgb.astype(np.uint8)

    return _finalize


rgb_feature = dt.Lambda(function=make_rgb_volume)
rgb_feature.__distributed__ = False

noise_feature = dt.Lambda(function=make_shot_noise)
noise_feature.__distributed__ = False

finalize_feature = dt.Lambda(function=make_finalize)
finalize_feature.__distributed__ = False


# ---------------------------------------------------------------------------
# Scatterer (fluorophore) construction
# ---------------------------------------------------------------------------

def _point(x: float, y: float, intensity: float) -> dt.PointParticle:
    """A fluorophore at pixel (x=col, y=row) with relative intensity.

    ``PointParticle`` handles the sub-pixel emission discretization for us.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dt.PointParticle(
            position=(y, x, 0),
            position_unit="pixel",
            intensity=float(intensity),
        )


def _ellipsoid(x: float, y: float, radius_px: float,
               intensity: float = 1.0) -> dt.Ellipsoid:
    """A solid circular fluorescent inclusion of radius ``radius_px`` pixels.

    ``dt.Ellipsoid`` is a true volumetric scatterer whose in-focus projection
    is a filled disk -- exactly the look of an out-of-focus inclusion.
    """
    r_m = radius_px * 1e-6 * 1e-4  # 100 nm per pixel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dt.Ellipsoid(
            position=(y, x, 0),
            position_unit="pixel",
            radius=r_m,
            intensity=float(intensity),
        )


def _red_blobs() -> dt.Ellipsoid:
    """Two large, solid red inclusions matching the reference blob sizes."""
    b1 = _ellipsoid(866, 117, 20.0, 1.0)
    b2 = _ellipsoid(501, 260, 17.5, 1.0)
    return b1 & b2


def _red_satellites() -> dt.PointParticle:
    """Small, brighter red satellite specks."""
    pts = [_point(x, y, a) for x, y, a in
           [(831, 61, 1.0), (881, 86, 0.9), (831, 94, 0.7),
            (940, 175, 0.6), (800, 250, 0.5), (900, 30, 0.5),
            (960, 240, 0.4), (820, 150, 0.4)]]
    s = pts[0]
    for p in pts[1:]:
        s = s & p
    return s


def _green_blobs() -> dt.Ellipsoid:
    """Diagonal band of solid green inclusions."""
    centers = [
        (576, 225, 15.5),
        (613, 213, 14.5),
        (663, 228, 15.0),
        (714, 196, 14.0),
        (763, 164, 16.5),
        (835, 163, 13.0),
    ]
    sample: dt.Ellipsoid | None = None
    for x, y, r in centers:
        e = _ellipsoid(x, y, r, 1.0)
        sample = e if sample is None else (sample & e)
    return sample  # type: ignore[return-value]


def _green_satellites() -> dt.PointParticle:
    """Small, brighter green satellite specks."""
    pts = [_point(x, y, a) for x, y, a in
           [(532, 205, 0.9), (620, 250, 0.7), (700, 150, 0.8),
            (870, 210, 0.55), (560, 170, 0.5), (770, 250, 0.45),
            (600, 150, 0.4)]]
    s = pts[0]
    for p in pts[1:]:
        s = s & p
    return s


def make_merge_images(**kwargs):
    """Factory for dt.Merge: add a list of images together (clipped to 0..1)."""
    def _merge(imgs: list) -> np.ndarray:
        acc = None
        for x in imgs:
            a = np.asarray(x, dtype=np.float64)
            acc = a if acc is None else (acc + a)
        return np.clip(acc, 0.0, 1.0)

    return _merge


def _channel_image(optics: dt.Fluorescence, blob_sample, sat_sample) -> dt.Feature:
    """Image one channel = soft blobs (heavy blur) + sharp satellites (light blur).

    The reference shows two distinct scales: broad, diffuse inclusions and a
    few tight bright specks. Imaging the two populations with different blur
    widths and adding them (via ``dt.Merge``) reproduces that two-tier texture.
    """
    blobs = (optics(blob_sample)
             >> dt.NormalizeMinMax()
             >> dt.GaussianBlur(sigma=11)
             >> dt.NormalizeMinMax())
    satellites = (optics(sat_sample)
                  >> dt.NormalizeMinMax()
                  >> dt.GaussianBlur(sigma=1.5)
                  >> dt.NormalizeMinMax())
    return (blobs & satellites) >> dt.Merge(function=make_merge_images)


# ---------------------------------------------------------------------------
# Microscope (one fluorescence optics per channel)
# ---------------------------------------------------------------------------

def _fluorescence(wavelength: float) -> dt.Fluorescence:
    return dt.Fluorescence(
        NA=1.2,                       # high-NA oil objective - diffraction-limited blur
        resolution=1e-6,              # 1 um per pixel before magnification
        magnification=10,             # -> 100 nm per pixel after magnification
        wavelength=wavelength,
        padding=(32, 32, 32, 32),
        output_region=(0, 0, H, W),
    )


def build_optics() -> tuple[dt.Fluorescence, dt.Fluorescence]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "NA / wavelength * resolution / magnification"
        g = _fluorescence(wavelength=525e-9)  # green channel
        r = _fluorescence(wavelength=620e-9)  # red channel
    return g, r


# ---------------------------------------------------------------------------
# Top-level pipeline (pure DeepTrack chaining)
# ---------------------------------------------------------------------------

def build_pipeline() -> dt.Feature:
    green_optics, red_optics = build_optics()
    green_img = _channel_image(green_optics, _green_blobs(), _green_satellites())
    red_img = _channel_image(red_optics, _red_blobs(), _red_satellites())

    # Combine the two channels, colorize, add shot noise, quantize.
    return ((green_img & red_img)
            >> rgb_feature
            >> noise_feature
            >> finalize_feature)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    pipeline = build_pipeline()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = np.asarray(pipeline.resolve(), dtype=np.uint8)

    # DeepTrack wraps outputs in batch/sequence dims -> keep the (H, W, 3) core
    if image.ndim > 3:
        image = image.reshape((-1, H, W, 3))
    image = image.reshape((H, W, 3))

    here = os.path.dirname(os.path.abspath(__file__))
    out_rgb = os.path.join(here, "simulated_image.png")
    Image.fromarray(image, mode="RGB").save(out_rgb)
    print(f"wrote {out_rgb} shape={image.shape}")

    # Also save the two normalised single-channel views for inspection.
    g, r = build_optics()
    gs = np.asarray(_channel_image(g, _green_blobs(), _green_satellites()).resolve(),
                    dtype=np.float64).reshape((H, W))
    rs = np.asarray(_channel_image(r, _red_blobs(), _red_satellites()).resolve(),
                    dtype=np.float64).reshape((H, W))
    side = np.concatenate(
        [(rs * 255).astype(np.uint8), np.zeros((H, 8), dtype=np.uint8),
         (gs * 255).astype(np.uint8)],
        axis=1,
    )
    out_chan = os.path.join(here, "simulated_channels.png")
    Image.fromarray(side, mode="L").save(out_chan)
    print(f"wrote {out_chan} shape={side.shape}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(image)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(here, "simulated_image_preview.png"), dpi=110)
        print("wrote simulated_image_preview.png")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"(no preview: {exc})")


if __name__ == "__main__":
    main()
