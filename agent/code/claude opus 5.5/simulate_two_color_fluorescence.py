"""Two-color (red / green) fluorescence simulation with DeepTrack2.

Reference: ``image.png`` in this folder — a dark-field-of-view, two-channel
fluorescence micrograph showing:

* red channel   : two bright, roughly round puncta at both ends of a diagonal
                  track, plus a sparse sprinkle of faint red speckles;
* green channel : a handful of bright blobs strung along the same diagonal,
                  embedded in a diffuse, cloud-like green haze.

Each colour channel is imaged with its own ``dt.Fluorescence`` microscope
(different emission wavelength), the channels are combined with ``&`` and
merged into an RGB image with a ``dt.Lambda`` operator.

All physical parameters below are *estimates* chosen to visually match the
reference image; calibrate them against real data if needed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe; the figure is also saved to disk

from pathlib import Path

import deeptrack as dt
import matplotlib.pyplot as plt
import numpy as np

# Field of view (pixels) — same ~1:3.1 aspect ratio as the reference image.
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 400
PADDING = (32, 32, 32, 32)

# Number of objects per channel (estimates from the reference image).
N_RED_PUNCTA = 2
N_RED_SPECKLES = 25
N_GREEN_BLOBS = 4
N_GREEN_HAZE = 120

OUTPUT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def sample_track(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample a diagonal track (start point, end point) in pixel coordinates.

    Positions are (row, column). In the reference the track goes from the
    lower-left (red puncta) to the upper-right (red puncta) through the
    middle of the image.
    """
    start = np.array([
        rng.uniform(0.55, 0.75) * IMAGE_HEIGHT,
        rng.uniform(0.30, 0.45) * IMAGE_WIDTH,
    ])
    length = rng.uniform(0.25, 0.35) * IMAGE_WIDTH
    angle = rng.uniform(np.deg2rad(15), np.deg2rad(30))  # rising to the right
    end = start + length * np.array([-np.sin(angle), np.cos(angle)])
    return start, end


def point_on_track(
    start: np.ndarray,
    end: np.ndarray,
    t_range: tuple[float, float],
    jitter: float,
) -> np.ndarray:
    """Random point on the segment ``start``–``end`` with Gaussian jitter."""
    t = np.random.uniform(*t_range)
    return start + t * (end - start) + np.random.normal(0, jitter, 2)


# --------------------------------------------------------------------------- #
# Custom operator wrapped with dt.Lambda: stack (red, green) -> RGB
# --------------------------------------------------------------------------- #
def merge_to_rgb(red_gain: float = 1.0, green_gain: float = 0.85):
    """Return a function mapping a [red, green] list of images to RGB."""

    def rgb_function(images):
        red, green = (np.asarray(im).squeeze() for im in images)
        red = np.clip(red * red_gain, 0, 1)
        green = np.clip(green * green_gain, 0, 1)
        blue = 0.15 * np.minimum(red, green)  # slight overlap tint
        return np.stack([red, green, blue], axis=-1)

    return rgb_function


def clip_unit():
    """Return a function clipping an image to [0, 1]."""

    def clip_function(image):
        return np.clip(image, 0.0, 1.0)

    return clip_function


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def build_channel_optics(wavelength: float) -> dt.Fluorescence:
    """Fluorescence microscope for one emission channel."""
    return dt.Fluorescence(
        NA=0.8,
        resolution=1e-6,
        magnification=10,
        wavelength=wavelength,
        padding=PADDING,
        output_region=(0, 0, IMAGE_HEIGHT, IMAGE_WIDTH),
    )


def build_pipeline(seed: int | None = None) -> dt.Feature:
    """Build the two-colour fluorescence pipeline.

    The track geometry is fixed per pipeline (shared by both channels);
    object positions, sizes and intensities are re-sampled on every
    ``.update()``.
    """
    rng = np.random.default_rng(seed)
    start, end = sample_track(rng)

    # ---------------- red channel ---------------- #
    red_endpoints = iter([])

    def red_end_position():
        nonlocal red_endpoints
        # Alternate between the two ends of the track.
        try:
            t = next(red_endpoints)
        except StopIteration:
            red_endpoints = iter([0.0, 1.05])
            t = next(red_endpoints)
        return (start + t * (end - start)
                + np.random.normal(0, 1.0, 2)) * dt.units.pixel

    red_puncta = dt.Ellipsoid(
        position=red_end_position,
        position_unit="pixel",
        z=lambda: np.random.normal(0, 1),
        radius=lambda: np.random.uniform(0.6e-6, 0.8e-6, 3),
        rotation=lambda: np.random.uniform(0, np.pi),
        intensity=lambda: np.random.uniform(0.9, 1.1),
    )

    red_speckles = dt.PointParticle(
        position=lambda: (
            point_on_track(start, end, (0.9, 1.8), jitter=10.0)
        ) * dt.units.pixel,
        position_unit="pixel",
        z=lambda: np.random.normal(0, 5),
        intensity=lambda: np.random.uniform(5.0, 40.0),
    )

    red_sample = (red_puncta ^ N_RED_PUNCTA) >> (red_speckles ^ N_RED_SPECKLES)

    # ---------------- green channel ---------------- #
    green_blobs = dt.Ellipsoid(
        position=lambda: (
            point_on_track(start, end, (0.25, 0.95), jitter=2.5)
        ) * dt.units.pixel,
        position_unit="pixel",
        z=lambda: np.random.normal(0, 2),
        radius=lambda: np.random.uniform(0.45e-6, 0.65e-6, 3),
        rotation=lambda: np.random.uniform(0, np.pi),
        intensity=lambda: np.random.uniform(0.5, 0.9),
    )

    green_haze = dt.PointParticle(
        position=lambda: (
            point_on_track(start, end, (0.1, 1.0), jitter=6.0)
        ) * dt.units.pixel,
        position_unit="pixel",
        z=lambda: np.random.normal(0, 8),
        intensity=lambda: np.random.uniform(10.0, 45.0),
    )

    green_sample = (green_blobs ^ N_GREEN_BLOBS) >> (green_haze ^ N_GREEN_HAZE)

    # ---------------- imaging ---------------- #
    red_optics = build_channel_optics(wavelength=610e-9)
    green_optics = build_channel_optics(wavelength=520e-9)

    camera = (
        dt.NormalizeMinMax()
        >> dt.Poisson(snr=lambda: np.random.uniform(40, 80), background=0.0)
        >> dt.Gaussian(mu=0.0, sigma=lambda: np.random.uniform(0.002, 0.008))
        >> dt.Lambda(function=clip_unit)
    )

    red_image = red_optics(red_sample) >> camera
    green_image = green_optics(green_sample) >> camera

    to_rgb = dt.Lambda(function=merge_to_rgb)
    to_rgb.__distributed__ = False  # receive both channels at once

    return (red_image & green_image) >> to_rgb


def main() -> None:
    n_examples = 4
    fig, axes = plt.subplots(n_examples + 1, 1, figsize=(10, 2.6 * (n_examples + 1)))

    reference_path = OUTPUT_DIR / "image.png"
    if reference_path.exists():
        axes[0].imshow(plt.imread(reference_path))
        axes[0].set_title("Reference")
    axes[0].axis("off")

    for i in range(n_examples):
        pipeline = build_pipeline(seed=i)
        image = np.asarray(pipeline.update().resolve())
        assert image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3), image.shape
        assert np.isfinite(image).all()

        plt.imsave(OUTPUT_DIR / f"simulated_{i}.png", np.clip(image, 0, 1))
        axes[i + 1].imshow(np.clip(image, 0, 1))
        axes[i + 1].set_title(f"Simulated #{i}")
        axes[i + 1].axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "simulated_overview.png", dpi=100)
    plt.show()
    print(f"Saved simulated images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
