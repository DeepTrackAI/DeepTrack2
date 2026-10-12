"""DeepTrack2 simulation of fission-yeast mitotic spindles for U-Net training.

Biology modelled (see ``image.png`` for the reference frame):

* Cdc11-CFP marks the two spindle pole bodies (SPBs) -> "red" channel.
* Ndc80-GFP marks the six kinetochores (KTs, 3 sister pairs) -> "green" channel.
* Wide-field fluorescence, time-lapse: defocus, diffuse nuclear Ndc80 haze,
  faint cytoplasmic CFP speckles, CFP/GFP bleed-through, shot + read noise.

Every call of :func:`SpindleSimulator.sample` draws one new spindle (length,
orientation, mitotic stage, KT arrangement, brightness, noise) and returns

* ``image``   : float32 array (2, H, W) — channel 0 = KT (GFP), 1 = SPB (CFP)
* ``target``  : float32 array (2, H, W) — Gaussian heatmaps of KT / SPB centres
* ``kt_pos``  : (n, 2) KT positions (row, col) in pixels
* ``spb_pos`` : (m, 2) SPB positions (row, col) in pixels

All physical parameters are *estimates*; the most important one to calibrate
against your microscope is ``PIXEL_SIZE_UM`` (camera pixel size in the
sample plane).
"""

from __future__ import annotations

import operator
from functools import reduce
from pathlib import Path

import deeptrack as dt
import numpy as np

# --------------------------------------------------------------------------- #
# Imaging constants (estimates — calibrate to your setup)
# --------------------------------------------------------------------------- #
PIXEL_SIZE_UM = 0.1          # sample-plane pixel size (resolution / magnification)
IMAGE_SIZE = 128             # training crops are IMAGE_SIZE x IMAGE_SIZE pixels
PADDING = (16, 16, 16, 16)
NA = 1.4
WAVELENGTH_GFP = 510e-9      # Ndc80-GFP emission
WAVELENGTH_CFP = 475e-9      # Cdc11-CFP emission

N_SPB = 2
N_KT = 6                     # 3 chromosomes x 2 sister kinetochores
N_HAZE = 10                  # dim Ndc80 puncta forming the diffuse nuclear haze
N_SPECKLES = 6               # faint cytoplasmic CFP speckles

HEATMAP_SIGMA_PX = 1.5       # width of the Gaussian training targets

UM = 1.0 / PIXEL_SIZE_UM     # pixels per micrometre


# --------------------------------------------------------------------------- #
# Spindle geometry (one shared draw per sample)
# --------------------------------------------------------------------------- #
def sample_spindle() -> dict:
    """Draw one spindle: centre, axis, length and mitotic stage.

    Stages (rough S. pombe mitosis):
      * ``prometaphase`` : short spindle, KTs scattered along/around it
      * ``metaphase``    : KT pairs clustered around the spindle middle
      * ``anaphase``     : sisters separated, KTs close to the SPBs
    """
    stage = np.random.choice(
        ["prometaphase", "metaphase", "anaphase"], p=[0.3, 0.45, 0.25]
    )
    length_um = {
        "prometaphase": np.random.uniform(0.8, 2.0),
        "metaphase": np.random.uniform(1.8, 3.2),
        "anaphase": np.random.uniform(3.0, 9.0),
    }[stage]
    length = length_um * UM

    angle = np.random.uniform(0, np.pi)
    axis = np.array([np.sin(angle), np.cos(angle)])           # (row, col)
    normal = np.array([-axis[1], axis[0]])

    margin = 0.5 * length * np.abs(axis) + 8
    centre = np.array([
        np.random.uniform(margin[0], IMAGE_SIZE - margin[0]),
        np.random.uniform(margin[1], IMAGE_SIZE - margin[1]),
    ]) if np.all(margin < IMAGE_SIZE / 2) else np.full(2, IMAGE_SIZE / 2)

    # Spindle may be tilted out of the focal plane (wide-field defocus).
    z_centre = np.random.normal(0, 0.2 * UM)
    z_tilt = np.random.normal(0, 0.06)  # dz per pixel along the axis

    return dict(
        stage=stage, length=length, axis=axis, normal=normal,
        centre=centre, z_centre=z_centre, z_tilt=z_tilt,
    )


def point_on_spindle(spindle: dict, t: float, lateral: float = 0.0) -> np.ndarray:
    """(row, col) of fraction ``t`` (0 = SPB 1, 1 = SPB 2) along the spindle."""
    return (
        spindle["centre"]
        + (t - 0.5) * spindle["length"] * spindle["axis"]
        + lateral * spindle["normal"]
    )


def z_on_spindle(spindle: dict, t: float) -> float:
    return spindle["z_centre"] + (t - 0.5) * spindle["length"] * spindle["z_tilt"]


def spb_fraction(index: int) -> float:
    return 0.0 if index == 0 else 1.0


def kt_fraction(spindle: dict, index: int, pair_centre: float) -> float:
    """Fraction along the spindle of kinetochore ``index`` (pair = index // 2)."""
    sister = -1 if index % 2 == 0 else 1
    stage = spindle["stage"]
    if stage == "anaphase":
        # Sisters segregated: one to each pole, slightly lagging behind it.
        lag = np.random.uniform(0.02, 0.15)
        return lag if sister < 0 else 1 - lag
    separation = np.random.uniform(0.2, 0.7) * UM / spindle["length"]
    return pair_centre + sister * separation / 2


# --------------------------------------------------------------------------- #
# Custom operators (wrapped with dt.Lambda)
# --------------------------------------------------------------------------- #
def camera(peak_photons: float = 200.0, background_photons: float = 10.0,
           read_noise: float = 2.0):
    """Wide-field camera: scale to photons, add background, shot + read noise."""

    def camera_function(image):
        image = np.asarray(image, dtype=np.float64)
        peak = image.max()
        signal = image / peak * peak_photons if peak > 0 else image
        photons = np.random.poisson(np.clip(signal + background_photons, 0, None))
        return photons + np.random.normal(0, read_noise, photons.shape)

    return camera_function


def bleed_through(gfp_into_cfp: float = 0.05, cfp_into_gfp: float = 0.05):
    """Mix the [KT (GFP), SPB (CFP)] channels to mimic spectral crosstalk."""

    def mix_function(images):
        gfp, cfp = (np.asarray(im, dtype=np.float64).squeeze() for im in images)
        return np.stack([gfp + cfp_into_gfp * cfp, cfp + gfp_into_cfp * gfp])

    return mix_function


def normalize_frame(image: np.ndarray) -> np.ndarray:
    """Per-channel robust normalisation; used for simulation AND real frames.

    ``image`` is (C, H, W). Background median -> 0, 99.9th percentile -> 1.
    """
    image = np.asarray(image, dtype=np.float32)
    out = np.empty_like(image)
    for c in range(image.shape[0]):
        low = np.median(image[c])
        high = np.percentile(image[c], 99.9)
        out[c] = (image[c] - low) / max(high - low, 1e-6)
    return np.clip(out, -1.0, 5.0)


def normalize():
    def normalize_function(image):
        return normalize_frame(image)

    return normalize_function


def render_heatmap(positions: np.ndarray, shape: tuple[int, int],
                   sigma: float = HEATMAP_SIGMA_PX) -> np.ndarray:
    """Max-combined Gaussian blobs (peak 1) at ``positions`` (row, col)."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    heatmap = np.zeros(shape, dtype=np.float32)
    for r, c in np.reshape(positions, (-1, 2)):
        blob = np.exp(-((rows - r) ** 2 + (cols - c) ** 2) / (2 * sigma**2))
        heatmap = np.maximum(heatmap, blob)
    return heatmap


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def build_optics(wavelength: float) -> dt.Fluorescence:
    return dt.Fluorescence(
        NA=NA,
        resolution=PIXEL_SIZE_UM * 1e-6 * 10,
        magnification=10,
        wavelength=wavelength,
        padding=PADDING,
        output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE),
    )


class SpindleSimulator:
    """Builds the DeepTrack2 pipeline and produces (image, target) samples.

    All scatterers depend on one shared ``spindle`` feature, so a single
    ``.update()`` draws a consistent spindle for both channels. (The
    scatterers are listed explicitly instead of using ``^``, because each
    ``^`` copy would draw its own spindle.)
    """

    def __init__(self) -> None:
        self.spindle = dt.Value(value=sample_spindle)
        # Shared per-sample KT pair centres (metaphase / prometaphase).
        self.pairs = dt.Value(value=lambda: np.random.normal(0.5, 0.12, 3))

        # ---------------- SPBs (Cdc11-CFP) ---------------- #
        self.spbs = [
            dt.Sphere(
                position=lambda spindle, i=i: point_on_spindle(
                    spindle, spb_fraction(i), np.random.normal(0, 0.3)
                ),
                z=lambda spindle, i=i: z_on_spindle(spindle, spb_fraction(i)),
                spindle=self.spindle.value,
                position_unit="pixel",
                radius=lambda: np.random.uniform(0.08, 0.15) * 1e-6,
                intensity=lambda: np.random.uniform(0.8, 1.2),
            )
            for i in range(N_SPB)
        ]

        # ---------------- KTs (Ndc80-GFP) ---------------- #
        def kt_position(spindle, pairs, i):
            stage = spindle["stage"]
            pair_centre = np.clip(pairs[i // 2], 0.2, 0.8)
            if stage == "prometaphase":
                pair_centre = np.random.uniform(0.0, 1.0)
            t = kt_fraction(spindle, i, pair_centre)
            lateral_sd = {"prometaphase": 0.35, "metaphase": 0.15,
                          "anaphase": 0.15}[stage] * UM
            return point_on_spindle(spindle, t, np.random.normal(0, lateral_sd))

        self.kts = [
            dt.Sphere(
                position=lambda spindle, pairs, i=i: kt_position(spindle, pairs, i),
                z=lambda spindle: z_on_spindle(spindle, 0.5)
                + np.random.normal(0, 0.15 * UM),
                spindle=self.spindle.value,
                pairs=self.pairs.value,
                position_unit="pixel",
                radius=lambda: np.random.uniform(0.06, 0.12) * 1e-6,
                intensity=lambda: np.random.uniform(0.5, 1.0),
            )
            for i in range(N_KT)
        ]

        # ---------------- distractors (not labelled) ---------------- #
        haze = [
            dt.PointParticle(
                position=lambda spindle: point_on_spindle(
                    spindle, np.random.uniform(-0.1, 1.1),
                    np.random.normal(0, 0.5 * UM),
                ),
                z=lambda: np.random.normal(0, 0.6 * UM),
                spindle=self.spindle.value,
                position_unit="pixel",
                intensity=lambda: np.random.uniform(0.0, 0.25),
            )
            for _ in range(N_HAZE)
        ]
        speckles = [
            dt.PointParticle(
                position=lambda spindle: point_on_spindle(
                    spindle, np.random.uniform(-0.5, 1.5),
                    np.random.normal(0, 1.5 * UM),
                ),
                z=lambda: np.random.normal(0, 0.5 * UM),
                spindle=self.spindle.value,
                position_unit="pixel",
                intensity=lambda: np.random.uniform(0.0, 0.15),
            )
            for _ in range(N_SPECKLES)
        ]

        kt_sample = reduce(operator.rshift, self.kts + haze)
        spb_sample = reduce(operator.rshift, self.spbs + speckles)

        def channel_camera() -> dt.Feature:
            return dt.Lambda(
                function=camera,
                peak_photons=lambda: np.random.uniform(40, 400),
                background_photons=lambda: np.random.uniform(5, 50),
                read_noise=lambda: np.random.uniform(1, 4),
            )

        kt_image = build_optics(WAVELENGTH_GFP)(kt_sample) >> channel_camera()
        spb_image = build_optics(WAVELENGTH_CFP)(spb_sample) >> channel_camera()

        mixer = dt.Lambda(
            function=bleed_through,
            gfp_into_cfp=lambda: np.random.uniform(0.0, 0.15),
            cfp_into_gfp=lambda: np.random.uniform(0.0, 0.15),
        )
        mixer.__distributed__ = False  # receives both channels at once

        self.pipeline = (kt_image & spb_image) >> mixer >> dt.Lambda(function=normalize)

        self.kt_positions = dt.TakeProperties(
            reduce(operator.rshift, self.kts), "position")
        self.spb_positions = dt.TakeProperties(
            reduce(operator.rshift, self.spbs), "position")

    def sample(self) -> dict:
        self.pipeline.update()
        image = np.asarray(self.pipeline.resolve(), dtype=np.float32)
        kt_pos = self._in_view(self.kt_positions.resolve())
        spb_pos = self._in_view(self.spb_positions.resolve())
        shape = image.shape[-2:]
        target = np.stack([render_heatmap(kt_pos, shape),
                           render_heatmap(spb_pos, shape)])
        return dict(image=image, target=target, kt_pos=kt_pos, spb_pos=spb_pos,
                    stage=self.spindle.value()["stage"])

    @staticmethod
    def _in_view(positions) -> np.ndarray:
        positions = np.reshape(np.asarray(positions, dtype=np.float32), (-1, 2))
        inside = np.all((positions >= 0) & (positions < IMAGE_SIZE), axis=1)
        return positions[inside]


def to_rgb(image: np.ndarray) -> np.ndarray:
    """(2, H, W) [KT, SPB] -> RGB with KT green and SPB red, like the reference."""
    kt, spb = (np.clip(image[c], 0, 1) for c in (0, 1))
    return np.stack([spb, kt, np.zeros_like(kt)], axis=-1)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(0)
    simulator = SpindleSimulator()
    n = 6
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.4))
    for j in range(n):
        s = simulator.sample()
        assert s["image"].shape == (2, IMAGE_SIZE, IMAGE_SIZE)
        assert np.isfinite(s["image"]).all()
        axes[0, j].imshow(to_rgb(s["image"]))
        axes[0, j].set_title(s["stage"])
        axes[1, j].imshow(to_rgb(s["image"]))
        axes[1, j].scatter(s["kt_pos"][:, 1], s["kt_pos"][:, 0], s=25,
                           facecolors="none", edgecolors="cyan", linewidths=0.8)
        axes[1, j].scatter(s["spb_pos"][:, 1], s["spb_pos"][:, 0], s=60,
                           facecolors="none", edgecolors="magenta", linewidths=1)
        axes[1, j].set_title(f"{len(s['kt_pos'])} KT, {len(s['spb_pos'])} SPB")
        for ax in axes[:, j]:
            ax.axis("off")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "spindle_simulation_examples.png"
    fig.savefig(out, dpi=90)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
