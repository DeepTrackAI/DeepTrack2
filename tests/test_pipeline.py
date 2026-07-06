import unittest

import warnings
from contextlib import contextmanager

import numpy as np

from deeptrack.backend import TORCH_AVAILABLE
from deeptrack.optical.optics import Fluorescence, Brightfield, Darkfield
from deeptrack.optical import scatterers
from tests import BackendTestBase

if TORCH_AVAILABLE:
    import torch


class TestScatterers_NumPy(BackendTestBase):
    BACKEND = "numpy"

    _EXPECTED_OPTICS_WARNING_PATTERNS = (
        r"Brightfield imaging from ScatteredVolume assumes a weak-phase / projection approximation.*",
        r"Darkfield imaging from ScatteredVolume is a very rough approximation.*",
        r"Approximating darkfield contrast from refractive index.*",
    )

    @contextmanager
    def _suppress_expected_optics_warnings(self):
        with warnings.catch_warnings():
            for pattern in self._EXPECTED_OPTICS_WARNING_PATTERNS:
                warnings.filterwarnings(
                    "ignore",
                    message=pattern,
                    category=UserWarning,
                )
            yield

    @property
    def array_type(self):
        if self.BACKEND == "numpy":
            return np.ndarray
        elif self.BACKEND == "torch":
            return torch.Tensor
        else:
            raise ValueError(f"Unsupported backend: {self.BACKEND}")

    def test__all__(self):
        from deeptrack import (
            PointParticle,
            Ellipse,
            Sphere,
            Ellipsoid,
            MieSphere,
            MieStratifiedSphere,
            Incoherent,
        )

    def to_numpy(self, x):
        return (
            x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        )

    def test_PointParticle_Fluorescence(self):

        scatterer = scatterers.PointParticle(
            intensity=100,
            position_unit="pixel",
            position=(16, 16),
        )

        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
        )

        output_image = optics(scatterer).resolve()
        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (32, 32, 1))

        arr_np = self.to_numpy(output_image)
        self.assertTrue(np.isfinite(arr_np).all())
        self.assertGreater(arr_np.max(), 0)

    def test_PointParticle_Fluorescence_upscale(self):

        scatterer = scatterers.PointParticle(
            intensity=100,
            position_unit="pixel",
            position=(16, 16),
        )

        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
        )

        optics_upscaled = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
            upscale=3,
        )

        output_image = optics(scatterer).resolve()
        arr_np = self.to_numpy(output_image)

        output_image_upscaled = optics_upscaled(scatterer).resolve()
        self.assertIsInstance(output_image_upscaled, self.array_type)
        self.assertEqual(output_image_upscaled.shape, (32, 32, 1))

        arr_np_upscaled = self.to_numpy(output_image_upscaled)
        self.assertTrue(np.isfinite(arr_np_upscaled).all())
        self.assertGreater(arr_np_upscaled.max(), 0)

        # Peak location should remain stable
        peak = np.unravel_index(
            np.argmax(arr_np[..., 0]), arr_np[..., 0].shape
        )
        peak_upscaled = np.unravel_index(
            np.argmax(arr_np_upscaled[..., 0]),
            arr_np_upscaled[..., 0].shape,
        )
        self.assertEqual(peak, peak_upscaled)

        # Total intensity should remain similar
        self.assertTrue(
            np.isclose(arr_np.sum(), arr_np_upscaled.sum(), rtol=1e-1)
        )

    def test_PointParticle_Fluorescence_upscale_asymmetric(self):

        scatterer = scatterers.PointParticle(
            intensity=100,
            position_unit="pixel",
            position=(16, 16),
        )

        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
        )

        optics_upscaled = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
            upscale=(1, 5, 2),
        )

        output_image = optics(scatterer).resolve()
        arr_np = self.to_numpy(output_image)

        output_image_upscaled = optics_upscaled(scatterer).resolve()
        self.assertIsInstance(output_image_upscaled, self.array_type)
        self.assertEqual(output_image_upscaled.shape, (32, 32, 1))

        arr_np_upscaled = self.to_numpy(output_image_upscaled)
        self.assertTrue(np.isfinite(arr_np_upscaled).all())
        self.assertGreater(arr_np_upscaled.max(), 0)

        # Peak location should remain stable
        peak = np.unravel_index(
            np.argmax(arr_np[..., 0]), arr_np[..., 0].shape
        )
        peak_upscaled = np.unravel_index(
            np.argmax(arr_np_upscaled[..., 0]),
            arr_np_upscaled[..., 0].shape,
        )
        self.assertEqual(peak, peak_upscaled)

        # Total intensity should remain similar
        self.assertTrue(
            np.isclose(arr_np.sum(), arr_np_upscaled.sum(), rtol=1e-1)
        )

    def test_Ellipse_Fluorescence(self):
        scatterer = scatterers.Ellipse(
            intensity=100,
            position=(16, 16),
            position_unit="pixel",
            radius=(3e-6, 2e-6),
            rotation=0.0,
            upsample=3,
        )

        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
        )

        output_image = optics(scatterer).resolve()

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (32, 32, 1))

        arr = self.to_numpy(output_image)
        self.assertTrue(np.isfinite(arr).all())
        self.assertGreater(arr.max(), 0)
        self.assertGreater(arr.sum(), 0)

    def test_Ellipse_Fluorescence_upscale(self):
        scatterer = scatterers.Ellipse(
            intensity=100,
            position=(16, 16),
            position_unit="pixel",
            radius=(3e-6, 2e-6),
            rotation=0.0,
            upsample=1,
        )

        optics1 = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
        )
        optics2 = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
            upscale=3,
        )

        out1 = self.to_numpy(optics1(scatterer).resolve())
        out2 = self.to_numpy(optics2(scatterer).resolve())

        self.assertEqual(out1.shape, (32, 32, 1))
        self.assertEqual(out2.shape, (32, 32, 1))

        self.assertTrue(np.isfinite(out1).all())
        self.assertTrue(np.isfinite(out2).all())

        self.assertGreater(out1.sum(), 0)
        self.assertGreater(out2.sum(), 0)

        self.assertTrue(np.isclose(out1.sum(), out2.sum(), rtol=1e-1))

    def test_Ellipse_Fluorescence_upscale_asymmetric(self):
        scatterer = scatterers.Ellipse(
            intensity=100,
            position=(16, 16),
            position_unit="pixel",
            radius=(3e-6, 2e-6),
            rotation=0.0,
            upsample=1,
        )

        optics1 = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
        )
        optics2 = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
            upscale=(1, 5, 2),
        )

        out1 = self.to_numpy(optics1(scatterer).resolve())
        out2 = self.to_numpy(optics2(scatterer).resolve())

        self.assertEqual(out1.shape, (32, 32, 1))
        self.assertEqual(out2.shape, (32, 32, 1))

        self.assertTrue(np.isfinite(out1).all())
        self.assertTrue(np.isfinite(out2).all())

        self.assertGreater(out1.sum(), 0)
        self.assertGreater(out2.sum(), 0)

        self.assertTrue(np.isclose(out1.sum(), out2.sum(), rtol=1e-1))

    def test_Ellipse_Brightfield(self):
        scatterer = scatterers.Ellipse(
            refractive_index=1.45,
            position=(16, 16),
            position_unit="pixel",
            radius=(3e-6, 2e-6),
            rotation=0.0,
            upsample=3,
        )

        optics = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
        )

        with self._suppress_expected_optics_warnings():
            output_image = optics(scatterer).resolve()

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (32, 32, 1))

        arr = self.to_numpy(output_image)
        self.assertTrue(np.isfinite(arr).all())
        self.assertGreater(np.abs(arr).sum(), 0)

    def test_Ellipse_Brightfield_upscale(self):
        scatterer = scatterers.Ellipse(
            refractive_index=1.45,
            position=(16, 16),
            position_unit="pixel",
            radius=(3e-6, 2e-6),
            rotation=0.0,
            upsample=1,
        )

        optics1 = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
        )
        optics2 = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
            upscale=3,
        )

        with self._suppress_expected_optics_warnings():
            out1 = self.to_numpy(optics1(scatterer).resolve())
            out2 = self.to_numpy(optics2(scatterer).resolve())

        self.assertEqual(out1.shape, (32, 32, 1))
        self.assertEqual(out2.shape, (32, 32, 1))

        self.assertTrue(np.isfinite(out1).all())
        self.assertTrue(np.isfinite(out2).all())

        self.assertGreater(np.abs(out1).sum(), 0)
        self.assertGreater(np.abs(out2).sum(), 0)

        self.assertTrue(np.isclose(out1.sum(), out2.sum(), rtol=1e-1))

    def test_Ellipse_Brightfield_upscale_asymmetric(self):
        scatterer = scatterers.Ellipse(
            refractive_index=1.45,
            position=(16, 16),
            position_unit="pixel",
            radius=(3e-6, 2e-6),
            rotation=0.0,
            upsample=1,
        )

        optics1 = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
        )
        optics2 = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
            upscale=(1, 5, 2),
        )

        with self._suppress_expected_optics_warnings():
            out1 = self.to_numpy(optics1(scatterer).resolve())
            out2 = self.to_numpy(optics2(scatterer).resolve())

        self.assertEqual(out1.shape, (32, 32, 1))
        self.assertEqual(out2.shape, (32, 32, 1))

        self.assertTrue(np.isfinite(out1).all())
        self.assertTrue(np.isfinite(out2).all())

        self.assertGreater(np.abs(out1).sum(), 0)
        self.assertGreater(np.abs(out2).sum(), 0)

        self.assertTrue(np.isclose(out1.sum(), out2.sum(), rtol=1e-1))

    def test_Ellipse_Brightfield_warns_projection_approximation(self):
        scatterer = scatterers.Ellipse(
            refractive_index=1.45,
            position=(16, 16),
            position_unit="pixel",
            radius=(3e-6, 2e-6),
            rotation=0.0,
            upsample=3,
        )

        optics = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
        )

        with self.assertWarnsRegex(
            UserWarning,
            r"Brightfield imaging from ScatteredVolume assumes a weak-phase / projection approximation\.",
        ):
            output_image = optics(scatterer).resolve()

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (32, 32, 1))

    def test_Sphere_Fluorescence(self):
        scatterer = scatterers.Sphere(
            intensity=100,
            position=(16, 16),
            position_unit="pixel",
            radius=5e-6,
            upsample=1,
        )

        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
        )

        output_image = optics(scatterer).resolve()

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (32, 32, 1))

        arr = self.to_numpy(output_image)
        self.assertTrue(np.isfinite(arr).all())
        self.assertGreater(arr.max(), 0)
        self.assertGreater(arr.sum(), 0)

    def test_Sphere_Fluorescence_upscale(self):
        scatterer = scatterers.Sphere(
            intensity=100,
            position=(16, 16),
            position_unit="pixel",
            radius=5e-6,
            upsample=1,
        )

        optics1 = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
        )
        optics2 = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
            upscale=3,
        )

        out1 = self.to_numpy(optics1(scatterer).resolve())
        out2 = self.to_numpy(optics2(scatterer).resolve())

        self.assertEqual(out1.shape, (32, 32, 1))
        self.assertEqual(out2.shape, (32, 32, 1))

        self.assertTrue(np.isfinite(out1).all())
        self.assertTrue(np.isfinite(out2).all())

        self.assertGreater(out1.sum(), 0)
        self.assertGreater(out2.sum(), 0)

        self.assertTrue(np.isclose(out1.sum(), out2.sum(), rtol=1e-1))

    def test_Sphere_Fluorescence_upscale_asymmetric(self):
        scatterer = scatterers.Sphere(
            intensity=100,
            position=(16, 16),
            position_unit="pixel",
            radius=5e-6,
            upsample=1,
        )

        optics1 = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
        )
        optics2 = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
            upscale=(1, 5, 2),
        )

        out1 = self.to_numpy(optics1(scatterer).resolve())
        out2 = self.to_numpy(optics2(scatterer).resolve())

        self.assertEqual(out1.shape, (32, 32, 1))
        self.assertEqual(out2.shape, (32, 32, 1))

        self.assertTrue(np.isfinite(out1).all())
        self.assertTrue(np.isfinite(out2).all())

        self.assertGreater(out1.sum(), 0)
        self.assertGreater(out2.sum(), 0)

        self.assertTrue(np.isclose(out1.sum(), out2.sum(), rtol=1e-1))

    def test_Sphere_Brightfield(self):
        scatterer = scatterers.Sphere(
            refractive_index=1.45,
            position=(16, 16),
            position_unit="pixel",
            radius=5e-6,
            upsample=3,
        )

        optics = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
        )

        with self._suppress_expected_optics_warnings():
            output_image = optics(scatterer).resolve()

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (32, 32, 1))

        arr = self.to_numpy(output_image)
        self.assertTrue(np.isfinite(arr).all())
        self.assertGreater(np.abs(arr).sum(), 0)

    def test_Sphere_Brightfield_upscale(self):
        scatterer = scatterers.Sphere(
            refractive_index=1.45,
            position=(16, 16),
            position_unit="pixel",
            radius=5e-6,
            upsample=1,
        )

        optics1 = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
        )
        optics2 = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
            upscale=3,
        )

        with self._suppress_expected_optics_warnings():
            out1 = self.to_numpy(optics1(scatterer).resolve())
            out2 = self.to_numpy(optics2(scatterer).resolve())

        self.assertEqual(out1.shape, (32, 32, 1))
        self.assertEqual(out2.shape, (32, 32, 1))

        self.assertTrue(np.isfinite(out1).all())
        self.assertTrue(np.isfinite(out2).all())

        self.assertGreater(np.abs(out1).sum(), 0)
        self.assertGreater(np.abs(out2).sum(), 0)

        self.assertTrue(np.isclose(out1.sum(), out2.sum(), rtol=1e-1))

    def test_Sphere_Brightfield_upscale_asymmetric(self):
        scatterer = scatterers.Sphere(
            refractive_index=1.45,
            position=(16, 16),
            position_unit="pixel",
            radius=5e-6,
            upsample=1,
        )

        optics1 = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
        )
        optics2 = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
            upscale=(1, 5, 2),
        )

        with self._suppress_expected_optics_warnings():
            out1 = self.to_numpy(optics1(scatterer).resolve())
            out2 = self.to_numpy(optics2(scatterer).resolve())

        self.assertEqual(out1.shape, (32, 32, 1))
        self.assertEqual(out2.shape, (32, 32, 1))

        self.assertTrue(np.isfinite(out1).all())
        self.assertTrue(np.isfinite(out2).all())

        self.assertGreater(np.abs(out1).sum(), 0)
        self.assertGreater(np.abs(out2).sum(), 0)

        self.assertTrue(np.isclose(out1.sum(), out2.sum(), rtol=1e-1))

    def test_Ellipsoid_Fluorescence(self):
        scatterer = scatterers.Ellipsoid(
            intensity=100,
            position=(16, 16),
            position_unit="pixel",
            radius=(5e-6, 3e-6, 2e-6),
            upsample=1,
        )

        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
        )

        with self._suppress_expected_optics_warnings():
            output_image = optics(scatterer).resolve()

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (32, 32, 1))

        arr = self.to_numpy(output_image)
        self.assertTrue(np.isfinite(arr).all())
        self.assertGreater(arr.max(), 0)
        self.assertGreater(arr.sum(), 0)

    def test_Ellipsoid_Fluorescence_upscale(self):
        scatterer = scatterers.Ellipsoid(
            intensity=100,
            position=(16, 16),
            position_unit="pixel",
            radius=(5e-6, 3e-6, 2e-6),
            upsample=1,
        )

        optics1 = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
        )
        optics2 = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
            upscale=3,
        )

        with self._suppress_expected_optics_warnings():
            out1 = self.to_numpy(optics1(scatterer).resolve())
            out2 = self.to_numpy(optics2(scatterer).resolve())

        self.assertEqual(out1.shape, (32, 32, 1))
        self.assertEqual(out2.shape, (32, 32, 1))

        self.assertTrue(np.isfinite(out1).all())
        self.assertTrue(np.isfinite(out2).all())

        self.assertGreater(out1.sum(), 0)
        self.assertGreater(out2.sum(), 0)

        self.assertTrue(np.isclose(out1.sum(), out2.sum(), rtol=1e-1))

    def test_Ellipsoid_Fluorescence_upscale_asymmetric(self):
        scatterer = scatterers.Ellipsoid(
            intensity=100,
            position=(16, 16),
            position_unit="pixel",
            radius=(5e-6, 3e-6, 2e-6),
            upsample=1,
        )

        optics1 = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
        )
        optics2 = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
            upscale=(1, 5, 2),
        )

        with self._suppress_expected_optics_warnings():
            out1 = self.to_numpy(optics1(scatterer).resolve())
            out2 = self.to_numpy(optics2(scatterer).resolve())

        self.assertEqual(out1.shape, (32, 32, 1))
        self.assertEqual(out2.shape, (32, 32, 1))

        self.assertTrue(np.isfinite(out1).all())
        self.assertTrue(np.isfinite(out2).all())

        self.assertGreater(out1.sum(), 0)
        self.assertGreater(out2.sum(), 0)

        self.assertTrue(np.isclose(out1.sum(), out2.sum(), rtol=1e-1))

    def test_Ellipsoid_Brightfield(self):
        scatterer = scatterers.Ellipsoid(
            refractive_index=1.45,
            position=(16, 16),
            position_unit="pixel",
            radius=(5e-6, 3e-6, 2e-6),
            upsample=3,
        )

        optics = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
        )

        with self._suppress_expected_optics_warnings():
            output_image = optics(scatterer).resolve()

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (32, 32, 1))

        arr = self.to_numpy(output_image)
        self.assertTrue(np.isfinite(arr).all())
        self.assertGreater(np.abs(arr).sum(), 0)

    def test_Ellipsoid_Brightfield_upscale(self):
        scatterer = scatterers.Ellipsoid(
            refractive_index=1.45,
            position=(16, 16),
            position_unit="pixel",
            radius=(5e-6, 3e-6, 2e-6),
            upsample=1,
        )

        optics1 = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
        )
        optics2 = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
            upscale=3,
        )

        with self._suppress_expected_optics_warnings():
            out1 = self.to_numpy(optics1(scatterer).resolve())
            out2 = self.to_numpy(optics2(scatterer).resolve())

        self.assertEqual(out1.shape, (32, 32, 1))
        self.assertEqual(out2.shape, (32, 32, 1))

        self.assertTrue(np.isfinite(out1).all())
        self.assertTrue(np.isfinite(out2).all())

        self.assertGreater(np.abs(out1).sum(), 0)
        self.assertGreater(np.abs(out2).sum(), 0)

        self.assertTrue(np.isclose(out1.sum(), out2.sum(), rtol=1e-1))

    def test_Ellipsoid_Brightfield_upscale_asymmetric(self):
        scatterer = scatterers.Ellipsoid(
            refractive_index=1.45,
            position=(16, 16),
            position_unit="pixel",
            radius=(5e-6, 3e-6, 2e-6),
            upsample=1,
        )

        optics1 = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
        )
        optics2 = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
            upscale=(1, 5, 2),
        )

        with self._suppress_expected_optics_warnings():
            out1 = self.to_numpy(optics1(scatterer).resolve())
            out2 = self.to_numpy(optics2(scatterer).resolve())

        self.assertEqual(out1.shape, (32, 32, 1))
        self.assertEqual(out2.shape, (32, 32, 1))

        self.assertTrue(np.isfinite(out1).all())
        self.assertTrue(np.isfinite(out2).all())

        self.assertGreater(np.abs(out1).sum(), 0)
        self.assertGreater(np.abs(out2).sum(), 0)

        self.assertTrue(np.isclose(out1.sum(), out2.sum(), rtol=1e-1))

    def test_Darkfield_ScatteredVolume_warns_rough_approximation(self):
        scatterer = scatterers.Ellipse(
            refractive_index=1.45,
            position=(16, 16),
            position_unit="pixel",
            radius=(3e-6, 2e-6),
            rotation=0.0,
            upsample=3,
        )

        optics = Darkfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
        )

        with self.assertWarnsRegex(
            UserWarning,
            r"Darkfield imaging from ScatteredVolume is a very rough approximation\.",
        ):
            output_image = optics(scatterer).resolve()

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (32, 32, 1))

    def test_Darkfield_refractive_index_warns_nonphysical_contrast(self):
        scatterer = scatterers.Ellipse(
            refractive_index=1.45,
            position=(16, 16),
            position_unit="pixel",
            radius=(3e-6, 2e-6),
            rotation=0.0,
            upsample=3,
        )

        optics = Darkfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 32, 32),
        )

        with self.assertWarnsRegex(
            UserWarning,
            r"Approximating darkfield contrast from refractive index\.",
        ):
            output_image = optics(scatterer).resolve()

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (32, 32, 1))


class TestScatterers_NumPy_Only(BackendTestBase):
    BACKEND = "numpy"

    def test_MieSphere_Brightfield(self):
        optics = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
            return_field=True,
        )

        scatterer = scatterers.MieSphere(
            radius=0.5e-6,
            refractive_index=1.45 + 0.1j,
            input_polarization=0.0,
            output_polarization=0.0,
            mode="geometric",
        )

        out = optics(scatterer).resolve()

        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (64, 64, 1))

        arr = out
        self.assertTrue(np.isfinite(arr.real).all())
        self.assertTrue(np.isfinite(arr.imag).all())
        self.assertGreater(np.abs(arr).sum(), 0)

    def test_MieSphere_Brightfield_modes(self):
        optics = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
            return_field=True,
        )

        common = dict(
            radius=0.5e-6,
            refractive_index=1.45 + 0.1j,
            input_polarization=0.0,
            output_polarization=0.0,
        )

        out_geom = optics(
            scatterers.MieSphere(mode="geometric", **common)
        ).resolve()
        out_hybrid = optics(
            scatterers.MieSphere(mode="hybrid", **common)
        ).resolve()

        self.assertEqual(out_geom.shape, (64, 64, 1))
        self.assertEqual(out_hybrid.shape, (64, 64, 1))

        self.assertTrue(np.isfinite(out_geom.real).all())
        self.assertTrue(np.isfinite(out_geom.imag).all())
        self.assertTrue(np.isfinite(out_hybrid.real).all())
        self.assertTrue(np.isfinite(out_hybrid.imag).all())

        self.assertGreater(np.abs(out_geom).sum(), 0)
        self.assertGreater(np.abs(out_hybrid).sum(), 0)

    def test_MieStratifiedSphere_Brightfield(self):
        optics = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
            return_field=True,
        )

        scatterer = scatterers.MieStratifiedSphere(
            radius=(0.5e-6, 1.0e-6),
            refractive_index=(1.45 + 0.1j, 1.52),
            input_polarization=0.0,
            output_polarization=0.0,
            mode="hybrid",
        )

        out = optics(scatterer).resolve()

        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (64, 64, 1))

        arr = out
        self.assertTrue(np.isfinite(arr.real).all())
        self.assertTrue(np.isfinite(arr.imag).all())
        self.assertGreater(np.abs(arr).sum(), 0)

    def test_Incoherent_MieSphere_Brightfield(self):
        optics = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 64, 64),
            return_field=True,
        )

        scatterer = scatterers.Incoherent(
            scatterers.MieSphere(
                radius=0.5e-6,
                refractive_index=1.45 + 0.1j,
            ),
            input_unpolarized=True,
            output_unpolarized=True,
        )

        out = optics(scatterer).resolve()
        arr = out

        self.assertEqual(arr.shape, (64, 64, 1))
        self.assertTrue(np.isfinite(arr).all())
        self.assertGreater(arr.sum(), 0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
class TestScatterers_Torch(TestScatterers_NumPy):
    BACKEND = "torch"


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
class TestMath_TorchOnly(BackendTestBase):
    BACKEND = "torch"

    def test_point_particle_intensity_gradient(self):

        # --- PointParticle intensity optimization ---
        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=4,
            output_region=(0, 0, 32, 32),
        )
        # target
        true_intensity = 2.0
        particle = scatterers.PointParticle(
            position=(16, 16),
            intensity=true_intensity,
        )
        target = optics(particle).update()().detach()
        # learnable parameter
        intensity = torch.tensor(0.5, requires_grad=True)
        # scatterer with learnable intensity
        particle = scatterers.PointParticle(
            position=(16, 16),
            intensity=intensity,
        )
        optimizer = torch.optim.Adam([intensity], lr=0.1)
        pipeline = optics(particle)
        prev_loss = None
        for _ in range(20):
            optimizer.zero_grad()
            image = pipeline.update()()
            loss = ((image - target) ** 2).mean()
            loss.backward()
            optimizer.step()
            self.assertIsNotNone(intensity.grad)
            if prev_loss is not None:
                self.assertNotEqual(loss.item(), prev_loss)
            prev_loss = loss.item()
        self.assertTrue(abs(intensity.item() - true_intensity) < 0.5)


if __name__ == "__main__":
    unittest.main()
