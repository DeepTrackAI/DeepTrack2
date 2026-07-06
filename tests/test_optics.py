import unittest

import numpy as np
import warnings
from contextlib import contextmanager

from deeptrack.optical import optics
from deeptrack.optical.scatterers import PointParticle, Sphere, MieSphere
from deeptrack import units_registry as u

from deeptrack.backend import TORCH_AVAILABLE, xp
from tests import BackendTestBase

if TORCH_AVAILABLE:
    import torch


class TestOptics_NumPy(BackendTestBase):
    BACKEND = "numpy"

    _EXPECTED_OPTICS_WARNING_PATTERNS = (
        r"Brightfield imaging from ScatteredVolume assumes a weak-phase / projection approximation.*",
        r"Darkfield imaging from ScatteredVolume is a very rough approximation.*",
        r"Approximating darkfield contrast from refractive index.*",
        r"Fluorescence scatterer has no 'intensity'.*",
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

    def test_Microscope(self):
        microscope_type = optics.Fluorescence()
        scatterer = PointParticle(intensity=100)
        microscope = optics.Microscope(
            sample=scatterer,
            objective=microscope_type,
        )
        output_image = microscope.get(None)
        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (128, 128, 1))

    def test_Optics(self):
        microscope = optics.Optics()
        scatterer = PointParticle()
        image = microscope(scatterer)
        self.assertIsInstance(image, optics.Microscope)

    def test_Fluorescence(self):
        microscope = optics.Fluorescence(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            padding=(10, 10, 10, 10),
            output_region=(0, 0, 64, 64),
        )
        scatterer = PointParticle(
            intensity=100,
            position_unit="pixel",
            position=(32, 32),
        )
        output_image = microscope(scatterer).resolve()

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (64, 64, 1))
        self.assertEqual(microscope.NA(), 0.7)

        img = output_image[..., 0]
        peak = np.unravel_index(int(xp.argmax(img)), img.shape)
        self.assertLessEqual(abs(peak[0] - 32), 1)
        self.assertLessEqual(abs(peak[1] - 32), 1)

    def test_Brightfield(self):
        microscope = optics.Brightfield(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )
        scatterer = PointParticle(
            refractive_index=1.45 + 0.1j,
            position_unit="pixel",
            position=(32, 32),
        )
        imaged_scatterer = microscope(scatterer)

        with self._suppress_expected_optics_warnings():
            output_image = imaged_scatterer.resolve()
        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (64, 64, 1))

    def test_Holography(self):
        microscope = optics.Holography(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )
        scatterer = PointParticle(
            refractive_index=1.45 + 0.1j,
            position_unit="pixel",
            position=(32, 32),
        )
        imaged_scatterer = microscope(scatterer)
        with self._suppress_expected_optics_warnings():
            output_image = imaged_scatterer.resolve()
        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (64, 64, 1))

    def test_Brightfield_Holography_equivalence(self):
        bf = optics.Brightfield(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )
        hg = optics.Holography(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )

        scatterer = PointParticle(
            refractive_index=1.45 + 0.1j,
            position_unit="pixel",
            position=(32, 32),
        )

        with self._suppress_expected_optics_warnings():
            img_bf = bf(scatterer).resolve()
            img_hg = hg(scatterer).resolve()

        err = float(xp.mean(xp.abs(img_bf - img_hg)))
        self.assertLess(err, 1e-10)

    def test_ISCAT(self):
        microscope = optics.ISCAT(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )
        scatterer = PointParticle(
            refractive_index=1.45 + 0.1j,
            position_unit="pixel",
            position=(32, 32),
        )
        imaged_scatterer = microscope(scatterer)

        with self._suppress_expected_optics_warnings():
            output_image = imaged_scatterer.resolve()

        self.assertEqual(microscope.illumination_angle(), 3.141592653589793)
        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (64, 64, 1))
        self.assertEqual(microscope.amp_factor(), 1)

    def _rot90_asymmetry_ratio(self, image: np.ndarray) -> float:
            """Returns max|img - rot90(img)| / std(img), a scale-independent
            measure of how far an image is from being rotationally symmetric.
            """
            img = np.asarray(image).squeeze()
            std = img.std()
            if std == 0:
                return 0.0
            return float(np.abs(img - np.rot90(img)).max() / std)

    def test_ISCAT_MieSphere_no_analyzer_is_symmetric(self):

        common_kwargs = dict(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 64, 64),
            padding=(32, 32, 32, 32),
        )
        scatterer_common = dict(
            position=(32, 32, 2),
            position_unit="pixel",
            radius=2e-6,
            refractive_index=1.42,
            phase_shift_correction=True,
            L=10,
        )

        microscope_no_analyzer = optics.ISCAT(**common_kwargs)

        microscope_fixed_linear = optics.ISCAT(
            input_polarization=0.0,
            output_polarization=0.0,
            **common_kwargs,
        )

        no_analyzer = MieSphere(
            **scatterer_common,
        )
        fixed_linear = MieSphere(
            **scatterer_common,
        )

        with self._suppress_expected_optics_warnings():
            img_no_analyzer = microscope_no_analyzer(no_analyzer).resolve()
            img_fixed_linear = microscope_fixed_linear(
                fixed_linear
            ).resolve()

        self.assertIsInstance(img_no_analyzer, self.array_type)
        self.assertEqual(img_no_analyzer.shape, (64, 64, 1))

        ratio_no_analyzer = self._rot90_asymmetry_ratio(img_no_analyzer)
        ratio_fixed_linear = self._rot90_asymmetry_ratio(img_fixed_linear)

        self.assertLess(ratio_no_analyzer, 3.0)
        self.assertLess(ratio_no_analyzer, ratio_fixed_linear / 3.0)

    def test_Holography_MieSphere_no_analyzer_is_symmetric(self):
        common_kwargs = dict(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 64, 64),
            padding=(32, 32, 32, 32),
            return_field=True,
        )
        scatterer_common = dict(
            position=(32, 32, 2),
            position_unit="pixel",
            radius=2e-6,
            refractive_index=1.42,
            L=10,
        )

        microscope = optics.Holography(**common_kwargs)

        no_analyzer = MieSphere(
            input_polarization=None,
            output_polarization=None,
            **scatterer_common,
        )
        fixed_linear = MieSphere(
            input_polarization=0.0,
            output_polarization=0.0,
            **scatterer_common,
        )

        with self._suppress_expected_optics_warnings():
            field_no_analyzer = np.asarray(
                microscope(no_analyzer).resolve()
            ).squeeze()
            field_fixed_linear = np.asarray(
                microscope(fixed_linear).resolve()
            ).squeeze()

        self.assertTrue(np.iscomplexobj(field_no_analyzer))

        for component in ("real", "imag"):
            ratio_no_analyzer = self._rot90_asymmetry_ratio(
                getattr(field_no_analyzer, component)
            )
            ratio_fixed_linear = self._rot90_asymmetry_ratio(
                getattr(field_fixed_linear, component)
            )
            self.assertLess(ratio_no_analyzer, 3.0)
            self.assertLessEqual(ratio_no_analyzer, ratio_fixed_linear)

    def test_ISCAT_Holography_MieSphere_no_analyzer_equivalence(self):

        common = dict(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 64, 64),
            padding=(32, 32, 32, 32),
        )
        scatterer_kwargs = dict(
            position=(32, 32, 2),
            position_unit="pixel",
            radius=2e-6,
            refractive_index=1.42,
            input_polarization=None,
            output_polarization=None,
            L=10,
        )

        holography = optics.Holography(
            phase_shift_correction=True,
            illumination_angle=np.pi,
            return_field=True,
            **common,
        )
        iscat = optics.ISCAT(
            return_field=True,
            **common,
        )

        with self._suppress_expected_optics_warnings():
            img_holo = holography(MieSphere(**scatterer_kwargs)).resolve()
            img_iscat = iscat(MieSphere(**scatterer_kwargs)).resolve()

        err = float(xp.mean(xp.abs(img_holo - img_iscat)))
        self.assertLess(err, 1e-10)

    def test_Darkfield(self):
        microscope = optics.Darkfield(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )
        scatterer = PointParticle(
            refractive_index=1.45 + 0.1j,
            position_unit="pixel",
            position=(32, 32),
        )
        imaged_scatterer = microscope(scatterer)

        with self._suppress_expected_optics_warnings():
            output_image = imaged_scatterer.resolve()

        self.assertEqual(microscope.illumination_angle(), 1.5707963267948966)
        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (64, 64, 1))

    def test_IlluminationGradient(self):
        illumination_gradient = optics.IlluminationGradient(
            gradient=(5e-5, 5e-5)
        )
        microscope = optics.Brightfield(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
            illumination=illumination_gradient,
        )
        scatterer = PointParticle(
            refractive_index=1.45 + 0.1j,
            position_unit="pixel",
            position=(32, 32),
        )
        imaged_scatterer = microscope(scatterer)
        with self._suppress_expected_optics_warnings():
            output_image = imaged_scatterer.resolve()
        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (64, 64, 1))

    def test_upscale_Brightfield(self):
        microscope = optics.Brightfield(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=5,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )
        scatterer = Sphere(
            refractive_index=1.45,
            radius=1e-6,
            z=2 * u.um,
            position_unit="pixel",
            position=(32, 32),
        )

        imaged_scatterer = microscope(scatterer)

        with self._suppress_expected_optics_warnings():
            output_image_no_upscale = imaged_scatterer.update()(upscale=1)
            output_image_2x_upscale = imaged_scatterer.update()(
                upscale=(2, 2, 1)
            )

        self.assertEqual(output_image_no_upscale.shape, (64, 64, 1))
        self.assertEqual(output_image_2x_upscale.shape, (64, 64, 1))
        # Ensure the upscaled image is almost the same as the original image

        rel_error = xp.abs(
            output_image_2x_upscale - output_image_no_upscale
        ).mean() / xp.mean(
            output_image_no_upscale
        )  # Mean relative error
        self.assertLess(rel_error, 0.1)

    def test_upscale_fluorescence(self):
        microscope = optics.Fluorescence(
            NA=0.5,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )
        scatterer = Sphere(
            intensity=100,
            radius=1e-6,
            z=2 * u.um,
            position_unit="pixel",
            position=(32, 32),
        )

        imaged_scatterer = microscope(scatterer)
        output_image_no_upscale = imaged_scatterer.update()(upscale=1)

        output_image_2x_upscale = imaged_scatterer.update()(upscale=(2, 2, 1))

        self.assertEqual(output_image_no_upscale.shape, (64, 64, 1))
        self.assertEqual(output_image_2x_upscale.shape, (64, 64, 1))
        # Ensure the upscaled image is almost the same as the original image

        rel_error = xp.abs(
            output_image_2x_upscale - output_image_no_upscale
        ).mean() / xp.mean(
            output_image_no_upscale
        )  # Mean relative error
        self.assertLess(rel_error, 0.1)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
class TestOptics_PyTorch(TestOptics_NumPy):
    BACKEND = "torch"


if __name__ == "__main__":
    unittest.main()